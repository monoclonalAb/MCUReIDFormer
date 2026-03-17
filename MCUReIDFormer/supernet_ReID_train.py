import argparse
import datetime
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler

from lib import utils
from lib.config import cfg, update_config_from_file
from lib.config_new import cfg_new, update_config_new_from_file
from lib.datasets import build_dataset
from lib.mcu_constraints import estimate_peak_sram_kb, fits_sram
from lib.multi_supernet import make_list, evolution_supernet
from model.supernet_transformer_ReID import Vision_TransformerSuper
from supernet_ReID_engine import (
    CenterLoss,
    TripletLoss,
    evaluate_reid,
    train_one_epoch_reid,
    sample_configs,
)


def sample_sram_legal_ratio(model, img_size, patch_size, rank_ratio, sample_num,
                            choices, sram_budget_kb):
    """Check what fraction of random subnets fit within the SRAM budget."""
    legal = 0
    for _ in range(sample_num):
        config = sample_configs(choices)
        sram_kb = estimate_peak_sram_kb(config, img_size, patch_size, rank_ratio)
        if sram_kb <= sram_budget_kb:
            legal += 1
    return legal / sample_num


def sample_average_mAP(model, sample_num, choices, device,
                       query_loader, gallery_loader, amp, output_path):
    """Sample random subnets and compute average mAP."""
    total_mAP = 0
    for i in range(sample_num):
        config = sample_configs(choices)
        stats = evaluate_reid(query_loader, gallery_loader, model, device,
                              amp=amp, mode='retrain', retrain_config=config)
        total_mAP += stats['mAP']
        print(f"  Sample {i+1}/{sample_num}: mAP={stats['mAP']:.4f}")
    avg = total_mAP / sample_num
    print(f"  Average mAP: {avg:.4f}")
    return avg


def get_args_parser():
    parser = argparse.ArgumentParser("Hardware-Aware ReIDFormer training", add_help=False)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument("--cfg", help="architecture config file", required=True, type=str)
    parser.add_argument("--cfg-new", help="hardware search space config file", default="", type=str)

    # ReID specific parameters
    parser.add_argument("--reid-dim", default=256, type=int, help="ReID embedding dimension")
    parser.add_argument("--triplet-margin", default=0.3, type=float, help="Margin for triplet loss")
    parser.add_argument("--center-loss-weight", default=0.0005, type=float, help="Center loss weight")
    parser.add_argument("--label-smooth", default=0.1, type=float, help="Label smoothing")

    # Model parameters
    parser.add_argument("--mode", type=str, default="super", choices=["super", "retrain"])
    parser.add_argument("--input-size", default=256, type=int, help="Input image size")
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT")

    parser.add_argument("--relative_position", action="store_true")
    parser.add_argument("--gp", action="store_true", help="Use global pooling")
    parser.add_argument("--change_qkv", action="store_true")
    parser.add_argument("--max_relative_position", type=int, default=14)
    parser.add_argument("--no_abs_pos", action="store_true")

    # Hardware-aware parameters
    parser.add_argument("--rank-ratio", type=float, default=0.9,
                        help="Low-rank decomposition ratio (default for single-level)")
    parser.add_argument("--sram-budget", type=float, default=320,
                        help="SRAM budget in KB (default: 320 for STM32F746)")
    parser.add_argument("--two-level", action="store_true",
                        help="Enable two-level search (outer: rank_ratio/patch_size)")
    parser.add_argument("--super-epoch", default=30, type=int,
                        help="Epochs per supernet in two-level search")
    parser.add_argument("--step-num", default=3, type=int,
                        help="Number of outer search space evolution steps")
    parser.add_argument("--sample-num", default=10, type=int,
                        help="Number of subnets to sample for evaluation")

    # Optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER")
    parser.add_argument("--opt-eps", default=1e-8, type=float, metavar="EPSILON")
    parser.add_argument("--clip-grad", type=float, default=1.0, metavar="NORM")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight-decay", type=float, default=0.0001)

    # Learning rate parameters
    parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER")
    parser.add_argument("--lr", type=float, default=3.5e-4, metavar="LR")
    parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--min-lr", type=float, default=1e-6, metavar="LR")
    parser.add_argument("--warmup-epochs", type=int, default=10, metavar="N")
    parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N")

    # Augmentation parameters
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--train-interpolation", type=str, default="bicubic")
    parser.add_argument("--reprob", type=float, default=0.5)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)

    # Dataset parameters
    parser.add_argument("--data-path", default="./data/ATRW/", type=str)
    parser.add_argument("--data-set", default="ATRW", type=str)
    parser.add_argument("--output_dir", default="./output_reid/", help="path to save")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.set_defaults(pin_mem=True)

    # Distributed training
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dist-eval", action="store_true", default=False)
    parser.set_defaults(amp=True)

    return parser


def train_single_supernet(args, cfg, device, data_loader_train, data_loader_query,
                          data_loader_gallery, cur_rank_ratio, cur_patch_size, output_path):
    """Train a single supernet with given (rank_ratio, patch_size) and return best mAP."""

    print(f"\n{'='*60}")
    print(f"Training supernet: rank_ratio={cur_rank_ratio}, patch_size={cur_patch_size}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    model = Vision_TransformerSuper(
        img_size=args.input_size,
        patch_size=cur_patch_size,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        rank_ratio=cur_rank_ratio,
        reid=True,
        reid_dim=args.reid_dim,
    )

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if this (rank_ratio, patch_size) was trained before
    checkpoint_path = output_path / "checkpoint.pth"
    start_epoch = 0
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    elif args.resume:
        # Load pretrained weights for new supernets
        if Path(args.resume).is_file():
            ckpt = torch.load(args.resume, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            # Filter incompatible keys
            filtered = {k: v for k, v in state_dict.items()
                       if not (k.startswith("head.") or k.startswith("pos_embed"))}
            missing, unexpected = model_without_ddp.load_state_dict(filtered, strict=False)
            print(f"Loaded pretrained weights: {len(missing)} missing, {len(unexpected)} unexpected")

    # Setup losses
    criterion_id = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)
    criterion_triplet = TripletLoss(margin=args.triplet_margin)
    criterion_center = CenterLoss(num_classes=args.nb_classes, feat_dim=args.reid_dim)
    criterion_center = criterion_center.to(device)
    center_optimizer = torch.optim.SGD(criterion_center.parameters(), lr=0.5)

    # Train for super_epoch epochs
    end_epoch = start_epoch + args.super_epoch
    best_mAP = 0.0

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_reid(
            model, criterion_id, criterion_triplet, criterion_center,
            data_loader_train, optimizer, center_optimizer,
            device, epoch, args.clip_grad,
            amp=args.amp, choices=choices, mode="super",
        )

        lr_scheduler.step(epoch)

        # Evaluate at the end of this supernet's training block
        if epoch == end_epoch - 1:
            test_stats = evaluate_reid(
                data_loader_query, data_loader_gallery,
                model, device, amp=args.amp,
                choices=choices, mode="super",
            )
            best_mAP = test_stats["mAP"]
            print(f"  mAP at epoch {epoch}: {best_mAP:.4f}")

        # Save checkpoint
        utils.save_on_master({
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }, checkpoint_path)

    # Compute SRAM fit ratio
    legal_ratio = sample_sram_legal_ratio(
        model, args.input_size, cur_patch_size, cur_rank_ratio,
        args.sample_num, choices, args.sram_budget
    )
    print(f"  SRAM legal ratio: {legal_ratio:.2%}")

    return best_mAP, legal_ratio, end_epoch


def main_two_level(args):
    """Two-level search: outer loop evolves (rank_ratio, patch_size),
    inner loop trains supernets and evaluates."""

    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)
    update_config_new_from_file(args.cfg_new)

    print(args)
    print(cfg)
    print(cfg_new)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    random.seed(args.seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    args_query = argparse.Namespace(**vars(args))
    dataset_query, _ = build_dataset(is_train=False, args=args_query, folder_name="query")
    args_gallery = argparse.Namespace(**vars(args))
    dataset_gallery, _ = build_dataset(is_train=False, args=args_gallery, folder_name="gallery")

    print(f"Train: {len(dataset_train)}, Query: {len(dataset_query)}, Gallery: {len(dataset_gallery)}")
    print(f"Identities: {args.nb_classes}")

    # Create data loaders
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # Build index lists for the search space
    rank_ratio_choices = list(range(1, len(cfg_new.SEARCH_SPACE.RANK_RATIO) + 1))
    patch_size_choices = list(range(1, len(cfg_new.SEARCH_SPACE.PATCH_SIZE) + 1))
    all_combinations = make_list(rank_ratio_choices, patch_size_choices)

    # Sample initial (rank_ratio, patch_size) combinations
    n_initial = min(5, len(all_combinations))
    selected = random.sample(all_combinations, n_initial)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    all_results = []  # [(rank_ratio, patch_size, mAP, sram_legal_ratio)]

    for step in range(args.step_num):
        print(f"\n{'#'*80}")
        print(f"OUTER STEP {step + 1}/{args.step_num}")
        print(f"{'#'*80}")

        step_results = []

        for idx, (rr_idx, ps_idx) in enumerate(selected):
            cur_rank_ratio = cfg_new.SEARCH_SPACE.RANK_RATIO[rr_idx - 1]
            cur_patch_size = cfg_new.SEARCH_SPACE.PATCH_SIZE[ps_idx - 1]

            supernet_dir = output_dir / f"supernet_{cur_rank_ratio}_{cur_patch_size}"

            best_mAP, legal_ratio, end_epoch = train_single_supernet(
                args, cfg, device, data_loader_train,
                data_loader_query, data_loader_gallery,
                cur_rank_ratio, cur_patch_size, supernet_dir
            )

            result = [cur_rank_ratio, cur_patch_size, best_mAP * 100, legal_ratio]
            step_results.append(result)
            all_results.append(result)

            print(f"  Result: rank_ratio={cur_rank_ratio}, patch_size={cur_patch_size}, "
                  f"mAP={best_mAP:.4f}, SRAM_legal={legal_ratio:.2%}")

        # Log step results
        log_path = output_dir / "two_level_log.txt"
        with open(log_path, "a") as f:
            f.write(f"\n--- Step {step + 1} ---\n")
            for r in step_results:
                f.write(f"rank_ratio={r[0]}, patch_size={r[1]}, "
                        f"mAP={r[2]:.2f}, SRAM_legal={r[3]:.2f}\n")

        # Evolve search space using accumulated results
        if len(all_results) >= 3 and step < args.step_num - 1:
            # Use the best current (rank_ratio, patch_size) as the anchor
            best_result = max(all_results, key=lambda x: x[2] * x[3])
            anchor_rr = best_result[0]
            anchor_ps = best_result[1]

            evolver = evolution_supernet(all_results, anchor_rr, anchor_ps)
            step_delta = evolver.evolution_step()

            new_rr = anchor_rr + step_delta[0]
            new_ps = int(anchor_ps + step_delta[1])

            # Snap to nearest valid choices
            rr_list = cfg_new.SEARCH_SPACE.RANK_RATIO
            ps_list = cfg_new.SEARCH_SPACE.PATCH_SIZE
            new_rr = min(rr_list, key=lambda x: abs(x - new_rr))
            new_ps = min(ps_list, key=lambda x: abs(x - new_ps))

            new_rr_idx = rr_list.index(new_rr) + 1
            new_ps_idx = ps_list.index(new_ps) + 1

            # Build new selection around the evolved point
            new_selected = [(new_rr_idx, new_ps_idx)]
            # Add neighbors
            for dr in [-1, 0, 1]:
                for dp in [-1, 0, 1]:
                    ri = new_rr_idx + dr
                    pi = new_ps_idx + dp
                    if 1 <= ri <= len(rr_list) and 1 <= pi <= len(ps_list):
                        if (ri, pi) not in new_selected:
                            new_selected.append((ri, pi))

            selected = new_selected[:n_initial]
            print(f"\nEvolved search space -> anchor: rank_ratio={new_rr}, patch_size={new_ps}")
            print(f"Next combinations: {[(rr_list[r-1], ps_list[p-1]) for r, p in selected]}")
        else:
            # Re-sample for next step
            selected = random.sample(all_combinations, n_initial)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Two-level search complete in {total_time/3600:.2f} hours")
    print(f"{'='*80}")

    best = max(all_results, key=lambda x: x[2] * x[3])
    print(f"Best: rank_ratio={best[0]}, patch_size={best[1]}, "
          f"mAP={best[2]:.2f}%, SRAM_legal={best[3]:.2%}")

    # Save best
    with open(output_dir / "best_supernet.yaml", "w") as f:
        yaml.dump({
            "SUPERNET": {
                "RANK_RATIO": float(best[0]),
                "PATCH_SIZE": int(best[1]),
            },
            "RESULTS": {
                "mAP": float(best[2]),
                "SRAM_legal_ratio": float(best[3]),
            }
        }, f)


def main_single_level(args):
    """Original single-level training with rank_ratio support."""

    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build datasets
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    args_query = argparse.Namespace(**vars(args))
    dataset_query, _ = build_dataset(is_train=False, args=args_query, folder_name="query")
    args_gallery = argparse.Namespace(**vars(args))
    dataset_gallery, _ = build_dataset(is_train=False, args=args_gallery, folder_name="gallery")

    print(f"Train: {len(dataset_train)}, Query: {len(dataset_query)}, Gallery: {len(dataset_gallery)}")
    print(f"Identities: {args.nb_classes}")

    # Create data loaders
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)

    # Create model with rank_ratio
    print(f"Creating Hardware-Aware ReID Vision Transformer")
    print(f"  rank_ratio={args.rank_ratio}, patch_size={args.patch_size}")
    print(cfg)

    model = Vision_TransformerSuper(
        img_size=args.input_size,
        patch_size=args.patch_size,
        embed_dim=cfg.SUPERNET.EMBED_DIM,
        depth=cfg.SUPERNET.DEPTH,
        num_heads=cfg.SUPERNET.NUM_HEADS,
        mlp_ratio=cfg.SUPERNET.MLP_RATIO,
        qkv_bias=True,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        rank_ratio=args.rank_ratio,
        reid=True,
        reid_dim=args.reid_dim,
    )

    choices = {
        "num_heads": cfg.SEARCH_SPACE.NUM_HEADS,
        "mlp_ratio": cfg.SEARCH_SPACE.MLP_RATIO,
        "embed_dim": cfg.SEARCH_SPACE.EMBED_DIM,
        "depth": cfg.SEARCH_SPACE.DEPTH,
    }

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion_id = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)
    criterion_triplet = TripletLoss(margin=args.triplet_margin)
    criterion_center = CenterLoss(num_classes=args.nb_classes, feat_dim=args.reid_dim)
    criterion_center = criterion_center.to(device)
    center_optimizer = torch.optim.SGD(criterion_center.parameters(), lr=0.5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        f.write(args_text)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    retrain_config = None
    if args.mode == "retrain" and "RETRAIN" in cfg:
        retrain_config = {
            "layer_num": cfg.RETRAIN.DEPTH,
            "embed_dim": [cfg.RETRAIN.EMBED_DIM] * cfg.RETRAIN.DEPTH,
            "num_heads": cfg.RETRAIN.NUM_HEADS,
            "mlp_ratio": cfg.RETRAIN.MLP_RATIO,
        }

    if args.eval:
        evaluate_reid(data_loader_query, data_loader_gallery, model, device,
                      mode=args.mode, retrain_config=retrain_config)
        return

    # Training
    print("Start training")
    start_time = time.time()
    best_mAP = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_reid(
            model, criterion_id, criterion_triplet, criterion_center,
            data_loader_train, optimizer, center_optimizer,
            device, epoch, args.clip_grad,
            amp=args.amp, choices=choices, mode=args.mode,
            retrain_config=retrain_config,
        )

        lr_scheduler.step(epoch)

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_stats = evaluate_reid(
                data_loader_query, data_loader_gallery,
                model, device, amp=args.amp,
                choices=choices, mode=args.mode,
                retrain_config=retrain_config,
            )

            is_best = test_stats["mAP"] > best_mAP
            best_mAP = max(best_mAP, test_stats["mAP"])
            print(f"Best mAP: {best_mAP:.2%}")

            if args.output_dir:
                utils.save_on_master({
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "mAP": test_stats["mAP"],
                }, output_dir / "checkpoint.pth")

                if is_best:
                    utils.save_on_master({
                        "model": model_without_ddp.state_dict(),
                        "mAP": test_stats["mAP"],
                        "epoch": epoch,
                    }, output_dir / "best_model.pth")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Best mAP: {best_mAP:.2%}")


def main(args):
    if args.two_level:
        if not args.cfg_new:
            raise ValueError("--cfg-new is required for two-level search")
        main_two_level(args)
    else:
        main_single_level(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Hardware-Aware ReIDFormer training", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
