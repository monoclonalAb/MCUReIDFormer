import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

from lib import utils
from lib.config import cfg, update_config_from_file
from lib.datasets import build_dataset
from model.supernet_transformer_ReID import Vision_TransformerSuper
from supernet_ReID_engine import (
    CenterLoss,
    TripletLoss,
    evaluate_reid,
    train_one_epoch_reid,
)


def get_args_parser():
    parser = argparse.ArgumentParser("AutoFormer ReID training", add_help=False)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=120, type=int)
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )

    # ReID specific parameters
    parser.add_argument(
        "--reid-dim", default=256, type=int, help="ReID embedding dimension"
    )
    parser.add_argument(
        "--triplet-margin", default=0.3, type=float, help="Margin for triplet loss"
    )
    parser.add_argument(
        "--center-loss-weight", default=0.0005, type=float, help="Center loss weight"
    )
    parser.add_argument(
        "--label-smooth", default=0.1, type=float, help="Label smoothing"
    )

    # Model parameters
    parser.add_argument(
        "--mode", type=str, default="retrain", choices=["super", "retrain"]
    )
    parser.add_argument("--input-size", default=256, type=int, help="Input image size")
    parser.add_argument("--patch_size", default=16, type=int)

    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT")

    parser.add_argument("--relative_position", action="store_true")
    parser.add_argument("--gp", action="store_true", help="Use global pooling")
    parser.add_argument("--change_qkv", action="store_true")
    parser.add_argument("--max_relative_position", type=int, default=14)
    parser.add_argument("--no_abs_pos", action="store_true")

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
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.set_defaults(amp=True)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    update_config_from_file(args.cfg)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Build datasets
    print("Building training dataset...")
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    print("Building query dataset...")
    args_query = argparse.Namespace(**vars(args))
    dataset_query, _ = build_dataset(
        is_train=False, args=args_query, folder_name="query"
    )

    print("Building gallery dataset...")
    args_gallery = argparse.Namespace(**vars(args))
    dataset_gallery, _ = build_dataset(
        is_train=False, args=args_gallery, folder_name="gallery"
    )

    print(
        f"Train samples: {len(dataset_train)}, Query: {len(dataset_query)}, Gallery: {len(dataset_gallery)}"
    )
    print(f"Number of identities: {args.nb_classes}")

    # Create data loaders
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Create model
    print(f"Creating ReID Vision Transformer")
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

    # Setup optimizer
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # Setup losses
    criterion_id = LabelSmoothingCrossEntropy(smoothing=args.label_smooth)
    criterion_triplet = TripletLoss(margin=args.triplet_margin)
    criterion_center = CenterLoss(num_classes=args.nb_classes, feat_dim=args.reid_dim)
    criterion_center = criterion_center.to(device)

    # Center loss optimizer
    center_optimizer = torch.optim.SGD(criterion_center.parameters(), lr=0.5)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        f.write(args_text)

    # Resume from checkpoint
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

    # Evaluation only
    if args.eval:
        test_stats = evaluate_reid(
            data_loader_query,
            data_loader_gallery,
            model,
            device,
            mode=args.mode,
            retrain_config=retrain_config,
        )
        return

    # Training
    print("Start training")
    start_time = time.time()
    best_mAP = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_reid(
            model,
            criterion_id,
            criterion_triplet,
            criterion_center,
            data_loader_train,
            optimizer,
            center_optimizer,
            device,
            epoch,
            args.clip_grad,
            amp=args.amp,
            choices=choices,
            mode=args.mode,
            retrain_config=retrain_config,
        )

        lr_scheduler.step(epoch)

        # Evaluate every 10 epochs or at the end
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            test_stats = evaluate_reid(
                data_loader_query,
                data_loader_gallery,
                model,
                device,
                amp=args.amp,
                choices=choices,
                mode=args.mode,
                retrain_config=retrain_config,
            )

            is_best = test_stats["mAP"] > best_mAP
            best_mAP = max(best_mAP, test_stats["mAP"])

            print(f"Best mAP: {best_mAP:.2%}")

            # Save checkpoint
            if args.output_dir:
                checkpoint_path = output_dir / "checkpoint.pth"
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "mAP": test_stats["mAP"],
                    },
                    checkpoint_path,
                )

                if is_best:
                    best_path = output_dir / "best_model.pth"
                    utils.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "mAP": test_stats["mAP"],
                            "epoch": epoch,
                        },
                        best_path,
                    )

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
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    print(f"Best mAP: {best_mAP:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "AutoFormer ReID training", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
