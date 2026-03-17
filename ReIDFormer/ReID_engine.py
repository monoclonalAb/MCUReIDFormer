import math
import random
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils.model import unwrap_model

from lib import utils


def sample_configs(choices):
    config = {}
    dimensions = ["mlp_ratio", "num_heads"]
    depth = random.choice(choices["depth"])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]
    config["embed_dim"] = [random.choice(choices["embed_dim"])] * depth
    config["layer_num"] = depth
    return config


class TripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining"""

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # For each anchor, find hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(torch.nn.Module):
    """Center loss for better feature learning"""

    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / batch_size
        return loss


def train_one_epoch_reid(
    model: torch.nn.Module,
    criterion_id: torch.nn.Module,
    criterion_triplet: TripletLoss,
    criterion_center: CenterLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    center_optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    amp: bool = True,
    choices=None,
    mode="super",
    retrain_config=None,
):
    model.train()
    criterion_id.train()

    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "loss_id", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_tri", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )
    metric_logger.add_meter(
        "loss_cen", utils.SmoothedValue(window_size=1, fmt="{value:.4f}")
    )

    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Sample random config for supernet training
        if mode == "super":
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == "retrain":
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if amp:
            with torch.cuda.amp.autocast():
                # Get both embedding and logits
                embeddings, logits = model(samples, return_feats=True)

                # ID loss (classification)
                loss_id = criterion_id(logits, targets)

                # Triplet loss (metric learning)
                loss_triplet = criterion_triplet(embeddings, targets)

                # Center loss
                loss_center = criterion_center(embeddings, targets)

                # Combined loss
                loss = loss_id + loss_triplet + 0.0005 * loss_center
        else:
            embeddings, logits = model(samples, return_feats=True)
            loss_id = criterion_id(logits, targets)
            loss_triplet = criterion_triplet(embeddings, targets)
            loss_center = criterion_center(embeddings, targets)
            loss = loss_id + loss_triplet + 0.0005 * loss_center

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        center_optimizer.zero_grad()

        if amp:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
            scaler.scale(loss).backward()
            if max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.step(center_optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            center_optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_id=loss_id.item())
        metric_logger.update(loss_tri=loss_triplet.item())
        metric_logger.update(loss_cen=loss_center.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_reid(
    query_loader,
    gallery_loader,
    model,
    device,
    amp=True,
    choices=None,
    mode="super",
    retrain_config=None,
):
    """Evaluate ReID performance using CMC and mAP"""
    model.eval()

    if mode == "super":
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    print("Extracting query features...")
    qf, q_pids, q_camids = extract_features(query_loader, model, device, amp)

    print("Extracting gallery features...")
    gf, g_pids, g_camids = extract_features(gallery_loader, model, device, amp)

    print("Computing distance matrix...")
    m, n = qf.size(0), gf.size(0)
    distmat = (
        torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.cpu().numpy()

    print("Computing CMC and mAP...")
    cmc, mAP = evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results:")
    print("mAP: {:.2%}".format(mAP))
    for r in [1, 5, 10, 20]:
        print("CMC Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))

    return {"mAP": mAP, "rank1": cmc[0], "rank5": cmc[4], "rank10": cmc[9]}


def extract_features(data_loader, model, device, amp=True):
    features = []
    pids = []
    camids = []

    for imgs, batch_pids, img_paths in data_loader:
        imgs = imgs.to(device)

        if amp:
            with torch.cuda.amp.autocast():
                feats = model(imgs, reid_infer=True)
        else:
            feats = model(imgs, reid_infer=True)

        features.append(feats)
        pids.extend(batch_pids.cpu().numpy())
        # Extract camera IDs from paths if available
        camids.extend([0] * len(batch_pids))  # ATRW doesn't have cameras

    features = torch.cat(features, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)

    return features, pids, camids


def evaluate_ranking(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute CMC and mAP"""
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Compute CMC
    all_cmc = []
    all_AP = []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # Remove gallery samples with same pid and camid
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # Compute AP
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    if num_valid_q == 0:
        raise RuntimeError("No valid query")

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
