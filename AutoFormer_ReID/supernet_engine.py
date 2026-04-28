import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import numpy as np

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    reid_mode=False):
    model.train()
    if not reid_mode:
        criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if reid_mode:
            samples, targets, _ = batch  # (images, pids, camids)
        else:
            samples, targets = batch

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

        if reid_mode:
            # ReID: no mixup, model returns (cls_score, feat)
            if amp:
                with torch.cuda.amp.autocast():
                    cls_score, feat = model(samples)
                    loss, loss_dict = criterion(cls_score, feat, targets)
            else:
                cls_score, feat = model(samples)
                loss, loss_dict = criterion(cls_score, feat, targets)
        else:
            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if amp:
                with torch.cuda.amp.autocast():
                    if teacher_model:
                        with torch.no_grad():
                            teach_output = teacher_model(samples)
                        _, teacher_label = teach_output.topk(1, 1, True, True)
                        outputs = model(samples)
                        loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                    else:
                        outputs = model(samples)
                        loss = criterion(outputs, targets)
            else:
                outputs = model(samples)
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        if reid_mode:
            metric_logger.update(id_loss=loss_dict['id_loss'])
            metric_logger.update(triplet_loss=loss_dict['triplet_loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_reid(query_loader, gallery_loader, model, device, amp=True,
                  choices=None, mode='super', retrain_config=None):
    """ReID evaluation: extract embeddings, compute distance matrix, return CMC/mAP."""
    model.eval()

    if mode == 'super':
        config = sample_configs(choices=choices)
    else:
        config = retrain_config
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)

    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    def extract_features(data_loader):
        feats, pids, camids = [], [], []
        for images, pid, camid in data_loader:
            images = images.to(device, non_blocking=True)
            if amp:
                with torch.cuda.amp.autocast():
                    feat = model(images)
            else:
                feat = model(images)
            feats.append(feat.cpu())
            pids.append(pid)
            camids.append(camid)
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0).numpy()
        camids = torch.cat(camids, dim=0).numpy()
        return feats, pids, camids

    q_feats, q_pids, q_camids = extract_features(query_loader)
    g_feats, g_pids, g_camids = extract_features(gallery_loader)

    # Cosine distance (features are L2-normalised)
    distmat = 1 - torch.mm(q_feats, g_feats.t()).numpy()

    cmc, mAP = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids)

    print(f"mAP: {mAP:.4f}  Rank-1: {cmc[0]:.4f}  Rank-5: {cmc[4]:.4f}  Rank-10: {cmc[9]:.4f}")
    return {'mAP': mAP, 'rank1': cmc[0], 'rank5': cmc[4], 'rank10': cmc[9]}


def compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Compute CMC curve and mAP following the Market-1501 evaluation protocol.

    For each query, gallery items with the same pid AND same camid are excluded.

    Args:
        distmat: [num_query, num_gallery] distance matrix
        q_pids, g_pids: identity labels
        q_camids, g_camids: camera ids
        max_rank: max rank for CMC computation

    Returns:
        (cmc, mAP) where cmc is a numpy array of length max_rank
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices = np.argsort(distmat, axis=1)  # sort gallery by ascending distance
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # Single-camera datasets (e.g. iPanda50, ATRW) encode a constant camid in filenames.
    # Applying the Market-1501 (same-pid & same-camid) filter would remove every true match.
    single_cam = np.unique(np.concatenate([q_camids, g_camids])).size == 1

    all_cmc = []
    all_AP = []

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        if single_cam:
            remove = np.zeros_like(order, dtype=bool)
        else:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        raw_cmc = matches[q_idx][keep]
        if not np.any(raw_cmc):
            # This query has no valid match in gallery (skip)
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1  # binarise

        all_cmc.append(cmc[:max_rank])

        # Compute average precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precision * raw_cmc).sum() / num_rel
        all_AP.append(ap)

    if len(all_cmc) == 0:
        print("Warning: no query has a valid gallery match — returning zeros")
        return np.zeros(max_rank, dtype=np.float32), 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(axis=0) / num_q  # average over all queries
    mAP = np.mean(all_AP)

    return all_cmc, mAP
