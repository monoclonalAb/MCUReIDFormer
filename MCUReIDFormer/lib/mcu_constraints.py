"""
MCU SRAM and Flash constraint estimation for ViT subnets.

Estimates peak memory usage during inference on a microcontroller,
where only one layer executes at a time and activations must fit in SRAM.
"""

import math


# Bytes per element (int8 quantized inference typical for MCU)
BYTES_PER_PARAM = 1      # INT8 weights
BYTES_PER_ACTIVATION = 1  # INT8 activations


def estimate_seq_length(img_size, patch_size):
    """Number of tokens = (img_size / patch_size)^2 + 1 (cls token)."""
    n_patches = (img_size // patch_size) ** 2
    return n_patches + 1


def estimate_flash_bytes(config, rank_ratio, num_classes, reid_dim=256):
    """
    Estimate total model size in bytes (stored in Flash).

    Parameters stored: all weight matrices across all layers + embeddings.
    With low-rank decomposition, each original W (m x n) becomes
    W1 (m x r) @ W2 (r x r) @ W3 (r x n) where r = rank_ratio * min(m, n).
    """
    depth = config['layer_num']
    embed_dims = config['embed_dim']
    mlp_ratios = config['mlp_ratio']
    num_heads_list = config['num_heads']

    total_params = 0

    for i in range(depth):
        d = embed_dims[i]
        ffn_dim = int(d * mlp_ratios[i])
        mlp_rank = int(min(d, ffn_dim) * rank_ratio)
        qkv_rank = int(d * rank_ratio)

        # QKV low-rank: d->r, r->r, r->3d
        total_params += d * qkv_rank + qkv_rank * qkv_rank + qkv_rank * 3 * d
        # Proj low-rank: d->r, r->r, r->d
        total_params += d * qkv_rank + qkv_rank * qkv_rank + qkv_rank * d
        # FFN up low-rank: d->r, r->r, r->ffn
        total_params += d * mlp_rank + mlp_rank * mlp_rank + mlp_rank * ffn_dim
        # FFN down low-rank: ffn->r, r->r, r->d
        total_params += ffn_dim * mlp_rank + mlp_rank * mlp_rank + mlp_rank * d
        # LayerNorm x2: 2 * d (weight + bias each)
        total_params += 4 * d

    # Patch embedding conv: 3 * patch_size^2 * embed_dim (approximate)
    # Positional embedding, cls token
    # Head + ReID head
    d0 = embed_dims[0]
    total_params += d0 * num_classes  # classification head
    total_params += d0 * reid_dim     # reid head
    total_params += reid_dim           # bnneck

    return total_params * BYTES_PER_PARAM


def estimate_peak_sram_bytes(config, img_size, patch_size, rank_ratio):
    """
    Estimate peak SRAM usage in bytes during inference.

    Peak memory occurs at the layer with the largest intermediate activations.
    For each transformer layer, we need to hold:
      - Input activations:  seq_len * embed_dim
      - Largest intermediate (one of):
        a) Attention scores: num_heads * seq_len * seq_len
        b) QKV output:      seq_len * 3 * embed_dim
        c) FFN intermediate: seq_len * ffn_dim (or seq_len * rank with low-rank)
      - Output activations: seq_len * embed_dim (can often alias input for residual)

    With low-rank decomposition, intermediates at the bottleneck are
    seq_len * rank instead of seq_len * full_dim, reducing peak SRAM.
    """
    seq_len = estimate_seq_length(img_size, patch_size)
    depth = config['layer_num']
    embed_dims = config['embed_dim']
    mlp_ratios = config['mlp_ratio']
    num_heads_list = config['num_heads']

    peak = 0

    for i in range(depth):
        d = embed_dims[i]
        ffn_dim = int(d * mlp_ratios[i])
        n_heads = num_heads_list[i]
        mlp_rank = int(min(d, ffn_dim) * rank_ratio)
        qkv_rank = int(d * rank_ratio)

        # Input buffer
        input_buf = seq_len * d

        # Attention path peak:
        # After qkv1: seq_len * qkv_rank (bottleneck)
        # After qkv3: seq_len * 3 * d (full QKV)
        # Attention scores: n_heads * seq_len * seq_len
        # We need input + max(qkv_output, attn_scores)
        attn_scores = n_heads * seq_len * seq_len
        qkv_output = seq_len * 3 * d
        attn_peak = input_buf + max(attn_scores, qkv_output)

        # FFN path peak:
        # After fc11: seq_len * mlp_rank (bottleneck)
        # After fc13: seq_len * ffn_dim (full FFN width)
        # With low-rank, the bottleneck limits peak; without it, ffn_dim dominates
        ffn_intermediate = seq_len * ffn_dim
        ffn_bottleneck = seq_len * mlp_rank
        ffn_peak = input_buf + max(ffn_intermediate, ffn_bottleneck)

        layer_peak = max(attn_peak, ffn_peak)
        peak = max(peak, layer_peak)

    return peak * BYTES_PER_ACTIVATION


def estimate_peak_sram_kb(config, img_size, patch_size, rank_ratio):
    """Peak SRAM in kilobytes."""
    return estimate_peak_sram_bytes(config, img_size, patch_size, rank_ratio) / 1024.0


def estimate_flash_kb(config, rank_ratio, num_classes, reid_dim=256):
    """Flash usage in kilobytes."""
    return estimate_flash_bytes(config, rank_ratio, num_classes, reid_dim) / 1024.0


def fits_sram(config, img_size, patch_size, rank_ratio, sram_budget_kb):
    """Check if a subnet fits within the SRAM budget."""
    usage = estimate_peak_sram_kb(config, img_size, patch_size, rank_ratio)
    return usage <= sram_budget_kb


def fits_flash(config, rank_ratio, num_classes, flash_budget_kb, reid_dim=256):
    """Check if a subnet's weights fit within the Flash budget."""
    usage = estimate_flash_kb(config, rank_ratio, num_classes, reid_dim)
    return usage <= flash_budget_kb
