# How AutoFormer ReID Works

AutoFormer ReID adapts the AutoFormer one-shot NAS framework from **image classification** to **re-identification**. The supernet / subnet sampling / evolutionary search machinery stays the same.
The changes are in what the model outputs, how it's trained, how data is loaded, and how "good" is measured.

## Step 1: Building the Supernet

A YAML file (e.g. `supernet-T.yaml`) specifies the **maximum** values:

```
SUPERNET:
    EMBED_DIM: 256
    DEPTH: 14
    NUM_HEADS: 4
    MLP_RATIO: 4.0
```

### Supernet ViT

#### Initialisation:
1. patch embedding:
    - splits the image into fixed-size patches, flattens them, and projects it to a vector via a linear layer
    - exists 256-layers of output filters `weight` (subnet can choose to use only `n` filters)
2. class token:
    - prepend a learnable [cls] token to the sequence
    - the single token carries the final classification output 
3. positional embedding:
    - adds learnable positional embeddings to the input sequence
4. transformer encoder blocks (repeated `n` times) - each block contains:
    1. aims to model the relationships between each patch
    - `layernorm` -> `multi-head self-attention` -> `residual connection`
        - `layernorm`
            - normalises each token's values to have mean=0, std=1 (prevent drift to very large/small numbers)
        - `multi-head self-attention`
            - LARGE GENERALISATION - calculates attention scores between every pair
        - `residual connection`
            - adds the attention output back to the original input 
    2. aims to process all the information gathered and draw conclusions
    - `layernorm` -> `ffn (linear -> gelu -> linear)` -> `residual connection`
        - `feedforward network`
            - linear
                - matrix multiply transforming first `n` tokens into `mlp_ratio x n` tokens
                - each value of `mlp_ratio x n` is a weighted sum of all `n` inputs
            - GELU
                - non-linear activation function (applies to the 672 values individually)
                - keeps positive values approximately the same, push negative values to approximately zero
            - linear
                - matrix multiple transforming `mlp_ratio x n` tokens into `n` tokens
                - each value of `n` is a weighted sum of all `mlp_ratio x n`
5a. inference
    - `final layernorm` -> take the [cls] token -> `bottlenext (LinearSuper: embed_dim -> 256)` -> L2 normalisation -> returns embedding
5b. testing
    - `final layernorm` -> take the [cls] token -> `bottlenext (LinearSuper: embed_dim -> 256)` -> BNNeck (BatchNorm1d) -> classifier (Linear -> ID logits)

