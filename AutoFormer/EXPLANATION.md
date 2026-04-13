# How AutoFormer Works

AutoFormer is a **one-shot Neural Architecture Search (NAS)** framework for Vision Transformers. 
Instead of training thousands of individual models to find the best one, it trains a single "supernet" that contains all possible sub-architectures inside it.

## Step 1: Building the Supernet

A YAML file (e.g. `supernet-T.yaml`) specifies the **maximum** values:

```
SUPERNET:
    EMBED_DIM: 256
    DEPTH: 14
    NUM_HEADS: 4
    MLP_RATIO: 4.0
```

It builds a normal ViT, but every layer uses elastic "Super" modules sized at max:

### Normal ViT
1. patch embedding:
    - splits the image into fixed-size patches, flattens them, and projects it to a vector via a linear layer
2. class token:
    - prepend a learnable [cls] token to the sequence
    - the single token carries the final classification output 
3. positional embedding:
    - adds learnable positional embeddings to the input sequence
4. transformer encoder blocks (repeated `n` times) - each block contains:
    - `layernorm -> multi-head self-attention` -> residual connectio
    - `layernorm -> ffn (linear -> gelu -> linear)` -> residual connection
5. `final layernorm` -> take the [cls] token -> `linear head` -> residual connection

### Supernet ViT
1. patch embedding:
    - exists a 256 output filters, but a subnet can choose to use only `n` filters
2. class token:
    - same as normal ViT
3. positional embedding:
    - same as normal ViT
4. transformer encoder blocks
    - can have up to 14 blocks, but subnet can choose to use only `n` blocks (unused blocks become identity/skip)
    - each block can have up to 4 heads, but subnet can choose to use only `n` heads (e.g. 2 heads means only the first 2 heads are active)
    - each block can have up to 4.0 MLP ratio, but subnet can choose to use only `n` MLP ratio ()

## Step 2: Training the Supernet

The YAML file also specifies the **search space** — the set of choices available to subnets:

```
SEARCH_SPACE:
    EMBED_DIM: [192, 216, 240]
    DEPTH: [12, 13, 14]
    NUM_HEADS: [3, 4]
    MLP_RATIO: [3.5, 4.0]
```

Training uses random subnet sampling — every training iteration picks a different subnet:

1. for each batch of images:
    - **randomly sample a subnet config** from the search space (e.g. depth=13, embed_dim=216, heads=[3,4,3,...], mlp_ratio=[3.5,4.0,3.5,...])
    - call `model.set_sample_config(config)` to activate only that subnet's weights
    - unused blocks become identity/skip operations, unused heads/dims are simply not indexed
    - forward pass through only the active subnet
    - compute loss, backpropagate, update weights
2. repeat for 500 epochs
    - over the course of training, thousands of different subnets get trained
    - they all share the same weight pool — a subnet just uses a slice of the full weights
    - this is what makes it "one-shot": one training run covers the entire search space

### Why this works
- each subnet only uses a contiguous slice of the supernet weights (e.g. first 216 of 256 dims, first 3 of 4 heads)
- because subnets share weights, training one subnet also partially trains overlapping subnets
- after enough iterations, every possible subnet has been sampled many times and has well-trained weights

## Step 3: Searching for the Best Subnet

After the supernet is trained, we need to find which subnet config performs best. This uses an **evolutionary search** (genetic algorithm):

1. random initialisation:
    - generate 50 random subnet configs from the search space
    - evaluate each on a validation set using the trained supernet weights
    - filter out configs that violate parameter constraints (e.g. must be between 5M-6M params)
2. for each generation (20 generations total):
    - **select** the top 10 subnets by accuracy
    - **mutate** (25 new candidates): pick a top-10 subnet, randomly tweak some of its choices (e.g. change one layer's heads from 3 to 4, or change depth from 13 to 14)
    - **crossover** (25 new candidates): pick two top-10 subnets, create an offspring by taking each dimension from a random parent
    - evaluate all 50 candidates on the validation set
3. after 20 generations:
    - return the highest accuracy subnet config

### How subnets are evaluated
- load the trained supernet checkpoint
- call `model.set_sample_config(candidate_config)` — same mechanism as training
- run inference on the validation set, measure top-1 accuracy
- no retraining needed — the supernet weights already work for any valid subnet

## Step 4: Retraining the Best Subnet

The best subnet config from search is written into a YAML file:

```
RETRAIN:
    DEPTH: 14
    EMBED_DIM: 240
    NUM_HEADS: [3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4]
    MLP_RATIO: [3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0, 3.5, 3.5, 3.5, 4.0, 4.0, 4.0, 4.0]
```

Retraining finetunes the supernet with this **fixed** config:

1. load the trained supernet checkpoint
2. for each batch of images:
    - always use the same fixed subnet config (no random sampling)
    - forward pass, compute loss, backpropagate
3. repeat for 300 epochs
    - this is essentially finetuning — the weights are already partially trained from step 2
    - fixing the config lets the subnet's specific weights fully converge without interference from other subnets

## Future Steps: Adapting for Re-Identification (ReID)

AutoFormer was built for **image classification** — predicting "what class is this?" For ReID, the question becomes "is this the same individual?" This requires learning an **embedding space** where same-identity images cluster together, rather than predicting a fixed set of labels.

The supernet / subnet sampling / evolutionary search machinery stays the same. The changes are in what the model outputs, what loss trains it, and how we measure "good".

### 1. Model head

Currently the model does: `cls_token -> linear -> class logits`

For ReID, replace with:
1. embedding layer:
    - `cls_token -> linear (e.g. 768 -> 256)` — this is the feature vector used for matching at inference
2. BNNeck (batch normalisation neck):
    - `embedding -> batch norm` — stabilises metric learning, separates the embedding space from the classification space
3. ID classification head (training only):
    - `BNNeck output -> linear -> ID logits` — treats each identity as a class, provides auxiliary supervision
    - thrown away at inference — only the embedding is kept

### 2. Loss functions

Classification uses cross-entropy only. ReID combines two losses:
1. ID loss (cross-entropy):
    - treats each identity as a class (e.g. 149 tigers = 149 classes)
    - applied to the ID classification head output
    - helps learn discriminative features
2. triplet loss:
    - given (anchor, positive, negative), pushes anchor closer to positive and away from negative
    - applied to the embedding output
    - directly optimises the distance metric we care about
3. total loss = ID loss + λ · triplet loss

### 3. Dataset loading

Classification uses `ImageFolder` which loads images by directory. ReID needs:
1. filename parsing:
    - filenames encode `{pid}_{camid}_{trackid}_{frameid}.jpg`
    - parse the pid (person/animal identity) from each filename
2. PK sampling:
    - each batch contains P random identities, K images each
    - this guarantees every batch has valid triplets (positives and negatives)
3. train/query/gallery splits:
    - train set for training
    - query set = probe images to search for
    - gallery set = database of images to search through

### 4. Evaluation

Classification uses top-1 accuracy. ReID uses a retrieval protocol:
1. extract embeddings for all query and gallery images
2. compute pairwise distance matrix (cosine or L2) between every query and every gallery image
3. for each query, rank all gallery images by distance
4. metrics:
    - **CMC curve** (rank-1, rank-5, rank-10) — "is the correct match in the top-k results?"
    - **mAP** (mean average precision) — average precision across all queries, accounts for multiple correct matches

### 5. NAS search changes

The evolutionary search currently uses top-1 classification accuracy as its fitness function. For ReID:
- replace the fitness function with **mAP** or **rank-1 accuracy**
- evaluate candidates using the query/gallery retrieval protocol instead of top-1 classification
- parameter constraints may need adjusting depending on deployment target (e.g. edge devices for wildlife cameras)
