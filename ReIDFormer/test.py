# test.py
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader
import numpy as np

from lib.datasets import ATRW   # ✅ assumes lib/datasets/ATRW.py exists
from lib.datasets import build_reid_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Test ATRW Dataset Loading")
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to ATRW dataset root, containing train/query/gallery folders')
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--visualize', action='store_true', help='Show a batch of images')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def build_loader(dataset, input_size, batch_size, num_workers, train_split=False):
    transform = build_reid_transform(is_train=train_split, input_size=input_size)
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train_split,
                        num_workers=num_workers, drop_last=train_split)
    return loader


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("✅ Testing ATRW Dataset")
    print(f"Data root: {args.data_path}")

    # Initialize datasets from pre-defined folders
    train_dataset = ATRW(root=os.path.join(args.data_path, "train"), split='train')
    query_dataset = ATRW(root=os.path.join(args.data_path, "query"), split='query')
    gallery_dataset = ATRW(root=os.path.join(args.data_path, "gallery"), split='gallery')

    print(f"Train samples: {len(train_dataset)}, IDs: {train_dataset.num_pids}")
    print(f"Query samples: {len(query_dataset)}, IDs: {query_dataset.num_pids}")
    print(f"Gallery samples: {len(gallery_dataset)}, IDs: {gallery_dataset.num_pids}")

    print("\n🎉 All splits loaded successfully. Dataset structure is correct for training/evaluation!")


if __name__ == '__main__':
    main()
