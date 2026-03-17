"""
Evolution search adapted for Re-Identification with hardware-aware constraints.
Searches for optimal architecture using mAP as fitness metric,
with SRAM and Flash budget enforcement for MCU deployment.
"""

import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from lib.datasets import build_dataset
from lib import utils
from lib.mcu_constraints import estimate_peak_sram_kb, estimate_flash_kb, fits_sram, fits_flash
from supernet_ReID_engine import evaluate_reid
from model.supernet_transformer_ReID import Vision_TransformerSuper
import argparse
import os
import yaml
from lib.config import cfg, update_config_from_file

def decode_cand_tuple(cand_tuple):
    """Decode candidate tuple into architecture config"""
    depth = cand_tuple[0]
    return (depth,
            list(cand_tuple[1:depth+1]),
            list(cand_tuple[depth + 1: 2 * depth + 1]),
            cand_tuple[-1])

class EvolutionSearcherReID(object):
    """Evolution searcher optimized for Re-ID using mAP as fitness,
    with SRAM/Flash hardware constraints."""

    def __init__(self, args, device, model, model_without_ddp, choices,
                 query_loader, gallery_loader, output_dir):
        self.device = device
        self.model = model
        self.model_without_ddp = model_without_ddp
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_scores = []
        self.cand_params = []
        self.choices = choices

        # Hardware constraints
        self.sram_budget_kb = args.sram_budget
        self.flash_budget_kb = args.flash_budget
        self.img_size = args.input_size
        self.patch_size = args.patch_size
        self.rank_ratio = args.rank_ratio
        self.reid_dim = args.reid_dim

    def save_checkpoint(self):
        """Save search state"""
        info = {}
        info['top_scores'] = self.top_scores
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, f"search_checkpoint-{self.epoch}.pth")
        torch.save(info, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self):
        """Load search state"""
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print(f'Loaded checkpoint from {self.checkpoint_path}')
        return True

    def is_legal(self, cand):
        """Check if candidate is legal (param + SRAM + Flash constraints) and evaluate it"""
        assert isinstance(cand, tuple)

        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}

        info = self.vis_dict[cand]

        if 'visited' in info:
            return False

        # Decode candidate
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
        sampled_config = {
            'layer_num': depth,
            'mlp_ratio': mlp_ratio,
            'num_heads': num_heads,
            'embed_dim': [embed_dim] * depth
        }

        # Check parameter constraints
        n_parameters = self.model_without_ddp.get_sampled_params_numel(sampled_config)
        info['params'] = n_parameters / 10.**6

        if info['params'] > self.parameters_limits:
            print(f'Parameters {info["params"]:.2f}M exceed limit {self.parameters_limits}M')
            return False

        if info['params'] < self.min_parameters_limits:
            print(f'Parameters {info["params"]:.2f}M below minimum {self.min_parameters_limits}M')
            return False

        # Check SRAM constraint
        if self.sram_budget_kb > 0:
            sram_kb = estimate_peak_sram_kb(sampled_config, self.img_size,
                                            self.patch_size, self.rank_ratio)
            info['sram_kb'] = sram_kb
            if not fits_sram(sampled_config, self.img_size, self.patch_size,
                           self.rank_ratio, self.sram_budget_kb):
                print(f'SRAM {sram_kb:.1f}KB exceeds budget {self.sram_budget_kb}KB')
                return False

        # Check Flash constraint
        if self.flash_budget_kb > 0:
            flash_kb = estimate_flash_kb(sampled_config, self.rank_ratio,
                                         self.model_without_ddp.num_classes,
                                         self.reid_dim)
            info['flash_kb'] = flash_kb
            if not fits_flash(sampled_config, self.rank_ratio,
                            self.model_without_ddp.num_classes,
                            self.flash_budget_kb, self.reid_dim):
                print(f'Flash {flash_kb:.1f}KB exceeds budget {self.flash_budget_kb}KB')
                return False

        print(f"Rank {utils.get_rank()}: Evaluating {cand}, params: {info['params']:.2f}M"
              + (f", SRAM: {info.get('sram_kb', 0):.1f}KB" if self.sram_budget_kb > 0 else ""))

        # Evaluate on ReID task
        eval_stats = evaluate_reid(
            self.query_loader,
            self.gallery_loader,
            self.model,
            self.device,
            amp=self.args.amp,
            mode='retrain',
            retrain_config=sampled_config
        )

        # Store metrics
        info['mAP'] = eval_stats['mAP']
        info['rank1'] = eval_stats['rank1']
        info['rank5'] = eval_stats['rank5']
        info['rank10'] = eval_stats['rank10']

        # Use mAP as primary fitness metric
        info['fitness'] = eval_stats['mAP']

        info['visited'] = True

        print(f"  -> mAP: {info['mAP']:.4f}, Rank-1: {info['rank1']:.4f}")

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        """Update top-k candidates"""
        assert k in self.keep_top_k
        print('Selecting top candidates...')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        """Generate random candidates"""
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand

    def get_random_cand(self):
        """Generate a random candidate"""
        cand_tuple = list()
        dimensions = ['mlp_ratio', 'num_heads']
        depth = random.choice(self.choices['depth'])
        cand_tuple.append(depth)

        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))

        cand_tuple.append(random.choice(self.choices['embed_dim']))
        return tuple(cand_tuple)

    def get_random(self, num):
        """Sample random candidates"""
        print(f'Sampling {num} random candidates...')
        cand_iter = self.stack_random_cand(self.get_random_cand)

        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print(f'Random candidate {len(self.candidates)}/{num}')

        print(f'Sampled {len(self.candidates)} random candidates')

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        """Generate candidates through mutation"""
        assert k in self.keep_top_k
        print('Generating mutations...')
        res = []
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(cand)
            random_s = random.random()

            # Mutate depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])
                if new_depth > depth:
                    mlp_ratio = mlp_ratio + [random.choice(self.choices['mlp_ratio'])
                                             for _ in range(new_depth - depth)]
                    num_heads = num_heads + [random.choice(self.choices['num_heads'])
                                            for _ in range(new_depth - depth)]
                else:
                    mlp_ratio = mlp_ratio[:new_depth]
                    num_heads = num_heads[:new_depth]
                depth = new_depth

            # Mutate mlp_ratio
            for i in range(depth):
                if random.random() < m_prob:
                    mlp_ratio[i] = random.choice(self.choices['mlp_ratio'])

            # Mutate num_heads
            for i in range(depth):
                if random.random() < m_prob:
                    num_heads[i] = random.choice(self.choices['num_heads'])

            # Mutate embed_dim
            if random.random() < s_prob:
                embed_dim = random.choice(self.choices['embed_dim'])

            result_cand = [depth] + mlp_ratio + num_heads + [embed_dim]
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f'Mutation {len(res)}/{mutation_num}')

        print(f'Generated {len(res)} mutations')
        return res

    def get_crossover(self, k, crossover_num):
        """Generate candidates through crossover"""
        assert k in self.keep_top_k
        print('Generating crossovers...')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50

            # Find parents with same length
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])

            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print(f'Crossover {len(res)}/{crossover_num}')

        print(f'Generated {len(res)} crossovers')
        return res

    def search(self):
        """Run evolution search"""
        print('='*80)
        print('Hardware-Aware Evolution Search for ReID')
        print(f'Population: {self.population_num}, Select: {self.select_num}')
        print(f'Mutation: {self.mutation_num}, Crossover: {self.crossover_num}')
        print(f'Epochs: {self.max_epochs}')
        print(f'Parameter range: {self.min_parameters_limits}M - {self.parameters_limits}M')
        print(f'SRAM budget: {self.sram_budget_kb}KB, Flash budget: {self.flash_budget_kb}KB')
        print(f'Rank ratio: {self.rank_ratio}, Patch size: {self.patch_size}')
        print('='*80)

        # Initialize population
        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print(f'\n{"="*80}')
            print(f'Epoch {self.epoch + 1}/{self.max_epochs}')
            print(f'{"="*80}')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            # Update top-k based on mAP (fitness)
            self.update_top_k(
                self.candidates,
                k=self.select_num,
                key=lambda x: self.vis_dict[x]['fitness']
            )
            self.update_top_k(
                self.candidates,
                k=50,
                key=lambda x: self.vis_dict[x]['fitness']
            )

            # Print top results
            print(f'\nTop {len(self.keep_top_k[50])} architectures:')
            sram_header = "SRAM(KB)" if self.sram_budget_kb > 0 else ""
            print(f'{"Rank":<6} {"mAP":<8} {"R-1":<8} {"R-5":<8} {"Params":<10} {sram_header:<10} {"Config"}')
            print('-' * 90)

            tmp_scores = []
            for i, cand in enumerate(self.keep_top_k[50]):
                info = self.vis_dict[cand]
                tmp_scores.append(info['fitness'])
                sram_str = f"{info.get('sram_kb', 0):.0f}" if self.sram_budget_kb > 0 else ""
                print(f'{i+1:<6} {info["mAP"]:.4f}   {info["rank1"]:.4f}   '
                      f'{info["rank5"]:.4f}   {info["params"]:.2f}M     {sram_str:<10} {str(cand)[:40]}...')

            self.top_scores.append(tmp_scores)

            # Generate new population
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob
            )
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            # Fill remaining slots with random
            remaining = self.population_num - len(self.candidates)
            if remaining > 0:
                self.get_random(self.population_num)

            self.epoch += 1
            self.save_checkpoint()

        # Final report
        print(f'\n{"="*80}')
        print('Search Complete!')
        print(f'{"="*80}')
        best_info = self.vis_dict[self.keep_top_k[50][0]]
        print(f'\nBest architecture (mAP: {best_info["mAP"]:.4f}):')
        print(self.keep_top_k[50][0])
        if 'sram_kb' in best_info:
            print(f'SRAM: {best_info["sram_kb"]:.1f}KB / {self.sram_budget_kb}KB')

        # Save best config
        best_cand = self.keep_top_k[50][0]
        depth, mlp_ratio, num_heads, embed_dim = decode_cand_tuple(best_cand)
        best_config = {
            'DEPTH': depth,
            'MLP_RATIO': mlp_ratio,
            'NUM_HEADS': num_heads,
            'EMBED_DIM': embed_dim
        }

        with open(os.path.join(self.output_dir, 'best_config.yaml'), 'w') as f:
            yaml.dump({'RETRAIN': best_config}, f)

        print(f'Best config saved to {self.output_dir}/best_config.yaml')

def get_args_parser():
    parser = argparse.ArgumentParser('Hardware-aware evolution search for ReID', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # Evolution parameters
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=40)
    parser.add_argument('--min-param-limits', type=float, default=5)

    # Config
    parser.add_argument('--cfg', help='config file', required=True, type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--scale', action='store_true')

    # Model
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=256, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--reid-dim', default=256, type=int)
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14)
    parser.add_argument('--no_abs_pos', action='store_true')

    # Hardware-aware parameters
    parser.add_argument('--rank-ratio', type=float, default=0.9,
                        help='Low-rank decomposition ratio (0.4-0.95)')
    parser.add_argument('--sram-budget', type=float, default=320,
                        help='SRAM budget in KB (0 to disable, default: 320 for STM32F746)')
    parser.add_argument('--flash-budget', type=float, default=1024,
                        help='Flash budget in KB (0 to disable, default: 1024 for STM32F746)')

    # Dataset
    parser.add_argument('--data-path', required=True, type=str)
    parser.add_argument('--data-set', default='ATRW', type=str)
    parser.add_argument('--output_dir', default='./search_output')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False)
    parser.add_argument('--pin-mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--amp', action='store_true')
    parser.set_defaults(amp=True)

    return parser

def main(args):
    update_config_from_file(args.cfg)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "search_config.yaml"), 'w') as f:
        f.write(args_text)

    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    # Build datasets
    args_query = argparse.Namespace(**vars(args))
    dataset_query, args.nb_classes = build_dataset(is_train=False, args=args_query, folder_name='query')

    args_gallery = argparse.Namespace(**vars(args))
    dataset_gallery, _ = build_dataset(is_train=False, args=args_gallery, folder_name='gallery')

    # Data loaders
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query, batch_size=int(2 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    data_loader_gallery = torch.utils.data.DataLoader(
        dataset_gallery, batch_size=int(2 * args.batch_size),
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    # Create model with low-rank decomposition
    print(f"Creating Hardware-Aware SuperVisionTransformer for ReID")
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
        drop_rate=0.0,
        drop_path_rate=0.1,
        gp=args.gp,
        num_classes=args.nb_classes,
        max_relative_position=args.max_relative_position,
        relative_position=args.relative_position,
        change_qkv=args.change_qkv,
        abs_pos=not args.no_abs_pos,
        rank_ratio=args.rank_ratio,
        reid=True,
        reid_dim=args.reid_dim
    )

    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    # Load pretrained weights
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['model']

            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('head.') or k.startswith('pos_embed'):
                    print(f"Skipping incompatible key: {k} (shape {v.shape})")
                    continue
                filtered_state_dict[k] = v

            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(filtered_state_dict, strict=False)
            print(f"=> Loaded pretrained weights (skipped pos_embed/head)")
            print(f"   Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

    # Search space
    choices = {
        'num_heads': cfg.SEARCH_SPACE.NUM_HEADS,
        'mlp_ratio': cfg.SEARCH_SPACE.MLP_RATIO,
        'embed_dim': cfg.SEARCH_SPACE.EMBED_DIM,
        'depth': cfg.SEARCH_SPACE.DEPTH
    }

    # Run search
    t = time.time()
    searcher = EvolutionSearcherReID(
        args, device, model, model_without_ddp, choices,
        data_loader_query, data_loader_gallery, args.output_dir
    )

    searcher.search()

    print(f'\nTotal search time: {(time.time() - t) / 3600:.2f} hours')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Hardware-aware evolution search for ReID', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
