"""
Search space evolution for hardware-aware NAS.

Adapts MCUFormer's approach: fits linear models to predict accuracy and
SRAM fit ratio as functions of (rank_ratio, patch_size), then computes
gradient-based steps to evolve toward better search space configurations.
"""

import itertools
import numpy as np
import random


def make_list(input1, input2):
    """Generate all (rank_ratio_idx, patch_size_idx) combinations."""
    return list(itertools.product(input1, input2))


def find_max_positions(lst):
    max_val = max(lst)
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == max_val:
            return i
    return len(lst) - 1


def find_min_positions(lst):
    min_val = min(lst)
    for i, val in enumerate(lst):
        if val == min_val:
            return i
    return 0


class evolution_supernet(object):
    """
    Evolves the (rank_ratio, patch_size) search space based on
    accuracy and SRAM fit ratio observations.

    result_array: list of [rank_ratio, patch_size, accuracy, sram_fit_ratio]
    """
    def __init__(self, result_array, rank_ratio, patch_size):
        self.result_array = result_array
        self.rank_ratio = rank_ratio
        self.patch_size = patch_size

    def fit_SRAM_plane(self):
        """Fit linear model: SRAM_violation ~ rank_ratio + patch_size."""
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        y = np.array([[1 - y2] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w.reshape(-1)[-2:]

    def get_SRAM_evolution_step(self):
        weight_array = self.fit_SRAM_plane()
        threshold_array = np.array([0.2, 2])
        step_array = np.array([0.05, 4])
        return (weight_array // threshold_array) * step_array

    def fit_error_plane(self):
        """Fit linear model: error ~ rank_ratio + patch_size."""
        X = np.array([[1, x1, x2] for x1, x2, y1, y2 in self.result_array])
        y = np.array([[100 - y1] for x1, x2, y1, y2 in self.result_array]).reshape(-1, 1)
        w = np.linalg.inv(X.T @ X) @ X.T @ y
        return w.reshape(-1)[-2:]

    def get_error_evolution_step(self):
        weight_array = self.fit_error_plane()
        threshold_array = np.array([10, 20])
        step_array = np.array([0.05, 4])
        return (weight_array // threshold_array) * step_array

    def evolution_step(self):
        """Compute combined step for (rank_ratio, patch_size)."""
        error_step = self.get_error_evolution_step()
        sram_step = self.get_SRAM_evolution_step()
        step = error_step + sram_step

        # Clamp patch_size step to valid range [16, 32]
        if self.patch_size + step[1] < 16:
            step[1] = 16 - self.patch_size
        if self.patch_size + step[1] > 32:
            step[1] = 32 - self.patch_size

        # Clamp rank_ratio step to valid range [0.4, 0.95]
        if self.rank_ratio + step[0] < 0.4:
            step[0] = 0.4 - self.rank_ratio
        if self.rank_ratio + step[0] > 0.95:
            step[0] = 0.95 - self.rank_ratio

        return step
