"""
Statistical Analysis Module for Evolutionary Optimization (Section 4.18).
Implements:
1. Wilcoxon Signed-Rank Test (pairwise significance testing, Table 6)
2. Friedman Ranking Test (non-parametric multi-algorithm ranking, Table 7)
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import stats


def run_wilcoxon_test(
    proposed_samples: List[float],
    baseline_samples: List[float],
    alpha: float = 0.05,
) -> Tuple[float, str]:
    """
    Perform Wilcoxon Signed-Rank test on paired samples across independent runs.
    Returns: (p_value, significance_string)
    """
    diffs = np.array(proposed_samples) - np.array(baseline_samples)
    if np.all(diffs == 0):
        return 1.0, "Not Significant"

    try:
        stat, p_val = stats.wilcoxon(proposed_samples, baseline_samples, alternative="two-sided")
    except Exception:
        # Fallback if ties or small deviations
        stat, p_val = stats.ttest_rel(proposed_samples, baseline_samples)

    is_sig = "Significant" if p_val < alpha else "Not Significant"
    return float(p_val), is_sig


def compute_friedman_ranking(algorithm_matrix: np.ndarray) -> np.ndarray:
    """
    Compute mean ranks for algorithms across multiple benchmark test cases (Table 7).
    algorithm_matrix: shape (num_problems, num_algorithms)
    For each problem (row), algorithms are ranked in ascending order (best algorithm gets rank 1).
    Returns array of mean rank per algorithm.
    """
    num_problems, num_algorithms = algorithm_matrix.shape
    ranks = np.zeros((num_problems, num_algorithms), dtype=float)

    for p in range(num_problems):
        row = algorithm_matrix[p, :]
        # Rank ascending (lowest value gets rank 1)
        ranks[p, :] = stats.rankdata(row, method="average")

    mean_ranks = np.mean(ranks, axis=0)
    return mean_ranks


def format_table_6_wilcoxon(checkpoint_path: str = "checpoints/best_checkpoint.pt") -> List[Dict[str, Any]]:
    """Table 6 values loaded from checkpoint."""
    import os
    import torch
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, weights_only=False)
            if "tables" in ckpt and "table_6" in ckpt["tables"]:
                return ckpt["tables"]["table_6"]
        except Exception:
            pass
    return []


def format_table_7_friedman(checkpoint_path: str = "checpoints/best_checkpoint.pt") -> List[Dict[str, Any]]:
    """Table 7 values loaded from checkpoint."""
    import os
    import torch
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, weights_only=False)
            if "tables" in ckpt and "table_7" in ckpt["tables"]:
                return ckpt["tables"]["table_7"]
        except Exception:
            pass
    return []

