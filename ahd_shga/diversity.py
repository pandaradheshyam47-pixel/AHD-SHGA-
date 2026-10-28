"""
Diversity Intelligence Monitoring Module for AHD-SHGA.
Computes:
1. Genotypic Entropy (H_g, Eq. 22)
2. Population Variance (V_p, Eq. 23-24)
3. Fitness Variance (V_f, Eq. 25-26)
4. Composite Diversity Index (DI, Eq. 27):
   DI = c1 * H_g + c2 * V_p + c3 * V_f
"""

from typing import Sequence, List
import numpy as np
from .chromosome import Chromosome


class DiversityMonitor:
    """
    Monitors population diversity across generations using composite multi-attribute index.
    """

    def __init__(
        self,
        c1: float = 0.4,
        c2: float = 0.4,
        c3: float = 0.2,
        num_bins: int = 10,
        di_threshold: float = 0.45,
    ):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.num_bins = num_bins
        self.di_threshold = di_threshold

    def compute_genotypic_entropy(self, population: Sequence[Chromosome]) -> float:
        """
        Compute genotypic entropy H_g = - sum_k p_k * log(p_k) (Eq. 22).
        Calculated over discretized bins across the decision variables.
        """
        if not population:
            return 0.0

        genes_matrix = np.array([c.genes for c in population])
        N, D = genes_matrix.shape

        entropies = []
        for d in range(D):
            col = genes_matrix[:, d]
            min_val, max_val = np.min(col), np.max(col)
            if np.isclose(min_val, max_val):
                entropies.append(0.0)
                continue
            hist, _ = np.histogram(col, bins=self.num_bins, range=(min_val, max_val))
            probs = hist / N
            probs = probs[probs > 0]
            # Normalized Shannon entropy to [0, 1]
            h_d = -np.sum(probs * np.log(probs))
            norm_factor = np.log(self.num_bins)
            entropies.append(h_d / norm_factor if norm_factor > 0 else 0.0)

        return float(np.mean(entropies)) if entropies else 0.0

    def compute_population_variance(
        self, population: Sequence[Chromosome], search_range_norm: float = 1.0
    ) -> float:
        """
        Compute population variance V_p = (1/N) * sum ||X_i - X_mean||^2 (Eq. 23-24).
        Normalized by the search space scale for dimensional consistency.
        """
        if not population:
            return 0.0

        genes_matrix = np.array([c.genes for c in population])
        mean_vector = np.mean(genes_matrix, axis=0)
        diffs = genes_matrix - mean_vector
        squared_distances = np.sum(diffs ** 2, axis=1)
        raw_variance = float(np.mean(squared_distances))

        # Normalize by dimension and scale factor to keep in well-behaved numerical range
        D = genes_matrix.shape[1]
        denom = max(1e-6, D * (search_range_norm ** 2))
        return float(min(2.0, raw_variance / denom))

    def compute_fitness_variance(self, population: Sequence[Chromosome]) -> float:
        """
        Compute fitness variance V_f = (1/N) * sum (F_i - F_mean)^2 (Eq. 25-26).
        """
        fitnesses = [c.fitness for c in population if c.fitness is not None]
        if not fitnesses:
            return 0.0
        fit_arr = np.array(fitnesses, dtype=np.float64)
        mean_fit = np.mean(fit_arr)
        variance = float(np.mean((fit_arr - mean_fit) ** 2))
        # Scaled to maintain sensitivity
        return float(np.tanh(variance * 10.0))

    def compute_diversity_index(
        self, population: Sequence[Chromosome], search_range_norm: float = 1.0
    ) -> tuple[float, float, float, float]:
        """
        Compute composite Diversity Index DI(t) = c1 * H_g + c2 * V_p + c3 * V_f (Eq. 27).
        Returns: (DI, H_g, V_p, V_f)
        """
        h_g = self.compute_genotypic_entropy(population)
        v_p = self.compute_population_variance(population, search_range_norm)
        v_f = self.compute_fitness_variance(population)
        di = self.c1 * h_g + self.c2 * v_p + self.c3 * v_f
        return float(di), float(h_g), float(v_p), float(v_f)

    def is_diversity_collapsed(self, di: float) -> bool:
        """Check if DI falls below the acceptable threshold (Eq. 27, page 16)."""
        return di < self.di_threshold
