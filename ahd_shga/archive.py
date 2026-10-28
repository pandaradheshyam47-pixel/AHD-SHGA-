"""
Elite Archive and Convergence Monitor for AHD-SHGA.
Implements:
1. Fixed-capacity memory archive of elite solutions (Eqs. 39-40)
2. Best solution identification (Eq. 41)
3. Multi-criteria convergence termination (Eq. 42)
"""

from typing import List, Optional, Sequence
import numpy as np
from .chromosome import Chromosome


class EliteArchive:
    """
    Maintains the m best-performing solutions discovered across generations.
    """

    def __init__(self, capacity: int = 10, epsilon: float = 1e-6):
        self.capacity = capacity
        self.epsilon = epsilon
        self.archive: List[Chromosome] = []
        self.best_history: List[float] = []

    def update(self, candidates: Sequence[Chromosome]) -> Chromosome:
        """
        Incorporate newly validated elite candidates and retain top m (Eq. 40).
        Returns the current global best candidate.
        """
        # Only candidates with true evaluated fitness can enter archive
        valid_candidates = [
            c.copy() for c in candidates if c.fitness is not None and c.evaluated_with_true
        ]
        pool = self.archive + valid_candidates

        # Sort pool ascending (minimization)
        pool.sort(key=lambda c: c.fitness if c.fitness is not None else float("inf"))

        # Deduplicate identical solutions within precision tolerance
        unique_elites: List[Chromosome] = []
        for cand in pool:
            is_dup = False
            for existing in unique_elites:
                if np.allclose(cand.genes, existing.genes, atol=1e-5):
                    is_dup = True
                    break
            if not is_dup:
                unique_elites.append(cand)
            if len(unique_elites) >= self.capacity:
                break

        self.archive = unique_elites
        best_cand = self.archive[0]
        self.best_history.append(float(best_cand.fitness))
        return best_cand

    def get_best(self) -> Optional[Chromosome]:
        """Return the best individual in the archive (Eq. 41)."""
        return self.archive[0] if self.archive else None

    def compute_improvement(self) -> float:
        """
        Compute absolute improvement in the best fitness value (Eq. 42):
        Delta_F(t) = |f(X_best(t)) - f(X_best(t-1))|
        """
        if len(self.best_history) < 2:
            return 1.0  # Initial large improvement
        return abs(self.best_history[-1] - self.best_history[-2])

    def has_converged(self, min_generations: int = 10) -> bool:
        """
        Check if improvement has stalled below epsilon over recent generations.
        """
        if len(self.best_history) < min_generations:
            return False
        delta_f = self.compute_improvement()
        return delta_f < self.epsilon
