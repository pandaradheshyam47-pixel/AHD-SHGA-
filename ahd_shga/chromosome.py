"""
Chromosome Representation for AHD-SHGA.
Encodes decision variables and self-evolving adaptive parameters:
C_i = [x_i1, x_i2, ..., x_iD, p_c,i, p_m,i, w_h,i] (Eq. 2)
"""

from typing import Optional, Sequence
import numpy as np


class Chromosome:
    """
    Individual candidate solution in the AHD-SHGA population.
    Carries both continuous/binary decision variables and local adaptive
    control parameters (crossover probability, mutation rate, hybrid weight).
    """

    def __init__(
        self,
        genes: np.ndarray,
        p_c: float,
        p_m: float,
        w_h: float,
        p_c_bounds: tuple[float, float] = (0.6, 0.9),
        p_m_bounds: tuple[float, float] = (0.01, 0.1),
        w_h_bounds: tuple[float, float] = (0.0, 1.0),
    ):
        self.genes = np.asarray(genes, dtype=np.float64)
        self.p_c = float(np.clip(p_c, p_c_bounds[0], p_c_bounds[1]))
        self.p_m = float(np.clip(p_m, p_m_bounds[0], p_m_bounds[1]))
        self.w_h = float(np.clip(w_h, w_h_bounds[0], w_h_bounds[1]))
        self.p_c_bounds = p_c_bounds
        self.p_m_bounds = p_m_bounds
        self.w_h_bounds = w_h_bounds

        # Fitness bookkeeping
        self.fitness: Optional[float] = None
        self.predicted_fitness: Optional[float] = None
        self.uncertainty: float = 0.0
        self.evaluated_with_true: bool = False

    @property
    def dimension(self) -> int:
        return len(self.genes)

    def to_binary(self, threshold: Optional[float] = None) -> np.ndarray:
        """
        Continuous-to-binary mapping via sigmoid transfer function (Eq. page 18).
        P(s_i = 1) = 1 / (1 + exp(-x_id))
        s_i = 1 if sigmoid output > r (where r in [0, 1] or deterministic threshold).
        """
        sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(self.genes, -30.0, 30.0)))
        if threshold is not None:
            return (sigmoid_val > threshold).astype(int)
        r = np.random.uniform(0.0, 1.0, size=self.genes.shape)
        return (sigmoid_val > r).astype(int)

    def get_effective_parameters(
        self,
        ref_p_c: float,
        ref_p_m: float,
        ref_w_h: float,
        eta: float = 0.5,
    ) -> tuple[float, float, float]:
        """
        Compute effective blended parameters using population-level reference (Eq. 3):
        p_c,i(t) = clip(p_c,i + eta * (p_c(t) - p_c,i), p_c_min, p_c_max)
        """
        eff_p_c = float(
            np.clip(
                self.p_c + eta * (ref_p_c - self.p_c),
                self.p_c_bounds[0],
                self.p_c_bounds[1],
            )
        )
        eff_p_m = float(
            np.clip(
                self.p_m + eta * (ref_p_m - self.p_m),
                self.p_m_bounds[0],
                self.p_m_bounds[1],
            )
        )
        eff_w_h = float(
            np.clip(
                self.w_h + eta * (ref_w_h - self.w_h),
                self.w_h_bounds[0],
                self.w_h_bounds[1],
            )
        )
        return eff_p_c, eff_p_m, eff_w_h

    def copy(self) -> "Chromosome":
        clone = Chromosome(
            genes=self.genes.copy(),
            p_c=self.p_c,
            p_m=self.p_m,
            w_h=self.w_h,
            p_c_bounds=self.p_c_bounds,
            p_m_bounds=self.p_m_bounds,
            w_h_bounds=self.w_h_bounds,
        )
        clone.fitness = self.fitness
        clone.predicted_fitness = self.predicted_fitness
        clone.uncertainty = self.uncertainty
        clone.evaluated_with_true = self.evaluated_with_true
        return clone

    @classmethod
    def random(
        cls,
        dimension: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        p_c_bounds: tuple[float, float] = (0.6, 0.9),
        p_m_bounds: tuple[float, float] = (0.01, 0.1),
        w_h_bounds: tuple[float, float] = (0.0, 1.0),
    ) -> "Chromosome":
        """
        Initialize chromosome uniformly at random across search space and
        parameter intervals (Eq. 5, page 10-11).
        """
        lb = np.asarray(lower_bounds, dtype=np.float64)
        ub = np.asarray(upper_bounds, dtype=np.float64)
        genes = lb + np.random.uniform(0.0, 1.0, size=dimension) * (ub - lb)
        p_c = np.random.uniform(p_c_bounds[0], p_c_bounds[1])
        p_m = np.random.uniform(p_m_bounds[0], p_m_bounds[1])
        w_h = np.random.uniform(w_h_bounds[0], w_h_bounds[1])
        return cls(
            genes=genes,
            p_c=p_c,
            p_m=p_m,
            w_h=w_h,
            p_c_bounds=p_c_bounds,
            p_m_bounds=p_m_bounds,
            w_h_bounds=w_h_bounds,
        )
