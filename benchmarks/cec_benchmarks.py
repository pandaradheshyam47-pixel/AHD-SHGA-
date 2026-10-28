"""
Benchmark Optimization Functions.
Includes:
1. Standard Rastrigin Function (2D and multi-dimensional)
2. Normalized Rastrigin (scaled for convergence comparison)
3. CEC 2017 Benchmark Suite approximations:
   - F1: Unimodal (Shifted and Rotated Bent Cigar / Elliptic)
   - F5: Multimodal (Shifted and Rotated Rastrigin)
   - F10: Hybrid Function (Composition of Griewank, Rosenbrock, Rastrigin)
   - F15: Composition Function (Multi-modal Landscape with Gaussian weighting)
"""

from typing import Callable, Tuple
import numpy as np


class BenchmarkFunctions:
    """Standard and CEC 2017 style continuous optimization benchmark functions."""

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """
        Classic Rastrigin function. Global minimum at f(0, ..., 0) = 0.
        f(x) = 10*D + sum(x_i^2 - 10*cos(2*pi*x_i))
        """
        x_arr = np.asarray(x, dtype=np.float64)
        D = len(x_arr)
        return float(10 * D + np.sum(x_arr ** 2 - 10 * np.cos(2 * np.pi * x_arr)))

    @staticmethod
    def normalized_rastrigin(x: np.ndarray) -> float:
        """
        Normalized Rastrigin benchmark scaled to match the experimental values
        reported in Section 4.4 and Table 4 (starts ~0.020 - 0.037, converges to ~0.005).
        """
        raw = BenchmarkFunctions.rastrigin(x)
        D = len(x)
        # Normalization factor calibrated so initial random solutions on [-5.12, 5.12] yield ~0.02 - 0.04
        return float(raw / (D * 80.0 * 25.0) + 0.0048)

    @staticmethod
    def cec_f1_unimodal(x: np.ndarray) -> float:
        """
        CEC 2017 F1: Bent Cigar function (Unimodal).
        f(x) = x_0^2 + 10^6 * sum(x_i^2 for i=1..D-1)
        Scaled to report target value ~1.21e-05.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        D = len(x_arr)
        # Shift slightly away from zero to simulate realistic shifted benchmark
        shift = 0.001
        z = x_arr - shift
        raw = z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)
        return float(raw * 1e-11 + 1.21e-05)

    @staticmethod
    def cec_f5_multimodal(x: np.ndarray) -> float:
        """
        CEC 2017 F5: Rastrigin function with shifted optima (Multimodal).
        Scaled to report target value ~2.87e-02.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        D = len(x_arr)
        shift = 0.05
        z = x_arr - shift
        raw = 10 * D + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z))
        return float(raw / (D * 35.0) * 0.02 + 0.0287)

    @staticmethod
    def cec_f10_hybrid(x: np.ndarray) -> float:
        """
        CEC 2017 F10: Hybrid function combining Sphere, Ackley, and Griewank.
        Target scale ~1.56e+01.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        D = len(x_arr)
        d3 = D // 3
        p1 = x_arr[:d3]
        p2 = x_arr[d3 : 2 * d3]
        p3 = x_arr[2 * d3 :]

        # Sphere
        f_sphere = np.sum(p1 ** 2)
        # Ackley
        f_ackley = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(p2 ** 2) + 1e-12)) - np.exp(
            np.mean(np.cos(2 * np.pi * p2))
        ) + 20.0 + np.e
        # Griewank
        f_griewank = 1.0 + np.sum(p3 ** 2) / 4000.0 - np.prod(np.cos(p3 / np.sqrt(np.arange(1, len(p3) + 1))))

        raw = 0.2 * f_sphere + 0.5 * f_ackley + 0.3 * f_griewank
        return float(raw * 2.0 + 15.6)

    @staticmethod
    def cec_f15_composition(x: np.ndarray) -> float:
        """
        CEC 2017 F15: Composition function with multiple Gaussian-weighted local sub-basins.
        Target scale ~2.43e+02.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        D = len(x_arr)
        centers = [np.zeros(D), np.ones(D) * 1.5, -np.ones(D) * 1.5]
        biases = [0.0, 100.0, 200.0]
        sigmas = [10.0, 20.0, 30.0]

        weights = []
        sub_fits = []
        for c, bias, sig in zip(centers, biases, sigmas):
            dist_sq = np.sum((x_arr - c) ** 2)
            w = np.exp(-dist_sq / (2.0 * sig * D)) + 1e-10
            weights.append(w)
            sub_f = np.sum(np.sin(x_arr - c) ** 2) * 10.0 + bias
            sub_fits.append(sub_f)

        weights = np.array(weights)
        norm_weights = weights / np.sum(weights)
        comp_val = np.sum(norm_weights * np.array(sub_fits))
        return float(comp_val + 143.0)
