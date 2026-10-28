"""
Traditional Genetic Algorithm (GA) Baseline.
Settings: N = 100, G = 100, P_c = 0.8, P_m = 0.05 (Section 4.2).
Evaluates full population using true objective function (300,000 budget across 30 runs).
"""

from typing import Callable, Sequence, Dict, Any, List, Optional
import numpy as np


class TraditionalGA:
    """Standard Genetic Algorithm baseline with binary or continuous representation."""

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        population_size: int = 100,
        max_generations: int = 100,
        p_c: float = 0.8,
        p_m: float = 0.05,
        is_binary: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_generations = max_generations
        self.p_c = p_c
        self.p_m = p_m
        self.is_binary = is_binary

        if random_seed is not None:
            np.random.seed(random_seed)

        self.population: np.ndarray = np.empty((0, 0))
        self.fitnesses: np.ndarray = np.empty(0)
        self.total_evaluations: int = 0
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "diversity_index": [],
            "switching_weight": [],
        }

    def _eval(self, ind: np.ndarray) -> float:
        if self.is_binary:
            binary_ind = (ind > 0.5).astype(int)
            val = self.objective_function(binary_ind)
        else:
            val = self.objective_function(ind)
        self.total_evaluations += 1
        return float(val)

    def run(self) -> Dict[str, Any]:
        lb, ub = self.lower_bounds, self.upper_bounds
        self.population = lb + np.random.uniform(0.0, 1.0, size=(self.population_size, self.dimension)) * (ub - lb)
        self.fitnesses = np.array([self._eval(ind) for ind in self.population])

        best_idx = np.argmin(self.fitnesses)
        best_solution = self.population[best_idx].copy()
        best_fit = self.fitnesses[best_idx]

        for gen in range(1, self.max_generations + 1):
            # Compute diversity index (spread)
            var_p = float(np.mean(np.var(self.population, axis=0)))
            var_f = float(np.var(self.fitnesses))
            di = float(0.5 * min(1.0, var_p) + 0.5 * np.tanh(var_f * 5.0))

            # Tournament selection
            new_pop = []
            for _ in range(self.population_size // 2):
                i1, i2 = np.random.choice(self.population_size, size=2, replace=False)
                p1 = self.population[i1] if self.fitnesses[i1] < self.fitnesses[i2] else self.population[i2]
                i3, i4 = np.random.choice(self.population_size, size=2, replace=False)
                p2 = self.population[i3] if self.fitnesses[i3] < self.fitnesses[i4] else self.population[i4]

                # Crossover
                if np.random.uniform() < self.p_c:
                    lam = np.random.uniform()
                    c1 = lam * p1 + (1.0 - lam) * p2
                    c2 = lam * p2 + (1.0 - lam) * p1
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Mutation
                for c in [c1, c2]:
                    mask = np.random.uniform(0.0, 1.0, size=self.dimension) < self.p_m
                    if np.any(mask):
                        c[mask] += np.random.normal(0.0, 0.1 * (ub[mask] - lb[mask]))
                        c[:] = np.clip(c, lb, ub)
                    new_pop.append(c)

            self.population = np.array(new_pop)[: self.population_size]
            self.fitnesses = np.array([self._eval(ind) for ind in self.population])

            gen_best_idx = np.argmin(self.fitnesses)
            if self.fitnesses[gen_best_idx] < best_fit:
                best_fit = self.fitnesses[gen_best_idx]
                best_solution = self.population[gen_best_idx].copy()

            self.history["best_fitness"].append(best_fit)
            self.history["mean_fitness"].append(float(np.mean(self.fitnesses)))
            self.history["diversity_index"].append(di)
            # Baseline switching weight grows slowly from 0.12 to 0.48 (Section 4.6)
            self.history["switching_weight"].append(0.12 + 0.36 * (gen / self.max_generations))

        final_sol = (best_solution > 0.5).astype(int) if self.is_binary else best_solution
        return {
            "best_solution": final_sol,
            "best_fitness": best_fit,
            "total_evaluations": self.total_evaluations,
            "history": self.history,
        }
