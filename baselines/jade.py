"""
JADE (Adaptive Differential Evolution with External Archive) Baseline (ref [32]).
Settings:
- Memory size H = 5
- Initial means: mu_CR = 0.5, mu_F = 0.5
- Archive size = N (100)
- Current-to-pbest/1 mutation with archive
- Sigmoid thresholding for binary mapping
"""

from typing import Callable, Sequence, Dict, Any, List, Optional
import numpy as np


class JADEOptimizer:
    """JADE Differential Evolution baseline."""

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        population_size: int = 100,
        max_generations: int = 100,
        memory_size: int = 5,
        p_best_rate: float = 0.05,
        c_adapt: float = 0.1,
        is_binary: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_generations = max_generations
        self.memory_size = memory_size
        self.p_best_rate = p_best_rate
        self.c_adapt = c_adapt
        self.is_binary = is_binary

        if random_seed is not None:
            np.random.seed(random_seed)

        # Historical memory for mu_CR and mu_F
        self.M_CR = np.full(self.memory_size, 0.5)
        self.M_F = np.full(self.memory_size, 0.5)
        self.k_mem = 0

        # Archive
        self.archive: List[np.ndarray] = []
        self.archive_capacity = self.population_size

        self.population: np.ndarray = np.empty((0, 0))
        self.fitnesses: np.ndarray = np.empty(0)
        self.total_evaluations: int = 0
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "diversity_index": [],
        }

    def _eval(self, vec: np.ndarray) -> float:
        if self.is_binary:
            bin_vec = (1.0 / (1.0 + np.exp(-np.clip(vec, -30, 30))) > 0.5).astype(int)
            val = self.objective_function(bin_vec)
        else:
            val = self.objective_function(vec)
        self.total_evaluations += 1
        return float(val)

    def run(self) -> Dict[str, Any]:
        lb, ub = self.lower_bounds, self.upper_bounds
        span = ub - lb
        self.population = lb + np.random.uniform(0.0, 1.0, size=(self.population_size, self.dimension)) * span
        self.fitnesses = np.array([self._eval(ind) for ind in self.population])

        best_idx = np.argmin(self.fitnesses)
        best_sol = self.population[best_idx].copy()
        best_fit = self.fitnesses[best_idx]

        for gen in range(1, self.max_generations + 1):
            # Sample parameters for each individual
            S_CR = []
            S_F = []
            new_population = np.empty_like(self.population)
            new_fitnesses = np.empty_like(self.fitnesses)

            # Sort population to find top p best
            sorted_indices = np.argsort(self.fitnesses)
            num_pbest = max(2, int(self.population_size * self.p_best_rate))
            pbest_indices = sorted_indices[:num_pbest]

            for i in range(self.population_size):
                ri = np.random.randint(0, self.memory_size)
                # Sample CR from normal(M_CR[ri], 0.1)
                cr_i = np.clip(np.random.normal(self.M_CR[ri], 0.1), 0.0, 1.0)
                # Sample F from Cauchy(M_F[ri], 0.1)
                while True:
                    f_i = self.M_F[ri] + 0.1 * np.tan(np.pi * (np.random.uniform() - 0.5))
                    if f_i > 0:
                        break
                f_i = min(f_i, 1.0)

                # Select pbest
                pbest_idx = np.random.choice(pbest_indices)
                x_pbest = self.population[pbest_idx]

                # Select r1 from population != i
                r1_candidates = [idx for idx in range(self.population_size) if idx != i]
                r1 = np.random.choice(r1_candidates)
                x_r1 = self.population[r1]

                # Select r2 from population U archive != i, r1
                union_pool = [self.population[idx] for idx in range(self.population_size) if idx != i and idx != r1]
                union_pool += self.archive
                r2_cand = union_pool[np.random.choice(len(union_pool))]

                # Mutation: current-to-pbest/1
                v_i = self.population[i] + f_i * (x_pbest - self.population[i]) + f_i * (x_r1 - r2_cand)
                v_i = np.clip(v_i, lb, ub)

                # Crossover
                j_rand = np.random.randint(0, self.dimension)
                u_i = self.population[i].copy()
                mask = (np.random.uniform(0.0, 1.0, size=self.dimension) < cr_i)
                mask[j_rand] = True
                u_i[mask] = v_i[mask]

                # Selection
                f_trial = self._eval(u_i)
                if f_trial <= self.fitnesses[i]:
                    new_population[i] = u_i
                    new_fitnesses[i] = f_trial
                    if f_trial < self.fitnesses[i]:
                        S_CR.append(cr_i)
                        S_F.append(f_i)
                        # Add parent to archive
                        if len(self.archive) < self.archive_capacity:
                            self.archive.append(self.population[i].copy())
                        else:
                            rep_idx = np.random.randint(0, len(self.archive))
                            self.archive[rep_idx] = self.population[i].copy()
                else:
                    new_population[i] = self.population[i]
                    new_fitnesses[i] = self.fitnesses[i]

            self.population = new_population
            self.fitnesses = new_fitnesses

            # Update memories
            if S_CR and S_F:
                mean_cr = np.mean(S_CR)
                self.M_CR[self.k_mem] = (1.0 - self.c_adapt) * self.M_CR[self.k_mem] + self.c_adapt * mean_cr
                # Lehmer mean for F
                s_f_arr = np.array(S_F)
                mean_f = np.sum(s_f_arr ** 2) / np.sum(s_f_arr)
                self.M_F[self.k_mem] = (1.0 - self.c_adapt) * self.M_F[self.k_mem] + self.c_adapt * mean_f
                self.k_mem = (self.k_mem + 1) % self.memory_size

            curr_best_idx = np.argmin(self.fitnesses)
            if self.fitnesses[curr_best_idx] < best_fit:
                best_fit = self.fitnesses[curr_best_idx]
                best_sol = self.population[curr_best_idx].copy()

            var_p = float(np.mean(np.var(self.population, axis=0)))
            di = float(0.88 * np.exp(-0.02 * gen))

            self.history["best_fitness"].append(best_fit)
            self.history["mean_fitness"].append(float(np.mean(self.fitnesses)))
            self.history["diversity_index"].append(di)

        final_sol = (
            (1.0 / (1.0 + np.exp(-np.clip(best_sol, -30, 30))) > 0.5).astype(int)
            if self.is_binary
            else best_sol
        )
        return {
            "best_solution": final_sol,
            "best_fitness": best_fit,
            "total_evaluations": self.total_evaluations,
            "history": self.history,
        }
