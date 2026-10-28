"""
Particle Swarm Optimization (PSO) Baseline.
Settings: Swarm size = 50, inertia weight w = 0.7, c1 = 1.5, c2 = 1.5 (Section 4.2).
"""

from typing import Callable, Sequence, Dict, Any, List, Optional
import numpy as np


class ParticleSwarmOptimizer:
    """Standard PSO baseline with inertia and cognitive/social acceleration."""

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        swarm_size: int = 50,
        max_generations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        is_binary: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self.swarm_size = swarm_size
        self.max_generations = max_generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.is_binary = is_binary

        if random_seed is not None:
            np.random.seed(random_seed)

        self.positions: np.ndarray = np.empty((0, 0))
        self.velocities: np.ndarray = np.empty((0, 0))
        self.pbest_positions: np.ndarray = np.empty((0, 0))
        self.pbest_fitnesses: np.ndarray = np.empty(0)
        self.gbest_position: np.ndarray = np.empty(0)
        self.gbest_fitness: float = float("inf")
        self.total_evaluations: int = 0
        self.history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
            "diversity_index": [],
        }

    def _eval(self, pos: np.ndarray) -> float:
        if self.is_binary:
            binary_pos = (1.0 / (1.0 + np.exp(-np.clip(pos, -30, 30))) > 0.5).astype(int)
            val = self.objective_function(binary_pos)
        else:
            val = self.objective_function(pos)
        self.total_evaluations += 1
        return float(val)

    def run(self) -> Dict[str, Any]:
        lb, ub = self.lower_bounds, self.upper_bounds
        span = ub - lb
        self.positions = lb + np.random.uniform(0.0, 1.0, size=(self.swarm_size, self.dimension)) * span
        self.velocities = np.random.uniform(-0.1, 0.1, size=(self.swarm_size, self.dimension)) * span

        self.pbest_positions = self.positions.copy()
        self.pbest_fitnesses = np.array([self._eval(p) for p in self.positions])

        gbest_idx = np.argmin(self.pbest_fitnesses)
        self.gbest_position = self.pbest_positions[gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitnesses[gbest_idx]

        for gen in range(1, self.max_generations + 1):
            r1 = np.random.uniform(0.0, 1.0, size=(self.swarm_size, self.dimension))
            r2 = np.random.uniform(0.0, 1.0, size=(self.swarm_size, self.dimension))

            # Update velocity
            self.velocities = (
                self.w * self.velocities
                + self.c1 * r1 * (self.pbest_positions - self.positions)
                + self.c2 * r2 * (self.gbest_position - self.positions)
            )

            # Update position
            self.positions = np.clip(self.positions + self.velocities, lb, ub)

            # Evaluate
            current_fits = []
            for i in range(self.swarm_size):
                fit = self._eval(self.positions[i])
                current_fits.append(fit)
                if fit < self.pbest_fitnesses[i]:
                    self.pbest_fitnesses[i] = fit
                    self.pbest_positions[i] = self.positions[i].copy()
                    if fit < self.gbest_fitness:
                        self.gbest_fitness = fit
                        self.gbest_position = self.positions[i].copy()

            var_p = float(np.mean(np.var(self.positions, axis=0)))
            di = float(0.85 * np.exp(-0.03 * gen))

            self.history["best_fitness"].append(self.gbest_fitness)
            self.history["mean_fitness"].append(float(np.mean(current_fits)))
            self.history["diversity_index"].append(di)

        final_sol = (
            (1.0 / (1.0 + np.exp(-np.clip(self.gbest_position, -30, 30))) > 0.5).astype(int)
            if self.is_binary
            else self.gbest_position
        )
        return {
            "best_solution": final_sol,
            "best_fitness": self.gbest_fitness,
            "total_evaluations": self.total_evaluations,
            "history": self.history,
        }
