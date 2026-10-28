"""
Main AHD-SHGA Optimizer Class.
Implements Algorithm 1 (Adaptive Hyper-Diversity Surrogate-Assisted Hybrid Genetic Algorithm)
with full support for standard optimization, binary fault identification, and ablation modes.
"""

from typing import Callable, List, Optional, Sequence, Dict, Any
import numpy as np

from .chromosome import Chromosome
from .feedback_controller import FeedbackController
from .diversity import DiversityMonitor
from .surrogate import GaussianProcessSurrogate
from .operators import EvolutionaryOperators
from .archive import EliteArchive


class AHDSHGAOptimizer:
    """
    Adaptive Hyper-Diversity Surrogate-Assisted Hybrid Genetic Algorithm.
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        dimension: int,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        population_size: int = 100,
        max_generations: int = 100,
        elite_true_k: int = 56,
        archive_capacity: int = 10,
        di_threshold: float = 0.45,
        recovery_window: int = 5,
        p_c_bounds: tuple[float, float] = (0.6, 0.9),
        p_m_bounds: tuple[float, float] = (0.01, 0.1),
        convergence_epsilon: float = 1e-6,
        is_binary_problem: bool = False,
        # Ablation study flags (Section 4.20)
        disable_diversity_regulation: bool = False,
        disable_parameter_evolution: bool = False,
        disable_hybrid_interaction: bool = False,
        disable_surrogate_assistance: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.objective_function = objective_function
        self.dimension = dimension
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_true_k = elite_true_k
        self.archive_capacity = archive_capacity
        self.di_threshold = di_threshold
        self.recovery_window = recovery_window
        self.p_c_bounds = p_c_bounds
        self.p_m_bounds = p_m_bounds
        self.convergence_epsilon = convergence_epsilon
        self.is_binary_problem = is_binary_problem

        # Ablation flags
        self.disable_diversity_regulation = disable_diversity_regulation
        self.disable_parameter_evolution = disable_parameter_evolution
        self.disable_hybrid_interaction = disable_hybrid_interaction
        self.disable_surrogate_assistance = disable_surrogate_assistance

        if random_seed is not None:
            np.random.seed(random_seed)

        # Core Modules
        self.diversity_monitor = DiversityMonitor(di_threshold=di_threshold)
        self.feedback_controller = FeedbackController(
            p_c_min=p_c_bounds[0],
            p_c_max=p_c_bounds[1],
            p_m_min=p_m_bounds[0],
            p_m_max=p_m_bounds[1],
            recovery_window=recovery_window,
        )
        self.surrogate = GaussianProcessSurrogate(
            length_scale=1.0,
            noise_level=1e-4,
            random_state=random_seed or 42,
        )
        self.operators = EvolutionaryOperators(
            tau=0.5,
            delta=0.5,
            eta_blend=0.5,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        self.archive = EliteArchive(
            capacity=archive_capacity,
            epsilon=convergence_epsilon,
        )

        # Metrics and runtime state
        self.population: List[Chromosome] = []
        self.total_true_evaluations: int = 0
        self.total_exploration_ops: int = 0
        self.total_refinement_ops: int = 0

        # Run history
        self.history: Dict[str, List[Any]] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "std_fitness": [],
            "diversity_index": [],
            "h_g": [],
            "v_p": [],
            "v_f": [],
            "p_c_ref": [],
            "p_m_ref": [],
            "w_h_ref": [],
            "lambda_s_ref": [],
            "true_evaluations_count": [],
            "exploration_ratio": [],
            "surrogate_rmse": [],
            "surrogate_mae": [],
            "is_recovering": [],
        }

    def _eval_individual(self, chromosome: Chromosome) -> float:
        """Evaluate chromosome using the true objective function."""
        if self.is_binary_problem:
            vector = chromosome.to_binary()
        else:
            vector = chromosome.genes

        fit = self.objective_function(vector)
        chromosome.fitness = float(fit)
        chromosome.evaluated_with_true = True
        self.total_true_evaluations += 1
        return float(fit)

    def initialize_population(self) -> None:
        """Step 1 & 2: Initialize population and adaptive parameters."""
        self.population = [
            Chromosome.random(
                dimension=self.dimension,
                lower_bounds=self.lower_bounds,
                upper_bounds=self.upper_bounds,
                p_c_bounds=self.p_c_bounds,
                p_m_bounds=self.p_m_bounds,
            )
            for _ in range(self.population_size)
        ]

        if self.disable_parameter_evolution:
            # Fixed parameters for w/o PE ablation
            for c in self.population:
                c.p_c = 0.8
                c.p_m = 0.05
                c.w_h = 0.5

        # Step 3: Evaluate initial population with true objective
        X_train, y_train = [], []
        for ind in self.population:
            fit = self._eval_individual(ind)
            X_train.append(ind.genes)
            y_train.append(fit)

        # Update initial archive
        self.archive.update(self.population)

        # Step 4: Train initial surrogate model (GPR)
        if not self.disable_surrogate_assistance:
            self.surrogate.fit(np.array(X_train), np.array(y_train))

    def run(self) -> Dict[str, Any]:
        """
        Execute Algorithm 1 optimization loop for max_generations or until convergence.
        """
        self.initialize_population()
        search_range_norm = float(np.mean(self.upper_bounds - self.lower_bounds))

        for gen in range(1, self.max_generations + 1):
            # Step 6 & 7: Diversity monitoring
            di, h_g, v_p, v_f = self.diversity_monitor.compute_diversity_index(
                self.population, search_range_norm=search_range_norm
            )

            # Step 8: Closed-loop feedback controller
            delta_f = self.archive.compute_improvement()
            if self.disable_diversity_regulation:
                ctrl = {
                    "p_c": 0.8,
                    "p_m": 0.05,
                    "w_h": 0.5,
                    "lambda_s": 0.8,
                    "is_recovering": False,
                }
            else:
                ctrl = self.feedback_controller.compute_control_variables(
                    di=di,
                    v_f=v_f,
                    delta_f=delta_f,
                    di_threshold=self.di_threshold,
                )

            # Step 17-19: Generate offspring population
            best_ind = self.archive.get_best()
            offspring_population: List[Chromosome] = []
            gen_explorations = 0
            gen_refinements = 0

            for ind in self.population:
                if self.disable_hybrid_interaction:
                    # w/o HI ablation: purely GA exploration
                    parents = self.operators.select_parents(self.population, 2)
                    c_eff, m_eff, _ = ind.get_effective_parameters(ctrl["p_c"], ctrl["p_m"], 0.0)
                    child = self.operators.apply_crossover(ind, parents[1], c_eff)
                    offspring = self.operators.apply_mutation(child, m_eff)
                    mode = "exploration"
                else:
                    offspring, mode = self.operators.evolve_candidate(
                        individual=ind,
                        population=self.population,
                        best_individual=best_ind,
                        ref_p_c=ctrl["p_c"],
                        ref_p_m=ctrl["p_m"],
                        ref_w_h=ctrl["w_h"],
                    )

                if mode == "exploration":
                    gen_explorations += 1
                else:
                    gen_refinements += 1

                offspring_population.append(offspring)

            self.total_exploration_ops += gen_explorations
            self.total_refinement_ops += gen_refinements

            # Step 20: Surrogate-assisted evaluation
            if self.disable_surrogate_assistance:
                # w/o SA: evaluate ALL candidates with true objective
                for off in offspring_population:
                    self._eval_individual(off)
                selected_elites = offspring_population
                rmse_val, mae_val = 0.0, 0.0
            else:
                offspring_genes = np.array([c.genes for c in offspring_population])
                pred_means, pred_stds = self.surrogate.predict(offspring_genes)

                for idx, off in enumerate(offspring_population):
                    off.predicted_fitness = float(pred_means[idx])
                    off.uncertainty = float(pred_stds[idx])
                    # Lower confidence bound for ranking: pred - 0.5 * std (minimization)
                    off.fitness = float(pred_means[idx])
                    off.evaluated_with_true = False

                # Rank by surrogate predicted fitness
                sort_indices = np.argsort(pred_means)
                num_true_k = min(self.elite_true_k, len(offspring_population))

                # Hybrid selection: top elites + exploration samples guided by lambda_s
                num_pure_elite = int(num_true_k * ctrl["lambda_s"])
                num_pure_elite = max(1, min(num_pure_elite, num_true_k))
                num_exploratory = num_true_k - num_pure_elite

                elite_indices = list(sort_indices[:num_pure_elite])
                remaining_indices = list(sort_indices[num_pure_elite:])
                if num_exploratory > 0 and len(remaining_indices) >= num_exploratory:
                    # Select candidates with highest uncertainty or random exploration
                    uncertainties = pred_stds[remaining_indices]
                    explore_pick = np.argsort(uncertainties)[-num_exploratory:]
                    elite_indices.extend([remaining_indices[p] for p in explore_pick])
                else:
                    elite_indices = list(sort_indices[:num_true_k])

                selected_elites = []
                true_X, true_y = [], []
                for idx in elite_indices:
                    cand = offspring_population[idx]
                    fit = self._eval_individual(cand)
                    true_X.append(cand.genes)
                    true_y.append(fit)
                    selected_elites.append(cand)

                # Compute surrogate accuracy metrics for this generation
                metrics = self.surrogate.evaluate_metrics(np.array(true_X), np.array(true_y))
                rmse_val = metrics["rmse"]
                mae_val = metrics["mae"]

                # Step 22: Incremental surrogate update
                self.surrogate.update(np.array(true_X), np.array(true_y))

            # Step 21: Update elite archive
            current_best = self.archive.update(selected_elites)

            # Form next generation population (elitist replacement)
            # Combine current population and offspring, sort by fitness
            combined = self.population + offspring_population
            combined.sort(key=lambda c: c.fitness if c.fitness is not None else float("inf"))
            self.population = combined[: self.population_size]

            # Record generation statistics
            fitness_values = [c.fitness for c in self.population if c.fitness is not None]
            best_fit = float(current_best.fitness) if current_best.fitness is not None else 0.0
            mean_fit = float(np.mean(fitness_values))
            std_fit = float(np.std(fitness_values))
            expl_ratio = (
                self.total_exploration_ops
                / max(1, self.total_exploration_ops + self.total_refinement_ops)
            )

            self.history["generation"].append(gen)
            self.history["best_fitness"].append(best_fit)
            self.history["mean_fitness"].append(mean_fit)
            self.history["std_fitness"].append(std_fit)
            self.history["diversity_index"].append(di)
            self.history["h_g"].append(h_g)
            self.history["v_p"].append(v_p)
            self.history["v_f"].append(v_f)
            self.history["p_c_ref"].append(ctrl["p_c"])
            self.history["p_m_ref"].append(ctrl["p_m"])
            self.history["w_h_ref"].append(ctrl["w_h"])
            self.history["lambda_s_ref"].append(ctrl["lambda_s"])
            self.history["true_evaluations_count"].append(self.total_true_evaluations)
            self.history["exploration_ratio"].append(expl_ratio)
            self.history["surrogate_rmse"].append(rmse_val)
            self.history["surrogate_mae"].append(mae_val)
            self.history["is_recovering"].append(ctrl["is_recovering"])

            # Step 24: Check termination
            if gen >= self.max_generations:
                break
            if gen > 15 and self.archive.has_converged():
                break

        final_best = self.archive.get_best()
        final_genes = final_best.genes if final_best is not None else self.population[0].genes
        final_solution = (
            final_best.to_binary() if self.is_binary_problem and final_best else final_genes
        )
        final_fitness = final_best.fitness if final_best and final_best.fitness is not None else 0.0

        return {
            "best_solution": final_solution,
            "best_fitness": final_fitness,
            "final_archive": self.archive.archive,
            "total_evaluations": self.total_true_evaluations,
            "history": self.history,
        }
