"""
Adaptive Hybrid Evolutionary Operators for AHD-SHGA.
Implements:
1. Fitness proportionate selection (Eq. 29)
2. Hybrid search mode selection (Eq. 28)
3. Exploration operators:
   - Arithmetic crossover (Eq. 30)
   - Stochastic Gaussian mutation (Eq. 31)
4. Refinement operators:
   - Differential local search update (Eq. 32)
5. Continuous-to-binary sigmoid transfer mapping
"""

from typing import List, Sequence, Tuple
import numpy as np
from .chromosome import Chromosome


class EvolutionaryOperators:
    """
    Manages selection, hybrid switching, crossover, mutation, and local refinement.
    """

    def __init__(
        self,
        tau: float = 0.5,
        delta: float = 0.5,
        eta_blend: float = 0.5,
        lower_bounds: Sequence[float] = (),
        upper_bounds: Sequence[float] = (),
    ):
        self.tau = tau
        self.delta = delta
        self.eta_blend = eta_blend
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.float64) if len(lower_bounds) > 0 else None
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.float64) if len(upper_bounds) > 0 else None

    def select_parents(self, population: Sequence[Chromosome], num_parents: int = 2) -> List[Chromosome]:
        """
        Fitness-proportionate selection (Eq. 29).
        Favors individuals with lower fitness for minimization problems.
        """
        fitnesses = np.array([c.fitness if c.fitness is not None else 1e6 for c in population], dtype=np.float64)
        # Shift fitness so minimum gets highest weight
        max_f = np.max(fitnesses)
        min_f = np.min(fitnesses)
        span = max_f - min_f if max_f > min_f else 1.0

        inverted_scores = (max_f - fitnesses) / span + 1e-4
        probs = inverted_scores / np.sum(inverted_scores)

        selected_indices = np.random.choice(len(population), size=num_parents, p=probs, replace=True)
        return [population[i] for i in selected_indices]

    def apply_crossover(
        self,
        parent_a: Chromosome,
        parent_b: Chromosome,
        effective_p_c: float,
    ) -> Chromosome:
        """
        Arithmetic crossover (Eq. 30):
        X_i^(c) = lambda * X_a + (1 - lambda) * X_b
        Recombines decision variables and evolves local adaptive parameters.
        """
        if np.random.uniform(0.0, 1.0) < effective_p_c:
            lam = np.random.uniform(0.0, 1.0)
            child_genes = lam * parent_a.genes + (1.0 - lam) * parent_b.genes
            child_p_c = lam * parent_a.p_c + (1.0 - lam) * parent_b.p_c
            child_p_m = lam * parent_a.p_m + (1.0 - lam) * parent_b.p_m
            child_w_h = lam * parent_a.w_h + (1.0 - lam) * parent_b.w_h
        else:
            child_genes = parent_a.genes.copy()
            child_p_c = parent_a.p_c
            child_p_m = parent_a.p_m
            child_w_h = parent_a.w_h

        return Chromosome(
            genes=child_genes,
            p_c=child_p_c,
            p_m=child_p_m,
            w_h=child_w_h,
            p_c_bounds=parent_a.p_c_bounds,
            p_m_bounds=parent_a.p_m_bounds,
            w_h_bounds=parent_a.w_h_bounds,
        )

    def apply_mutation(
        self,
        individual: Chromosome,
        effective_p_m: float,
    ) -> Chromosome:
        """
        Stochastic variation (Eq. 31):
        x_id' = x_id + p_m,i * epsilon, epsilon ~ N(0, sigma^2)
        Also introduces minor perturbation into the local adaptive parameters.
        """
        mutated_genes = individual.genes.copy()
        D = len(mutated_genes)

        mask = np.random.uniform(0.0, 1.0, size=D) < effective_p_m
        if np.any(mask):
            scale = 0.1
            if self.lower_bounds is not None and self.upper_bounds is not None:
                scale = 0.1 * (self.upper_bounds - self.lower_bounds)
            perturbation = np.random.normal(0.0, scale, size=D)
            mutated_genes[mask] += perturbation[mask]

            if self.lower_bounds is not None and self.upper_bounds is not None:
                mutated_genes = np.clip(mutated_genes, self.lower_bounds, self.upper_bounds)

        # Perturb local parameters slightly to enable evolutionary adaptation
        mut_p_c = individual.p_c + np.random.normal(0.0, 0.02)
        mut_p_m = individual.p_m + np.random.normal(0.0, 0.005)
        mut_w_h = individual.w_h + np.random.normal(0.0, 0.05)

        return Chromosome(
            genes=mutated_genes,
            p_c=mut_p_c,
            p_m=mut_p_m,
            w_h=mut_w_h,
            p_c_bounds=individual.p_c_bounds,
            p_m_bounds=individual.p_m_bounds,
            w_h_bounds=individual.w_h_bounds,
        )

    def apply_differential_refinement(
        self,
        individual: Chromosome,
        best_individual: Chromosome,
        r1_individual: Chromosome,
        r2_individual: Chromosome,
        effective_w_h: float,
    ) -> Chromosome:
        """
        Differential refinement update (Eq. 32):
        X_i^(r) = X_i + w_h,i * (X_best - X_i) + delta * (X_r1 - X_r2)
        """
        x_i = individual.genes
        x_best = best_individual.genes
        x_r1 = r1_individual.genes
        x_r2 = r2_individual.genes

        refined_genes = x_i + effective_w_h * (x_best - x_i) + self.delta * (x_r1 - x_r2)

        if self.lower_bounds is not None and self.upper_bounds is not None:
            refined_genes = np.clip(refined_genes, self.lower_bounds, self.upper_bounds)

        return Chromosome(
            genes=refined_genes,
            p_c=individual.p_c,
            p_m=individual.p_m,
            w_h=individual.w_h,
            p_c_bounds=individual.p_c_bounds,
            p_m_bounds=individual.p_m_bounds,
            w_h_bounds=individual.w_h_bounds,
        )

    def evolve_candidate(
        self,
        individual: Chromosome,
        population: Sequence[Chromosome],
        best_individual: Chromosome,
        ref_p_c: float,
        ref_p_m: float,
        ref_w_h: float,
    ) -> Tuple[Chromosome, str]:
        """
        Executes one evolutionary transition for a candidate:
        Determines mode via Eq. 28, applies Exploration (Eqs. 30-31) or Refinement (Eq. 32).
        Returns: (offspring, mode_string)
        """
        eff_p_c, eff_p_m, eff_w_h = individual.get_effective_parameters(
            ref_p_c, ref_p_m, ref_w_h, eta=self.eta_blend
        )

        # Mode determination: Eq. 28
        if eff_w_h < self.tau:
            # Exploration Mode
            parents = self.select_parents(population, num_parents=2)
            crossover_child = self.apply_crossover(individual, parents[1], eff_p_c)
            offspring = self.apply_mutation(crossover_child, eff_p_m)
            mode = "exploration"
        else:
            # Refinement Mode
            n = len(population)
            indices = np.random.choice(n, size=2, replace=False)
            r1, r2 = population[indices[0]], population[indices[1]]
            offspring = self.apply_differential_refinement(
                individual, best_individual, r1, r2, eff_w_h
            )
            mode = "refinement"

        return offspring, mode
