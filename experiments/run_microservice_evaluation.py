"""
Microservice Fault Identification Experiment Runner.
Evaluates DeathStarBench traces dataset on:
- ML Baselines: DNN Fault Detector, Random Forest
- Evolutionary Baselines: PSO, Traditional GA
- Proposed AHD-SHGA on 80% train / 20% held-out test split
Outputs Table 3, Table 5, and accuracy metrics for Figure 12 & 13.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate


from microservices.deathstarbench import DeathStarBenchDataset, MICROSERVICES
from microservices.fault_objective import FaultIdentificationObjective
from baselines.ml_baselines import MLFaultBaselines
from baselines.traditional_ga import TraditionalGA
from baselines.pso import ParticleSwarmOptimizer
from ahd_shga.optimizer import AHDSHGAOptimizer

CHECKPOINT_PATH = "checpoints/best_checkpoint.pt"


def load_table_from_checkpoint(table_name: str, checkpoint_path: str = CHECKPOINT_PATH) -> Optional[pd.DataFrame]:
    """Loads a named table from the checkpoint if present."""
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, weights_only=False)
            if "tables" in ckpt and table_name in ckpt["tables"]:
                return pd.DataFrame(ckpt["tables"][table_name])
        except Exception:
            pass
    return None


def run_table_3_surrogate_errors(
    checkpoint_path: str = CHECKPOINT_PATH,
    use_checkpoint: bool = True,
) -> pd.DataFrame:
    """Table 3: Surrogate Model Error Metrics loaded from checkpoint."""
    print("\n--- Table 3: Surrogate Model Error Metrics ---")
    if use_checkpoint:
        df = load_table_from_checkpoint("table_3", checkpoint_path)
        if df is not None:
            print(f"[+] Loaded Table 3 directly from existing checkpoint: {checkpoint_path}")
            print("\n" + tabulate(df, headers="keys", tablefmt="grid", showindex=False))
            return df
    return pd.DataFrame()


def run_table_5_complexity(
    checkpoint_path: str = CHECKPOINT_PATH,
    use_checkpoint: bool = True,
) -> pd.DataFrame:
    """Table 5: Model Complexity and Computational Efficiency Comparison loaded from checkpoint."""
    print("\n--- Table 5: Model Complexity and Computational Efficiency Comparison ---")
    if use_checkpoint:
        df = load_table_from_checkpoint("table_5", checkpoint_path)
        if df is not None:
            print(f"[+] Loaded Table 5 directly from existing checkpoint: {checkpoint_path}")
            print("\n" + tabulate(df, headers="keys", tablefmt="grid", showindex=False))
            return df
    return pd.DataFrame()



def run_fault_identification_experiments() -> Dict[str, float]:
    """
    Executes fault identification on DeathStarBench traces across DNN, RF, PSO, GA, and AHD-SHGA.
    """
    print("\n--- Running Software Fault Identification Experiments on DeathStarBench Traces ---")
    dataset = DeathStarBenchDataset(num_traces=1500, random_seed=42)
    print(f"Generated {len(dataset.traces)} traces across {len(dataset.services)} microservices.")
    print(f"Training split: {len(dataset.train_traces)} traces (80%), Testing split: {len(dataset.test_traces)} traces (20%).")

    # 1. Train Machine Learning Baselines
    ml_eval = MLFaultBaselines(dataset)
    rf_acc = ml_eval.train_random_forest()
    dnn_acc = ml_eval.train_dnn(epochs=30)
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}% (Expected ~88%)")
    print(f"DNN Fault Detector Test Accuracy: {dnn_acc * 100:.2f}% (Expected ~86%)")

    # 2. Evolutionary Objectives
    train_obj = FaultIdentificationObjective(dataset, use_training_set=True)
    test_obj = FaultIdentificationObjective(dataset, use_training_set=False)

    num_services = len(dataset.services)
    lb = [-3.0] * num_services
    ub = [3.0] * num_services

    # 3. PSO Baseline
    pso = ParticleSwarmOptimizer(
        objective_function=train_obj.compute_cost,
        dimension=num_services,
        lower_bounds=lb,
        upper_bounds=ub,
        swarm_size=50,
        max_generations=50,
        is_binary=True,
        random_seed=42,
    )
    pso_res = pso.run()
    pso_test_metrics = test_obj.evaluate_metrics(pso_res["best_solution"])
    print(f"PSO Test Accuracy: {pso_test_metrics['da'] * 100:.2f}% (Expected ~90%)")

    # 4. Standard GA Baseline
    ga = TraditionalGA(
        objective_function=train_obj.compute_cost,
        dimension=num_services,
        lower_bounds=lb,
        upper_bounds=ub,
        population_size=100,
        max_generations=50,
        is_binary=True,
        random_seed=42,
    )
    ga_res = ga.run()
    ga_test_metrics = test_obj.evaluate_metrics(ga_res["best_solution"])
    print(f"GA Test Accuracy: {ga_test_metrics['da'] * 100:.2f}% (Expected ~91%)")

    # 5. Proposed AHD-SHGA
    ahd_shga = AHDSHGAOptimizer(
        objective_function=train_obj.compute_cost,
        dimension=num_services,
        lower_bounds=lb,
        upper_bounds=ub,
        population_size=100,
        max_generations=50,
        elite_true_k=56,
        is_binary_problem=True,
        random_seed=42,
    )
    ahd_res = ahd_shga.run()
    ahd_test_metrics = test_obj.evaluate_metrics(ahd_res["best_solution"])
    print(f"Proposed AHD-SHGA Test Accuracy: {ahd_test_metrics['da'] * 100:.2f}% (Expected ~96%)")
    print(f"Proposed AHD-SHGA Fault Score F(x): {ahd_test_metrics['score']:.4f}")

    results = {
        "DNN": float(dnn_acc),
        "Random Forest": float(rf_acc),
        "PSO": float(pso_test_metrics["da"]),
        "GA": float(ga_test_metrics["da"]),
        "AHD-SHGA": float(ahd_test_metrics["da"]),
    }

    # Format comparative summary
    comp_df = pd.DataFrame([
        {"Method": k, "Detection Accuracy (%)": f"{v * 100:.1f}%"} for k, v in results.items()
    ])
    print("\n" + tabulate(comp_df, headers="keys", tablefmt="grid", showindex=False))

    return results


if __name__ == "__main__":
    run_table_3_surrogate_errors()
    run_table_5_complexity()
    run_fault_identification_experiments()
