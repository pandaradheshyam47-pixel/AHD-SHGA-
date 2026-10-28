"""
Master Execution Pipeline.
Executes all benchmark and microservice fault identification experiments,
generates all 12 publication figures, and outputs all comparative tables.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_visualizations import generate_all_figures
from experiments.run_benchmarks import (
    run_table_4_comparison,
    run_table_6_wilcoxon,
    run_table_7_friedman,
    run_table_9_cec_comparison,
)
from experiments.run_ablation import run_ablation_experiments
from experiments.run_microservice_evaluation import (
    run_table_3_surrogate_errors,
    run_table_5_complexity,
    run_fault_identification_experiments,
)


def main():
    print("================================================================================")
    print("  FEEDBACK-DRIVEN EVOLUTIONARY OPTIMIZATION (AHD-SHGA) REPLICATION SUITE")
    print("================================================================================")
    start_time = time.time()

    results_dir = os.path.abspath("results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Generate Figures 2 to 13
    print("\n[Phase 1/4] Generating all 12 publication figures...")
    generate_all_figures(output_dir=figures_dir)

    # 2. Benchmark Experiments & Comparative Tables
    print("\n[Phase 2/4] Running Benchmark Experiments and Statistical Analysis...")
    df_table_4 = run_table_4_comparison(runs_per_algorithm=2)
    df_table_4.to_csv(os.path.join(results_dir, "table_4_performance_comparison.csv"), index=False)

    df_table_6 = run_table_6_wilcoxon()
    df_table_6.to_csv(os.path.join(results_dir, "table_6_wilcoxon_tests.csv"), index=False)

    df_table_7 = run_table_7_friedman()
    df_table_7.to_csv(os.path.join(results_dir, "table_7_friedman_ranking.csv"), index=False)

    df_table_9 = run_table_9_cec_comparison()
    df_table_9.to_csv(os.path.join(results_dir, "table_9_cec2017_comparison.csv"), index=False)

    # 3. Ablation Experiments
    print("\n[Phase 3/4] Running Ablation Experiments (Table 8)...")
    df_table_8 = run_ablation_experiments(num_test_runs=2)
    df_table_8.to_csv(os.path.join(results_dir, "table_8_ablation_study.csv"), index=False)

    # 4. Microservice Fault Identification
    print("\n[Phase 4/4] Running DeathStarBench Microservice Fault Identification...")
    df_table_3 = run_table_3_surrogate_errors()
    df_table_3.to_csv(os.path.join(results_dir, "table_3_surrogate_errors.csv"), index=False)

    df_table_5 = run_table_5_complexity()
    df_table_5.to_csv(os.path.join(results_dir, "table_5_model_complexity.csv"), index=False)

    run_fault_identification_experiments()

    elapsed = time.time() - start_time
    print("\n================================================================================")
    print(f"  EXECUTION COMPLETE IN {elapsed:.2f} SECONDS")
    print(f"  Figures saved to: {figures_dir}")
    print(f"  Tables saved to:  {results_dir}")
    print("================================================================================")


if __name__ == "__main__":
    main()
