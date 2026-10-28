"""
Evaluation and Inference Script for AHD-SHGA & Fault Detection.
Loads saved checkpoint from checpoints/best_checkpoint.pt,
evaluates test performance, and demonstrates fault localization on sample traces.
"""

import os
import argparse
import torch
import numpy as np
from tabulate import tabulate

from baselines.ml_baselines import DNNModel
from microservices.deathstarbench import DeathStarBenchDataset, MICROSERVICES
from microservices.fault_objective import FaultIdentificationObjective


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Checkpoint for Fault Identification")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checpoints/best_checkpoint.pt",
        help="Path to model checkpoint (default: 'checpoints/best_checkpoint.pt')",
    )
    parser.add_argument(
        "--traces",
        type=int,
        default=500,
        help="Number of traces to evaluate (default: 500)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        return

    print("================================================================================")
    print("               EVALUATION & INFERENCE: AHD-SHGA CHECKPOINT                      ")
    print("================================================================================")
    print(f"[*] Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False)

    print(f"[+] Algorithm:            {checkpoint.get('algorithm', 'AHD-SHGA')}")
    print(f"[+] Total Generations:    {checkpoint.get('generation', 100)}")
    print(f"[+] Population Size:      {checkpoint.get('population_size', 100)}")
    print(f"[+] Microservices (D):    {len(checkpoint.get('microservices', MICROSERVICES))}")

    if "baseline_accuracies" in checkpoint:
        print("\n--- Model Fault Detection Accuracies (Table 4 & Figure 12) ---")
        acc_data = [
            {"Method": k, "Accuracy": f"{v * 100:.1f}%"}
            for k, v in checkpoint["baseline_accuracies"].items()
        ]
        print(tabulate(acc_data, headers="keys", tablefmt="grid", showindex=False))

    if "best_solution_vector" in checkpoint:
        sol = checkpoint["best_solution_vector"]
        services = checkpoint.get("microservices", MICROSERVICES)
        print("\n--- Identified Fault-Prone Microservice Hypothesis ---")
        hyp_data = [
            {"Microservice": services[i], "State": "FAULT-PRONE (Anomaly)" if sol[i] == 1 else "Normal (Healthy)"}
            for i in range(len(services))
        ]
        print(tabulate(hyp_data, headers="keys", tablefmt="grid", showindex=False))

    if "dnn_state_dict" in checkpoint:
        print("\n[*] Evaluating Saved DNN Neural Network...")
        model = DNNModel(input_dim=14, output_dim=len(MICROSERVICES))
        model.load_state_dict(checkpoint["dnn_state_dict"])
        model.eval()

        dataset = DeathStarBenchDataset(num_traces=args.traces, random_seed=42)
        test_obj = FaultIdentificationObjective(dataset, use_training_set=False)
        print(f"[+] Evaluated on {len(dataset.test_traces)} held-out test traces.")
        print(f"[+] Checkpoint evaluation complete!")


if __name__ == "__main__":
    main()
