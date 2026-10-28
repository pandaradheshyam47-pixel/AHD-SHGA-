# Feedback-Driven Evolutionary Optimization for Stable and Efficient Fault Identification (AHD-SHGA)

This repository provides the complete, production-grade implementation of the research paper:

> **"Feedback-Driven Evolutionary Optimization for Stable and Efficient Fault Identification"**  
> *Radhe Shyam Panda, Manisha Sharma, Rajesh Tiwari, Vikas Pandey (2026)*

---

## 🔬 Overview & Key Scientific Principles

Evolutionary algorithms frequently face the fundamental trade-off between **exploration** (maintaining population diversity to avoid local optima) and **exploitation** (intensifying search around promising regions). Traditional methods apply heuristic parameter-tuning or static rules, often leading to **premature diversity collapse** or excessive computational cost.

The **AHD-SHGA** (*Adaptive Hyper-Diversity Surrogate-Assisted Hybrid Genetic Algorithm*) shifts evolutionary optimization from an unguided heuristic into a **closed-loop feedback-regulated control system**:

1. **Self-Evolving Chromosome Encoding**:  
   Every chromosome $C_i$ carries its own local operating parameters alongside decision variables:
   $$C_i = [x_{i1}, x_{i2}, \dots, x_{iD}, p_{c,i}, p_{m,i}, w_{h,i}]$$
   where $p_{c,i}$ is crossover probability, $p_{m,i}$ is mutation rate, and $w_{h,i}$ is hybrid switching weight.

2. **Unified Feedback Controller**:  
   Monitors the population state dynamics $S(t) = [DI(t), V_f(t), \Delta f(t)]$ and dynamically derives population-level reference control parameters:
   $$p_c(t) = p_c^{\min} + (p_c^{\max} - p_c^{\min}) \cdot \sigma(DI(t))$$
   $$p_m(t) = p_m^{\max} - (p_m^{\max} - p_m^{\min}) \cdot \sigma(DI(t))$$
   $$w_h(t) = \lambda_1 DI(t) + \lambda_2 V_f(t)$$
   $$\lambda_s(t) = \frac{1}{1 + e^{-k \cdot \Delta f(t)}}$$

3. **Composite Diversity Monitoring**:  
   Tracks genotypic entropy $H_g$, population variance $V_p$, and fitness variance $V_f$:
   $$DI(t) = c_1 H_g + c_2 V_p + c_3 V_f$$

4. **Diversity Recovery Override Mechanism**:  
   When $DI(t) < DI_{th}$, triggers an immediate 4-stage intervention for $R = 5$ generations:
   - **Mutation Boost**: $p_{m,i}(t) \leftarrow p_m^{\max}$
   - **Crossover Suppression**: $p_{c,i}(t) \leftarrow p_c^{\min}$
   - **Exploration Bias**: $w_{h,i}(t) \leftarrow \min(w_{h,i}(t), \tau - \epsilon)$
   - **Surrogate Evaluation Reduction**: $\lambda_s(t) \leftarrow \lambda_s^{\min}$

5. **Gaussian Process Regression (GPR) Surrogate**:  
   Approximates expensive evaluations using an RBF covariance kernel with predictive uncertainty estimation. Selects the top $k = 56$ candidates for true evaluation, reducing computational cost by $\sim 44\%$ while maintaining superior search accuracy.

6. **Distributed Software Fault Identification**:  
   Applied to the **DeathStarBench Traces Dataset** (microservices architecture: API Gateway, User Service, Post Service, Media Service, Storage Service, Recommendation Service, Database) to detect fault-prone services using multi-objective fitness:
   $$F(x) = \alpha \cdot DA(x) - \beta \cdot LD(x) - \gamma \cdot FP(x)$$
   achieving **96% fault detection accuracy**.

---

## 📁 Repository Structure

```text
implementation/
├── ahd_shga/                     # Core AHD-SHGA Framework Package
│   ├── __init__.py
│   ├── chromosome.py             # Chromosome encoding & parameter blending (Eqs. 2-5)
│   ├── diversity.py              # Genotypic entropy, variance & composite DI (Eqs. 22-27)
│   ├── feedback_controller.py    # Closed-loop controller & recovery mechanism (Eqs. 15-21)
│   ├── surrogate.py              # Gaussian Process Regression with RBF kernel (Eqs. 12-14)
│   ├── operators.py              # Crossover, mutation & differential refinement (Eqs. 28-32)
│   ├── archive.py                # Elite archive & convergence check (Eqs. 39-42)
│   └── optimizer.py              # Full Algorithm 1 implementation & ablation toggles
├── baselines/                    # Comparative Baseline Algorithms
│   ├── __init__.py
│   ├── traditional_ga.py         # Standard Genetic Algorithm
│   ├── pso.py                    # Particle Swarm Optimization
│   ├── jade.py                   # JADE with success-history & external archive (ref [32])
│   ├── ml_baselines.py           # PyTorch 3-layer DNN & Random Forest classifiers
│   └── metaheuristics.py         # NGA, MGEA, GA-SMLD, LLM-GWO, HA-GA-NNQ, LLM-ECG, etc.
├── benchmarks/                   # Continuous Benchmark Optimization Suite
│   ├── __init__.py
│   └── cec_benchmarks.py         # Rastrigin, CEC 2017 F1, F5, F10, F15
├── microservices/                # Distributed Microservice Fault Domain
│   ├── __init__.py
│   ├── deathstarbench.py         # Call graph, trace generator & 80/20 train/test split
│   └── fault_objective.py        # Multi-objective fault fitness evaluation (Eq. 7)
├── evaluation/                   # Evaluation Metrics & Statistical Analysis
│   ├── __init__.py
│   ├── metrics.py                # BFV, MFV, SDF, CR, DPI, EEBM, CEM (Eqs. 43-46)
│   └── statistical_tests.py      # Wilcoxon Signed-Rank & Friedman Ranking tests
├── experiments/                  # Experiment Orchestration & Visualization
│   ├── __init__.py
│   ├── plot_utils.py             # High-DPI publication styling
│   ├── run_visualizations.py     # Generates Figures 2 to 13
│   ├── run_benchmarks.py         # Executes benchmark tests & Tables 4, 6, 7, 9
│   ├── run_ablation.py           # Executes ablation study & Table 8
│   ├── run_microservice_evaluation.py # DeathStarBench experiments & Tables 3, 5
│   └── run_all.py                # Master execution runner
├── tests/                        # Pytest Test Suite
│   └── test_all.py               # Unit & integration tests for all modules

```

---

## 🚀 Quickstart & Execution

### 1. Full Training Program
Train the full evolutionary AHD-SHGA optimizer, deep neural network detector, and random forest models on the DeathStarBench dataset, saving the best checkpoint to `checpoints/best_checkpoint.pt`:
```bash
# Full training on all models with 100 generations
python train.py --mode all --generations 100 --pop_size 100

# Train only AHD-SHGA evolutionary search
python train.py --mode ahd_shga --generations 100

# Train deep neural network baseline
python train.py --mode dnn --dnn_epochs 60
```

### 2. Evaluate Checkpoint
Load the saved checkpoint from `checpoints/best_checkpoint.pt` and evaluate against held-out test traces:
```bash
python eval.py --checkpoint checpoints/best_checkpoint.pt
```

### 3. Run Unit Tests
Verify all core components, mathematical mappings, and integration logic:
```bash
python -m pytest tests/test_all.py -v
```

### 2. Generate All Figures (Figures 2 to 13)
Generate publication-quality 300 DPI figures:
```bash
python experiments/run_visualizations.py
```

### 3. Run Benchmark Suite (Tables 4, 6, 7, 9)
Execute Rastrigin and CEC 2017 benchmark evaluations:
```bash
python experiments/run_benchmarks.py
```

### 4. Run Ablation Study (Table 8)
Evaluate impact of individual mechanisms (w/o DR, w/o PE, w/o HI, w/o SA):
```bash
python experiments/run_ablation.py
```

### 5. Run Microservice Software Fault Identification
Evaluate on DeathStarBench traces with 80% train / 20% test split:
```bash
python experiments/run_microservice_evaluation.py
```

### 6. Run Complete Pipeline (End-to-End)
```bash
python experiments/run_all.py
```

---

## 📊 Summary of Results Replicated

| Table / Figure | Description | Key Findings |
|---|---|---|
| **Figure 2** | Rastrigin Convergence Curves | AHD-SHGA drops steadily from ~0.020 to **0.005** with the narrowest confidence interval. |
| **Figure 3** | Population Diversity ($DI$) | AHD-SHGA preserves diversity at **0.62**, preventing collapse seen in baselines ($0.22 - 0.30$). |
| **Figure 4** | Exploration-Exploitation Balance | Adaptive hybrid weight transitions smoothly from **0.21** to **0.79**. |
| **Figure 5 & Table 3** | Surrogate Prediction Accuracy | GPR predictions tightly match true evaluations: **RMSE = 0.00042**, **MAE = 0.00031**. |
| **Figure 6** | Computational Cost | True evaluations cut by **44%** ($168,000$ vs $300,000$). |
| **Figure 7** | 2D Landscape Trajectories | Visualizes candidate flow across multi-modal valleys into global minimum. |
| **Figures 8-11** | Microservice Trace Profiling | Call graph dependencies, latency distributions, operation heatmap, and Gantt timeline. |
| **Figure 12** | Fault Detection Accuracy | AHD-SHGA achieves **96%**, outperforming GA (91%), PSO (90%), RF (88%), and DNN (86%). |
| **Figure 13** | Fault Score Progression | Optimizes score from 0.60 to **0.96** over 100 generations. |
| **Table 4** | Algorithm Performance Comparison | AHD-SHGA wins on BFV (0.005), CR (0.90), DPI (0.82), EEBM (0.88), CEM (0.85). |
| **Tables 6 & 7** | Statistical Significance | Wilcoxon p-values $< 0.005$; Friedman mean rank **1.15** across 13 algorithms. |
| **Table 8** | Ablation Study | Removing surrogate drops efficiency to 0.60; removing diversity regulation drops BFV to 0.008. |
| **Table 9** | CEC 2017 Benchmarks | Outperforms LSHADE, CMA-ES, and PSO across F1, F5, F10, F15. |
