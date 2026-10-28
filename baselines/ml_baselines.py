"""
Machine Learning Fault Detection Baselines (Section 4.2 & 4.14).
1. Deep Neural Network (DNN): 3 hidden layers, ReLU activation, Adam optimizer.
2. Random Forest (RF): 100 estimators, maximum depth 10.
Evaluated on the held-out 20% DeathStarBench test traces.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import torch
import torch.nn as nn
import torch.optim as optim

from microservices.deathstarbench import DeathStarBenchDataset


class DNNModel(nn.Module):
    """3-hidden layer Feedforward Neural Network with ReLU activations."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLFaultBaselines:
    """Trains and tests standard ML fault detection baselines on trace data."""

    def __init__(self, dataset: DeathStarBenchDataset):
        self.dataset = dataset
        self.services = dataset.services
        self.num_services = len(self.services)

        self.X_train, self.y_train = self._extract_features(dataset.train_traces)
        self.X_test, self.y_test = self._extract_features(dataset.test_traces)

    def _extract_features(self, traces: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract latency vector and status indicators for each service as input features X,
        and ground-truth binary fault vector as target y.
        """
        N = len(traces)
        # Features: observed latency per service (7) + status code error flag per service (7) = 14 features
        X = np.zeros((N, self.num_services * 2), dtype=np.float32)
        y = np.zeros((N, self.num_services), dtype=np.float32)

        for i, tr in enumerate(traces):
            y[i, :] = tr["ground_truth_vector"]
            lat_map = {sp["service_name"]: sp["latency"] for sp in tr["spans"]}
            err_map = {sp["service_name"]: (1.0 if sp["status_code"] >= 500 else 0.0) for sp in tr["spans"]}

            for s_idx, s_name in enumerate(self.services):
                nom_lat = self.dataset.nominal_latencies[s_name]
                obs_lat = lat_map.get(s_name, nom_lat)
                # Feature 1: normalized latency
                X[i, s_idx] = obs_lat / nom_lat
                # Feature 2: error status flag
                X[i, self.num_services + s_idx] = err_map.get(s_name, 0.0)

        return X, y

    def train_random_forest(self) -> float:
        """
        Random Forest with 100 estimators and max depth 10.
        Returns accuracy on test set (Figure 12: ~88%).
        """
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf = MultiOutputClassifier(base_rf)
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)
        # Overall element-wise detection accuracy: (TP + TN) / Total
        accuracy = float(np.mean(y_pred == self.y_test))
        return accuracy

    def train_dnn(self, epochs: int = 50, lr: float = 0.005) -> float:
        """
        Deep Neural Network with 3 hidden layers, ReLU activation, Adam optimizer.
        Returns accuracy on test set (Figure 12: ~86%).
        """
        torch.manual_seed(42)
        input_dim = self.X_train.shape[1]
        output_dim = self.num_services

        model = DNNModel(input_dim, output_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        X_tr_t = torch.tensor(self.X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(self.y_train, dtype=torch.float32)
        X_te_t = torch.tensor(self.X_test, dtype=torch.float32)

        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = model(X_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).numpy()
            binary_preds = (preds > 0.5).astype(int)
            accuracy = float(np.mean(binary_preds == self.y_test))

        return accuracy
