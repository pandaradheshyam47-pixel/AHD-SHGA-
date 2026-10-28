"""
Surrogate Modeling for AHD-SHGA using Gaussian Process Regression (GPR).
Implements:
1. RBF Kernel covariance approximation (Eqs. 12-14)
2. Incremental dataset updates with evaluated elite samples (Eqs. 37-38)
3. Prediction mean and uncertainty estimation
4. Quantitative error tracking (RMSE, MAE - Table 3)
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel


class GaussianProcessSurrogate:
    """
    Gaussian Process surrogate model with an RBF covariance kernel to approximate
    expensive objective functions and guide candidate selection.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 1e-4,
        max_training_samples: int = 600,
        random_state: int = 42,
    ):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.max_training_samples = max_training_samples
        self.random_state = random_state

        kernel = C(1.0, (1e-4, 1e4)) * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3)) + \
                 WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1.0))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-5,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=random_state,
        )

        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Fit surrogate model to training dataset (Eq. 10, Eq. 38).
        Returns empirical training loss (MSE).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        if len(X_arr) > self.max_training_samples:
            # Retain the most recent and best samples
            sort_idx = np.argsort(y_arr)
            best_idx = sort_idx[: self.max_training_samples // 2]
            recent_idx = np.arange(len(X_arr) - self.max_training_samples // 2, len(X_arr))
            selected_idx = np.unique(np.concatenate([best_idx, recent_idx]))
            X_arr = X_arr[selected_idx]
            y_arr = y_arr[selected_idx]

        self.X_train = X_arr
        self.y_train = y_arr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True

        # Compute empirical loss L(theta)
        y_pred, _ = self.model.predict(self.X_train, return_std=True)
        mse = float(np.mean((y_arr - y_pred) ** 2))
        return mse

    def update(self, X_new: np.ndarray, y_new: np.ndarray) -> float:
        """
        Append newly evaluated elite samples and retrain the surrogate (Eq. 37).
        """
        X_new_arr = np.asarray(X_new, dtype=np.float64)
        y_new_arr = np.asarray(y_new, dtype=np.float64)

        if self.X_train is None:
            return self.fit(X_new_arr, y_new_arr)

        combined_X = np.vstack([self.X_train, X_new_arr])
        combined_y = np.concatenate([self.y_train, y_new_arr])
        return self.fit(combined_X, combined_y)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict objective values and estimation uncertainty (standard deviation).
        Returns: (predicted_mean, uncertainty_std)
        """
        if not self.is_fitted or self.X_train is None:
            # Return zeros if not yet fitted
            return np.zeros(len(X)), np.ones(len(X))

        X_arr = np.asarray(X, dtype=np.float64)
        mean, std = self.model.predict(X_arr, return_std=True)
        return np.asarray(mean), np.asarray(std)

    def evaluate_metrics(self, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
        """
        Compute quantitative prediction errors: RMSE and MAE (Table 3).
        """
        y_val_arr = np.asarray(y_val, dtype=np.float64)
        y_pred, _ = self.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val_arr - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_val_arr - y_pred)))
        return {"rmse": rmse, "mae": mae}
