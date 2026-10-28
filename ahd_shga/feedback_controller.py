"""
Unified Feedback Control Mechanism and Diversity Recovery for AHD-SHGA.
Implements:
1. Population-level reference control variables (Eqs. 15-21)
   S(t) = [DI(t), V_f(t), Delta_f(t)]
   U(t) = [p_c(t), p_m(t), w_h(t), lambda_s(t)]
2. Diversity Recovery Mechanism (Section 3.4.1) with window R = 5
"""

import numpy as np


class FeedbackController:
    """
    Closed-loop feedback controller that maps population state dynamics
    to evolutionary control parameters.
    """

    def __init__(
        self,
        p_c_min: float = 0.6,
        p_c_max: float = 0.9,
        p_m_min: float = 0.01,
        p_m_max: float = 0.10,
        lambda_1: float = 0.6,
        lambda_2: float = 0.4,
        k: float = 5.0,
        tau: float = 0.5,
        epsilon: float = 0.05,
        lambda_s_min: float = 0.1,
        recovery_window: int = 5,
        di_center: float = 0.5,
        di_slope: float = 4.0,
    ):
        self.p_c_min = p_c_min
        self.p_c_max = p_c_max
        self.p_m_min = p_m_min
        self.p_m_max = p_m_max
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.k = k
        self.tau = tau
        self.epsilon = epsilon
        self.lambda_s_min = lambda_s_min
        self.recovery_window = recovery_window
        self.di_center = di_center
        self.di_slope = di_slope

        # State tracking
        self.recovery_counter: int = 0
        self.is_recovering: bool = False

    def _sigmoid(self, z: float) -> float:
        """Standard sigmoid activation."""
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))

    def _diversity_sigmoid(self, di: float) -> float:
        """
        Sigmoid mapped to diversity index centered around di_center.
        """
        return self._sigmoid(self.di_slope * (di - self.di_center))

    def compute_control_variables(
        self,
        di: float,
        v_f: float,
        delta_f: float,
        di_threshold: float,
    ) -> dict[str, float]:
        """
        Compute feedback control values p_c(t), p_m(t), w_h(t), lambda_s(t) (Eqs. 18-21),
        and apply the 4-part Diversity Recovery Mechanism if DI(t) < DI_th or within recovery window.
        """
        # Base feedback mapping
        sig_di = self._diversity_sigmoid(di)

        # Eq. 18: p_c(t) = p_c_min + (p_c_max - p_c_min) * sigma(DI(t))
        p_c_ref = self.p_c_min + (self.p_c_max - self.p_c_min) * sig_di

        # Eq. 19: p_m(t) = p_m_max - (p_m_max - p_m_min) * sigma(DI(t))
        p_m_ref = self.p_m_max - (self.p_m_max - self.p_m_min) * sig_di

        # Eq. 20: w_h(t) = lambda1 * DI(t) + lambda2 * V_f(t)
        w_h_ref = float(np.clip(self.lambda_1 * di + self.lambda_2 * v_f, 0.0, 1.0))

        # Eq. 21: lambda_s(t) = 1 / (1 + exp(-k * delta_f(t)))
        lambda_s_ref = self._sigmoid(self.k * delta_f)

        # Check diversity recovery trigger (Section 3.4.1)
        if di < di_threshold:
            self.recovery_counter = self.recovery_window
            self.is_recovering = True
        elif self.recovery_counter > 0:
            self.recovery_counter -= 1
            self.is_recovering = (self.recovery_counter > 0)
        else:
            self.is_recovering = False

        # Apply Diversity Recovery overrides if active (Section 3.4.1)
        if self.is_recovering:
            # (i) Mutation boost
            p_m_ref = self.p_m_max
            # (ii) Crossover suppression
            p_c_ref = self.p_c_min
            # (iii) Exploration bias: cap at tau - epsilon
            w_h_ref = min(w_h_ref, self.tau - self.epsilon)
            # (iv) Surrogate evaluation reduction: lambda_s <- lambda_s_min
            lambda_s_ref = self.lambda_s_min

        return {
            "p_c": float(np.clip(p_c_ref, self.p_c_min, self.p_c_max)),
            "p_m": float(np.clip(p_m_ref, self.p_m_min, self.p_m_max)),
            "w_h": float(np.clip(w_h_ref, 0.0, 1.0)),
            "lambda_s": float(np.clip(lambda_s_ref, self.lambda_s_min, 1.0)),
            "is_recovering": self.is_recovering,
            "recovery_counter": self.recovery_counter,
        }
