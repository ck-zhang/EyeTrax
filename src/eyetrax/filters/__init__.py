from __future__ import annotations

import cv2
import numpy as np


def make_kalman(
    state_dim: int = 4,
    meas_dim: int = 2,
    dt: float = 1.0,
    process_var: float = 50.0,
    measurement_var: float = 0.2,
    init_state: np.ndarray | None = None,
) -> cv2.KalmanFilter:
    """
    Factory returning a cv2.KalmanFilter
    """
    kf = cv2.KalmanFilter(state_dim, meas_dim)

    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(state_dim, dtype=np.float32) * process_var
    kf.measurementNoiseCov = np.eye(meas_dim, dtype=np.float32) * measurement_var
    kf.errorCovPost = np.eye(state_dim, dtype=np.float32)

    kf.statePre = np.zeros((state_dim, 1), np.float32)
    kf.statePost = np.zeros((state_dim, 1), np.float32)

    if init_state is not None:
        init_state = np.asarray(init_state, np.float32).reshape(state_dim, 1)
        kf.statePre[:] = init_state
        kf.statePost[:] = init_state

    return kf


from .base import BaseSmoother
from .kalman import KalmanSmoother
from .kalman_ema import KalmanEMASmoother
from .kde import KDESmoother
from .noop import NoSmoother

__all__ = [
    "make_kalman",
    "BaseSmoother",
    "KalmanSmoother",
    "KalmanEMASmoother",
    "KDESmoother",
    "NoSmoother",
]
