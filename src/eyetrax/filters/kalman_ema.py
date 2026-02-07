"""
Kalman + EMA hybrid filter for gaze estimation.

Combines kalman filtering ( dynamic prediction with process/measurement models)
With exponential moving average (low-pass smoothing ) for superior short-term
stability and noise rejection.

Design:
- Kalman filter predicts and corrects using motion model and measurements
- EMA smooths the Kalman output to reduce high-frequency jitter
-Tunable EMA alpha ( 0 = ono smoothing, 1 = fully smoothed )
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from eyetrax.utils.screen import get_screen_size

from . import make_kalman
from .base import BaseSmoother

class KalmanEMASmoother(BaseSmoother):
    """
    Hybrid Kalman + EMA smoother for gaze estimation.

    Applies a 2 stage filtering pipeline:
    1. Kalman filter for dynamic tracking and prediction
    2. Exponential Moving Average (EMA) for jitter reduction

    This hybrid approach provides:
    - Better stability on noisy measurements ( via EMA )
    - Better responsiveness to true motion ( via Kalman dynamics )
    - Lower latency than pure smoothing approaches ( Kalman prediction is fast )
    - Tunable smoothing level via ema_alpha parameter

    Attributes:
        kf ( cv2.KalmanFilter) : Internal Kalman filter for motion prediction.
        ema_alpha (float) : EMA smoothing factor( 0.0 = no smoothing, 1.0 = full ).
        ema_x (float or None) : Running EMA estimate for x coordinate.
        ema_y (float or None) : Running EMA estimate for y coordinate
        """
    def __init__(selfself, kf: cv2.KalmanFilter|None = None, ema_alphas: float = 0.25) -> None:
        """
        Initialize Kalman + EMA hybrid smoother
        Args:
            kf : cv2.KalmanFilter instance. If None, creates a default Kalman Filter.
            ema_alpha: EMA smoothing factor in range [0.0, 1.0]
            0.5: Moderate smoothing, good jitter reduction
            0.75: Strong smoothing , less responsive to changes
            1.0: Fully smoothed, very slow to respond
        Raises:
            ValueError: If ema_alpha is not in [0.0, 1.0]
        """

