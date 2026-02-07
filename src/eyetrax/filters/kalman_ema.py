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
from fontTools.misc.cython import returns
from pandas.core.ops import kleene_xor

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
    def __init__(self, kf: cv2.KalmanFilter|None = None, ema_alpha: float = 0.25) -> None:
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
        super().__init__()

        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0.0, 1.0], got {ema_alpha}")

        try:
            import cv2
            self.kf = kf if isinstance(kf, cv2.KalmanFilter) else make_kalman()
        except ImportError:
            self.kf = make_kalman()

        self.ema_alpha = float(ema_alpha)
        self.ema_x: float | None = None
        self.ema_y: float | None = None

    def step(self, x: int, y: int) -> Tuple[int, int]:
        """
        Process a measurement and return filtered gaze position.

        2-stage pipeline:
        1. Kalman predict + correct: motion-aware filtering with dynamics
        2. EMA smoothing: exponential averaging to reduce jitter

        Args:
            x: Measured x coordinate (pixels).
            y: Measured y coordinate (pixels).
        """
        # Stage 1: Kalman Filtering
        meas = np.array([[float(x)],[float(y)]], dtype=np.float32)

        # Initialize Kalman on first call
        if not np.any(self.kf.statePost):
            self.kf.statePre[:2] = meas
            self.kf.statePost[:2] = meas

        # Kalman predict + correct
        pred = self.kf.predict()
        self.kf.correct(meas)

        kx = float(pred[0,0])
        ky = float(pred[1,0])

        # Stage 2: EMA smoothing on Kalman output
        if self.ema_alpha == 0.0:
            # No EMA smoothing
            return int(kx), int(ky)
        if self.ema_x is None:
            # Initialize EMA on first call
            self.ema_x = kx
            self.ema_y = ky
        else:
            # Apply exponential moving average
            # # new_ema = (1-alpha) * old_ema + alpha* measurement
            # # When alpha is small we trust the old estimate more ( more smoothing )
            a = self.ema_alpha
            self.ema_x = (1.0 - a) * self.ema_x + a*kx
            self.ema_y = (1.0 - a) * self.ema_y + a* ky
        return int(self.ema_x), int(self.ema_y)

    def tune(self, gaze_estimator, *, camera_index: int = 0)->None:
        """
        Tune Kalman Filter's measurement noise covariance using live gaze data.

        Same tuning procedure as KalmanSmoother: collects gaze samples at three
        screen locations to estimate measurement noise and adjust filter response

        Args:
            gaze_estimator: GazeEstimator instance for feature extraction.
            camera_index: Camera index to use for tuning (default 0).
        """
        # Delegate to Kalman tuning (  both filters share same Kalman parameters )
        screen_width, screen_height = get_screen_size()

        points_tpl = [
            (screen_width // 2, screen_height //4 ),
            (screen_width //4, 3* screen_height //4),
            (3* screen_width //4 , 3* screen_height //4 ),
        ]
        points = [
            dict(
                position=pos,
                start_time=None,
                data_collection_started = False,
                collection_start_time = None,
                collected_gaze = []
            )
        for pos in points_tpl
        ]

        proximity_threshold = screen_width /5
        initial_delay = 0.5
        data_collection_duration = 0.5

        import cv2
        cv2.namedWindow("Fine Tuning", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Fine Tuning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cap = cv2.VideoCapture(camera_index)
        gaze_positions = []

        while points:
            ret, frame = cap.read()
            if not ret:
                continue
            features, blink_detected = gaze_estimator.extract_features(frame)
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Look at the poitns until they disappear"
            size, _ = cv2.getTextSize(text, font, 1.5, 2)

            cv2.putText(
                canvas,
                text,
                ((screen_width - size[0])//2, screen_height - 50),
                font,
                1.5,
                ( 255, 255, 255),
                2

            )

            import time
            now = time.time()

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                gaze_x, gaze_y = map(int, gaze_point)
                cv2.circle(canvas, (gaze_x, gaze_y), 10, (255,0,0), -1)

                for point in points[:]:
                    dx, dy = (
                        gaze_x - point["position"][0],
                        gaze_y - point["position"][1]
                    )
                    if np.hypot(dx,dy) <= proximity_threshold:
                        if point["start_time"] is None:
                            point["start_time"] = now
                        elapsed = now - point["start_time"]
                        if (
                            not point["data_collection_started"]
                            and elapsed >= initial_delay
                        ):
                            point["data_collection_started"]= True
                            point["collection_start_time"]=now

                        if point["data_collection_started"]:
                            data_elapsed = now - point["collection_start_time"]
                            point["collected_gaze"].append([gaze_x, gaze_y])
                            shake = int(
                                5 + (data_elapsed / data_collection_duration)*20
                            )
                            shaken = (
                                point["position"][0]
                                + int(np.random.uniform(-shake,shake)),
                                point["position"][1]
                                + int(np.random.uniform(-shake, shake)),
                            )
                            cv2.circle(canvas, shaken, 20, (0,255,0), -1)
                            if data_elapsed >= data_collection_duration:
                                gaze_positions.extend(point["collected_gaze"])
                                points.remove(point)
                        else:
                            cv2.circle(canvas, point["position"], 25, (0,255,255), 2)
                    else:
                        point.update(start_time = None,
                                     data_collection_started=False,
                                     collection_start_time = None,
                                     collected_gaze = [])
                cv2.imshow("Fine Tuning", canvas)
                if cv2.waitKey(1) == 27:
                    cap.release()
                    cv2.destroyWindow("Fine Tuning")
                    return

            cap.release()
            cv2.destroyWindow("Fine Tuning")

            gaze_positions = np.array(gaze_positions)
            print(f"[kalman_ema.tune] gaze_positions shape: {gaze_positions.shape}, dtype:{gaze_positions.dtype}")
            print(f"[kalman_ema.tune] gaze_positions: {gaze_positions}")
            if gaze_positions.shape[0] < 2:
                print("[kalman_ema] Insufficient gaze data for tuning; skipping")
                return
            var = np.var(gaze_positions, axis = 0)
            #var[var == 0] = 1e-4
            if np.isscalar(var):
                if var == 0:
                    var = 1e-4
            else:
                 var[var == 0] = 1e-4
            self.kf.measurementNoiseCov = np.array(
                [[var[0],0],[0,var[1]]], dtype = np.float32
            )