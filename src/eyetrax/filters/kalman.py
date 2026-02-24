from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np

from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera

from . import make_kalman
from .base import BaseSmoother


class KalmanSmoother(BaseSmoother):

    def __init__(self, kf=None) -> None:
        super().__init__()

        try:
            import cv2

            self.kf = kf if isinstance(kf, cv2.KalmanFilter) else make_kalman()
        except ImportError:
            self.kf = make_kalman()

    def step(self, x: int, y: int) -> Tuple[int, int]:
        meas = np.array([[float(x)], [float(y)]], dtype=np.float32)

        if not np.any(self.kf.statePost):
            self.kf.statePre[:2] = meas
            self.kf.statePost[:2] = meas

        pred = self.kf.predict()
        self.kf.correct(meas)

        return int(pred[0, 0]), int(pred[1, 0])

    def tune(self, gaze_estimator, *, camera_index: int = 0):
        """
        Quick fine‑tuning pass to adjust Kalman filter's measurementNoiseCov
        """
        screen_width, screen_height = get_screen_size()

        points_tpl = [
            (screen_width // 2, screen_height // 4),
            (screen_width // 4, 3 * screen_height // 4),
            (3 * screen_width // 4, 3 * screen_height // 4),
        ]

        points = [
            dict(
                position=pos,
                start_time=None,
                data_collection_started=False,
                collection_start_time=None,
                collected_gaze=[],
            )
            for pos in points_tpl
        ]

        proximity_threshold = screen_width / 5
        initial_delay = 0.5
        data_collection_duration = 0.5

        cv2.namedWindow("Fine Tuning", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Fine Tuning", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        cap = open_camera(camera_index)
        gaze_positions = []

        while points:
            ret, frame = cap.read()
            if not ret:
                continue

            features, blink_detected = gaze_estimator.extract_features(frame)
            canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            for point in points:
                cv2.circle(canvas, point["position"], 20, (0, 255, 0), -1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Look at the points until they disappear"
            size, _ = cv2.getTextSize(text, font, 1.5, 2)
            cv2.putText(
                canvas,
                text,
                ((screen_width - size[0]) // 2, screen_height - 50),
                font,
                1.5,
                (255, 255, 255),
                2,
            )

            now = time.time()

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                gaze_x, gaze_y = map(int, gaze_point)
                cv2.circle(canvas, (gaze_x, gaze_y), 10, (255, 0, 0), -1)

                for point in points[:]:
                    dx, dy = (
                        gaze_x - point["position"][0],
                        gaze_y - point["position"][1],
                    )
                    if np.hypot(dx, dy) <= proximity_threshold:
                        if point["start_time"] is None:
                            point["start_time"] = now
                        elapsed = now - point["start_time"]

                        if (
                            not point["data_collection_started"]
                            and elapsed >= initial_delay
                        ):
                            point["data_collection_started"] = True
                            point["collection_start_time"] = now

                        if point["data_collection_started"]:
                            data_elapsed = now - point["collection_start_time"]
                            point["collected_gaze"].append([gaze_x, gaze_y])
                            shake = int(
                                5 + (data_elapsed / data_collection_duration) * 20
                            )
                            shaken = (
                                point["position"][0]
                                + int(np.random.uniform(-shake, shake)),
                                point["position"][1]
                                + int(np.random.uniform(-shake, shake)),
                            )
                            cv2.circle(canvas, shaken, 20, (0, 255, 0), -1)
                            if data_elapsed >= data_collection_duration:
                                gaze_positions.extend(point["collected_gaze"])
                                points.remove(point)
                        else:
                            cv2.circle(canvas, point["position"], 25, (0, 255, 255), 2)
                    else:
                        point.update(
                            start_time=None,
                            data_collection_started=False,
                            collection_start_time=None,
                            collected_gaze=[],
                        )

            cv2.imshow("Fine Tuning", canvas)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyWindow("Fine Tuning")
                return

        cap.release()
        cv2.destroyWindow("Fine Tuning")

        gaze_positions = np.array(gaze_positions)
        if gaze_positions.shape[0] < 2:
            return

        var = np.var(gaze_positions, axis=0)
        var[var == 0] = 1e-4
        self.kf.measurementNoiseCov = np.array(
            [[var[0], 0], [0, var[1]]], dtype=np.float32
        )
