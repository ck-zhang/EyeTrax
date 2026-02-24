from __future__ import annotations

import random
import time
from typing import List, Tuple

import cv2
import numpy as np

from eyetrax.calibration.nine_point import run_9_point_calibration
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import open_camera


class BlueNoiseSampler:
    def __init__(self, w: int, h: int, margin: float = 0.08):
        self.w, self.h = w, h
        self.mx, self.my = int(w * margin), int(h * margin)

    def sample(self, n: int, k: int = 30) -> List[Tuple[int, int]]:
        pts: List[Tuple[int, int]] = []
        for _ in range(n):
            best, best_d2 = None, -1
            for _ in range(k):
                x = random.randint(self.mx, self.w - self.mx)
                y = random.randint(self.my, self.h - self.my)
                d2 = (
                    min((x - px) ** 2 + (y - py) ** 2 for px, py in pts) if pts else 1e9
                )
                if d2 > best_d2:
                    best, best_d2 = (x, y), d2
            pts.append(best)
        return pts


def _draw_live_pred(canvas, frame, gaze_estimator):
    ft, blink = gaze_estimator.extract_features(frame)
    if ft is None or blink:
        return None
    x_pred, y_pred = gaze_estimator.predict(np.array([ft]))[0]
    draw_cursor(canvas, int(x_pred), int(y_pred), alpha=1.0)
    return ft


def _pulse_and_capture_live(
    gaze_estimator: GazeEstimator,
    cap: cv2.VideoCapture,
    pts: List[Tuple[int, int]],
    sw: int,
    sh: int,
):
    feats, targs = [], []
    for x, y in pts:
        pulse_start = time.time()
        while time.time() - pulse_start < 1.0:
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), np.uint8)
            rad = 15 + int(15 * abs(np.sin((time.time() - pulse_start) * 6)))
            cv2.circle(canvas, (x, y), rad, (0, 255, 0), -1)
            _draw_live_pred(canvas, frame, gaze_estimator)
            cv2.imshow("Adaptive Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None, None

        cap_start = time.time()
        while time.time() - cap_start < 1.0:
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), np.uint8)
            cv2.circle(canvas, (x, y), 20, (0, 255, 0), -1)
            t = (time.time() - cap_start) / 1.0
            ang = 360 * (1 - (t * t * (3 - 2 * t)))
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
            ft = _draw_live_pred(canvas, frame, gaze_estimator)
            cv2.imshow("Adaptive Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None, None
            if ft is not None:
                feats.append(ft)
                targs.append([x, y])
    return feats, targs


def run_adaptive_calibration(
    gaze_estimator: GazeEstimator,
    *,
    num_random_points: int = 60,
    retrain_every: int = 10,
    show_predictions: bool = True,
    camera_index: int = 0,
) -> None:
    run_9_point_calibration(gaze_estimator, camera_index=camera_index)

    sw, sh = get_screen_size()
    sampler = BlueNoiseSampler(sw, sh)
    points = sampler.sample(num_random_points)

    cap = open_camera(camera_index)
    cv2.namedWindow("Adaptive Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Adaptive Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    all_feats, all_targs = [], []

    for chunk_start in range(0, len(points), retrain_every):
        chunk = points[chunk_start : chunk_start + retrain_every]
        feats, targs = _pulse_and_capture_live(gaze_estimator, cap, chunk, sw, sh)
        if feats is None:
            break
        all_feats.extend(feats)
        all_targs.extend(targs)

        gaze_estimator.train(np.asarray(all_feats), np.asarray(all_targs))

    cap.release()
    cv2.destroyWindow("Adaptive Calibration")
