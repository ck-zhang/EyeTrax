import os

import cv2
import numpy as np
import pyvirtualcam

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_dense_grid_calibration,
    run_lissajous_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import (
    KDESmoother,
    KalmanEMASmoother,
    KalmanSmoother,
    NoSmoother,
    make_kalman,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames


def run_virtualcam():
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    confidence_level = args.confidence
    ema_alpha = args.ema_alpha

    gaze_estimator = GazeEstimator(model_name=args.model)

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[virtualcam] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "dense":
            run_dense_grid_calibration(
                gaze_estimator,
                camera_index=camera_index,
                rows=args.grid_rows,
                cols=args.grid_cols,
                margin_ratio=args.grid_margin,
            )
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kalman_ema":
        kalman = make_kalman()
        smoother = KalmanEMASmoother(kalman, ema_alpha=ema_alpha)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    green_bg = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    green_bg[:] = (0, 255, 0)

    with camera(camera_index) as cap:
        cam_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        with pyvirtualcam.Camera(
            width=screen_width,
            height=screen_height,
            fps=cam_fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        ) as cam:
            print(f"Virtual camera started: {cam.device}")
            for frame in iter_frames(cap):
                features, blink_detected = gaze_estimator.extract_features(frame)

                if features is not None and not blink_detected:
                    gaze_point = gaze_estimator.predict(np.array([features]))[0]
                    x, y = map(int, gaze_point)
                    x_pred, y_pred = smoother.step(x, y)
                    contours = smoother.debug.get("contours", [])
                else:
                    x_pred = y_pred = None
                    contours = []

                output = green_bg.copy()
                if contours:
                    cv2.drawContours(output, contours, -1, (0, 0, 255), 3)
                if x_pred is not None and y_pred is not None:
                    draw_cursor(
                        output,
                        x_pred,
                        y_pred,
                        alpha=1.0,
                        radius_outer=10,
                        radius_inner=0,
                        color_outer=(0, 0, 255),
                    )

                cam.send(output)
                cam.sleep_until_next_frame()


if __name__ == "__main__":
    run_virtualcam()
