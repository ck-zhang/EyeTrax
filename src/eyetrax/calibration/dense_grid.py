"""
Dense grid calibration routine for arbitrary-resolution grid calibration.
"""

import cv2
import numpy as np

from eyetrax.calibration.common import (
_pulse_and_capture,
compute_grid_points_from_shape,
wait_for_face_and_countdown
)

from eyetrax.utils.screen import get_screen_size

def run_dense_grid_calibration(
        gaze_estimator,
        *,
        rows: int = 6,
        cols: int = 6,
        camera_index: int = 0,
        pulse_d: float = 0.9,
        cd_d: float = 0.9,
        margin_ratio:float = 0.10,
        order: str = "default"
) -> None:
    """
    Run dense grid calibration with arbitrary grid dimensions
    Args:
        gaze_estimator: GazeEstimator instance to train
        rows: Number of rows in calibration grid ( default = 5 )
        cols: Number of columns in calibration grid ( default = 5)
        camera_index: Index of camera to use ( default = 0 )
        pulse_d: Duration ( seconds ) of pulsing animation ( default 0.9 )
        cd_d: Duration ( seconds ) of capture phase ( default 0.9 )
        margin_ratio: Fraction of screen to use as margin from edges ( default 0.10).
        order: Grid ordering strategy: "default" or "serpentine" (default "default")
    Returns:
        None. Trains the gaze_estimator model
    Raises:
        ValueError: If rows or cols <=0
    Example:
        >>> estimator = GazeEstimator()
        >>> run_desnse_grid_calibration(estimator, rows = 8, cols = 8)
    """
    sw, sh = get_screen_size()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_index}")
    try:
        if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur = 2 ):
            return
        pts = compute_grid_points_from_shape(
            rows, cols, sw, sh, margin_ratio = margin_ratio, order = order
        )
        res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh, pulse_d = pulse_d, cd_d = cd_d)
        if res is None:
            return
        feats, targs = res
        if feats:
            gaze_estimator.train(np.array(feats), np.array(targs))
            print(
                f"[dense_grid] Calibrated with {len(feats)} samples"
                f"from {rows}*{cols} grid"
            )
        else:
            print("[dense_grid] No features collected; calibration may have failed")
    finally:
        cap.release()
        cv2.destroyAllWindows()


