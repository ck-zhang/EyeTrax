from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import sys
import time
import urllib.request

import cv2
import numpy as np

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.models import BaseModel, create_model


_DEFAULT_FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "eyetrax"})
        with urllib.request.urlopen(req, timeout=30) as resp, tmp.open("wb") as fh:
            total = resp.headers.get("Content-Length")
            total_i = int(total) if total and total.isdigit() else None

            downloaded = 0
            last_report = 0.0
            start = time.time()

            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)

                if not sys.stderr.isatty():
                    continue

                now = time.time()
                if now - last_report < 0.2:
                    continue
                last_report = now

                if total_i:
                    pct = 100.0 * (downloaded / max(total_i, 1))
                    sys.stderr.write(
                        f"\r[eyetrax] Downloading FaceLandmarker model: "
                        f"{pct:5.1f}% ({downloaded / 1e6:.1f}/{total_i / 1e6:.1f} MB)"
                    )
                else:
                    sys.stderr.write(
                        f"\r[eyetrax] Downloading FaceLandmarker model: "
                        f"{downloaded / 1e6:.1f} MB"
                    )
                sys.stderr.flush()

            if sys.stderr.isatty():
                dur = time.time() - start
                sys.stderr.write(
                    f"\r[eyetrax] Downloaded FaceLandmarker model "
                    f"({downloaded / 1e6:.1f} MB) in {dur:.1f}s\n"
                )
                sys.stderr.flush()
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)


def _ensure_face_landmarker_task(
    model_path: str | os.PathLike[str] | None,
) -> Path:
    if model_path:
        p = Path(model_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"FaceLandmarker model not found: {p}")
        return p

    env = os.environ.get("EYETRAX_FACE_LANDMARKER_MODEL")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"EYETRAX_FACE_LANDMARKER_MODEL points to missing file: {p}"
            )
        return p

    cache_path = (
        Path.home() / ".cache" / "eyetrax" / "mediapipe" / "face_landmarker.task"
    )
    if cache_path.exists():
        return cache_path

    print(
        f"[eyetrax] FaceLandmarker model missing; downloading to {cache_path}",
        file=sys.stderr,
    )
    try:
        _download_file(_DEFAULT_FACE_LANDMARKER_TASK_URL, cache_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download the MediaPipe FaceLandmarker model. "
            "If you're offline or downloads are blocked, manually download "
            "`face_landmarker.task` and pass `--face-model /path/to/face_landmarker.task` "
            "or set `EYETRAX_FACE_LANDMARKER_MODEL=/path/to/face_landmarker.task`."
        ) from e
    return cache_path


def _create_face_landmarker(*, model_path: str | os.PathLike[str] | None):
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
        from mediapipe.tasks.python.core.base_options import BaseOptions  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Failed to import mediapipe, which is required for face landmarks. "
            "If you're seeing NumPy 2.x / TensorFlow import errors, install with "
            "`numpy<2` (e.g. `pip install 'numpy<2'`) and reinstall eyetrax."
        ) from e

    task_path = _ensure_face_landmarker_task(model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        face_landmarker_model: str | os.PathLike[str] | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
    ):
        self._mp, self._face_landmarker = _create_face_landmarker(
            model_path=face_landmarker_model
        )
        self.model: BaseModel = create_model(model_name, **(model_kwargs or {}))

        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history
        self._mp_last_ts_ms = 0

    def extract_features(self, image):
        """
        Takes in image and returns landmarks around the eye region
        Normalization with outer-eye midpoint as anchor
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_rgb,
        )
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._mp_last_ts_ms:
            ts_ms = self._mp_last_ts_ms + 1
        self._mp_last_ts_ms = ts_ms

        result = self._face_landmarker.detect_for_video(mp_image, ts_ms)
        if not result.face_landmarks:
            return None, False

        landmarks = result.face_landmarks[0]

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        eye_center = (left_corner + right_corner) / 2.0
        shifted_points = all_points - eye_center
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - eye_center
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - eye_center)
        right_corner_rot = R.T @ (right_corner - eye_center)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        if all_points.shape[0] <= max(subset_indices):
            raise RuntimeError(
                "FaceLandmarker returned too few landmarks for the configured indices. "
                f"Got {all_points.shape[0]} landmarks, but need index {max(subset_indices)}. "
                "Use a FaceLandmarker model that outputs iris landmarks "
                "(e.g. the default face_landmarker.task)."
            )
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr

        return features, blink_detected

    def save_model(self, path: str | Path):
        """
        Pickle model
        """
        self.model.save(path)

    def load_model(self, path: str | Path):
        self.model = BaseModel.load(path)

    def train(self, X, y, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.model.train(X, y, variable_scaling)

    def predict(self, X):
        """
        Predicts gaze location
        """
        return self.model.predict(X)

    def close(self) -> None:
        if getattr(self, "_face_landmarker", None) is not None:
            self._face_landmarker.close()
            self._face_landmarker = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
