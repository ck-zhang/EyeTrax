from __future__ import annotations

from contextlib import contextmanager

import cv2


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """
    Open a camera by index.

    Compatibility fallback: if camera 0 fails to open, try camera 1.
    """
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        return cap

    cap.release()
    if index == 0:
        cap = cv2.VideoCapture(1)
        if cap.isOpened():
            return cap
        cap.release()
        raise RuntimeError("cannot open camera 0 (fallback to camera 1 also failed)")

    raise RuntimeError(f"cannot open camera {index}")


@contextmanager
def fullscreen(name: str):
    """
    Open a window in full-screen mode
    """
    cv2.namedWindow(name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        yield
    finally:
        cv2.destroyWindow(name)


@contextmanager
def camera(index: int = 0):
    """
    Context manager returning an opened VideoCapture
    """
    cap = open_camera(index)
    try:
        yield cap
    finally:
        cap.release()


def iter_frames(cap: cv2.VideoCapture):
    """
    Infinite generator yielding successive frames
    """
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        yield frame
