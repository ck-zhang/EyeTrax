from __future__ import annotations

from typing import Tuple

from .kalman import KalmanSmoother


class KalmanEMASmoother(KalmanSmoother):
    """
    Apply EMA smoothing on top of Kalman output.

    `ema_alpha` controls smoothing strength:
    - 0.0: EMA disabled (pure Kalman output)
    - closer to 1.0: stronger smoothing / more lag
    """

    def __init__(self, kf=None, ema_alpha: float = 0.25) -> None:
        super().__init__(kf=kf)

        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError(f"ema_alpha must be in [0.0, 1.0], got {ema_alpha}")

        self.ema_alpha = float(ema_alpha)
        self.ema_x: float | None = None
        self.ema_y: float | None = None

    def step(self, x: int, y: int) -> Tuple[int, int]:
        kx, ky = super().step(x, y)

        a = self.ema_alpha
        if a == 0.0:
            return kx, ky

        if self.ema_x is None or self.ema_y is None:
            self.ema_x = float(kx)
            self.ema_y = float(ky)
        else:
            self.ema_x = a * self.ema_x + (1.0 - a) * float(kx)
            self.ema_y = a * self.ema_y + (1.0 - a) * float(ky)

        return int(self.ema_x), int(self.ema_y)

