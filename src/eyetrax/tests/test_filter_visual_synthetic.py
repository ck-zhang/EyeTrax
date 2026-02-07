""" Visual Comparison with synthetic data ( no calibration needed )"""

import cv2
import numpy as np
from eyetrax.filters import KalmanSmoother, KalmanEMASmoother
from eyetrax.utils.screen import get_screen_size

def main():
    """Compare filters on synthetic circular gaze path."""
    sw, sh = get_screen_size()

    # 2 filters to compare
    kalman = KalmanSmoother()
    kalman_ema = KalmanEMASmoother(ema_alpha=0.5)

    frame = 0
    while True:
        t = frame * 0.05
        center_x, center_y = sw//2, sh //2
        radius_x, radius_y = 200,150

        clean_x = center_x + radius_x*np.cos(t)
        clean_y = center_y + radius_y*np.sin(t)

        noisy_x = clean_x + np.random.normal(0,10)
        noisy_y = clean_y + np.random.normal(0,10)

        x_k, y_k = kalman.step(int(noisy_x), int(noisy_y))
        x_ema, y_ema = kalman_ema.step(int(noisy_x), int(noisy_y))

        canvas = np.zeros((sh, sw, 3), dtype = np.uint8)
        canvas[:] = (40,40,40)

        mid = sw // 2

        cv2.circle(canvas, (x_k, y_k), 30, (0,255,0), -1)
        cv2.putText(canvas, "Kalman", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(canvas, f"({x_k}, {y_k})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        cv2.circle(canvas, (mid+(x_ema-mid), y_ema), 30, (0,0,255), -1)
        cv2.putText(canvas, "Kalman+EMA", (mid+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(canvas, f"({x_ema}, {y_ema})", ( mid+20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        cv2.line(canvas, (mid,0), (mid,sh), (100, 100, 100), 2)

        cv2.circle(canvas, (int(clean_x), int(clean_y)), 5, (100,100,100), -1)
        cv2.putText(canvas, f"Frame:{frame}",(20,sh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Filter Comparison (Synthetic)", canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        frame += 1

    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main()