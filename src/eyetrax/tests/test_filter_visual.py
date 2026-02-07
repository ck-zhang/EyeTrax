""" Visual comparison of filters in real-time"""

import cv2
import numpy as np
import time

from eyetrax.gaze import GazeEstimator
from eyetrax.calibration import run_9_point_calibration, run_5_point_calibration
from eyetrax.filters import KalmanSmoother, KalmanEMASmoother
from eyetrax.utils.screen import get_screen_size

from eyetrax.utils.screen import get_screen_size

def main():
    """Compare filters side-by-side on live camera."""
    gaze_est = GazeEstimator()

    # Calibration
    print("\n" +"="*60)
    print("Calibration requirement")
    print("="*60)
    print("Running 9 point calibration....")
    print("Look at each point as it appears on screen")
    run_5_point_calibration(gaze_est, camera_index = 0)

    #2 filters to compare
    kalman = KalmanSmoother()
    kalman_ema = KalmanEMASmoother(ema_alpha = 0.5)

    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(0)
    time.sleep(1)
    if not cap.isOpened():
        print("Error: couldn't open camera")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # minimize buffer
    print("Comparing Kalman vs Kalman+EMA ( a = 0.5 )")
    print("Press 'q' to quit")
    print()

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            continue

        features, blink = gaze_est.extract_features(frame)

        # create side-by-side comparison
        canvas = np.zeros((sh,sw,3),dtype=np.uint8)
        canvas[:] = (40,40,40)

        # Split screen : left = Kalman, right = Kalman + EMA

        mid = sw // 2

        #print(f"features:{features}, blink:{blink}")
        if features is not None and not blink:
            x_raw, y_raw =  gaze_est.predict(np.array([features]))[0]
            x_raw, y_raw = int(x_raw), int(y_raw)

            # Kalman output ( left side )
            x_k, y_k = kalman.step(x_raw, y_raw)

            # Kalman+EMA output (right side)
            x_ema, y_ema = kalman_ema.step(x_raw, y_raw)

            # draw on left half
            cv2.circle(canvas, (x_k, y_k), 30, (0,255,0), -1)
            cv2.putText(canvas, "Kalman", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.putText(canvas, f"({x_k}, {y_k}", (50,100),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

            # draw on right half
            cv2.circle(canvas, (mid+(x_ema-mid), y_ema), 30, (0,0,255), -1)
            cv2.putText(canvas, "Kalman+EMA", (mid+50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(canvas, f"({x_ema}, {y_ema}", (mid + 50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)

            # draw dividing line
            cv2.line(canvas, (mid,0), (mid, sh), (100,100,100), 2)
            frame_count += 1
        else:
            # Show waiting for face message
            cv2.putText(canvas, "Waiting for face detection...", (100, sh//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

        cv2.putText(canvas, f"Frames: {frame_count}", (20,sh-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        cv2.imshow("Filter Comparison", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n Quitting")
            return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
