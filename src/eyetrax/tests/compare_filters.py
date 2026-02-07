"""Compare filter performance on synthetic noisy data"""
import numpy as np
import cv2
from eyetrax.filters import KalmanSmoother, KalmanEMASmoother, NoSmoother

def test_filters_on_noisy_data():
    """Test filters on synthetic noisy gaze trajectory."""
    np.random.seed(42)

    #Simulate gaze following a smooth path with noise
    n_frames = 200

    # Ground truth : smooth circular path
    t = np.linspace(0, 2*np.pi, n_frames)
    clean_x = 960 + 300*np.cos(t)
    clean_y = 54- + 200 * np.sin(t)

    # add measurement noise
    noise_std = 15 # pixels
    noisy_x = clean_x + np.random.normal(0, noise_std, n_frames)
    noisy_y = clean_y + np.random.normal(0, noise_std, n_frames)

    # Test different filters
    filters = {
        'No Filter': NoSmoother(),
        'Kalman'   : KalmanSmoother(),
        'Kalman+EMA (a = 0.25)': KalmanEMASmoother(ema_alpha = 0.25),
        'Kalman+EMA (a = 0.50)': KalmanEMASmoother(ema_alpha=0.5),
        'Kalman+EMA (a = 0.75)': KalmanEMASmoother(ema_alpha=0.75)
    }
    results = {}

    for name, smoother in filters.items():
        output = []
        for x, y in zip(noisy_x, noisy_y):
            x_filt, y_filt = smoother.step(int(x), int(y))
            output.append((x_filt, y_filt))
        output = np.array(output)

        # Compute metrics
        error = np.sqrt((output[:, 0] - clean_x)**2+(output[:,1] - clean_y)**2)
        rmse = np.sqrt(np.mean(error**2))
        variance = np.var(output,axis=0).mean()

        results[name] = {
            'output': output,
            'rmse' : rmse,
            'variance' : variance,
            'error_mean': np.mean(error),
            'error_std' : np.std(error)
         }

    # print comparison
    print("\n"+"="*80)
    print("FILTER COMPARISON ON SYNTHETIC NOISY DATA")
    print("="*80)
    print(f"Ground truth:circular path, noise std={noise_std} px \n")
    print(f"{'Filter':<25} {'RMSE':<12} {'Variance':<12} {'Mean Error':<12} {'Std Error':<12}")

    for name in sorted(results.keys()):
        res = results[name]
        print(f"{name:<25} {res['rmse']:<12.2f} {res['variance']:<12.2f} {res['error_mean']:<12.2f} {res['error_std']:<12.2f}")

    print("="*80)
    print("\n Interpretation:")
    print("-Lower RMSE = closer to ground truth")
    print("Lower variance = smoother output ( less jitter )")
    print("Higher latency = lag on quick movements")
    print("\n Tip: compare kalman vs kalman + EMA at different alpha values")
    print("Higher alpha = more smootheing = more lag")




if __name__ == "__main__":
    test_filters_on_noisy_data()