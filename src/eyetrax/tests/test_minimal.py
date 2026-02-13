""" Minimal reproducible test"""
import numpy as np
import sys
import traceback

# Test 1 : Import works?

try:
    from eyetrax.filters import KalmanEMASmoother, make_kalman_ema
    print(" PASS Imports successful")
except ImportError as e:
    print(f" FAIL Import failed : {e}")
    exit(1)

# Test 2 : Instantiation works?
try:
    smoother = KalmanEMASmoother(ema_alpha=0.3)
    print("PASS Instantiation successful")
except Exception as e:
    print(f"Instantiation failed : {e}")
    exit(1)

# Test 3 : Can I run a step?
try:
    for i in range(10):
        x = 500+i*10 # moving up
        y = 300+i*5  # moving down
        x_out, y_out = smoother.step(x, y)
        print(f"  Step {i}: ({x}, {y}) -> ({x_out}, {y_out})")
    print("PASS step() works")
except Exception as e:
    print(f"FAIL step() fails : {e}")
    traceback.print_exc()
    exit(1)

# Test 4 : Can I vary alpha?
try:
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = KalmanEMASmoother(ema_alpha = alpha)
        x,y = s.step(500,300)
        print(f"alpha = {alpha}: step() returned ({x}, {y})")
    print("PASS Alpha variation works")
except Exception as e:
    print(f" FAIL Alpha variation fails : {e}")
    exit(1)

# Test 5 : test CLI changes
try:
    print("Testing CLI argument parsing")
    #simulate command line args
    sys.argv = ["prog", "--filter", "kalman_ema", "--ema-alpha", "0.3"]
    from eyetrax.cli import parse_common_args
    args = parse_common_args()

    print(f"CLI parsed successfully")
    print(f"filter: {args.filter}")
    print(f"ema_alpha : {args.ema_alpha}")

    assert args.filter == "kalman_ema", f"expectedfilter='kalman_ema', got '{args.filter}'"
    assert args.ema_alpha == 0.3, f"Expected ema_alpha=0.3, got{args.ema_alpha}"
    print("\n CLI tests passed!")
except Exception as e:
    print(f"FAILS CLI tests - Exception {e}")
    traceback.print_exc()
    exit(1)
