# EyeTrax

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17188537.svg)](https://doi.org/10.5281/zenodo.17188537)
[![PyPI version](https://img.shields.io/pypi/v/eyetrax.svg)](https://pypi.org/project/eyetrax/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)
[![GitHub stars](https://img.shields.io/github/stars/ck-zhang/EyeTrax.svg?style=social)](https://github.com/ck-zhang/EyeTrax)

![Demo](https://github.com/user-attachments/assets/1b953a10-442f-4c4a-95e0-52a68f1488bc)

EyeTrax is a Python library that provides **webcam-based eye tracking**.
Extract facial features, train a model and predict gaze with an easy‑to‑use interface.

## Features

- Real‑time gaze estimation
- Multiple calibration workflows
- Optional filtering (Kalman / Kalman+EMA / KDE)
- Model persistence – save / load a trained `GazeEstimator`
- Virtual-camera overlay that integrates with streaming software (e.g., OBS) via the bundled **`eyetrax-virtualcam`** CLI

## Installation

### From [PyPI](https://pypi.org/project/eyetrax/)

```bash
pip install eyetrax
```

### From source

```bash
git clone https://github.com/ck-zhang/eyetrax && cd eyetrax

# editable install — pick one
python -m pip install -e .
pip install uv && uv sync
```

## Demo

The **EyeTrax** package provides two command‑line entry points

| Command | Purpose |
|---------|---------|
| `eyetrax-demo` | Run an on‑screen gaze overlay demo |
| `eyetrax-virtualcam` | Stream the overlay to a virtual webcam |

Options

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--filter` | `kalman`, `kalman_ema`, `kde`, `none` | `none` | Smoothing filter |
| `--ema-alpha` *(kalman_ema only)* | *0–1* | `0.25` | EMA smoothing strength |
| `--camera` | *int* | `0` | Physical webcam index |
| `--calibration` | `9p`, `5p`, `lissajous`, `dense` | `9p` | Calibration routine |
| `--grid-rows` *(dense only)* | *int* | `5` | Calibration grid rows |
| `--grid-cols` *(dense only)* | *int* | `5` | Calibration grid columns |
| `--grid-margin` *(dense only)* | *float* | `0.10` | Margin ratio from edges |
| `--background` *(demo only)* | *path* | — | Background image |
| `--confidence` *(KDE only)* | *0–1* | `0.5` | Contour probability |

## Quick Examples

```bash
eyetrax-demo --filter kalman

# Kalman + EMA smoothing (tune EMA strength)
eyetrax-demo --filter kalman_ema --ema-alpha 0.5

# Dense grid calibration (higher spatial coverage)
eyetrax-demo --calibration dense --grid-rows 7 --grid-cols 7
```

```bash
eyetrax-virtualcam --filter kde --calibration 5p
```

### Virtual camera demo

https://github.com/user-attachments/assets/de4a0b63-8631-4c16-9901-9f83bc0bb766

## Library Usage

```python
from eyetrax import GazeEstimator, run_9_point_calibration
import cv2

# Create estimator and calibrate
estimator = GazeEstimator()
run_9_point_calibration(estimator)

# Save model
estimator.save_model("gaze_model.pkl")

# Load model
estimator = GazeEstimator()
estimator.load_model("gaze_model.pkl")

cap = cv2.VideoCapture(0)

while True:
    # Extract features from frame
    ret, frame = cap.read()
    features, blink = estimator.extract_features(frame)

    # Predict screen coordinates
    if features is not None and not blink:
        x, y = estimator.predict([features])[0]
        print(f"Gaze: ({x:.0f}, {y:.0f})")
```

## More

If you find EyeTrax useful, consider starring the repo or contributing. If you use it in your research, please cite it. The project is available under the MIT license.

**BibTeX**
```bibtex
@software{Zhang2025_EyeTrax,
  author       = {Chenkai Zhang},
  title        = {EyeTrax},
  version      = {0.2.2},
  date         = {2025-04-23},
  url          = {https://pypi.org/project/eyetrax/},
  repository   = {https://github.com/ck-zhang/EyeTrax},
  doi          = {10.5281/zenodo.17188537},
  keywords     = {eye tracking, computer vision}
}
```

**APA style**
```
Zhang, C. (2025). EyeTrax (0.2.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17188537
```
