from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_gray(image_path: str | Path, downsample: float = 0.5) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if downsample <= 0 or downsample > 1:
        raise ValueError("downsample must be in (0, 1]")

    if downsample < 1:
        image = cv2.resize(image, None, fx=downsample, fy=downsample, interpolation=cv2.INTER_AREA)

    return image


def compute_dense_flow(
    prev_image: np.ndarray,
    next_image: np.ndarray,
    farneback_params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 21,
        "iterations": 5,
        "poly_n": 7,
        "poly_sigma": 1.5,
        "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    }
    if farneback_params:
        params.update(farneback_params)

    flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, **params)
    u = flow[..., 0]
    v = flow[..., 1]
    magnitude = np.sqrt(u ** 2 + v ** 2)
    return u, v, magnitude


def extract_uv_features(u: np.ndarray, v: np.ndarray, magnitude: np.ndarray) -> dict[str, float]:
    return {
        "uv_u_mean": float(np.mean(u)),
        "uv_v_mean": float(np.mean(v)),
        "uv_magnitude_mean": float(np.mean(magnitude)),
        "uv_magnitude_std": float(np.std(magnitude)),
        "uv_magnitude_max": float(np.max(magnitude)),
        "uv_magnitude_p95": float(np.percentile(magnitude, 95)),
        "uv_magnitude_p99": float(np.percentile(magnitude, 99)),
    }
