"""OpenCV-based feature extraction from images.

Each function computes a single statistic.  extract_features() returns a
complete dictionary suitable for the ML model.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — higher means sharper."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(laplacian))


def compute_brightness(gray: np.ndarray) -> float:
    """Mean grayscale pixel intensity."""
    return float(np.mean(gray))


def compute_contrast(gray: np.ndarray) -> float:
    """Standard deviation of grayscale pixel intensities."""
    return float(np.std(gray))


def compute_noise_estimate(gray: np.ndarray) -> float:
    """Estimate noise via the median absolute deviation of the Laplacian.

    This is a fast, lightweight approximation commonly used in
    no-reference image quality assessment.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = np.median(np.abs(laplacian)) * 1.4826
    return float(sigma)


def compute_entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()  # normalise to probabilities
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    return float(abs(-np.sum(hist * np.log2(hist))))


def compute_saturation(img_bgr: np.ndarray) -> float:
    """Mean HSV saturation channel."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 1]))


def compute_underexposure_pct(gray: np.ndarray) -> float:
    """Percentage of pixels with intensity below 30 (very dark)."""
    return float(np.mean(gray < 30) * 100)


def compute_overexposure_pct(gray: np.ndarray) -> float:
    """Percentage of pixels with intensity above 225 (very bright)."""
    return float(np.mean(gray > 225) * 100)


def compute_edge_density(gray: np.ndarray) -> float:
    """Fraction of edge pixels via Canny detection."""
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0) * 100)


def compute_dynamic_range(gray: np.ndarray) -> float:
    """Intensity range between 5th and 95th percentile."""
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    return p95 - p5


def compute_colorfulness(img_bgr: np.ndarray) -> float:
    """Colorfulness metric based on Hasler & Susstrunk.

    Uses the mean and std of the red-green and blue-yellow opponent channels.
    """
    b, g, r = cv2.split(img_bgr.astype(float))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    return float(np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2))


def compute_texture_complexity(gray: np.ndarray) -> float:
    """Texture complexity via FFT energy in high-frequency bands."""
    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    # Create high-frequency mask (outside inner 20% of spectrum)
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) * 0.2
    mask = ((Y - cy)**2 + (X - cx)**2) > radius**2
    mag = np.abs(fshift)
    total_energy = np.sum(mag**2)
    if total_energy == 0:
        return 0.0
    hf_energy = np.sum(mag[mask]**2)
    return float(hf_energy / total_energy * 100)


def extract_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Compute all image statistics and return as a dictionary.

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR colour space (as loaded by cv2.imread).

    Returns
    -------
    dict with keys matching the ImageStatistics schema plus additional
    ML features for quality prediction.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return {
        # Core metrics (returned in API responses)
        "sharpness": round(compute_sharpness(gray), 2),
        "brightness": round(compute_brightness(gray), 2),
        "contrast": round(compute_contrast(gray), 2),
        "noise_estimate": round(compute_noise_estimate(gray), 2),
        "entropy": round(compute_entropy(gray), 2),
        "saturation": round(compute_saturation(img_bgr), 2),
        # Extended features (used by ML model)
        "underexposure_pct": round(compute_underexposure_pct(gray), 2),
        "overexposure_pct": round(compute_overexposure_pct(gray), 2),
        "edge_density": round(compute_edge_density(gray), 2),
        "dynamic_range": round(compute_dynamic_range(gray), 2),
        "colorfulness": round(compute_colorfulness(img_bgr), 2),
        "texture_complexity": round(compute_texture_complexity(gray), 2),
    }
