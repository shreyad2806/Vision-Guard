"""OpenCV-based feature extraction from images.

Two extraction modes:
  1. extract_model_features(img_bgr) — the 5 features used by the ML model
     (must match ml/train.py exactly: resize 224×224, Canny 100/200).
  2. extract_all_features(img_bgr) — 12 detailed statistics for the UI
     (sharpness, brightness, contrast, noise, entropy, saturation, etc.).
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Model feature extraction — must match ml/train.py
# ---------------------------------------------------------------------------

def extract_model_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Extract the 20 features used by the trained model (v4.0).

    Preprocessing matches training exactly:
      - Resize to 224x224
      - Grayscale for luminance features
      - HSV for saturation, hue
      - BGR for color features
      - Canny(100, 200) for edge_density
      - Sobel for gradient_magnitude
      - FFT for texture_complexity
    """
    resized = cv2.resize(img_bgr, (224, 224))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    b_ch, g_ch, r_ch = cv2.split(resized.astype(float))

    # Core features
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(np.mean(hsv[:, :, 1]))
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Noise
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_estimate = float(np.median(np.abs(laplacian)) * 1.4826)

    # Entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = float(abs(-np.sum(hist * np.log2(hist))))

    # Colorfulness
    rg = np.abs(r_ch - g_ch)
    yb = np.abs(0.5 * (r_ch + g_ch) - b_ch)
    colorfulness = float(np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

    # Exposure
    luminance_p10 = float(np.percentile(gray, 10))
    luminance_p90 = float(np.percentile(gray, 90))
    dark_pixel_pct = float(np.mean(gray < 30) * 100)
    bright_pixel_pct = float(np.mean(gray > 225) * 100)
    luminance_median = float(np.median(gray))

    # Gradient
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # Contrast
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))
    percentile_range = float(luminance_p90 - luminance_p10)

    # Noise-to-signal
    noise_to_signal = noise_estimate / max(brightness, 1.0)

    # Color stats
    channel_std = float(np.std([np.mean(r_ch), np.mean(g_ch), np.mean(b_ch)]))
    hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hue_hist = hue_hist / hue_hist.sum()
    hue_hist = hue_hist[hue_hist > 0]
    hue_entropy = float(abs(-np.sum(hue_hist * np.log2(hue_hist))))

    # Structure
    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) * 0.2
    fft_mask = ((Y - cy)**2 + (X - cx)**2) > radius**2
    mag = np.abs(fshift)
    total_energy = np.sum(mag**2)
    texture_complexity = float(np.sum(mag[fft_mask]**2) / total_energy * 100) if total_energy > 0 else 0.0

    return {
        "brightness": brightness, "contrast": contrast, "sharpness": sharpness,
        "saturation": saturation, "edge_density": edge_density,
        "noise_estimate": noise_estimate, "entropy": entropy, "colorfulness": colorfulness,
        "luminance_p10": luminance_p10, "luminance_p90": luminance_p90,
        "dark_pixel_pct": dark_pixel_pct, "bright_pixel_pct": bright_pixel_pct,
        "luminance_median": luminance_median, "gradient_magnitude": gradient_magnitude,
        "dynamic_range": dynamic_range, "percentile_range": percentile_range,
        "noise_to_signal": noise_to_signal, "channel_std": channel_std,
        "hue_entropy": hue_entropy, "texture_complexity": texture_complexity,
    }


# ---------------------------------------------------------------------------
# Detailed feature extraction — used for UI statistics and issue detection
# ---------------------------------------------------------------------------

def compute_noise_estimate(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sigma = np.median(np.abs(laplacian)) * 1.4826
    return float(sigma)


def compute_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(abs(-np.sum(hist * np.log2(hist))))


def compute_underexposure_pct(gray: np.ndarray) -> float:
    return float(np.mean(gray < 30) * 100)


def compute_overexposure_pct(gray: np.ndarray) -> float:
    return float(np.mean(gray > 225) * 100)


def compute_dynamic_range(gray: np.ndarray) -> float:
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    return p95 - p5


def compute_colorfulness(img_bgr: np.ndarray) -> float:
    b, g, r = cv2.split(img_bgr.astype(float))
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)
    return float(np.sqrt(rg_std**2 + yb_std**2) + 0.3 * np.sqrt(rg_mean**2 + yb_mean**2))


def compute_texture_complexity(gray: np.ndarray) -> float:
    f = np.fft.fft2(gray.astype(float))
    fshift = np.fft.fftshift(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = min(h, w) * 0.2
    mask = ((Y - cy)**2 + (X - cx)**2) > radius**2
    mag = np.abs(fshift)
    total_energy = np.sum(mag**2)
    if total_energy == 0:
        return 0.0
    hf_energy = np.sum(mag[mask]**2)
    return float(hf_energy / total_energy * 100)

# UI/issue-detection edge density is expressed as a percentage (0-100).
# This intentionally differs from extract_model_features(), where
# edge_density remains a fraction (0-1) to match ml/train.py.

def extract_all_features(img_bgr: np.ndarray) -> dict[str, float]:
    """Compute all image statistics for the UI display."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(np.mean(gray)), 2),
        "contrast": round(float(np.std(gray)), 2),
        "noise_estimate": round(compute_noise_estimate(gray), 2),
        "entropy": round(compute_entropy(gray), 2),
        "saturation": round(float(np.mean(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[:, :, 1])), 2),
        "underexposure_pct": round(compute_underexposure_pct(gray), 2),
        "overexposure_pct": round(compute_overexposure_pct(gray), 2),
        "edge_density": round(float(np.mean(cv2.Canny(gray, 50, 150) > 0) * 100), 2),
        "dynamic_range": round(compute_dynamic_range(gray), 2),
        "colorfulness": round(compute_colorfulness(img_bgr), 2),
        "texture_complexity": round(compute_texture_complexity(gray), 2),
    }


# Keep backward compatibility
extract_features = extract_all_features
