# Controlled-Condition Robustness Report — VisionGuard v4.0

## Purpose

The controlled-condition evaluation tests whether VisionGuard responds differently to common visual-quality degradations. Each condition modifies a synthetic base image in a specific way, and the full production pipeline processes the result.

## Controlled-Condition Robustness Summary

| Condition | Quality | Readiness | Status | Primary Issue | Issues |
|---|---:|---:|---|---|---:|
| Clean | 60.8 | 45.8 | LIMITED READINESS | Underexposure | 1 |
| Blurred | 13.7 | 0.0 | CRITICAL / REJECT | Severe Blur | 4 |
| Underexposed | 24.7 | 0.0 | CRITICAL / REJECT | Severe Blur | 3 |
| Overexposed | 66.7 | 21.7 | NOT READY | Overexposure | 2 |
| Noisy | 29.6 | 0.0 | CRITICAL / REJECT | Underexposure | 2 |
| Severely Degraded | 13.7 | 0.0 | CRITICAL / REJECT | Underexposure | 3 |

## Detailed Results

### Clean

- **Quality Score:** 60.8
- **Raw Prediction:** 62.4524
- **Readiness Score:** 45.8
- **Readiness Status:** LIMITED READINESS
- **Issues Detected:** 1

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Underexposure | high | 1.00 | dark_pixel_ratio | 32.90 | 25.00 |

**Key Features:**
- brightness: 120.6
- sharpness: 1188.4
- contrast: 83.9
- noise_estimate: 5.9
- edge_density: 0.0494
- dark_pixel_pct: 32.6
- bright_pixel_pct: 0.0

### Blurred

- **Quality Score:** 13.7
- **Raw Prediction:** 13.1704
- **Readiness Score:** 0.0
- **Readiness Status:** CRITICAL / REJECT
- **Issues Detected:** 4

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Severe Blur | critical | 1.00 | laplacian_variance | 1.05 | 15.00 |
| Underexposure | high | 1.00 | dark_pixel_ratio | 22.81 | 25.00 |
| Visual Degradation | high | 0.90 | degradation_signal_count | 2.00 | 2.00 |
| Potential Visual Defect | high | 0.85 | visual_anomaly_signal_count | 2.00 | 2.00 |

**Key Features:**
- brightness: 120.6
- sharpness: 2.3
- contrast: 72.7
- noise_estimate: 1.5
- edge_density: 0.0000
- dark_pixel_pct: 23.1
- bright_pixel_pct: 0.0

### Underexposed

- **Quality Score:** 24.7
- **Raw Prediction:** 23.5253
- **Readiness Score:** 0.0
- **Readiness Status:** CRITICAL / REJECT
- **Issues Detected:** 3

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Severe Blur | critical | 1.00 | laplacian_variance | 8.72 | 15.00 |
| Underexposure | critical | 1.00 | dark_pixel_ratio | 100.00 | 50.00 |
| Severe Visual Degradation | critical | 1.00 | degradation_signal_count | 5.00 | 3.00 |

**Key Features:**
- brightness: 13.9
- sharpness: 17.1
- contrast: 10.0
- noise_estimate: 0.0
- edge_density: 0.0000
- dark_pixel_pct: 100.0
- bright_pixel_pct: 0.0

### Overexposed

- **Quality Score:** 66.7
- **Raw Prediction:** 65.5179
- **Readiness Score:** 21.7
- **Readiness Status:** NOT READY
- **Issues Detected:** 2

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Overexposure | critical | 1.00 | bright_pixel_ratio | 67.10 | 30.00 |
| Potential Visual Defect | high | 0.85 | visual_anomaly_signal_count | 2.00 | 2.00 |

**Key Features:**
- brightness: 187.1
- sharpness: 970.1
- contrast: 96.1
- noise_estimate: 0.0
- edge_density: 0.0146
- dark_pixel_pct: 0.0
- bright_pixel_pct: 66.9

### Noisy

- **Quality Score:** 29.6
- **Raw Prediction:** 30.1572
- **Readiness Score:** 0.0
- **Readiness Status:** CRITICAL / REJECT
- **Issues Detected:** 2

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Underexposure | high | 1.00 | dark_pixel_ratio | 23.72 | 25.00 |
| Image Noise | critical | 1.00 | noise_estimate | 118.61 | 35.00 |

**Key Features:**
- brightness: 124.6
- sharpness: 7765.7
- contrast: 76.6
- noise_estimate: 80.1
- edge_density: 0.2018
- dark_pixel_pct: 24.9
- bright_pixel_pct: 0.7

### Severely Degraded

- **Quality Score:** 13.7
- **Raw Prediction:** 12.0304
- **Readiness Score:** 0.0
- **Readiness Status:** CRITICAL / REJECT
- **Issues Detected:** 3

| Issue | Severity | Confidence | Metric | Value | Threshold |
|---|---|---:|---|---:|---:|
| Underexposure | critical | 1.00 | dark_pixel_ratio | 99.76 | 50.00 |
| Image Noise | high | 1.00 | noise_estimate | 23.72 | 15.00 |
| Severe Visual Degradation | critical | 1.00 | degradation_signal_count | 3.00 | 3.00 |

**Key Features:**
- brightness: 14.5
- sharpness: 245.4
- contrast: 3.5
- noise_estimate: 14.8
- edge_density: 0.0000
- dark_pixel_pct: 100.0
- bright_pixel_pct: 0.0

## Analysis

### Clean Image

The clean image receives a quality score of **60.8** with readiness **45.8**. The system detects 1 issue(s): underexposure. This may reflect the synthetic nature of the base image having limited texture/complexity.

### Blur

Blurring reduces quality to **13.7** (vs clean: 60.8). Sharpness drops from 1188.4 to 2.3. Edge density changes from 0.0494 to 0.0000. The system detects: severe_blur, underexposure, visual_degradation, potential_visual_defect.

### Underexposure

Underexposure reduces quality to **24.7**. Brightness drops from 120.6 to 13.9. Dark pixel percentage increases from 32.6% to 100.0%. The system detects: severe_blur, underexposure, severe_visual_degradation.

### Overexposure

Overexposure results in quality **66.7**. Brightness increases to 187.1 (from 120.6). Bright pixel percentage increases from 0.0% to 66.9%. The system detects: overexposure, potential_visual_defect.

### Noise

Noise results in quality **29.6**. Noise estimate increases from 5.9 to 80.1. The system detects: underexposure, image_noise.

### Severe Degradation

Severe degradation results in quality **13.7**. Brightness drops to 14.5. Sharpness drops to 245.4. Entropy drops to 3.85 (indicating very low information content). The system detects: underexposure, image_noise, severe_visual_degradation.

## Quality/Readiness Differentiation

| Condition | Quality | Readiness |
|---|---:|---:|
| Clean | 60.8 | 45.8 |
| Blurred | 13.7 | 0.0 |
| Underexposed | 24.7 | 0.0 |
| Overexposed | 66.7 | 21.7 |
| Noisy | 29.6 | 0.0 |
| Severely Degraded | 13.7 | 0.0 |

- **Quality range:** 13.7 to 66.7 (spread: 52.9)
- **Readiness range:** 0.0 to 45.8 (spread: 45.8)
- **All conditions produce same quality:** NO (expected)
- **All conditions produce same readiness:** NO (expected)

## Key Finding

VisionGuard produces **condition-sensitive** quality and readiness assessments. Different visual conditions produce meaningfully different quality scores, readiness levels, and detected issues. The system responds to measurable image characteristics (sharpness, brightness, noise, information content) rather than returning a fixed score regardless of input quality.

## Limitations

- Synthetic test images may not fully represent real-world degradation patterns
- Controlled conditions test one degradation at a time; real images may have multiple simultaneous degradations
- Quality scores are model predictions, not ground-truth measurements
- The issue detector uses rule-based thresholds, not learned detection
- No issue-level ground truth exists to validate detection accuracy
