# Feature Importance Analysis — VisionGuard v4.0

## Method

Feature importance is computed using **permutation importance** on the **validation set** (2,443 samples).
For each feature, the feature values are randomly permuted 10 times, and the decrease in Spearman correlation is measured.
Features whose permutation causes a larger drop in Spearman correlation are more important.

This method is model-agnostic and does not require the model to expose `feature_importances_`.

## Results

| Rank | Feature | Importance | Std | Percentage |
|------|---------|-----------|-----|------------|
| 1 | sharpness | 0.9524 | 0.0286 | 41.0% |
| 2 | gradient_magnitude | 0.2822 | 0.0069 | 12.2% |
| 3 | colorfulness | 0.1566 | 0.0066 | 6.7% |
| 4 | entropy | 0.1396 | 0.0156 | 6.0% |
| 5 | saturation | 0.0920 | 0.0053 | 4.0% |
| 6 | noise_to_signal | 0.0853 | 0.0098 | 3.7% |
| 7 | edge_density | 0.0826 | 0.0044 | 3.6% |
| 8 | hue_entropy | 0.0718 | 0.0075 | 3.1% |
| 9 | luminance_p10 | 0.0688 | 0.0055 | 3.0% |
| 10 | dark_pixel_pct | 0.0679 | 0.0044 | 2.9% |
| 11 | channel_std | 0.0573 | 0.0056 | 2.5% |
| 12 | texture_complexity | 0.0557 | 0.0052 | 2.4% |
| 13 | dynamic_range | 0.0403 | 0.0040 | 1.7% |
| 14 | brightness | 0.0365 | 0.0024 | 1.6% |
| 15 | bright_pixel_pct | 0.0328 | 0.0031 | 1.4% |
| 16 | contrast | 0.0286 | 0.0027 | 1.2% |
| 17 | luminance_median | 0.0220 | 0.0025 | 0.9% |
| 18 | noise_estimate | 0.0210 | 0.0023 | 0.9% |
| 19 | luminance_p90 | 0.0163 | 0.0025 | 0.7% |
| 20 | percentile_range | 0.0124 | 0.0018 | 0.5% |

**Sum of importances:** 2.3222

## Interpretation

### 1. sharpness (41.0%)

Measures image detail and focus via Laplacian variance. Contributes most strongly to predictions — sharper images generally have higher quality scores.

### 2. gradient_magnitude (12.2%)

Mean edge strength via Sobel operators. Complements sharpness by capturing gradient information across the image.

### 3. colorfulness (6.7%)

Measures chromatic variation using Hasler & Susstrunk method. Colorful images tend to be perceived as higher quality; grayscale or washed-out images score lower.

### 4. entropy (6.0%)

Measures pixel intensity information content. Higher entropy indicates more varied texture and detail, contributing to perceived quality.

### 5. saturation (4.0%)

Mean HSV saturation. Captures color intensity — moderate saturation is associated with higher quality.

### 6. noise_to_signal (3.7%)

Ratio of noise estimate to mean brightness. Higher values indicate noisier images relative to their brightness.

### 7. edge_density (3.6%)

Fraction of Canny edge pixels. Structural information density.

### 8. hue_entropy (3.1%)

Entropy of hue histogram distribution. Captures color diversity — images with varied hues tend to score higher.

### 9. luminance_p10 (3.0%)

10th percentile of luminance. Indicates shadow region brightness — helps identify underexposure.

### 10. dark_pixel_pct (2.9%)

Percentage of very dark pixels (< 30). High values indicate severe underexposure or shadow crush.

### 11. channel_std (2.5%)

Standard deviation of RGB channel means. Captures color balance variation.

### 12. texture_complexity (2.4%)

High-frequency energy ratio from FFT. Captures fine texture detail beyond what edge detection measures.

### 13. dynamic_range (1.7%)

Intensity range between 5th and 95th percentiles. Wider ranges indicate better tonal variety.

### 14. brightness (1.6%)

Mean grayscale intensity. Basic exposure measure — very dark or very bright images score lower.

### 15. bright_pixel_pct (1.4%)

Percentage of very bright pixels (> 225). High values indicate overexposure or highlight clipping.

### 16. contrast (1.2%)

Standard deviation of grayscale intensity. Measures overall tonal spread.

### 17. luminance_median (0.9%)

Median luminance value. Robust central tendency of brightness.

### 18. noise_estimate (0.9%)

Robust noise estimate via median absolute deviation of Laplacian.

### 19. luminance_p90 (0.7%)

90th percentile of luminance. Indicates highlight region brightness — helps identify overexposure.

### 20. percentile_range (0.5%)

Intensity range between 10th and 90th percentiles. Similar to dynamic range but excludes extreme tails.

## Key Observations

1. **Sharpness dominates** — the model relies heavily on Laplacian variance for quality prediction. This aligns with perceptual IQA literature where sharpness is consistently one of the strongest quality predictors.

2. **Color information matters** — colorfulness and saturation together contribute significantly, suggesting the model captures aesthetic quality beyond mere technical fidelity.

3. **Exposure features are distributed** — rather than a single brightness feature, the model uses luminance percentiles, dark/bright pixel percentages, and noise-to-signal ratio to capture different aspects of exposure quality.

4. **Structural features add unique information** — gradient_magnitude, texture_complexity, and edge_density provide complementary structural information that raw brightness/contrast alone cannot capture.

5. **All 20 features contribute** — no feature has zero importance, confirming that the feature engineering step added genuinely useful information.

## Limitations

- Permutation importance measures feature importance **on the validation set** — importance may differ on other distributions.
- Importance does not imply **causation** — it measures how much the model relies on each feature, not whether the feature *causes* quality differences.
- Correlated features may share importance — removing one may not decrease performance if another correlated feature compensates.
- This analysis does not use SHAP or other game-theoretic attribution methods.
