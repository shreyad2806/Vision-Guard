# VisionGuard Detection Capabilities

## Required Detection Capabilities

### 1. Blur / Insufficient Sharpness
Primary metrics:
- Sharpness
- Edge Density

Detection:
- Triggered when measured sharpness or edge detail falls below configured thresholds.

Impact:
- May reduce reliability of downstream object, vehicle, pedestrian, or infrastructure detection.

---

### 2. Underexposure
Primary metrics:
- Brightness
- Underexposure Ratio

Detection:
- Triggered when dark-pixel ratio or brightness indicates insufficient illumination.

Impact:
- Important visual details may not be visible to downstream analytics.

---

### 3. Overexposure
Primary metrics:
- Overexposure Ratio
- Brightness

Detection:
- Triggered when bright-pixel clipping exceeds configured thresholds.

Impact:
- Highlight regions may lose important visual information.

---

### 4. Image Noise
Primary metrics:
- Noise Estimate

Detection:
- Triggered when estimated image noise exceeds configured thresholds.

Impact:
- May reduce computer vision reliability.

---

### 5. Image Corruption or Severe Degradation
Detection:
- Composite detection based on multiple severe feature failures.

Impact:
- Image may be unsuitable for downstream analytics.

---

### 6. Potential Visual Defect
Detection:
- Anomalous visual characteristics identified through unusual feature combinations.

Impact:
- Image may require review before analytics consumption.

---

### 7. Additional Quality Issues
Additional justified detections may include:
- Low Contrast
- Low Dynamic Range
- Low Edge Detail
- Low Information Content

Each issue must be supported by measurable image features.