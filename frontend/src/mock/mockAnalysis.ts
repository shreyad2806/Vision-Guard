import type { AnalysisResult } from "../types/analysis";

export const mockAcceptable: AnalysisResult = {
  analysis_id: "ana_001",
  filename: "warehouse_entrance_042.jpg",
  quality_score: 87,
  quality_label: "Excellent",
  analysis_confidence: 94,
  issues: [
    {
      type: "image_noise",
      severity: "low",
      metric: "noise_estimate",
      value: 8.3,
      threshold: 10.0,
      impact: "Slight luminance noise detected in shadow regions.",
      confidence: 0.72,
      description:
        "Slight luminance noise detected in shadow regions. This level of noise is unlikely to affect object detection or classification accuracy.",
    },
  ],
  statistics: {
    sharpness: 78.4,
    brightness: 132.1,
    contrast: 54.7,
    noise_estimate: 8.3,
    entropy: 6.82,
    saturation: 41.2,
  },
  explanations: [
    "Brightness is within the expected range.",
    "Contrast is sufficient for visible detail separation.",
    "Sharpness is sufficient for visible edge detail.",
    "Saturation is within the expected range for natural color reproduction.",
    "Edge density is sufficient for structural feature extraction.",
  ],
  summary:
    "The image demonstrates excellent overall quality (score: 87/100). Minor issues were noted but do not materially affect quality assessment.",
  recommendation: "Image is suitable for automated analysis.",
  created_at: "2026-08-27T09:15:00Z",
  processing_time_ms: 1240,
  model_version: "visionguard-iqa-v2.0",
  analytics_readiness_score: 78,
  analytics_readiness_status: "READY",
  analytics_readiness_details: {
    base_quality_score: 87.0,
    blur_penalty: 0.0,
    exposure_penalty: 0.0,
    noise_penalty: 5.0,
    corruption_penalty: 0.0,
    information_penalty: 4.0,
  },
  context: "CCTV Surveillance",
  context_impacts: [],
  issue_explanations: [
    {
      issue: "Mild Noise",
      evidence: { metric: "noise_estimate", value: 8.3, threshold: 10.0 },
      why_it_matters:
        "Slight noise may affect fine-grained texture analysis in edge cases.",
      recommendation:
        "Consider denoising filters if high precision is required.",
    },
  ],
};

export const mockDegraded: AnalysisResult = {
  analysis_id: "ana_002",
  filename: "parking_lot_cam_117.png",
  quality_score: 52,
  quality_label: "Fair",
  analysis_confidence: 89,
  issues: [
    {
      type: "insufficient_sharpness",
      severity: "high",
      metric: "laplacian_variance",
      value: 18.7,
      threshold: 25.0,
      impact: "Object detection and classification may be unreliable.",
      confidence: 0.91,
      description:
        "The image appears blurry or lacks sufficient edge detail. Edge information is below the expected threshold for reliable visual analysis.",
    },
    {
      type: "underexposure",
      severity: "moderate",
      metric: "mean_brightness",
      value: 68.9,
      threshold: 80.0,
      impact: "Shadow detail may be lost, reducing detection confidence.",
      confidence: 0.84,
      description:
        "The image appears darker than the expected range. Shadow detail may be lost.",
    },
    {
      type: "low_color_information",
      severity: "low",
      metric: "saturation",
      value: 22.8,
      threshold: 25.0,
      impact: "Color-based features may have reduced discriminative power.",
      confidence: 0.76,
      description:
        "Colors appear relatively muted or desaturated. This may affect color-based feature extraction.",
    },
  ],
  statistics: {
    sharpness: 34.2,
    brightness: 68.9,
    contrast: 38.1,
    noise_estimate: 18.7,
    entropy: 5.41,
    saturation: 22.8,
  },
  explanations: [
    "Brightness is somewhat below the optimal range.",
    "Image contrast is moderate but below the optimal range.",
    "Sharpness is significantly below the calibrated threshold. The image may be blurry.",
    "Image saturation is moderately below the calibrated range.",
    "Edge density is below the median range for training images.",
  ],
  summary:
    "The image exhibits moderate quality issues (Insufficient Sharpness, Underexposure, Low Color Information) with a quality score of 52/100. These conditions may reduce the reliability of downstream analysis.",
  recommendation:
    "Consider image enhancement or manual review before analytics.",
  created_at: "2026-08-27T07:42:00Z",
  processing_time_ms: 980,
  model_version: "visionguard-iqa-v2.0",
  analytics_readiness_score: 42,
  analytics_readiness_status: "LIMITED READINESS",
  analytics_readiness_details: {
    base_quality_score: 52.0,
    blur_penalty: 18.0,
    exposure_penalty: 8.0,
    noise_penalty: 0.0,
    corruption_penalty: 0.0,
    information_penalty: 5.0,
  },
  context: "Traffic Monitoring",
  context_impacts: [
    {
      issue_type: "insufficient_sharpness",
      context: "Traffic Monitoring",
      impact:
        "Vehicle detection and license-plate recognition may be unreliable.",
    },
    {
      issue_type: "underexposure",
      context: "Traffic Monitoring",
      impact:
        "Low-light conditions may reduce vehicle tracking accuracy at dawn/dusk.",
    },
  ],
  issue_explanations: [
    {
      issue: "Insufficient Sharpness",
      evidence: {
        metric: "laplacian_variance",
        value: 18.7,
        threshold: 25.0,
      },
      why_it_matters:
        "Blurred images reduce object detection and classification accuracy.",
      recommendation:
        "Improve camera focus or use a higher-resolution sensor.",
    },
    {
      issue: "Underexposure",
      evidence: {
        metric: "mean_brightness",
        value: 68.9,
        threshold: 80.0,
      },
      why_it_matters:
        "Dark images lose shadow detail needed for reliable analysis.",
      recommendation:
        "Adjust camera exposure settings or add supplemental lighting.",
    },
    {
      issue: "Low Color Information",
      evidence: { metric: "saturation", value: 22.8, threshold: 25.0 },
      why_it_matters:
        "Desaturated images reduce color-based feature extraction accuracy.",
      recommendation:
        "Check camera white-balance settings and lighting conditions.",
    },
  ],
};

export const mockDefective: AnalysisResult = {
  analysis_id: "ana_003",
  filename: "construction_zone_009.webp",
  quality_score: 18,
  quality_label: "Poor",
  analysis_confidence: 96,
  issues: [
    {
      type: "overexposure",
      severity: "high",
      metric: "overexposure_pct",
      value: 28.5,
      threshold: 10.0,
      impact: "Highlight clipping may obscure important visual details.",
      confidence: 0.89,
      description:
        "The image may be overexposed. Highlight clipping may have occurred, resulting in loss of detail in bright areas.",
    },
    {
      type: "severe_blur",
      severity: "high",
      metric: "laplacian_variance",
      value: 5.2,
      threshold: 15.0,
      impact: "Object detection and tracking will likely fail.",
      confidence: 0.92,
      description:
        "The image appears blurry or lacks sufficient edge detail. Edge information is below the expected threshold for reliable visual analysis.",
    },
    {
      type: "low_color_information",
      severity: "high",
      metric: "saturation",
      value: 8.1,
      threshold: 25.0,
      impact: "Color-based analysis features are severely compromised.",
      confidence: 0.87,
      description:
        "Colors appear relatively muted or desaturated.",
    },
  ],
  statistics: {
    sharpness: 11.6,
    brightness: 198.3,
    contrast: 15.2,
    noise_estimate: 32.4,
    entropy: 3.87,
    saturation: 8.1,
  },
  explanations: [
    "Brightness exceeds the calibrated range, indicating overexposure.",
    "Image contrast is below the calibrated range.",
    "Sharpness is significantly below the calibrated threshold. The image may be blurry.",
    "Image saturation is well below the calibrated range. Colors are heavily desaturated.",
    "Edge density is very low, suggesting limited structural content.",
  ],
  summary:
    "Significant quality degradation detected (Overexposure, Severe Blur, Low Color Information) with a quality score of 18/100. Image quality is substantially compromised. Review or recapture is recommended.",
  recommendation:
    "Image is unreliable for automated analysis. Recapture or manual review is required.",
  created_at: "2026-08-26T22:08:00Z",
  processing_time_ms: 1120,
  model_version: "visionguard-iqa-v2.0",
  analytics_readiness_score: 5,
  analytics_readiness_status: "CRITICAL",
  analytics_readiness_details: {
    base_quality_score: 18.0,
    blur_penalty: 25.0,
    exposure_penalty: 15.0,
    noise_penalty: 0.0,
    corruption_penalty: 0.0,
    information_penalty: 10.0,
  },
  context: "Drone Imagery",
  context_impacts: [
    {
      issue_type: "severe_blur",
      context: "Drone Imagery",
      impact:
        "Aerial mapping, 3D reconstruction, and ground-truth labeling may be unreliable.",
    },
    {
      issue_type: "overexposure",
      context: "Drone Imagery",
      impact:
        "Sun glare or reflection overexposure may obscure terrain features.",
    },
  ],
  issue_explanations: [
    {
      issue: "Overexposure",
      evidence: {
        metric: "overexposure_pct",
        value: 28.5,
        threshold: 10.0,
      },
      why_it_matters:
        "Highlight clipping destroys detail needed for reliable visual analysis.",
      recommendation:
        "Adjust exposure compensation or use HDR imaging.",
    },
    {
      issue: "Severe Blur",
      evidence: {
        metric: "laplacian_variance",
        value: 5.2,
        threshold: 15.0,
      },
      why_it_matters:
        "Critical blur destroys fine detail needed for object detection and tracking.",
      recommendation:
        "Recapture with stabilised camera, faster shutter, or improved autofocus.",
    },
    {
      issue: "Low Color Information",
      evidence: { metric: "saturation", value: 8.1, threshold: 25.0 },
      why_it_matters:
        "Heavily desaturated images reduce color-based analysis reliability.",
      recommendation:
        "Check camera sensor and lighting conditions.",
    },
  ],
};

/** Additional history entries for the history page */
export const mockHistory: AnalysisResult[] = [
  mockAcceptable,
  mockDegraded,
  mockDefective,
  {
    analysis_id: "ana_004",
    filename: "loading_dock_023.jpg",
    quality_score: 72,
    quality_label: "Good",
    analysis_confidence: 91,
    issues: [
      {
        type: "overexposure",
        severity: "low",
        metric: "overexposure_pct",
        value: 7.2,
        threshold: 10.0,
        impact: "Mild highlight clipping in upper regions.",
        confidence: 0.68,
        description:
          "Mild highlight clipping in the upper-right quadrant. Main subject area remains well-exposed.",
      },
    ],
    statistics: {
      sharpness: 72.1,
      brightness: 156.8,
      contrast: 48.3,
      noise_estimate: 6.2,
      entropy: 7.01,
      saturation: 38.9,
    },
    explanations: [
      "Brightness is somewhat above the optimal range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is sufficient for visible edge detail.",
      "Saturation is within the expected range.",
      "Edge density is sufficient for structural feature extraction.",
    ],
    summary:
      "The image demonstrates good overall quality (score: 72/100). Minor conditions detected (Overexposure) but are unlikely to significantly affect downstream analysis.",
    recommendation:
      "Image is suitable for most analytics tasks.",
    created_at: "2026-08-26T15:30:00Z",
    processing_time_ms: 1050,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 65,
    analytics_readiness_status: "READY",
    analytics_readiness_details: {
      base_quality_score: 72.0,
      blur_penalty: 0.0,
      exposure_penalty: 5.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 2.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [],
    issue_explanations: [
      {
        issue: "Mild Overexposure",
        evidence: {
          metric: "overexposure_pct",
          value: 7.2,
          threshold: 10.0,
        },
        why_it_matters:
          "Mild highlight clipping may affect bright-region analysis.",
        recommendation:
          "Adjust exposure settings for more consistent lighting.",
      },
    ],
  },
  {
    analysis_id: "ana_005",
    filename: "security_cam_feed_088.png",
    quality_score: 41,
    quality_label: "Fair",
    analysis_confidence: 87,
    issues: [
      {
        type: "insufficient_sharpness",
        severity: "high",
        metric: "laplacian_variance",
        value: 14.3,
        threshold: 25.0,
        impact: "Object detection and classification may be unreliable.",
        confidence: 0.88,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "low_color_information",
        severity: "moderate",
        metric: "saturation",
        value: 18.5,
        threshold: 25.0,
        impact: "Color-based features may have reduced discriminative power.",
        confidence: 0.81,
        description:
          "Colors appear relatively muted or desaturated.",
      },
    ],
    statistics: {
      sharpness: 29.8,
      brightness: 89.4,
      contrast: 42.6,
      noise_estimate: 21.3,
      entropy: 5.92,
      saturation: 18.5,
    },
    explanations: [
      "Brightness is somewhat below the optimal range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is significantly below the calibrated threshold. The image may be blurry.",
      "Image saturation is moderately below the calibrated range.",
      "Edge density is below the median range for training images.",
    ],
    summary:
      "The image exhibits moderate quality issues (Insufficient Sharpness, Low Color Information) with a quality score of 41/100. These conditions may reduce the reliability of downstream analysis.",
    recommendation:
      "Consider image enhancement or manual review before analytics.",
    created_at: "2026-08-25T11:22:00Z",
    processing_time_ms: 940,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 32,
    analytics_readiness_status: "NOT READY",
    analytics_readiness_details: {
      base_quality_score: 41.0,
      blur_penalty: 18.0,
      exposure_penalty: 0.0,
      noise_penalty: 5.0,
      corruption_penalty: 0.0,
      information_penalty: 5.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [
      {
        issue_type: "insufficient_sharpness",
        context: "CCTV Surveillance",
        impact:
          "Face recognition and person-tracking accuracy will be significantly reduced.",
      },
    ],
    issue_explanations: [
      {
        issue: "Insufficient Sharpness",
        evidence: {
          metric: "laplacian_variance",
          value: 14.3,
          threshold: 25.0,
        },
        why_it_matters:
          "Blurred images reduce object detection and classification accuracy.",
        recommendation:
          "Improve camera focus or use a higher-resolution sensor.",
      },
      {
        issue: "Low Color Information",
        evidence: { metric: "saturation", value: 18.5, threshold: 25.0 },
        why_it_matters:
          "Desaturated images reduce color-based feature extraction accuracy.",
        recommendation:
          "Check camera white-balance settings and lighting conditions.",
      },
    ],
  },
  {
    analysis_id: "ana_006",
    filename: "warehouse_interior_012.jpg",
    quality_score: 91,
    quality_label: "Excellent",
    analysis_confidence: 97,
    issues: [],
    statistics: {
      sharpness: 84.7,
      brightness: 128.3,
      contrast: 56.1,
      noise_estimate: 4.8,
      entropy: 7.14,
      saturation: 44.6,
    },
    explanations: [
      "Brightness is within the expected range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is sufficient for visible edge detail.",
      "Saturation is within the expected range.",
      "Edge density is sufficient for structural feature extraction.",
    ],
    summary:
      "Image quality is excellent with no significant issues detected. The image is well-suited for all downstream analysis tasks.",
    recommendation:
      "No action required — image meets all quality standards.",
    created_at: "2026-08-25T08:05:00Z",
    processing_time_ms: 890,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 88,
    analytics_readiness_status: "HIGHLY READY",
    analytics_readiness_details: {
      base_quality_score: 91.0,
      blur_penalty: 0.0,
      exposure_penalty: 0.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 3.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [],
    issue_explanations: [],
  },
  {
    analysis_id: "ana_007",
    filename: "exterior_cam_north_005.png",
    quality_score: 55,
    quality_label: "Fair",
    analysis_confidence: 85,
    issues: [
      {
        type: "underexposure",
        severity: "high",
        metric: "mean_brightness",
        value: 52.7,
        threshold: 80.0,
        impact: "Shadow detail may be lost, reducing detection confidence.",
        confidence: 0.87,
        description:
          "The image appears darker than the expected range. Shadow detail may be lost.",
      },
    ],
    statistics: {
      sharpness: 58.3,
      brightness: 52.7,
      contrast: 31.4,
      noise_estimate: 14.9,
      entropy: 4.88,
      saturation: 27.3,
    },
    explanations: [
      "Brightness is somewhat below the optimal range.",
      "Image contrast is moderate but below the optimal range.",
      "Sharpness is below the optimal range but some edge detail remains.",
      "Image saturation is moderately below the calibrated range.",
      "Edge density is below the median range for training images.",
    ],
    summary:
      "The image exhibits moderate quality issues (Underexposure) with a quality score of 55/100. These conditions may reduce the reliability of downstream analysis.",
    recommendation:
      "Adjust camera exposure settings or add supplemental lighting.",
    created_at: "2026-08-24T19:48:00Z",
    processing_time_ms: 1180,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 38,
    analytics_readiness_status: "NOT READY",
    analytics_readiness_details: {
      base_quality_score: 55.0,
      blur_penalty: 0.0,
      exposure_penalty: 15.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 5.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [
      {
        issue_type: "underexposure",
        context: "CCTV Surveillance",
        impact:
          "Person detection and face recognition accuracy will be significantly reduced.",
      },
    ],
    issue_explanations: [
      {
        issue: "Underexposure",
        evidence: {
          metric: "mean_brightness",
          value: 52.7,
          threshold: 80.0,
        },
        why_it_matters:
          "Dark images lose shadow detail needed for reliable analysis.",
        recommendation:
          "Adjust camera exposure settings or add supplemental lighting.",
      },
    ],
  },
  {
    analysis_id: "ana_008",
    filename: "traffic_camera_01.jpg",
    quality_score: 82,
    quality_label: "Excellent",
    analysis_confidence: 93,
    issues: [],
    statistics: {
      sharpness: 76.2,
      brightness: 141.5,
      contrast: 51.8,
      noise_estimate: 7.1,
      entropy: 6.94,
      saturation: 36.4,
    },
    explanations: [
      "Brightness is within the expected range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is sufficient for visible edge detail.",
      "Saturation is within the expected range.",
      "Edge density is sufficient for structural feature extraction.",
    ],
    summary:
      "The image demonstrates excellent overall quality (score: 82/100). No significant quality issues detected.",
    recommendation:
      "No action required — image meets all quality standards.",
    created_at: "2026-08-24T14:15:00Z",
    processing_time_ms: 1020,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 74,
    analytics_readiness_status: "READY",
    analytics_readiness_details: {
      base_quality_score: 82.0,
      blur_penalty: 0.0,
      exposure_penalty: 0.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 4.0,
    },
    context: "Traffic Monitoring",
    context_impacts: [],
    issue_explanations: [],
  },
  {
    analysis_id: "ana_009",
    filename: "factory_floor_cam_003.png",
    quality_score: 28,
    quality_label: "Poor",
    analysis_confidence: 92,
    issues: [
      {
        type: "insufficient_sharpness",
        severity: "high",
        metric: "laplacian_variance",
        value: 10.1,
        threshold: 25.0,
        impact: "Object detection and classification may be unreliable.",
        confidence: 0.89,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "low_color_information",
        severity: "moderate",
        metric: "saturation",
        value: 19.8,
        threshold: 25.0,
        impact: "Color-based features may have reduced discriminative power.",
        confidence: 0.73,
        description:
          "Colors appear relatively muted or desaturated.",
      },
    ],
    statistics: {
      sharpness: 14.3,
      brightness: 112.7,
      contrast: 22.1,
      noise_estimate: 28.9,
      entropy: 4.12,
      saturation: 19.8,
    },
    explanations: [
      "Brightness is within the expected range.",
      "Image contrast is below the calibrated range.",
      "Sharpness is significantly below the calibrated threshold. The image may be blurry.",
      "Image saturation is moderately below the calibrated range.",
      "Edge density is below the median range for training images.",
    ],
    summary:
      "Significant quality degradation detected (Insufficient Sharpness, Low Color Information) with a quality score of 28/100. Image quality is substantially compromised. Review or recapture is recommended.",
    recommendation:
      "Image is unreliable for automated analysis. Recapture or manual review is required.",
    created_at: "2026-08-23T16:40:00Z",
    processing_time_ms: 1150,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 18,
    analytics_readiness_status: "NOT READY",
    analytics_readiness_details: {
      base_quality_score: 28.0,
      blur_penalty: 18.0,
      exposure_penalty: 0.0,
      noise_penalty: 8.0,
      corruption_penalty: 0.0,
      information_penalty: 5.0,
    },
    context: "Infrastructure Inspection",
    context_impacts: [
      {
        issue_type: "insufficient_sharpness",
        context: "Infrastructure Inspection",
        impact:
          "Small structural defects may not be reliably visible.",
      },
    ],
    issue_explanations: [
      {
        issue: "Insufficient Sharpness",
        evidence: {
          metric: "laplacian_variance",
          value: 10.1,
          threshold: 25.0,
        },
        why_it_matters:
          "Blurred images reduce object detection and classification accuracy.",
        recommendation:
          "Improve camera focus or use a higher-resolution sensor.",
      },
      {
        issue: "Low Color Information",
        evidence: { metric: "saturation", value: 19.8, threshold: 25.0 },
        why_it_matters:
          "Desaturated images reduce color-based feature extraction accuracy.",
        recommendation:
          "Check camera white-balance settings and lighting conditions.",
      },
    ],
  },
  {
    analysis_id: "ana_010",
    filename: "parking_structure_007.jpg",
    quality_score: 63,
    quality_label: "Good",
    analysis_confidence: 88,
    issues: [
      {
        type: "overexposure",
        severity: "moderate",
        metric: "overexposure_pct",
        value: 12.3,
        threshold: 10.0,
        impact: "Bright regions may obscure detail in exposed areas.",
        confidence: 0.82,
        description:
          "The image may be overexposed. Highlight clipping may have occurred.",
      },
    ],
    statistics: {
      sharpness: 61.4,
      brightness: 168.2,
      contrast: 35.7,
      noise_estimate: 12.3,
      entropy: 5.78,
      saturation: 31.2,
    },
    explanations: [
      "Brightness is somewhat above the optimal range.",
      "Image contrast is moderate but below the optimal range.",
      "Sharpness is sufficient for visible edge detail.",
      "Image saturation is moderately below the calibrated range.",
      "Edge density is below the median range for training images.",
    ],
    summary:
      "The image demonstrates good overall quality (score: 63/100). Minor conditions detected (Overexposure) but are unlikely to significantly affect downstream analysis.",
    recommendation:
      "Image is suitable for most analytics tasks.",
    created_at: "2026-08-23T09:05:00Z",
    processing_time_ms: 1080,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 55,
    analytics_readiness_status: "LIMITED READINESS",
    analytics_readiness_details: {
      base_quality_score: 63.0,
      blur_penalty: 0.0,
      exposure_penalty: 8.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 3.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [],
    issue_explanations: [
      {
        issue: "Overexposure",
        evidence: {
          metric: "overexposure_pct",
          value: 12.3,
          threshold: 10.0,
        },
        why_it_matters:
          "Highlight clipping may affect bright-region analysis.",
        recommendation:
          "Adjust exposure compensation or use HDR imaging.",
      },
    ],
  },
  {
    analysis_id: "ana_011",
    filename: "warehouse_loading_bay_019.png",
    quality_score: 94,
    quality_label: "Excellent",
    analysis_confidence: 98,
    issues: [],
    statistics: {
      sharpness: 88.9,
      brightness: 131.6,
      contrast: 57.3,
      noise_estimate: 3.2,
      entropy: 7.28,
      saturation: 42.1,
    },
    explanations: [
      "Brightness is within the expected range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is sufficient for visible edge detail.",
      "Saturation is within the expected range.",
      "Edge density is sufficient for structural feature extraction.",
    ],
    summary:
      "Image quality is excellent with no significant issues detected. The image is well-suited for all downstream analysis tasks.",
    recommendation:
      "No action required — image meets all quality standards.",
    created_at: "2026-08-22T21:30:00Z",
    processing_time_ms: 920,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 91,
    analytics_readiness_status: "HIGHLY READY",
    analytics_readiness_details: {
      base_quality_score: 94.0,
      blur_penalty: 0.0,
      exposure_penalty: 0.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 3.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [],
    issue_explanations: [],
  },
  {
    analysis_id: "ana_012",
    filename: "gate_entry_cam_015.webp",
    quality_score: 45,
    quality_label: "Fair",
    analysis_confidence: 86,
    issues: [
      {
        type: "insufficient_sharpness",
        severity: "moderate",
        metric: "laplacian_variance",
        value: 20.5,
        threshold: 25.0,
        impact: "Object detection accuracy may be reduced.",
        confidence: 0.79,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "underexposure",
        severity: "moderate",
        metric: "mean_brightness",
        value: 76.4,
        threshold: 80.0,
        impact: "Shadow detail may be lost, reducing detection confidence.",
        confidence: 0.77,
        description:
          "The image appears darker than the expected range. Shadow detail may be lost.",
      },
    ],
    statistics: {
      sharpness: 42.1,
      brightness: 76.4,
      contrast: 36.9,
      noise_estimate: 16.8,
      entropy: 5.53,
      saturation: 24.7,
    },
    explanations: [
      "Brightness is somewhat below the optimal range.",
      "Image contrast is moderate but below the optimal range.",
      "Sharpness is below the optimal range but some edge detail remains.",
      "Image saturation is moderately below the calibrated range.",
      "Edge density is below the median range for training images.",
    ],
    summary:
      "The image exhibits moderate quality issues (Insufficient Sharpness, Underexposure) with a quality score of 45/100. These conditions may reduce the reliability of downstream analysis.",
    recommendation:
      "Consider image enhancement or manual review before analytics.",
    created_at: "2026-08-22T17:55:00Z",
    processing_time_ms: 1010,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 35,
    analytics_readiness_status: "NOT READY",
    analytics_readiness_details: {
      base_quality_score: 45.0,
      blur_penalty: 10.0,
      exposure_penalty: 8.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 5.0,
    },
    context: "Smart Campus",
    context_impacts: [
      {
        issue_type: "insufficient_sharpness",
        context: "Smart Campus",
        impact:
          "Student/visitor detection and facial recognition accuracy will be reduced.",
      },
      {
        issue_type: "underexposure",
        context: "Smart Campus",
        impact:
          "Low-light conditions may reduce occupancy monitoring accuracy.",
      },
    ],
    issue_explanations: [
      {
        issue: "Insufficient Sharpness",
        evidence: {
          metric: "laplacian_variance",
          value: 20.5,
          threshold: 25.0,
        },
        why_it_matters:
          "Blurred images reduce object detection and classification accuracy.",
        recommendation:
          "Improve camera focus or use a higher-resolution sensor.",
      },
      {
        issue: "Underexposure",
        evidence: {
          metric: "mean_brightness",
          value: 76.4,
          threshold: 80.0,
        },
        why_it_matters:
          "Dark images lose shadow detail needed for reliable analysis.",
        recommendation:
          "Adjust camera exposure settings or add supplemental lighting.",
      },
    ],
  },
  {
    analysis_id: "ana_013",
    filename: "server_room_cam_002.jpg",
    quality_score: 88,
    quality_label: "Excellent",
    analysis_confidence: 95,
    issues: [],
    statistics: {
      sharpness: 80.3,
      brightness: 125.8,
      contrast: 52.4,
      noise_estimate: 5.6,
      entropy: 7.06,
      saturation: 39.8,
    },
    explanations: [
      "Brightness is within the expected range.",
      "Contrast is sufficient for visible detail separation.",
      "Sharpness is sufficient for visible edge detail.",
      "Saturation is within the expected range.",
      "Edge density is sufficient for structural feature extraction.",
    ],
    summary:
      "The image demonstrates excellent overall quality (score: 88/100). No significant quality issues detected.",
    recommendation:
      "No action required — image meets all quality standards.",
    created_at: "2026-08-22T10:12:00Z",
    processing_time_ms: 960,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 82,
    analytics_readiness_status: "HIGHLY READY",
    analytics_readiness_details: {
      base_quality_score: 88.0,
      blur_penalty: 0.0,
      exposure_penalty: 0.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 3.0,
    },
    context: "CCTV Surveillance",
    context_impacts: [],
    issue_explanations: [],
  },
  {
    analysis_id: "ana_014",
    filename: "construction_perimeter_041.png",
    quality_score: 12,
    quality_label: "Critical",
    analysis_confidence: 98,
    issues: [
      {
        type: "underexposure",
        severity: "high",
        metric: "mean_brightness",
        value: 48.9,
        threshold: 80.0,
        impact: "Shadow detail may be lost, reducing detection confidence.",
        confidence: 0.95,
        description:
          "The image appears darker than the expected range. Shadow detail may be lost.",
      },
      {
        type: "low_color_information",
        severity: "high",
        metric: "saturation",
        value: 4.2,
        threshold: 25.0,
        impact: "Color-based analysis features are severely compromised.",
        confidence: 0.97,
        description:
          "Colors appear relatively muted or desaturated.",
      },
    ],
    statistics: {
      sharpness: 5.2,
      brightness: 48.9,
      contrast: 8.7,
      noise_estimate: 41.3,
      entropy: 2.91,
      saturation: 4.2,
    },
    explanations: [
      "Brightness is somewhat below the optimal range.",
      "Image contrast is below the calibrated range.",
      "Sharpness is significantly below the calibrated threshold. The image may be blurry.",
      "Image saturation is well below the calibrated range. Colors are heavily desaturated.",
      "Edge density is very low, suggesting limited structural content.",
    ],
    summary:
      "Image quality is critically degraded (score: 12/100). The image is unreliable for automated analysis. Recapture or manual review is required.",
    recommendation:
      "Image is unreliable for automated analysis. Recapture or manual review is required.",
    created_at: "2026-08-21T23:18:00Z",
    processing_time_ms: 1340,
    model_version: "visionguard-iqa-v2.0",
    analytics_readiness_score: 2,
    analytics_readiness_status: "CRITICAL",
    analytics_readiness_details: {
      base_quality_score: 12.0,
      blur_penalty: 0.0,
      exposure_penalty: 15.0,
      noise_penalty: 0.0,
      corruption_penalty: 0.0,
      information_penalty: 10.0,
    },
    context: "Infrastructure Inspection",
    context_impacts: [
      {
        issue_type: "underexposure",
        context: "Infrastructure Inspection",
        impact:
          "Structural defects in shadowed regions may be invisible to detection algorithms.",
      },
      {
        issue_type: "low_color_information",
        context: "Infrastructure Inspection",
        impact:
          "Color-based corrosion or material degradation indicators will be unreliable.",
      },
    ],
    issue_explanations: [
      {
        issue: "Underexposure",
        evidence: {
          metric: "mean_brightness",
          value: 48.9,
          threshold: 80.0,
        },
        why_it_matters:
          "Dark images lose shadow detail needed for reliable analysis.",
        recommendation:
          "Adjust camera exposure settings or add supplemental lighting.",
      },
      {
        issue: "Low Color Information",
        evidence: { metric: "saturation", value: 4.2, threshold: 25.0 },
        why_it_matters:
          "Heavily desaturated images reduce color-based analysis reliability.",
        recommendation:
          "Check camera sensor and lighting conditions.",
      },
    ],
  },
];

/** Lookup helper */
export function getMockAnalysisById(id: string): AnalysisResult | undefined {
  return mockHistory.find((a) => a.analysis_id === id);
}
