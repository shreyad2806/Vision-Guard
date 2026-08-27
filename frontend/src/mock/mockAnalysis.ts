import type { AnalysisResult } from "../types/analysis";

export const mockAcceptable: AnalysisResult = {
  analysis_id: "ana_001",
  filename: "warehouse_entrance_042.jpg",
  quality_score: 87,
  quality_label: "Excellent",
  analysis_confidence: 94,
  issues: [
    {
      type: "minor_noise",
      title: "Minor Noise",
      severity: "LOW",
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
  created_at: "2026-08-27T09:15:00Z",
  processing_time_ms: 1240,
  model_version: "visionguard-iqa-v1.1",
};

export const mockDegraded: AnalysisResult = {
  analysis_id: "ana_002",
  filename: "parking_lot_cam_117.png",
  quality_score: 52,
  quality_label: "Fair",
  analysis_confidence: 89,
  issues: [
    {
      type: "low_sharpness",
      title: "Low Sharpness",
      severity: "HIGH",
      confidence: 0.91,
      description:
        "The image appears blurry or lacks sufficient edge detail. Edge information is below the expected threshold for reliable visual analysis.",
    },
    {
      type: "low_brightness",
      title: "Low Brightness",
      severity: "MEDIUM",
      confidence: 0.84,
      description:
        "The image appears darker than the expected range. Shadow detail may be lost.",
    },
    {
      type: "low_saturation",
      title: "Low Saturation",
      severity: "LOW",
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
    "The image exhibits moderate quality issues (Low Sharpness, Low Brightness, Low Saturation) with a quality score of 52/100. These conditions may reduce the reliability of downstream analysis. Consider image enhancement or manual review.",
  created_at: "2026-08-27T07:42:00Z",
  processing_time_ms: 980,
  model_version: "visionguard-iqa-v1.1",
};

export const mockDefective: AnalysisResult = {
  analysis_id: "ana_003",
  filename: "construction_zone_009.webp",
  quality_score: 18,
  quality_label: "Poor",
  analysis_confidence: 96,
  issues: [
    {
      type: "high_brightness",
      title: "High Brightness",
      severity: "HIGH",
      confidence: 0.89,
      description:
        "The image may be overexposed. Highlight clipping may have occurred, resulting in loss of detail in bright areas.",
    },
    {
      type: "low_sharpness",
      title: "Low Sharpness",
      severity: "HIGH",
      confidence: 0.92,
      description:
        "The image appears blurry or lacks sufficient edge detail. Edge information is below the expected threshold for reliable visual analysis.",
    },
    {
      type: "low_saturation",
      title: "Low Saturation",
      severity: "HIGH",
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
    "Significant quality degradation detected (High Brightness, Low Sharpness, Low Saturation) with a quality score of 18/100. Image quality is substantially compromised. Review or recapture is recommended.",
  created_at: "2026-08-26T22:08:00Z",
  processing_time_ms: 1120,
  model_version: "visionguard-iqa-v1.1",
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
        type: "high_brightness",
        title: "High Brightness",
        severity: "LOW",
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
      "The image demonstrates good overall quality (score: 72/100). Minor conditions detected (High Brightness) but are unlikely to significantly affect downstream analysis.",
    created_at: "2026-08-26T15:30:00Z",
    processing_time_ms: 1050,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_005",
    filename: "security_cam_feed_088.png",
    quality_score: 41,
    quality_label: "Fair",
    analysis_confidence: 87,
    issues: [
      {
        type: "low_sharpness",
        title: "Low Sharpness",
        severity: "HIGH",
        confidence: 0.88,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "low_saturation",
        title: "Low Saturation",
        severity: "MEDIUM",
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
      "The image exhibits moderate quality issues (Low Sharpness, Low Saturation) with a quality score of 41/100. These conditions may reduce the reliability of downstream analysis. Consider image enhancement or manual review.",
    created_at: "2026-08-25T11:22:00Z",
    processing_time_ms: 940,
    model_version: "visionguard-iqa-v1.1",
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
    created_at: "2026-08-25T08:05:00Z",
    processing_time_ms: 890,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_007",
    filename: "exterior_cam_north_005.png",
    quality_score: 55,
    quality_label: "Fair",
    analysis_confidence: 85,
    issues: [
      {
        type: "low_brightness",
        title: "Low Brightness",
        severity: "HIGH",
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
      "The image exhibits moderate quality issues (Low Brightness) with a quality score of 55/100. These conditions may reduce the reliability of downstream analysis. Consider image enhancement or manual review.",
    created_at: "2026-08-24T19:48:00Z",
    processing_time_ms: 1180,
    model_version: "visionguard-iqa-v1.1",
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
      "The image demonstrates excellent overall quality (score: 82/100). Minor issues were noted but do not materially affect quality assessment.",
    created_at: "2026-08-24T14:15:00Z",
    processing_time_ms: 1020,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_009",
    filename: "factory_floor_cam_003.png",
    quality_score: 28,
    quality_label: "Poor",
    analysis_confidence: 92,
    issues: [
      {
        type: "low_sharpness",
        title: "Low Sharpness",
        severity: "HIGH",
        confidence: 0.89,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "low_saturation",
        title: "Low Saturation",
        severity: "MEDIUM",
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
      "Significant quality degradation detected (Low Sharpness, Low Saturation) with a quality score of 28/100. Image quality is substantially compromised. Review or recapture is recommended.",
    created_at: "2026-08-23T16:40:00Z",
    processing_time_ms: 1150,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_010",
    filename: "parking_structure_007.jpg",
    quality_score: 63,
    quality_label: "Good",
    analysis_confidence: 88,
    issues: [
      {
        type: "high_brightness",
        title: "High Brightness",
        severity: "MEDIUM",
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
      "The image demonstrates good overall quality (score: 63/100). Minor conditions detected (High Brightness) but are unlikely to significantly affect downstream analysis.",
    created_at: "2026-08-23T09:05:00Z",
    processing_time_ms: 1080,
    model_version: "visionguard-iqa-v1.1",
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
    created_at: "2026-08-22T21:30:00Z",
    processing_time_ms: 920,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_012",
    filename: "gate_entry_cam_015.webp",
    quality_score: 45,
    quality_label: "Fair",
    analysis_confidence: 86,
    issues: [
      {
        type: "low_sharpness",
        title: "Low Sharpness",
        severity: "MEDIUM",
        confidence: 0.79,
        description:
          "The image appears blurry or lacks sufficient edge detail.",
      },
      {
        type: "low_brightness",
        title: "Low Brightness",
        severity: "MEDIUM",
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
      "The image exhibits moderate quality issues (Low Sharpness, Low Brightness) with a quality score of 45/100. These conditions may reduce the reliability of downstream analysis. Consider image enhancement or manual review.",
    created_at: "2026-08-22T17:55:00Z",
    processing_time_ms: 1010,
    model_version: "visionguard-iqa-v1.1",
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
      "The image demonstrates excellent overall quality (score: 88/100). Minor issues were noted but do not materially affect quality assessment.",
    created_at: "2026-08-22T10:12:00Z",
    processing_time_ms: 960,
    model_version: "visionguard-iqa-v1.1",
  },
  {
    analysis_id: "ana_014",
    filename: "construction_perimeter_041.png",
    quality_score: 12,
    quality_label: "Critical",
    analysis_confidence: 98,
    issues: [
      {
        type: "low_brightness",
        title: "Low Brightness",
        severity: "HIGH",
        confidence: 0.95,
        description:
          "The image appears darker than the expected range. Shadow detail may be lost.",
      },
      {
        type: "low_saturation",
        title: "Low Saturation",
        severity: "HIGH",
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
    created_at: "2026-08-21T23:18:00Z",
    processing_time_ms: 1340,
    model_version: "visionguard-iqa-v1.1",
  },
];

/** Lookup helper */
export function getMockAnalysisById(id: string): AnalysisResult | undefined {
  return mockHistory.find((a) => a.analysis_id === id);
}
