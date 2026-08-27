import type { AnalysisResult } from "../types/analysis";

export const mockAcceptable: AnalysisResult = {
  analysis_id: "ana_001",
  filename: "warehouse_entrance_042.jpg",
  quality_score: 87,
  quality_label: "ACCEPTABLE",
  analysis_confidence: 94,
  issues: [
    {
      type: "Minor Noise",
      severity: "LOW",
      confidence: 72,
      explanation:
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
  summary:
    "The image exhibits good overall quality with adequate sharpness and balanced exposure. Minor noise is present in darker regions but does not materially impact downstream visual analysis. The image is suitable for use in monitoring pipelines.",
  created_at: "2026-08-27T09:15:00Z",
  processing_time_ms: 1240,
};

export const mockDegraded: AnalysisResult = {
  analysis_id: "ana_002",
  filename: "parking_lot_cam_117.png",
  quality_score: 52,
  quality_label: "DEGRADED",
  analysis_confidence: 89,
  issues: [
    {
      type: "Blur",
      severity: "HIGH",
      confidence: 91,
      explanation:
        "Insufficient edge detail detected across the frame. The image may significantly affect object detection accuracy and should not be used for precision monitoring tasks.",
    },
    {
      type: "Underexposure",
      severity: "MEDIUM",
      confidence: 84,
      explanation:
        "Image brightness is below the recommended threshold. Key visual features in shadowed areas may be lost, reducing reliability of downstream analysis.",
    },
    {
      type: "Image Noise",
      severity: "MEDIUM",
      confidence: 76,
      explanation:
        "Moderate chroma and luminance noise detected. This degrades visual clarity and may introduce artifacts in feature extraction pipelines.",
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
  summary:
    "The image shows significant blur and moderate noise with underexposure affecting shadow detail. These conditions may reduce the reliability of downstream computer vision systems. Manual review or image recapture is recommended.",
  created_at: "2026-08-27T07:42:00Z",
  processing_time_ms: 980,
};

export const mockDefective: AnalysisResult = {
  analysis_id: "ana_003",
  filename: "construction_zone_009.webp",
  quality_score: 18,
  quality_label: "DEFECTIVE",
  analysis_confidence: 96,
  issues: [
    {
      type: "Severe Degradation",
      severity: "HIGH",
      confidence: 97,
      explanation:
        "Critical level of visual degradation detected. The image lacks sufficient detail for any meaningful computer vision analysis.",
    },
    {
      type: "Image Corruption",
      severity: "HIGH",
      confidence: 94,
      explanation:
        "Signs of data corruption or transfer errors. Blocking artifacts and discontinuous pixel patterns indicate the file may be partially damaged.",
    },
    {
      type: "Overexposure",
      severity: "HIGH",
      confidence: 89,
      explanation:
        "Large areas of the image are clipped to pure white, resulting in irreversible loss of visual information.",
    },
    {
      type: "Blur",
      severity: "HIGH",
      confidence: 92,
      explanation:
        "Extreme lack of sharpness across the entire frame. No recoverable edge detail is present.",
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
  summary:
    "This image is severely compromised by multiple overlapping defects including corruption artifacts, extreme overexposure, and pervasive blur. The visual data is unreliable and should not be used in any downstream monitoring or analytics pipeline. Immediate recapture from a properly calibrated source is required.",
  created_at: "2026-08-26T22:08:00Z",
  processing_time_ms: 1120,
};

/** Additional history entries for the history page */
export const mockHistory: AnalysisResult[] = [
  mockAcceptable,
  mockDegraded,
  mockDefective,
  {
    analysis_id: "ana_004",
    filename: "loading_dock_023.jpg",
    quality_score: 79,
    quality_label: "ACCEPTABLE",
    analysis_confidence: 91,
    issues: [
      {
        type: "Overexposure",
        severity: "LOW",
        confidence: 68,
        explanation:
          "Mild highlight clipping in the upper-right quadrant. Main subject area remains well-exposed and analyzable.",
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
    summary:
      "Generally acceptable quality with minor overexposure in peripheral regions. The primary monitoring area is well-rendered and suitable for downstream analysis.",
    created_at: "2026-08-26T15:30:00Z",
    processing_time_ms: 1050,
  },
  {
    analysis_id: "ana_005",
    filename: "security_cam_feed_088.png",
    quality_score: 41,
    quality_label: "DEGRADED",
    analysis_confidence: 87,
    issues: [
      {
        type: "Blur",
        severity: "HIGH",
        confidence: 88,
        explanation:
          "Motion blur detected from camera vibration. Subject edges are significantly smeared.",
      },
      {
        type: "Image Noise",
        severity: "MEDIUM",
        confidence: 81,
        explanation:
          "High ISO noise visible throughout. Grain patterns suggest low-light sensor amplification.",
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
    summary:
      "Motion blur from camera instability combined with high ISO noise significantly degrades image quality. Not recommended for automated analysis without stabilization preprocessing.",
    created_at: "2026-08-25T11:22:00Z",
    processing_time_ms: 940,
  },
  {
    analysis_id: "ana_006",
    filename: "warehouse_interior_012.jpg",
    quality_score: 91,
    quality_label: "ACCEPTABLE",
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
    summary:
      "Excellent image quality with strong edge definition, balanced exposure, and minimal noise. Well-suited for all downstream visual analysis tasks.",
    created_at: "2026-08-25T08:05:00Z",
    processing_time_ms: 890,
  },
  {
    analysis_id: "ana_007",
    filename: "exterior_cam_north_005.png",
    quality_score: 63,
    quality_label: "DEGRADED",
    analysis_confidence: 85,
    issues: [
      {
        type: "Underexposure",
        severity: "HIGH",
        confidence: 87,
        explanation:
          "Overall scene is significantly underexposed. Critical detail in mid-tones and shadows is lost.",
      },
      {
        type: "Potential Visual Defect",
        severity: "MEDIUM",
        confidence: 73,
        explanation:
          "Anomalous dark region in the lower-left corner may indicate lens obstruction or sensor defect.",
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
    summary:
      "Severe underexposure limits the utility of this image for visual monitoring. An anomalous dark region suggests possible hardware obstruction. Recommend checking camera placement and exposure settings.",
    created_at: "2026-08-24T19:48:00Z",
    processing_time_ms: 1180,
  },
  {
    analysis_id: "ana_008",
    filename: "traffic_camera_01.jpg",
    quality_score: 82,
    quality_label: "ACCEPTABLE",
    analysis_confidence: 93,
    issues: [
      {
        type: "Minor Noise",
        severity: "LOW",
        confidence: 65,
        explanation:
          "Faint sensor noise visible in uniform sky regions. Does not impact vehicle detection.",
      },
    ],
    statistics: {
      sharpness: 76.2,
      brightness: 141.5,
      contrast: 51.8,
      noise_estimate: 7.1,
      entropy: 6.94,
      saturation: 36.4,
    },
    summary:
      "Good overall quality with strong lane markings and vehicle outlines. Minor noise in sky area is negligible for traffic monitoring purposes.",
    created_at: "2026-08-24T14:15:00Z",
    processing_time_ms: 1020,
  },
  {
    analysis_id: "ana_009",
    filename: "factory_floor_cam_003.png",
    quality_score: 35,
    quality_label: "DEFECTIVE",
    analysis_confidence: 92,
    issues: [
      {
        type: "Severe Degradation",
        severity: "HIGH",
        confidence: 95,
        explanation:
        "Image shows extensive compression artifacts rendering fine details unrecognizable.",
      },
      {
        type: "Blur",
        severity: "HIGH",
        confidence: 89,
        explanation:
        "Focus failure across the entire frame. No sharp edges detected.",
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
    summary:
      "Image is severely degraded with compression artifacts and focus failure. Not suitable for any automated inspection tasks.",
    created_at: "2026-08-23T16:40:00Z",
    processing_time_ms: 1150,
  },
  {
    analysis_id: "ana_010",
    filename: "parking_structure_007.jpg",
    quality_score: 71,
    quality_label: "DEGRADED",
    analysis_confidence: 88,
    issues: [
      {
        type: "Overexposure",
        severity: "MEDIUM",
        confidence: 82,
        explanation:
        "Sun glare causes highlight clipping in the central area, obscuring parked vehicles.",
      },
      {
        type: "Image Noise",
        severity: "LOW",
        confidence: 70,
        explanation:
        "Mild luminance noise in shadowed regions beneath the structure.",
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
    summary:
      "Sun glare significantly impacts visibility in the central monitoring zone. Peripheral areas remain analyzable but overall quality is degraded.",
    created_at: "2026-08-23T09:05:00Z",
    processing_time_ms: 1080,
  },
  {
    analysis_id: "ana_011",
    filename: "warehouse_loading_bay_019.png",
    quality_score: 94,
    quality_label: "ACCEPTABLE",
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
    summary:
      "Excellent image quality. Sharp focus, balanced lighting, and minimal noise. Ideal for all downstream computer vision tasks.",
    created_at: "2026-08-22T21:30:00Z",
    processing_time_ms: 920,
  },
  {
    analysis_id: "ana_012",
    filename: "gate_entry_cam_015.webp",
    quality_score: 57,
    quality_label: "DEGRADED",
    analysis_confidence: 86,
    issues: [
      {
        type: "Blur",
        severity: "MEDIUM",
        confidence: 79,
        explanation:
        "Slight motion blur on moving subjects. Static background remains acceptable.",
      },
      {
        type: "Underexposure",
        severity: "MEDIUM",
        confidence: 77,
        explanation:
        "Low ambient light conditions result in reduced contrast and detail loss in darker areas.",
      },
      {
        type: "Image Noise",
        severity: "LOW",
        confidence: 64,
        explanation:
        "Chroma noise visible in uniform colored surfaces.",
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
    summary:
      "Multiple quality issues from low-light conditions reduce image reliability. Subject identification may be compromised for moving objects.",
    created_at: "2026-08-22T17:55:00Z",
    processing_time_ms: 1010,
  },
  {
    analysis_id: "ana_013",
    filename: "server_room_cam_002.jpg",
    quality_score: 88,
    quality_label: "ACCEPTABLE",
    analysis_confidence: 95,
    issues: [
      {
        type: "Minor Noise",
        severity: "LOW",
        confidence: 58,
        explanation:
        "Negligible noise in uniform ceiling panels. No impact on equipment monitoring.",
      },
    ],
    statistics: {
      sharpness: 80.3,
      brightness: 125.8,
      contrast: 52.4,
      noise_estimate: 5.6,
      entropy: 7.06,
      saturation: 39.8,
    },
    summary:
      "High-quality capture suitable for equipment monitoring and thermal overlay alignment. Minor noise is well within acceptable thresholds.",
    created_at: "2026-08-22T10:12:00Z",
    processing_time_ms: 960,
  },
  {
    analysis_id: "ana_014",
    filename: "construction_perimeter_041.png",
    quality_score: 12,
    quality_label: "DEFECTIVE",
    analysis_confidence: 98,
    issues: [
      {
        type: "Image Corruption",
        severity: "HIGH",
        confidence: 99,
        explanation:
        "File header corruption detected. Image data is partially unreadable.",
      },
      {
        type: "Severe Degradation",
        severity: "HIGH",
        confidence: 97,
        explanation:
        "Complete loss of structural detail. No usable visual information present.",
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
    summary:
        "File is critically corrupted and contains no usable visual data. The source camera or storage medium should be inspected immediately.",
    created_at: "2026-08-21T23:18:00Z",
    processing_time_ms: 1340,
  },
];

/** Lookup helper */
export function getMockAnalysisById(id: string): AnalysisResult | undefined {
  return mockHistory.find((a) => a.analysis_id === id);
}
