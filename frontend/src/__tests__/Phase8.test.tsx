/**
 * Phase 8 — Comprehensive Frontend Tests
 *
 * Covers:
 * 1. Initial page state
 * 2. Image selection
 * 3. Analyze flow
 * 4. Successful response rendering
 * 5. Explainability display
 * 6. Smart-city context
 * 7. History
 * 8. Error handling
 * 9. Previous analysis display
 * 10. No hard-coded production demo data
 * 11. Type safety (API contract)
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import type { AnalysisResult } from "../types/analysis";
import {
  mockAcceptable,
  mockDegraded,
  mockHistory,
} from "../mock/mockAnalysis";

// ---------------------------------------------------------------------------
// Mock API module
// ---------------------------------------------------------------------------
vi.mock("../services/api", () => ({
  analyzeImage: vi.fn(),
  getAnalyses: vi.fn(),
  getAnalysisById: vi.fn(),
  SUPPORTED_CONTEXTS: [
    "CCTV Surveillance",
    "Traffic Monitoring",
    "Crowd Monitoring",
    "Drone Imagery",
    "Infrastructure Inspection",
    "Smart Campus",
  ],
}));

import {
  analyzeImage,
  getAnalyses,
  getAnalysisById,
  SUPPORTED_CONTEXTS,
} from "../services/api";

const mockedAnalyze = vi.mocked(analyzeImage);
const mockedGetAnalyses = vi.mocked(getAnalyses);
const mockedGetById = vi.mocked(getAnalysisById);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function createTestImage(name = "test.jpg", size = 1024): File {
  const buffer = new ArrayBuffer(size);
  return new File([buffer], name, { type: "image/jpeg" });
}

function wrapper({ children }: { children: React.ReactNode }) {
  return <MemoryRouter initialEntries={["/"]}>{children}</MemoryRouter>;
}

function historyWrapper(id?: string) {
  const entry = id ? `/history/${id}` : "/history";
  return function HistoryWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <MemoryRouter initialEntries={[entry]}>
        <Routes>
          <Route path="/" element={children} />
          <Route path="/history" element={children} />
          <Route path="/history/:id" element={children} />
        </Routes>
      </MemoryRouter>
    );
  };
}

async function uploadFile(
  user: ReturnType<typeof userEvent.setup>,
  container: HTMLElement,
  file: File,
) {
  const input = container.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  await user.upload(input, file);
}

import Dashboard from "../pages/Dashboard";
import HistoryPage from "../pages/History";
import AnalysisDetail from "../pages/AnalysisDetail";
import QualityScore from "../components/analysis/QualityScore";
import IssueCard from "../components/analysis/IssueCard";
import AnalyticsReadiness from "../components/analysis/AnalyticsReadiness";
import ContextImpact from "../components/analysis/ContextImpact";
import IssueExplanations from "../components/analysis/IssueExplanations";
import AnalysisResultComponent from "../components/analysis/AnalysisResult";
import EmptyState from "../components/common/EmptyState";
import ErrorState from "../components/common/ErrorState";
import ImageUploader from "../components/upload/ImageUploader";

/**
 * Helper: wait for a text to appear, using getAllByText for elements
 * that may render in both desktop + mobile views.
 */
function waitForText(text: string | RegExp) {
  return waitFor(() => {
    const els = screen.getAllByText(text);
    expect(els.length).toBeGreaterThanOrEqual(1);
  });
}

// =========================================================================
// 1. Initial Page State
// =========================================================================

describe("1. Initial Page State", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows empty state when no image is selected", () => {
    render(<Dashboard />, { wrapper });
    expect(screen.getByText("No image selected")).toBeInTheDocument();
    expect(screen.getByText(/Upload an image to begin/)).toBeInTheDocument();
  });

  it("analyze button is disabled when no image is selected", () => {
    render(<Dashboard />, { wrapper });
    const btn = screen.getByRole("button", { name: "Analyze Image" });
    expect(btn).toBeDisabled();
  });

  it("shows the pipeline context selector with default CCTV Surveillance", () => {
    render(<Dashboard />, { wrapper });
    expect(screen.getByRole("combobox")).toHaveValue("CCTV Surveillance");
  });

  it("shows all 6 supported contexts in dropdown", () => {
    render(<Dashboard />, { wrapper });
    const options = within(screen.getByRole("combobox")).getAllByRole("option");
    expect(options).toHaveLength(6);
    expect(options.map((o) => o.textContent)).toEqual([
      "CCTV Surveillance",
      "Traffic Monitoring",
      "Crowd Monitoring",
      "Drone Imagery",
      "Infrastructure Inspection",
      "Smart Campus",
    ]);
  });
});

// =========================================================================
// 2. Image Selection
// =========================================================================

describe("2. Image Selection", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows image preview after selecting a valid file", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    const file = createTestImage("photo.jpg");

    await uploadFile(user, container, file);

    const imgs = screen.getAllByRole("img", { name: "photo.jpg" });
    expect(imgs.length).toBeGreaterThanOrEqual(1);
  });

  it("hides 'No image selected' after image is chosen", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    expect(screen.getByText("No image selected")).toBeInTheDocument();

    await uploadFile(user, container, createTestImage("test.png"));

    expect(screen.queryByText("No image selected")).not.toBeInTheDocument();
  });

  it("displays the correct filename", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });

    await uploadFile(user, container, createTestImage("my_custom_image.webp"));

    expect(screen.getByText("my_custom_image.webp")).toBeInTheDocument();
  });

  it("enables the Analyze button after image selection", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    const btn = screen.getByRole("button", { name: "Analyze Image" });
    expect(btn).toBeDisabled();

    await uploadFile(user, container, createTestImage("ready.jpg"));

    expect(btn).not.toBeDisabled();
  });
});

// =========================================================================
// 3. Analyze Flow
// =========================================================================

describe("3. Analyze Flow", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows loading state during analysis", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockReturnValue(new Promise(() => {}));

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    expect(screen.getByText("Analyzing Visual Quality")).toBeInTheDocument();
    expect(screen.getByText(/This may take a few seconds/)).toBeInTheDocument();
  });

  it("disables analyze button while request is in progress", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockReturnValue(new Promise(() => {}));

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    // Button disappears during loading (replaced by AnalysisProgress)
    // Verify the loading indicator appears instead of the button
    await waitFor(() => {
      expect(screen.getByText("Analyzing Visual Quality")).toBeInTheDocument();
    });
    // Analyze button should no longer be in the DOM
    expect(screen.queryByRole("button", { name: "Analyze Image" })).not.toBeInTheDocument();
  });

  it("hides the upload zone during loading", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockReturnValue(new Promise(() => {}));

    await uploadFile(user, container, createTestImage("loaded.jpg"));
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    // Upload zone hidden during loading (Analyzing progress shown instead)
    await waitFor(() => {
      expect(screen.queryByText(/Drag and drop/)).not.toBeInTheDocument();
    });
  });
});

// =========================================================================
// 4. Successful Response Rendering
// =========================================================================

describe("4. Successful Response Rendering", () => {
  beforeEach(() => vi.clearAllMocks());

  it("displays all Phase 1-6 backend fields for a successful analysis", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockAcceptable);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    // Wait for result — use getAllByText since score appears in two places
    await waitForText("87");

    // Phase 1 — quality
    expect(screen.getAllByText("Excellent").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/visionguard-iqa-v2\.0/)).toBeInTheDocument();

    // Phase 3 — analytics readiness
    expect(screen.getAllByText("READY").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Penalty Breakdown")).toBeInTheDocument();

    // Phase 4 — context
    expect(screen.getAllByText("CCTV Surveillance").length).toBeGreaterThanOrEqual(1);

    // Phase 6 — explainability
    expect(screen.getByText("Explainability")).toBeInTheDocument();

    // Summary + Recommendation
    expect(screen.getByText(/87\/100/)).toBeInTheDocument();
    expect(
      screen.getByText(/Image is suitable for automated analysis/),
    ).toBeInTheDocument();

    // Metadata
    expect(screen.getByText("Analysis Metadata")).toBeInTheDocument();
  });

  it("renders issue cards with severity, metric, value, threshold, and impact", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockDegraded);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("52");

    // 3 issues — each appears in IssueCard + IssueExplanations
    expect(screen.getAllByText(/Insufficient Sharpness/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Underexposure").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/Low Color Information/).length).toBeGreaterThanOrEqual(2);

    // Metric value — 18.70
    expect(screen.getByText("18.70")).toBeInTheDocument();
    // Threshold — 25.00 appears in two issue cards
    expect(screen.getAllByText("25.00").length).toBeGreaterThanOrEqual(2);
  });

  it("renders quality metrics (statistics) from backend", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockDegraded);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("52");

    expect(screen.getByText("34.2")).toBeInTheDocument(); // sharpness
    expect(screen.getByText("68.9")).toBeInTheDocument(); // brightness
    expect(screen.getByText("Image Quality Metrics")).toBeInTheDocument();
  });

  it("renders analytics readiness penalty breakdown", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockDegraded);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("LIMITED READINESS");

    expect(screen.getByText("Penalty Breakdown")).toBeInTheDocument();
    expect(screen.getByText("Blur")).toBeInTheDocument();
    expect(screen.getByText("Exposure")).toBeInTheDocument();
    expect(screen.getByText("-18.0")).toBeInTheDocument(); // blur penalty
    expect(screen.getByText("-8.0")).toBeInTheDocument(); // exposure penalty
  });
});

// =========================================================================
// 5. Explainability
// =========================================================================

describe("5. Explainability", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows issue explanations with evidence and recommendation", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockDegraded);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("52");

    expect(screen.getByText("Explainability")).toBeInTheDocument();

    // Explanation issues — each appears in IssueCard + IssueExplanations
    expect(screen.getAllByText("Insufficient Sharpness").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Underexposure").length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/Low Color Information/).length).toBeGreaterThanOrEqual(2);

    // Evidence detail — check value and threshold exist
    expect(screen.getByText("18.70")).toBeInTheDocument(); // value
    // 25.00 appears in multiple issue cards + explanation evidence
    expect(screen.getAllByText("25.00").length).toBeGreaterThanOrEqual(1);

    // Why it matters
    expect(
      screen.getByText(/Blurred images reduce object detection/),
    ).toBeInTheDocument();

    // Recommendation
    expect(screen.getByText(/Improve camera focus/)).toBeInTheDocument();
  });

  it("does not show explanations for undetected issues", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockAcceptable);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("87");

    // mockAcceptable has 1 explanation for "Mild Noise"
    expect(screen.getByText("Mild Noise")).toBeInTheDocument();
    // Should NOT have explanation for undetected issue
    expect(
      screen.queryByText(/Insufficient Sharpness/),
    ).not.toBeInTheDocument();
  });

  it("shows no explanation section for healthy images with zero issues", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue({
      ...mockAcceptable,
      issues: [],
      issue_explanations: [],
    });

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("87");

    expect(screen.queryByText("Explainability")).not.toBeInTheDocument();
  });
});

// =========================================================================
// 6. Smart-City Context
// =========================================================================

describe("6. Smart-City Context", () => {
  beforeEach(() => vi.clearAllMocks());

  it("displays the selected context and backend impacts", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue(mockDegraded);

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitForText("52");

    expect(screen.getByText("Smart-City Impact")).toBeInTheDocument();
    expect(
      screen.getAllByText("Traffic Monitoring").length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getByText(/Vehicle detection and license-plate recognition/),
    ).toBeInTheDocument();
  });

  it("allows changing the context selector", async () => {
    const user = userEvent.setup();
    render(<Dashboard />, { wrapper });

    const select = screen.getByRole("combobox");
    await user.selectOptions(select, "Infrastructure Inspection");
    expect(select).toHaveValue("Infrastructure Inspection");

    await user.selectOptions(select, "Smart Campus");
    expect(select).toHaveValue("Smart Campus");
  });

  it("passes selected context to analyzeImage", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockResolvedValue({
      ...mockAcceptable,
      context: "Drone Imagery",
    });

    await uploadFile(user, container, createTestImage());
    await user.selectOptions(screen.getByRole("combobox"), "Drone Imagery");
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitFor(() => {
      expect(mockedAnalyze).toHaveBeenCalledWith(
        expect.any(File),
        "Drone Imagery",
      );
    });
  });
});

// =========================================================================
// 7. History
// =========================================================================

describe("7. History", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows loading skeleton while fetching history", () => {
    mockedGetAnalyses.mockReturnValue(new Promise(() => {}));
    render(<HistoryPage />, { wrapper: historyWrapper() });

    const skeletons = document.querySelectorAll(".skeleton-pulse");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("displays existing analyses after loading", async () => {
    mockedGetAnalyses.mockResolvedValue(mockHistory);
    render(<HistoryPage />, { wrapper: historyWrapper() });

    await waitFor(() => {
      expect(screen.getByText("Analysis History")).toBeInTheDocument();
    });

    // Filenames appear in both desktop + mobile views
    expect(
      screen.getAllByText("warehouse_entrance_042.jpg").length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("parking_lot_cam_117.png").length,
    ).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("construction_zone_009.webp").length,
    ).toBeGreaterThanOrEqual(1);

    expect(
      screen.getByText(`${mockHistory.length} Analyses`),
    ).toBeInTheDocument();
  });

  it("displays empty state when API returns an empty array", async () => {
    mockedGetAnalyses.mockResolvedValue([]);
    render(<HistoryPage />, { wrapper: historyWrapper() });

    await waitFor(() => {
      expect(screen.getByText("No analyses yet")).toBeInTheDocument();
    });

    expect(
      screen.getByText(/Upload an image and run an analysis/),
    ).toBeInTheDocument();
  });

  it("shows quality scores for each history entry", async () => {
    mockedGetAnalyses.mockResolvedValue(mockHistory);
    render(<HistoryPage />, { wrapper: historyWrapper() });

    await waitFor(() => {
      expect(
        screen.getAllByText("warehouse_entrance_042.jpg").length,
      ).toBeGreaterThanOrEqual(1);
    });

    const scoreValues = document.querySelectorAll(".score-bar__value");
    expect(scoreValues.length).toBeGreaterThanOrEqual(mockHistory.length);
  });

  it("does not crash on error from API", async () => {
    mockedGetAnalyses.mockRejectedValue(new Error("Network error"));
    render(<HistoryPage />, { wrapper: historyWrapper() });

    await waitFor(() => {
      expect(
        screen.getByText(/Unable to load analysis history/),
      ).toBeInTheDocument();
    });
  });
});

// =========================================================================
// 8. Error Handling
// =========================================================================

describe("8. Error Handling", () => {
  beforeEach(() => vi.clearAllMocks());

  const errorCases = [
    {
      label: "400 invalid image",
      message: "The request was invalid. Please check the uploaded file.",
      match: /request was invalid/,
    },
    {
      label: "413 oversized image",
      message: "The uploaded file exceeds the maximum allowed size.",
      match: /exceeds the maximum/,
    },
    {
      label: "500 backend failure",
      message: "An internal server error occurred. Please try again later.",
      match: /internal server error/,
    },
    {
      label: "network failure",
      message:
        "Unable to reach the server. Please check your network connection and try again.",
      match: /Unable to reach the server/,
    },
  ];

  for (const { label, message, match } of errorCases) {
    it(`shows user-friendly error for ${label}`, async () => {
      const user = userEvent.setup();
      const { container } = render(<Dashboard />, { wrapper });
      mockedAnalyze.mockRejectedValue(new Error(message));

      await uploadFile(user, container, createTestImage());
      await user.click(screen.getByRole("button", { name: "Analyze Image" }));

      await waitFor(() => {
        // Error may appear in multiple places (sidebar panel + results area)
        const els = screen.getAllByText(match);
        expect(els.length).toBeGreaterThanOrEqual(1);
      });
    });
  }

  it("leaves loading state after error", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockRejectedValue(new Error("Something broke"));

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitFor(() => {
      expect(screen.getAllByText(/Something broke/).length).toBeGreaterThanOrEqual(1);
    });

    // Upload zone visible again — after error the file is still selected, so we see the preview
    expect(container.querySelector('.upload-preview')).toBeInTheDocument();
  });

  it("shows Try Again button in error state", async () => {
    const user = userEvent.setup();
    const { container } = render(<Dashboard />, { wrapper });
    mockedAnalyze.mockRejectedValue(new Error("Server error"));

    await uploadFile(user, container, createTestImage());
    await user.click(screen.getByRole("button", { name: "Analyze Image" }));

    await waitFor(() => {
      const buttons = screen.getAllByRole("button", { name: /try again/i });
      expect(buttons.length).toBeGreaterThanOrEqual(1);
    });
  });
});

// =========================================================================
// 9. Previous Analysis Display
// =========================================================================

describe("9. Previous Analysis Display (AnalysisDetail)", () => {
  beforeEach(() => vi.clearAllMocks());

  it("displays stored analysis data when viewing a previous analysis", async () => {
    mockedGetById.mockResolvedValue(mockDegraded);
    render(<AnalysisDetail />, {
      wrapper: historyWrapper("ana_002"),
    });

    await waitFor(() => {
      expect(
        screen.getAllByText("parking_lot_cam_117.png").length,
      ).toBeGreaterThanOrEqual(1);
    });

    // Quality score (appears in both QualityScore and AnalysisResult)
    expect(screen.getAllByText("52").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Fair").length).toBeGreaterThanOrEqual(1);

    // Readiness
    expect(screen.getAllByText("42").length).toBeGreaterThanOrEqual(1);
    expect(
      screen.getAllByText("LIMITED READINESS").length,
    ).toBeGreaterThanOrEqual(1);

    // Issues — each appears in IssueCard + IssueExplanations
    expect(screen.getAllByText(/Insufficient Sharpness/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText("Underexposure").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Low Color Information/).length).toBeGreaterThanOrEqual(2);

    // Context
    expect(screen.getByText("Smart-City Impact")).toBeInTheDocument();

    // Explanations
    expect(screen.getByText("Explainability")).toBeInTheDocument();
  });

  it("shows not-found state for missing analysis", async () => {
    mockedGetById.mockResolvedValue(null);
    render(<AnalysisDetail />, {
      wrapper: historyWrapper("nonexistent"),
    });

    await waitFor(() => {
      expect(screen.getByText("Analysis Not Found")).toBeInTheDocument();
    });
  });

  it("shows error state when API call fails", async () => {
    mockedGetById.mockRejectedValue(new Error("Server error"));
    render(<AnalysisDetail />, {
      wrapper: historyWrapper("ana_001"),
    });

    await waitFor(() => {
      expect(
        screen.getByText(/Unable to Load Analysis/),
      ).toBeInTheDocument();
    });
  });
});

// =========================================================================
// 10. No Hard-Coded Production Demo Data
// =========================================================================

describe("10. No Hard-Coded Production Demo Data", () => {
  it("AnalysisResult component does not contain hard-coded scores", () => {
    const source = AnalysisResultComponent.toString();
    expect(source).not.toMatch(/quality_score\s*[:=]\s*\d+/);
    expect(source).not.toMatch(/readiness_score\s*[:=]\s*\d+/);
  });

  it("QualityScore renders arbitrary props, not hard-coded values", () => {
    const { unmount: u1 } = render(
      <QualityScore score={42} label="Fair" analysisConfidence={75} />,
      { wrapper },
    );
    expect(document.body.textContent).toContain("42");
    expect(document.body.textContent).toContain("Fair");
    u1?.();

    const { unmount: u2 } = render(
      <QualityScore score={0} label="Critical" analysisConfidence={0} />,
      { wrapper },
    );
    expect(document.body.textContent).toContain("0");
    u2?.();

    render(
      <QualityScore
        score={100}
        label="Excellent"
        analysisConfidence={100}
      />,
      { wrapper },
    );
    expect(document.body.textContent).toContain("100");
  });

  it("AnalysisResult renders with arbitrary values, not mock data", () => {
    const result: AnalysisResult = {
      analysis_id: "test_001",
      filename: "test.jpg",
      quality_score: 73,
      quality_label: "Good",
      analysis_confidence: 88,
      issues: [],
      statistics: {
        sharpness: 55,
        brightness: 120,
        contrast: 45,
        noise_estimate: 12,
        entropy: 6.5,
        saturation: 35,
      },
      summary: "Test summary",
      created_at: "2026-08-29T00:00:00Z",
      processing_time_ms: 1500,
      analytics_readiness_score: 60,
      analytics_readiness_status: "READY",
      analytics_readiness_details: {
        base_quality_score: 73,
        blur_penalty: 5,
        exposure_penalty: 0,
        noise_penalty: 5,
        corruption_penalty: 0,
        information_penalty: 3,
      },
    };

    const { container } = render(
      <AnalysisResultComponent result={result} />,
      { wrapper },
    );
    expect(container.textContent).toContain("73");
    expect(container.textContent).toContain("Good");
    expect(container.textContent).toContain("60");
  });
});

// =========================================================================
// 11. Type Safety — API Contract
// =========================================================================

describe("11. Type Safety — API Contract", () => {
  it("AnalysisResult type has all required Phase 1-6 fields (compile-time check)", () => {
    const result: AnalysisResult = {
      analysis_id: "type_check",
      filename: "test.jpg",
      quality_score: 50,
      quality_label: "Good",
      analysis_confidence: 85,
      issues: [],
      statistics: {
        sharpness: 50,
        brightness: 120,
        contrast: 45,
        noise_estimate: 10,
        entropy: 6,
        saturation: 35,
      },
      summary: "type check",
      created_at: "2026-01-01T00:00:00Z",
      processing_time_ms: 1000,
      analytics_readiness_score: 50,
      analytics_readiness_status: "READY",
      analytics_readiness_details: {
        base_quality_score: 50,
        blur_penalty: 0,
        exposure_penalty: 0,
        noise_penalty: 0,
        corruption_penalty: 0,
        information_penalty: 0,
      },
      context: "CCTV Surveillance",
      context_impacts: [
        {
          issue_type: "test",
          context: "CCTV Surveillance",
          impact: "test impact",
        },
      ],
      issue_explanations: [
        {
          issue: "Test Issue",
          evidence: { metric: "test_metric", value: 10, threshold: 5 },
          why_it_matters: "test why",
          recommendation: "test rec",
        },
      ],
    };

    expect(result.analysis_id).toBe("type_check");
    expect(result.quality_score).toBe(50);
    expect(result.analytics_readiness_score).toBe(50);
    expect(result.analytics_readiness_details.blur_penalty).toBe(0);
    expect(result.context).toBe("CCTV Surveillance");
    expect(result.context_impacts).toHaveLength(1);
    expect(result.issue_explanations).toHaveLength(1);
    expect(result.issue_explanations![0].evidence.value).toBe(10);
  });

  it("Issue type has all required fields from backend", () => {
    const issue = mockDegraded.issues[0];
    expect(issue).toHaveProperty("type");
    expect(issue).toHaveProperty("severity");
    expect(issue).toHaveProperty("metric");
    expect(issue).toHaveProperty("value");
    expect(issue).toHaveProperty("threshold");
    expect(issue).toHaveProperty("impact");
    expect(issue).toHaveProperty("confidence");
    expect(typeof issue.value).toBe("number");
    expect(typeof issue.threshold).toBe("number");
    expect(typeof issue.confidence).toBe("number");
  });

  it("AnalyticsReadinessDetails has all 6 penalty fields", () => {
    const d = mockAcceptable.analytics_readiness_details;
    expect(d).toHaveProperty("base_quality_score");
    expect(d).toHaveProperty("blur_penalty");
    expect(d).toHaveProperty("exposure_penalty");
    expect(d).toHaveProperty("noise_penalty");
    expect(d).toHaveProperty("corruption_penalty");
    expect(d).toHaveProperty("information_penalty");
  });

  it("SUPPORTED_CONTEXTS has exactly 6 values", () => {
    expect(SUPPORTED_CONTEXTS).toHaveLength(6);
    expect(SUPPORTED_CONTEXTS).toContain("CCTV Surveillance");
    expect(SUPPORTED_CONTEXTS).toContain("Traffic Monitoring");
    expect(SUPPORTED_CONTEXTS).toContain("Crowd Monitoring");
    expect(SUPPORTED_CONTEXTS).toContain("Drone Imagery");
    expect(SUPPORTED_CONTEXTS).toContain("Infrastructure Inspection");
    expect(SUPPORTED_CONTEXTS).toContain("Smart Campus");
  });

  it("IssueExplanation has evidence with metric/value/threshold", () => {
    const exp = mockDegraded.issue_explanations![0];
    expect(exp).toHaveProperty("issue");
    expect(exp).toHaveProperty("evidence");
    expect(exp).toHaveProperty("why_it_matters");
    expect(exp).toHaveProperty("recommendation");
    expect(exp.evidence).toHaveProperty("metric");
    expect(exp.evidence).toHaveProperty("value");
    expect(exp.evidence).toHaveProperty("threshold");
  });
});

// =========================================================================
// Component Unit Tests
// =========================================================================

describe("Component Unit Tests", () => {
  beforeEach(() => vi.clearAllMocks());

  describe("EmptyState", () => {
    it("renders title and message", () => {
      render(
        <EmptyState title="No data" message="Nothing here yet." />,
        { wrapper },
      );
      expect(screen.getByText("No data")).toBeInTheDocument();
      expect(screen.getByText("Nothing here yet.")).toBeInTheDocument();
    });
  });

  describe("ErrorState", () => {
    it("renders with default title", () => {
      render(<ErrorState message="Oops" />, { wrapper });
      expect(screen.getByText("Something went wrong")).toBeInTheDocument();
      expect(screen.getByText("Oops")).toBeInTheDocument();
    });

    it("renders custom title", () => {
      render(<ErrorState title="Custom" message="msg" />, { wrapper });
      expect(screen.getByText("Custom")).toBeInTheDocument();
    });

    it("renders retry button when onRetry provided", () => {
      const onRetry = vi.fn();
      render(<ErrorState message="fail" onRetry={onRetry} />, { wrapper });
      expect(
        screen.getByRole("button", { name: /try again/i }),
      ).toBeInTheDocument();
    });
  });

  describe("QualityScore", () => {
    it("renders score, label, and confidence", () => {
      render(
        <QualityScore
          score={85}
          label="Excellent"
          analysisConfidence={92}
        />,
        { wrapper },
      );
      expect(screen.getByText("85")).toBeInTheDocument();
      expect(screen.getByText("Excellent")).toBeInTheDocument();
      expect(screen.getByText("92%")).toBeInTheDocument();
    });
  });

  describe("IssueCard", () => {
    it("renders issue type, severity, metric, value, threshold", () => {
      render(
        <IssueCard
          issue={{
            type: "severe_blur",
            severity: "high",
            metric: "laplacian_variance",
            value: 3.5,
            threshold: 15.0,
            impact: "Detection will fail",
            confidence: 0.95,
          }}
        />,
        { wrapper },
      );
      expect(screen.getByText(/Severe Blur/)).toBeInTheDocument();
      expect(screen.getByText("high")).toBeInTheDocument();
      expect(screen.getByText("3.50")).toBeInTheDocument();
      expect(screen.getByText("15.00")).toBeInTheDocument();
      expect(screen.getByText("95%")).toBeInTheDocument();
    });
  });

  describe("AnalyticsReadiness", () => {
    it("renders score, status, and penalty breakdown", () => {
      render(
        <AnalyticsReadiness
          score={42}
          status="LIMITED READINESS"
          details={{
            base_quality_score: 52,
            blur_penalty: 18,
            exposure_penalty: 8,
            noise_penalty: 0,
            corruption_penalty: 0,
            information_penalty: 5,
          }}
        />,
        { wrapper },
      );
      expect(screen.getByText("42")).toBeInTheDocument();
      expect(screen.getByText("LIMITED READINESS")).toBeInTheDocument();
      expect(screen.getByText("Penalty Breakdown")).toBeInTheDocument();
      expect(screen.getByText("Blur")).toBeInTheDocument();
      // Penalty is rendered as -{value.toFixed(1)} => "-18.0"
      expect(screen.getByText("-18.0")).toBeInTheDocument();
      expect(screen.getByText("-8.0")).toBeInTheDocument();
    });
  });

  describe("ContextImpact", () => {
    it("renders context and impacts", () => {
      render(
        <ContextImpact
          context="Traffic Monitoring"
          impacts={[
            {
              issue_type: "blur",
              context: "Traffic Monitoring",
              impact: "License plate recognition unreliable",
            },
          ]}
        />,
        { wrapper },
      );
      expect(screen.getByText("Smart-City Impact")).toBeInTheDocument();
      expect(
        screen.getByText("Traffic Monitoring"),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/License plate recognition/),
      ).toBeInTheDocument();
    });

    it("renders empty message when no impacts", () => {
      render(
        <ContextImpact context="CCTV Surveillance" impacts={[]} />,
        { wrapper },
      );
      expect(
        screen.getByText(/No context-specific impacts/),
      ).toBeInTheDocument();
    });
  });

  describe("IssueExplanations", () => {
    it("renders explanation with evidence, why, recommendation", () => {
      const { container } = render(
        <IssueExplanations
          explanations={[
            {
              issue: "Underexposure",
              evidence: {
                metric: "mean_brightness",
                value: 45.2,
                threshold: 80.0,
              },
              why_it_matters: "Dark images lose detail",
              recommendation: "Add lighting",
            },
          ]}
        />,
        { wrapper },
      );
      expect(screen.getByText("Underexposure")).toBeInTheDocument();
      // Evidence values — check they exist in the rendered text
      expect(container.textContent).toContain("45.20");
      expect(container.textContent).toContain("80.00");
      expect(container.textContent).toContain("mean_brightness");
      expect(screen.getByText("Dark images lose detail")).toBeInTheDocument();
      expect(screen.getByText("Add lighting")).toBeInTheDocument();
    });

    it("returns null for empty explanations", () => {
      const { container } = render(
        <IssueExplanations explanations={[]} />,
        { wrapper },
      );
      expect(container.firstChild).toBeNull();
    });
  });

  describe("ImageUploader", () => {
    it("renders empty state with drop zone", () => {
      render(
        <ImageUploader
          onImageSelected={vi.fn()}
          onRemove={vi.fn()}
          selectedFile={null}
          previewUrl={null}
        />,
        { wrapper },
      );
      expect(screen.getByText(/Drag and drop/)).toBeInTheDocument();
      expect(screen.getByText(/Supported formats/)).toBeInTheDocument();
    });

    it("renders preview mode when file is selected", () => {
      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      render(
        <ImageUploader
          onImageSelected={vi.fn()}
          onRemove={vi.fn()}
          selectedFile={file}
          previewUrl="blob:test"
        />,
        { wrapper },
      );
      expect(
        screen.getByRole("img", { name: "test.jpg" }),
      ).toBeInTheDocument();
      expect(screen.getByText("test.jpg")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /remove/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /replace/i }),
      ).toBeInTheDocument();
    });

    it("calls onRemove when remove button clicked", async () => {
      const user = userEvent.setup();
      const onRemove = vi.fn();
      const file = new File(["test"], "test.jpg", { type: "image/jpeg" });
      render(
        <ImageUploader
          onImageSelected={vi.fn()}
          onRemove={onRemove}
          selectedFile={file}
          previewUrl="blob:test"
        />,
        { wrapper },
      );
      await user.click(screen.getByRole("button", { name: /remove/i }));
      expect(onRemove).toHaveBeenCalledOnce();
    });
  });
});
