import axios from "axios";
import type { AnalysisResult } from "../types/analysis";
import {
  mockAcceptable,
  mockHistory,
  getMockAnalysisById,
} from "../mock/mockAnalysis";

/** Base URL configurable via environment — swapped to real backend later. */
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const http = axios.create({
  baseURL: API_BASE,
  timeout: 60_000,
});

/** Simulate network delay for mock mode */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

/** Parse an axios/API error into a user-friendly message. */
function parseApiError(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = err.response?.data?.detail;

    if (!err.response) {
      // Network failure
      return "Unable to reach the server. Please check your network connection and try again.";
    }

    if (status === 413) {
      return detail ?? "The uploaded file exceeds the maximum allowed size.";
    }
    if (status === 400) {
      return detail ?? "The request was invalid. Please check the uploaded file.";
    }
    if (status === 422) {
      return detail ?? "The server could not process the request.";
    }
    if (status === 500) {
      return detail ?? "An internal server error occurred. Please try again later.";
    }
    if (status === 503) {
      return detail ?? "The service is temporarily unavailable. Please try again later.";
    }

    return detail ?? `Server error (${status}). Please try again.`;
  }

  if (err instanceof Error) {
    return err.message;
  }
  return "An unexpected error occurred.";
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Supported smart-city pipeline contexts. */
export const SUPPORTED_CONTEXTS = [
  "CCTV Surveillance",
  "Traffic Monitoring",
  "Crowd Monitoring",
  "Drone Imagery",
  "Infrastructure Inspection",
  "Smart Campus",
] as const;

export type PipelineContext = (typeof SUPPORTED_CONTEXTS)[number];

/** Upload an image for analysis. */
export async function analyzeImage(
  file: File,
  context: PipelineContext = "CCTV Surveillance",
): Promise<AnalysisResult> {
  if (API_BASE) {
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await http.post<AnalysisResult>(
        `/api/v1/analyze?context=${encodeURIComponent(context)}`,
        form,
      );
      return data;
    } catch (err) {
      throw new Error(parseApiError(err), { cause: err });
    }
  }

  // Mock mode
  await delay(2400);
  return { ...mockAcceptable, context };
}

/** Fetch the list of past analyses. */
export async function getAnalyses(): Promise<AnalysisResult[]> {
  if (API_BASE) {
    try {
      const { data } = await http.get<AnalysisResult[]>("/api/v1/analyses");
      return data;
    } catch (err) {
      throw new Error(parseApiError(err), { cause: err });
    }
  }

  await delay(600);
  return [...mockHistory];
}

/** Fetch a single analysis by id. */
export async function getAnalysisById(
  id: string,
): Promise<AnalysisResult | null> {
  if (API_BASE) {
    try {
      const { data } = await http.get<AnalysisResult>(`/api/v1/analyses/${id}`);
      return data;
    } catch (err) {
      if (axios.isAxiosError(err) && err.response?.status === 404) {
        return null;
      }
      throw new Error(parseApiError(err), { cause: err });
    }
  }

  await delay(400);
  return getMockAnalysisById(id) ?? null;
}
