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
  timeout: 30_000,
});

/** Simulate network delay for mock mode */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// Public API — swap the bodies with real http calls when the backend is ready
// ---------------------------------------------------------------------------

/** Upload an image for analysis (mock returns the acceptable sample). */
export async function analyzeImage(_file: File): Promise<AnalysisResult> {
  if (API_BASE) {
    const form = new FormData();
    form.append("file", _file);
    const { data } = await http.post<AnalysisResult>("/api/v1/analyze", form);
    return data;
  }

  // Mock: simulate staged analysis
  await delay(2400);
  return { ...mockAcceptable };
}

/** Fetch the list of past analyses. */
export async function getAnalyses(): Promise<AnalysisResult[]> {
  if (API_BASE) {
    const { data } = await http.get<AnalysisResult[]>("/api/v1/analyses");
    return data;
  }

  await delay(600);
  return [...mockHistory];
}

/** Fetch a single analysis by id. */
export async function getAnalysisById(
  id: string,
): Promise<AnalysisResult | null> {
  if (API_BASE) {
    const { data } = await http.get<AnalysisResult>(`/api/v1/analyses/${id}`);
    return data;
  }

  await delay(400);
  return getMockAnalysisById(id) ?? null;
}
