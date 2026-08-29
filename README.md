# VisionGuard

## AI-Powered Image Quality & Defect Detection

VisionGuard is a full-stack computer-vision and machine-learning system for evaluating image quality, detecting visual degradation, and estimating suitability for downstream computer-vision analytics. All analysis runs locally — no external AI or vision APIs are used.

**Stack:** Python · FastAPI · OpenCV · scikit-learn · React · TypeScript · Docker

---

## Status

| Component | Status |
|-----------|--------|
| Phases 1–9 | Completed and tested |
| Phase 10 (deployment) | Pending |
| Frontend URL | TBD |
| Backend URL | TBD |

---

## Overview

- **Image upload** with validation (extension, size, integrity)
- **OpenCV feature extraction** — 12+ engineered image-quality measurements
- **ML quality prediction** — trained gradient-boosted model with calibration
- **Issue detection** — rule-based identification of blur, exposure, noise, degradation, and defects
- **Analytics readiness** — evaluates suitability for downstream CV analytics using quality score minus issue-specific penalties
- **Explainability** — issue-driven explanations with evidence, impact, and recommendations
- **Smart-city contexts** — domain-specific impact descriptions for six application areas
- **Analysis history** — persistent storage and retrieval of past analyses

---

## How It Works

```
Image → Validation → OpenCV Feature Extraction → ML Inference → Calibration
→ Quality Score → Issue Detection → Analytics Readiness → Explainability
→ API → React Dashboard
```

---

## AI / ML Approach

VisionGuard uses a **hybrid approach** combining engineered computer-vision features with a trained quality model:

| Component | Technology | Role |
|-----------|-----------|------|
| Feature extraction | OpenCV | 12+ interpretable image measurements |
| Quality model | `HistGradientBoostingRegressor` (scikit-learn) | Predicts quality from 5 engineered features |
| Calibrator | `IsotonicRegression` (scikit-learn) | Maps raw prediction to 0–100 quality scale |
| Issue detector | Rule-based | Identifies specific degradation conditions |
| Readiness scorer | Penalty-based | Evaluates downstream analytics suitability |
| Explainability | Template-driven | Generates issue-specific explanations |

This is a classical machine-learning system using engineered image features, not a deep-learning model. The `HistGradientBoostingRegressor` was selected for fast inference and compatibility with the feature-based pipeline. The `IsotonicRegression` calibrator ensures monotonic mapping from raw model output to the application's quality scale.

---

## Quality Features & Issue Detection

### Extracted Features

| Feature | Purpose |
|---------|---------|
| Brightness | Mean pixel intensity — detects under/overexposure |
| Contrast | Intensity standard deviation — detects low-contrast scenes |
| Sharpness | Laplacian variance — detects blur and focus issues |
| Saturation | HSV colour intensity — detects desaturation |
| Edge Density | Canny edge pixel percentage — measures structural complexity |
| Noise Estimate | Robust noise via MAD of Laplacian — detects sensor noise |
| Entropy | Information content via histogram — detects information loss |
| Underexposure % | Dark pixel ratio (intensity < 30) — quantifies shadow clipping |
| Overexposure % | Bright pixel ratio (intensity > 225) — quantifies highlight clipping |
| Dynamic Range | p95 − p5 intensity difference — measures tonal range |
| Colorfulness | RGB channel statistics — measures colour diversity |
| Texture Complexity | FFT high-frequency energy ratio — measures surface texture |

### Detected Issues

| Issue | Evidence Metric | Detection Method |
|-------|----------------|-----------------|
| Insufficient sharpness | `laplacian_variance` | Sharpness < 40 (high), < 15 (critical) |
| Underexposure | `dark_pixel_ratio` | Underexposure % ≥ 15 or brightness < 70 |
| Overexposure | `bright_pixel_ratio` | Overexposure % ≥ 10 or brightness > 185 |
| Image noise | `noise_estimate` | Noise ≥ 15 (high), ≥ 35 (critical) |
| Severe visual degradation | `degradation_signal_count` | ≥ 3 feature signals below thresholds |
| Potential visual defect | `visual_anomaly_signal_count` | ≥ 2 anomalous feature combinations |
| Low contrast | `contrast` | Contrast < 20 (when dynamic range ≥ 40) |
| Limited dynamic range | `dynamic_range` | Dynamic range < 35 |
| Low colour information | `saturation` | Saturation < 15 (when brightness > 80) |

Each detected issue contains: **type**, **severity** (low/moderate/high/critical), **confidence**, **metric**, **observed value**, **threshold**, and **impact**.

---

## Explainability

VisionGuard provides two levels of explainability:

**1. Feature-level explanations** ("WHY THIS DECISION?") — per-feature textual assessments based on calibrated thresholds.

**2. Issue-driven explanations** — for every detected issue:

```
Detected: Underexposure
Evidence: dark_pixel_ratio = 24.18 (threshold: 25.0)
Why it matters: Underexposure reduces contrast in shadow areas,
  lowering detection confidence for people and vehicles.
Recommendation: Increase camera exposure or add supplementary lighting.
```

All explanation text is generated from the actual issue type, severity, and metric values — nothing is hard-coded to specific images.

---

## Smart-City Contexts

Six pipeline contexts provide domain-specific impact descriptions for detected issues:

| Context | Application |
|---------|------------|
| CCTV Surveillance | Security cameras, person re-identification |
| Traffic Monitoring | Vehicle detection, license-plate recognition |
| Crowd Monitoring | Person detection, crowd-density estimation |
| Drone Imagery | Aerial survey, land-use classification |
| Infrastructure Inspection | Structural defect detection |
| Smart Campus | Occupancy detection, parking analytics |

Each context maps every issue type to a domain-specific impact message (e.g., blur + Traffic Monitoring → "Vehicle detection and license-plate recognition may be unreliable").

---

## Evaluation

### Benchmark (Phase 5)

600 images from 5 established computer-vision datasets, processed across 5 smart-city contexts:

| Context | Images | Avg Quality | Avg Readiness | Dataset |
|---------|--------|-------------|---------------|---------|
| CCTV Surveillance | 100 | 48.73 | 45.97 | MOT17 |
| Drone Imagery | 200 | 38.00 | 23.29 | VisDrone |
| Infrastructure Inspection | 100 | 48.34 | 28.53 | RoadDefects-ISeg |
| Low-Light Evaluation | 100 | 25.61 | 0.59 | LOL |
| Traffic Monitoring | 100 | 55.61 | 42.28 | BDD100K |

These are aggregate quality/readiness outputs across each dataset, not classification accuracy metrics. The low-light context scored lowest (avg 25.61), consistent with the LOL dataset's characteristics. Traffic monitoring scored highest (55.61), reflecting generally well-exposed driving scenes.

### Testing

| Suite | Tests | Status |
|-------|-------|--------|
| Backend (`pytest`) | 237 | All passing |
| Frontend (`vitest`) | 57 | All passing |
| **Total** | **294** | **All passing** |

### Scoring Integrity (36 dedicated tests)

Verified that:
1. Quality score is generated by the ML pipeline (different inputs → different raw predictions → different calibrated scores)
2. Analytics readiness is independently calculated (penalties applied per issue type/severity)
3. Issue confidence is generated by the detector (varies by issue type: 0.8–1.0)
4. No hard-coded production scores in backend or frontend
5. Different image conditions produce different results
6. Frontend displays exact values returned by the API

---

## Screenshots

### Dashboard — Upload & Context Selection
![Dashboard](docs/screenshots/Screenshot%202026-08-29%20053114.png)
*Image uploaded with pipeline context selector ready.*

### Analysis Results — Quality Score & Detected Issues
![Analysis results](docs/screenshots/Screenshot%202026-08-29%20053130.png)
*Quality score, detected issues with severity and evidence.*

### Analytics Readiness & Quality Metrics
![Readiness and metrics](docs/screenshots/Screenshot%202026-08-29%20053152.png)
*Readiness score with penalty breakdown and feature metrics.*

### Explainability & Recommendations
![Explainability](docs/screenshots/Screenshot%202026-08-29%20053210.png)
*Issue-driven explainability with evidence, impact, and recommendations.*

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/analyze` | Upload image for quality analysis |
| `GET` | `/health` | Backend health and model status |
| `GET` | `/api/v1/model-info` | Model and pipeline information |
| `GET` | `/api/v1/analyses` | List analysis history |
| `GET` | `/api/v1/analyses/{id}` | Retrieve a specific analysis |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@image.jpg" \
  -G --data-urlencode "context=CCTV Surveillance"
```

Returns structured JSON containing quality score, label, confidence, detected issues, statistics, analytics readiness, context impacts, explanations, recommendation, and metadata.

Interactive API documentation is available at `/docs` when running locally (FastAPI/OpenAPI).

---

## Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
uvicorn apps.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

### Docker

```bash
docker compose up --build
```

Frontend at http://localhost, backend at http://localhost:8000, API docs at http://localhost:8000/docs.

---

## Docker / Deployment

```
Browser → :80 (nginx) → React SPA
                      → /api/* → :8000 (FastAPI)
                                → local ML pipeline
                                → SQLite (persistent volume)
```

**Verified:** Backend container starts and becomes healthy. Frontend container starts after backend health check passes. ML model loads successfully. Frontend proxies API requests to backend through Docker networking. CORS configured for local development.

---

## Project Structure

```
visionguard/
├── backend/
│   ├── apps/ml/          # ML pipeline (inference, features, issues, readiness, explainability)
│   ├── apps/api/         # FastAPI routes
│   ├── apps/schemas/     # Pydantic models
│   ├── models/           # Trained model artifacts
│   └── tests/            # 237 tests
├── frontend/
│   ├── src/components/   # React analysis components
│   ├── src/pages/        # Dashboard, History, Detail
│   └── __tests__/        # 57 tests
├── benchmark/            # Phase 5 evaluator and results
├── datasets/             # Benchmark images
├── docs/screenshots/     # Application screenshots
├── ml/                   # Training scripts
└── docker-compose.yml
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///./data/visionguard_v2.db` | Database connection |
| `MODEL_PATH` | `./models/model.joblib` | Trained model path |
| `MODEL_VERSION` | `visionguard-iqa-v2.0` | Model version |
| `MAX_UPLOAD_SIZE_MB` | `20` | Max upload size |
| `ALLOWED_ORIGINS` | `http://localhost,...` | CORS origins |

`.env` is not committed. Secrets are not committed. Model artifacts and datasets are excluded via `.gitignore`.

---

## Assessment Alignment

| Area | Weight | Evidence |
|------|--------|----------|
| CV understanding & feature reasoning | 15% | 12+ OpenCV-engineered features; two extraction modes; feature-appropriate issue detection |
| AI / ML implementation | 25% | `HistGradientBoostingRegressor` + `IsotonicRegression` calibration; hybrid architecture; confidence estimation from tree disagreement |
| Model evaluation & rigor | 15% | 600-image benchmark across 5 datasets; 36 scoring-integrity tests; score-range validation |
| Backend / API | 15% | FastAPI + Pydantic; 5 endpoints; structured responses; validation; persistence; healthcheck |
| Frontend | 10% | React + TypeScript dashboard; upload, analysis, history, explainability; renders API values |
| Deployment | 10% | Docker Compose; backend + frontend Dockerfiles; nginx proxy; healthcheck |
| Code quality & docs | 10% | 294 automated tests; typed Python + TypeScript; phase-based development |

---

## Limitations

- **Quality score ≠ task-specific accuracy.** A score of 80 does not guarantee success for a specific downstream task (e.g., license-plate recognition).
- **Threshold-based detection.** Issue detection uses fixed thresholds tuned for general smart-city imagery; specialised domains may require tuning.
- **Training distribution.** The model was trained on a specific dataset distribution. Images significantly outside that distribution may produce less reliable scores.
- **Single-image analysis.** VisionGuard analyses individual images, not video sequences or temporal patterns.
- **Calibration scope.** The `IsotonicRegression` calibrator was fitted on validation data from the training distribution.
- **Benchmark outputs.** The benchmark results are aggregate quality/readiness averages, not classification accuracy.

---

## Current Status

**Phases 1–9:** Completed and tested

**Phase 10:** Deployment pending

---

*AI-Powered Image Quality & Defect Detection — Internship Technical Assessment*
