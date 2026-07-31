# GeoLocator Demo (Gradio Space) — Implementation Plan

## 1) Goal & Scope
Build a simple, polished web demo that predicts latitude/longitude from a single image and visualizes results on a map. The demo will run as a **Hugging Face Space (Gradio)** and showcase:
- **Single image input** (upload or paste URL)
- **Top‑k predictions** with scores
- **Interactive map** showing prediction pins and (optionally) ground truth with great‑circle error
- **Baseline comparison** (our model vs. baseline)
- **Explore Tab** with **Best** or **Worst** examples from the test set (precomputed offline)

**Out of scope for MVP:** multi-page site, user accounts, complex analytics, filtering by country/urban/edge cases.

---

## 2) Feature Breakdown

### A. Predict Tab (MVP)
**Inputs**
- Image: drag‑and‑drop upload or URL text input
- Controls: Top‑k (1–5), “Show baseline” toggle
- Optional: Ground truth lat/lon fields (for error computation)

**Outputs**
- Metrics card: predicted lat/lon, runtime (ms), (optional) error to GT (km)
- Map: pins for top‑k predictions, ground truth pin (if provided), line showing error for top‑1
- Table: top‑k list (lat, lon, score)
- Baseline: baseline top‑1 coordinate and error (if GT supplied)

**Behavior**
- Predict runs on button click
- Input image is downscaled/normalized to model spec before inference
- Model is preloaded and kept warm within the Space session

### B. Explore Tab (Simple)
- Two subviews only: **Best** and **Worst**
- Each shows a small, curated gallery (10 items) from the test set with:
  - Thumbnail, short caption (error in km)
- Clicking an item:
  - Loads the image into Predict Tab
  - Prefills GT lat/lon
  - Displays **precomputed** prediction + error instantly (no re‑inference)

---

## 3) Data & Evaluation Artifacts

### Offline Evaluation (run locally or on a notebook)
- Use the existing training codebase to run evaluation on the test set
- Generate a compact table with, at minimum: `id`, `img_path_or_url`, `gt_lat`, `gt_lon`, `pred_lat`, `pred_lon`, `err_km`, `baseline_lat`, `baseline_lon`, `baseline_err_km`
- Sort by `err_km` and export two lightweight JSONs for the Space:
  - `best.json` (lowest error, N≈50)
  - `worst.json` (highest error, N≈50)
- Prepare **thumbnails** for gallery usage (≤512 px long side) to keep the Space lightweight

### Hosting the Artifacts
- **Small JSONs** (`best.json`, `worst.json`) and **thumbnails** live in the Space repo under `/data` and `/assets` respectively
- If needed, host the full evaluation table and full‑res images in an **HF Datasets** repo; the Space will only load the small JSONs by default

---

## 4) UX & Layout

**Overall**: One page with two tabs. Clean, minimal, responsive.

- **Header**: Title + one‑line description
- **Tab 1 — Predict**
  - Left column: image input, controls, ground truth fields, Predict button
  - Right column: map (top), metrics card + top‑k table (bottom), baseline summary
- **Tab 2 — Explore**
  - Toggle buttons: **Best** | **Worst**
  - Gallery grid: 5 columns desktop / 2–3 columns mobile
  - Click → switches to Predict tab with that example loaded and precomputed results shown

**Visual style**: light theme with subtle earth tones; clear markers/colors for model vs. baseline vs. ground truth.

---

## 5) Model Integration

**Assumption**: The model is already defined and trained in a separate module that, when imported, provides a callable prediction function.

**Prediction contract (internal to the Space)**
- Input: PIL image (or NumPy array) and `top_k`
- Output: `{ pred: {lat, lon}, topk: [{lat, lon, score}, ...], runtime_ms }`
- Baseline: a simple function returning a baseline `{lat, lon}` (and optional `topk`) for comparison
- Error: haversine great‑circle distance in km, computed when GT is available

**Performance**
- Load model at import time; ensure `eval()` / inference mode
- Downscale images to the trained input size (≤1024 px long side) to limit latency
- Consider ONNX export if you need CPU performance; otherwise enable GPU hardware in the Space

---

## 6) Repository Structure (Space)
```
/ (root)
  /assets/                 # gallery thumbnails (Best/Worst)
  /data/                   # best.json, worst.json
  README.md                # brief how-to and credits
  requirements.txt         # gradio, torch/onnx, pillow, numpy, folium, etc.
  app.py                   # Gradio UI & app wiring (no training code)
  predict.py               # Model loading and predict/baseline helpers (import prebuilt model)
```

**Notes**
- Weights: either pulled from an HF Model repo (preferred) or stored in the Space via LFS (slower clone)
- Secrets: likely none; if needed, use Space Secrets for tokens

---

## 7) Hosting on Hugging Face Spaces (Gradio)

**Steps**
1. **Create the Space**: Hugging Face → Spaces → Create → SDK = Gradio → Visibility = Public (or Private)
2. **Select Hardware**: start with **CPU**; switch to **GPU (e.g., T4)** if latency is high
3. **Push Code**: commit `requirements.txt`, `app.py`, `predict.py`, `README.md`, `/assets`, `/data`
4. **Model Weights**: load from HF Model repo in `predict.py` (pin a commit for reproducibility)
5. **Build & Launch**: Space auto-builds. Verify the Predict tab flow end‑to‑end
6. **Explore Tab**: validate that Best/Worst galleries load quickly; clicking items populates Predict tab
7. **Tuning**: if cold starts/latency are noticeable, upgrade hardware or reduce image size; enable queuing
8. **Share**: distribute the Space URL

**Operational Considerations**
- **Cold start**: first request after idle can be slower; keep the Space warm during demos by running a quick ping
- **Queue**: leave enabled to manage concurrent access gracefully
- **Runtime limits**: keep per‑request inference < ~60–90s
- **Repo size**: keep thumbnails small to speed up builds and page load

---

## 8) Testing & QA Checklist
- Predict tab: upload from file and from URL; verify top‑k table and map markers
- Ground truth fields: confirm error calculation and polyline rendering
- Baseline toggle: ensure baseline coordinate and error display correctly
- Explore tab: Best and Worst lists appear quickly; click → Predict tab prefilled; numbers match precomputed JSON
- Edge cases: no image uploaded, invalid URL, missing GT, top‑k = 1 and = 5
- Performance: measure median runtime on CPU vs. GPU Space; adjust input size accordingly

---

## 9) Deliverables
- Hugging Face Space URL (public or private)
- Source repository with:
  - `app.py`, `predict.py`, `requirements.txt`, `README.md`
  - `/data/best.json`, `/data/worst.json`
  - `/assets/` thumbnails
- Short demo script and screenshots for reports/presentations