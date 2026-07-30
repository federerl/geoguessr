# GeoLocator Demo (Gradio Space)

This repository hosts a Gradio app for a geolocation demo that predicts latitude/longitude from a single image and visualizes results on a map.

## Structure
- `app.py` — Gradio UI with Predict and Explore tabs
- `predict.py` — Model loading, inference wrapper, baseline, metrics, and Folium map builder
- `requirements.txt` — Python dependencies
- `data/` — Small JSONs for Explore (`best.json`, `worst.json`)
- `assets/` — Thumbnails used by Explore

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open the printed Gradio URL in your browser.

## Weights from Hugging Face Hub
Set environment variables so the app can pull weights on startup:
```bash
export HF_REPO_ID="your-username/your-model-repo"
export HF_FILENAME="model.pt"   # or a TorchScript/ONNX file
```

## Deploy to Hugging Face Spaces
1) Create a Space (SDK: Gradio). 2) Upload this repo (or link via Git). 3) Ensure `requirements.txt` is present. 4) Optionally select GPU hardware if latency is high.

## Explore data format
`data/best.json` and `data/worst.json` are **not checked into this repo** (they're generated from an offline evaluation run over the test set — see `src/GeoLocSFTTest/simple_run_images_first/preds.csv` for the raw predictions this is derived from). Without them, the app's Predict tab still works normally; the Explore tab just shows empty Best/Worst galleries. To populate it, generate the two JSONs plus matching thumbnails in `assets/`, each entry shaped like:
```json
{
  "id": "abc123",
  "img_url": "https://.../full.jpg",
  "thumb": "assets/abc123_512.jpg",
  "gt_lat": 37.7749,
  "gt_lon": -122.4194,
  "pred_lat": 37.6000,
  "pred_lon": -122.3000,
  "err_km": 20.5,
  "baseline_lat": 32.93,
  "baseline_lon": -11.29,
  "baseline_err_km": 6196.0
}
```

## Notes
- The UI falls back to a simple baseline if a model is not available.
- Keep thumbnails small (≤512 px long side) to speed up Space load times.

