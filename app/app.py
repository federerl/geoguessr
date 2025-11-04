from __future__ import annotations

import json
from pathlib import Path
import io
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image
import requests

from predict import (
    predict as run_model_predict,
    baseline as compute_baseline,
    folium_map,
    format_metrics_md,
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"


def load_url_image(url: str) -> Optional[Image.Image]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def ensure_image(image: Optional[Image.Image], url: Optional[str]) -> Optional[Image.Image]:
    if image is not None:
        return image
    if url:
        return load_url_image(url)
    return None


def do_predict(
    image: Optional[Image.Image],
    url: str,
    gt_lat: Optional[float],
    gt_lon: Optional[float],
    show_baseline: bool,
) -> Tuple[str, str]:
    img = ensure_image(image, url)
    if img is None:
        return "<div style='color:#b00'>Please provide an image or URL.</div>", ""

    result = run_model_predict(img)
    pred = result["pred"]
    runtime_ms = result["runtime_ms"]

    gt = None
    if gt_lat is not None and gt_lon is not None:
        gt = {"lat": float(gt_lat), "lon": float(gt_lon)}

    base = compute_baseline(img) if show_baseline else None
    map_html = folium_map(pred=pred, gt=gt, baseline_coord=base)
    md = format_metrics_md(pred=pred, runtime_ms=runtime_ms, gt=gt, baseline_coord=base)
    return map_html, md


def load_curated(which: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / ("best.json" if which == "Best" else "worst.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())  # type: ignore
    except Exception:
        return []


def gallery_items(entries: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for e in entries:
        img_path = e.get("thumb") or e.get("img_url")
        caption = f"err_km: {e.get('err_km', '—')}"
        if img_path:
            items.append((str(img_path), caption))
    return items


def render_from_precomputed(
    entry_idx: int,
    which: str,
    best_list: List[Dict[str, Any]],
    worst_list: List[Dict[str, Any]],
) -> Tuple[str, float, float, str]:
    entries = best_list if which == "Best" else worst_list
    if not entries or entry_idx < 0 or entry_idx >= len(entries):
        return "", gr.update(), gr.update(), ""
    e = entries[entry_idx]
    pred = {"lat": float(e["pred_lat"]), "lon": float(e["pred_lon"])}
    gt = {"lat": float(e["gt_lat"]), "lon": float(e["gt_lon"])}
    map_html = folium_map(pred=pred, gt=gt, baseline_coord={"lat": e.get("baseline_lat", 0.0), "lon": e.get("baseline_lon", 0.0)} if ("baseline_lat" in e and "baseline_lon" in e) else None)
    md = format_metrics_md(pred=pred, runtime_ms=0.0, gt=gt, baseline_coord=None)
    return map_html, float(e["gt_lat"]), float(e["gt_lon"]), md


with gr.Blocks(title="GeoLocator Demo") as demo:
    gr.Markdown("# GeoLocator — Predict latitude/longitude from an image")
    with gr.Tabs():
        with gr.TabItem("Predict"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    inp_img = gr.Image(type="pil", label="Image")
                    url_box = gr.Textbox(label="Image URL")
                    load_btn = gr.Button("Load URL")
                    gt_lat = gr.Number(label="Ground Truth Latitude", value=None)
                    gt_lon = gr.Number(label="Ground Truth Longitude", value=None)
                    show_baseline = gr.Checkbox(value=True, label="Show baseline")
                    predict_btn = gr.Button("Predict")
                with gr.Column(scale=2):
                    out_map = gr.HTML()
                    out_md = gr.Markdown()

            def _load_url(u: str) -> Optional[Image.Image]:
                return load_url_image(u)

            load_btn.click(_load_url, inputs=[url_box], outputs=[inp_img])
            predict_btn.click(
                do_predict,
                inputs=[inp_img, url_box, gt_lat, gt_lon, show_baseline],
                outputs=[out_map, out_md],
            )

        with gr.TabItem("Explore"):
            best_state = gr.State(load_curated("Best"))
            worst_state = gr.State(load_curated("Worst"))
            which = gr.Radio(["Best", "Worst"], value="Best", label="Which examples")
            gal = gr.Gallery(label="Curated examples", columns=5, height=320)
            with gr.Row():
                sel_idx = gr.Number(label="Selected index", value=-1, interactive=True)
                precomp_map = gr.HTML()
                precomp_md = gr.Markdown()

            def _refresh_gallery(w: str, b: List[Dict[str, Any]], wst: List[Dict[str, Any]]):
                entries = b if w == "Best" else wst
                return gallery_items(entries)

            which.change(_refresh_gallery, inputs=[which, best_state, worst_state], outputs=[gal])

            def _on_select(data: List[Any], w: str):
                # Gradio 4 returns selected data; we keep a simple index via sel_idx for clarity
                return len(data) - 1 if data else -1

            gal.select(_on_select, inputs=[gal, which], outputs=[sel_idx])

            def _render_sel(idx: float, w: str, b: List[Dict[str, Any]], wst: List[Dict[str, Any]]):
                return render_from_precomputed(int(idx), w, b, wst)

            # Update Predict's GT fields and show a precomputed map/metrics here
            sel_idx.change(
                _render_sel,
                inputs=[sel_idx, which, best_state, worst_state],
                outputs=[precomp_map, gt_lat, gt_lon, precomp_md],
            )


if __name__ == "__main__":
    demo.launch()


