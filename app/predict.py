from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from huggingface_hub import hf_hub_download
import folium


# --------- Configuration ---------
MAX_LONG_SIDE_PX: int = 1024
BASELINE_LAT: float = 32.938720
BASELINE_LON: float = -11.295957


@dataclass
class PredictionResult:
    pred_lat: float
    pred_lon: float
    runtime_ms: float


class ModelWrapper:
    """Light wrapper around the trained regression model.

    This implementation expects either:
    - A Torch model you reconstruct and load_state_dict into, or
    - A TorchScript model file loaded via torch.jit.load
    If none is available, prediction gracefully falls back to baseline.
    """

    def __init__(self, model_obj: Optional[object] = None, device: Optional[str] = None) -> None:
        self.model = model_obj
        if device is None:
            if torch is not None and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        if self.model is not None and torch is not None:
            try:
                self.model.eval()
                if hasattr(self.model, "to"):
                    self.model.to(self.device)
            except Exception:
                pass

    def preprocess(self, image: Image.Image) -> Any:
        image = image.convert("RGB")
        w, h = image.size
        long_side = max(w, h)
        if long_side > MAX_LONG_SIDE_PX:
            scale = MAX_LONG_SIDE_PX / float(long_side)
            image = image.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)
        if torch is not None and self.model is not None:
            arr = np.asarray(image).astype(np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            x = torch.from_numpy(arr).unsqueeze(0)
            return x.to(self.device)
        return image

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def __call__(self, image: Image.Image) -> Tuple[float, float]:
        if self.model is None or torch is None:
            return BASELINE_LAT, BASELINE_LON
        x = self.preprocess(image)
        try:
            y = self.model(x)
            if isinstance(y, (list, tuple)):
                y = y[0]
            if hasattr(y, "detach"):
                y = y.detach().cpu().numpy()
            if isinstance(y, np.ndarray):
                y = y.squeeze()
                lat = float(np.clip(y[0], -90.0, 90.0))
                lon = float(np.clip(y[1], -180.0, 180.0))
                return lat, lon
        except Exception:
            pass
        return BASELINE_LAT, BASELINE_LON


GLOBAL_MODEL: Optional[ModelWrapper] = None


def load_model(hf_repo_id: str, filename: str, device: str = "auto") -> None:
    global GLOBAL_MODEL
    model_obj = None
    if torch is not None:
        try:
            local_path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
            # Try TorchScript first; otherwise, users can adapt to their architecture here.
            try:
                model_obj = torch.jit.load(local_path, map_location="cpu")
            except Exception:
                model_obj = None
        except Exception:
            model_obj = None
    GLOBAL_MODEL = ModelWrapper(model_obj=model_obj, device=None if device == "auto" else device)


def ensure_model_loaded_from_env() -> None:
    if getattr(ensure_model_loaded_from_env, "_loaded", False):
        return
    repo_id = os.environ.get("HF_REPO_ID")
    filename = os.environ.get("HF_FILENAME")
    if repo_id and filename:
        load_model(repo_id, filename)
    else:
        # Fallback to baseline-only behavior
        GLOBAL_MODEL = ModelWrapper(model_obj=None)
    ensure_model_loaded_from_env._loaded = True  # type: ignore


def predict(image: Image.Image) -> Dict[str, Any]:
    ensure_model_loaded_from_env()
    t0 = time.time()
    lat, lon = GLOBAL_MODEL(image) if GLOBAL_MODEL is not None else (BASELINE_LAT, BASELINE_LON)
    runtime_ms = (time.time() - t0) * 1000.0
    return {"pred": {"lat": float(lat), "lon": float(lon)}, "runtime_ms": float(runtime_ms)}


def baseline(_image: Image.Image) -> Dict[str, float]:
    return {"lat": BASELINE_LAT, "lon": BASELINE_LON}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    lat1r, lon1r, lat2r, lon2r = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(r * c)


def _add_marker(m: folium.Map, lat: float, lon: float, color: str, label: str) -> None:
    folium.CircleMarker(
        location=(lat, lon), radius=6, color=color, fill=True, fill_opacity=0.9,
        tooltip=label
    ).add_to(m)


def folium_map(
    pred: Optional[Dict[str, float]],
    gt: Optional[Dict[str, float]] = None,
    baseline_coord: Optional[Dict[str, float]] = None,
    top1_polyline: bool = True,
) -> str:
    center_lat, center_lon = 0.0, 0.0
    if pred is not None:
        center_lat, center_lon = pred["lat"], pred["lon"]
    elif gt is not None:
        center_lat, center_lon = gt["lat"], gt["lon"]

    m = folium.Map(location=(center_lat, center_lon), zoom_start=3, tiles="OpenStreetMap")

    if pred is not None:
        _add_marker(m, pred["lat"], pred["lon"], color="#1f77b4", label="Prediction")
    if gt is not None:
        _add_marker(m, gt["lat"], gt["lon"], color="#2ca02c", label="Ground Truth")
    if baseline_coord is not None:
        _add_marker(m, baseline_coord["lat"], baseline_coord["lon"], color="#7f7f7f", label="Baseline")

    if top1_polyline and (pred is not None) and (gt is not None):
        folium.PolyLine(
            locations=[(pred["lat"], pred["lon"]), (gt["lat"], gt["lon"])],
            color="#1f77b4", weight=2, opacity=0.8
        ).add_to(m)

    return m._repr_html_()


def format_metrics_md(
    pred: Dict[str, float],
    runtime_ms: float,
    gt: Optional[Dict[str, float]] = None,
    baseline_coord: Optional[Dict[str, float]] = None,
) -> str:
    parts = [
        f"**Predicted**: lat {pred['lat']:.5f}, lon {pred['lon']:.5f}",
        f"**Runtime**: {runtime_ms:.1f} ms",
    ]
    if gt is not None:
        err = haversine_km(pred["lat"], pred["lon"], gt["lat"], gt["lon"])
        parts.append(f"**Error to GT**: {err:.2f} km")
    if baseline_coord is not None and gt is not None:
        berr = haversine_km(baseline_coord["lat"], baseline_coord["lon"], gt["lat"], gt["lon"])
        parts.append(f"**Baseline error**: {berr:.2f} km")
    return "\n\n".join(parts)


