# GeoGuessr — Image Geolocation with Deep Learning

A deep-learning project that predicts where a photo was taken (latitude/longitude and country) from the image alone, trained on a subset of the [OSV-5M](https://huggingface.co/datasets/osv5m/osv5m) street-view dataset. Built for CSSE416.

## What's in here

| Path | Description |
|---|---|
| `app/` | Gradio web demo — upload an image, get a predicted location plotted on a map, compare against a baseline |
| `src/train/` | Data cleaning/EDA (`data_analysis.ipynb`, `clean.ipynb`), a coordinate-based baseline (`baseline.ipynb`), and the main training notebook (`GeoGuessrModel.ipynb`) |
| `src/Resnet/` | ResNet18/34 regression models (PyTorch Lightning) predicting `(lat, lon)` directly, at increasing dataset sizes |
| `src/Classifier/` | A country-classification model (ResNet/ViT backbones via `timm`, focal loss + class-imbalance handling) as an alternative to direct coordinate regression |
| `src/GeoLocSFTTest/` | A smaller, quick fine-tuning experiment (frozen ResNet18 + Haversine loss) with a saved checkpoint and evaluation output |
| `src/Resnet_regressor.py` | Standalone `LightningModule` definition for the ResNet regressor, reused across notebooks |
| `src/extract_exif.py`, `src/flatten_directories.py`, `src/create_stratified_subset.py` | Data-prep utilities: pull GPS EXIF tags from images, flatten nested folders, and build stratified train/test subsets |

## Approach

Two modeling strategies were explored:
1. **Direct regression** — a CNN backbone (ResNet18/34) with a 2-output head trained to predict `(lat, lon)` directly, optimized with an MSE or Haversine-distance loss.
2. **Classification** — predicting the country/region as a class label, which sidesteps the wraparound and non-uniform-density issues of raw coordinate regression.

Both are evaluated against a simple baseline (mean/circular-mean coordinate predictor) using great-circle (Haversine) distance in kilometers as the error metric.

## Running the training/research code

```bash
pip install -r src/requirements.txt
jupyter lab
```

Notebooks expect an OSV-5M-derived CSV + image directory (see `src/train/data_analysis.ipynb` for the expected layout); paths were originally configured for the course's shared server and will need to point at your own local data.

## Running the demo app

```bash
cd app
pip install -r requirements.txt
python app.py
```

See [`app/README.md`](app/README.md) for details, including how the app loads model weights from the Hugging Face Hub.

## License

MIT — see [LICENSE](LICENSE).
