# geoai_rooftop_dhaka

Automated rooftop solar PV segmentation and energy potential estimation for Dhaka, Bangladesh.

## Data Provenance

| Component | Source | Role |
|-----------|--------|------|
| Segmentation training | PV01 Rooftop (Jiang et al.) | Model training (Chinese UAV imagery) |
| Building footprints | Dhaka Rooftop Potential (energydata.info) | Spatial constraint & validation |
| PV yield raster | Bangladesh PVOUT (World Bank) | Yield estimation |

**Disclaimer:**  PV01 is used as a proxy dataset; Dhaka building footprints are for post-processing, not training labels.

## Quick Start

```bash
pip install -e ".[dev]"
python -c "from pv_segmentation.utils.determinism import setup_determinism; setup_determinism(42)"
pytest tests/ -v
```


## Known limitations
Mask2Former: Uses `facebook/mask2former-swin-tiny-coco-instance` with `num_labels=2`. The load report shows some Swin params reinitialized due to transformers version drift; for production use, prefer `facebook/mask2former-swin-tiny-ade-semantic`.

SAM+LoRA: SAM's ViT-B uses a fused qkv projection rather than separate `q_proj/v_proj`. LoRA is applied to the fused projection, adapting q, k, and v jointly rather than only q and v. This deviates from the paper's described LoRA placement but is the only viable target in SAM's architecture

SAM+LoRA null prompt: The mask decoder is invoked with zero-valued sparse and dense prompts, which is SAM's "no prompt" condition. The decoder attends to image features only and produces a dense mask. If the training loss stalls at the class prior (all-background or all-foreground), a learnable null prompt parameter can be added; the current scaffold intentionally keeps prompts fixed to isolate the LoRA adaptation effect on the image encoder.


Ensure data/processed/manifest_split.csv has stem, split, row, col, unit, acq, block.

Run scripts/prepare_dataset.py → produces processed/images/, processed/masks/*.tif, manifest_geom.csv, splits/fixed_seed42.json.

Confirm train.py --smoke runs end-to-end against this layout. If it does, the dataset/datamodule side is consistent.

Create data/ancillary/ and populate climate (or wire pvout into yield_model.py), tariffs, and emission factor.

Create configs/economics.yaml and configs/monte_carlo.yaml from the placeholders.

Implement analysis/mosaic.py (needs manifest_geom.csv).

Train, predict_test.py, evaluate_ensemble.py, then run_analysis.py --steps mosaic vectorize capacity yield economics emissions priority monte_carlo


# one-time
python -m scripts.download.download_s2 --years 2016 2017 ... 2026 --out data/chips
python -m scripts.download.download_landsat --years 2010 2011 2012 2013 2014 2015 --out data/chips
python -m scripts.download.rasterize_masks \
    --footprints data/processed/footprints.gpkg --out data/chips/masks

# build splits from the manifest (prep_splits.py, extended for `block`)
python scripts/prep_splits.py --manifest data/manifest.csv --seed 42


# 1. build the index (safe to re-run after adding years)
python -m scripts.download.build_manifest

# 2. build splits (audit JSON is what the paper reports)
python scripts/prep_splits.py \
    --scales s2 --strategy fixed \
    --min-roof-ratio 0.01

# 3. patch experiment.yaml
#    scale: s2
#    year: 2026          # or null
#    use_valid_mask: true

# 4. train (smoke first)
python scripts/train.py --model unet --smoke --seed 42
python scripts/train.py --model unet --seed 42

# 5. predict + ensemble + paired eval
python scripts/predict_test.py --scale s2 --year 2026
python scripts/evaluate_ensemble.py
python scripts/evaluate_paired.py     