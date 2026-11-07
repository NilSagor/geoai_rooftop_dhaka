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