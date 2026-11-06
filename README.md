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