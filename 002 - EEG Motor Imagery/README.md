# CS002: EEG Motor Imagery Classification

## Status: Pipeline complete (auto-generated)
*Last updated: 2026-09-11 10:40:19 IST*

## Dataset
- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset
- **Subject:** 1 (single-subject pilot)
- **Runs:** 4, 8, 12 (motor imagery: left/right hand)
- **Channels:** 64 EEG, 160 Hz
- **Trials used:** 45 imagery epochs after processing

## Key results

### Best electrode configuration
**Central 3×3 Laplacian (C1,C2,C3,C4,Cz,CP1,CP2,CP3,CP4)** — CV accuracy: **0.644 ± 0.178**

### Model / config snapshot
| Item | Score |
|------|-------|
| C3/C4 mu-beta RF | 0.578 |
| Central 3×3 Laplacian (C1,C2,C3,C4,Cz,CP1,CP2,CP3,CP4) | 0.644 |
| Random Forest (full PSD) | see notebook |
| XGBoost | see notebook |
| EEGNetLite | see notebook |
| TabPFN | skipped (needs TABPFN_TOKEN) |
| Stage B enriched | 0.5555555555555556 |

### Notes
- PSD features use log10 + StandardScaler for RF/XGB/SHAP/config scoring
- Bandpass highcut capped below Nyquist (~40 Hz) for 160 Hz data
- TabPFN skipped without Prior Labs token
- Agentic layer is a schema stub (not a full agent run)
- Single-subject slice — treat accuracies as exploratory

### Stage A signal quality
- Flagged trials: 5 / 45

## Artifacts
- `eeg_motor_imagery_executed.ipynb`
- `plots/` (SHAP beeswarm, band importance, topomaps, alpha C3/C4)
- `data/processed/` (features, models, scores)

## Reproduce
```bash
cd /home/box/case-studies/_nospace/cs002_work
/home/box/case-studies/venv/bin/python cs002_pipeline.py --list
/home/box/case-studies/venv/bin/python cs002_pipeline.py --step N
```
