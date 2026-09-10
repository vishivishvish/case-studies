# Case-Studies Virtual Environment (CPU-Only)

## Activation
```bash
source /home/box/case-studies/venv/bin/activate
```

## What's Installed
Installed from `/home/box/case-studies/requirements-box-cpu.txt` into the venv at `/home/box/case-studies/venv/`.

### Core Data Science
- numpy, pandas, scipy, scikit-learn, matplotlib, seaborn

### Specialized ML
- xgboost, shap, statsmodels

### Image/Visualization
- pillow

### Jupyter Ecosystem
- jupyter, nbconvert, ipykernel, jupyterlab, notebook

### Deep Learning (CPU-only PyTorch)
- torch (from https://download.pytorch.org/whl/cpu)

### Audio Processing (CS12 - Spoken Word Recognition)
- librosa, soundfile, soxr

### Transformers / NLP (CS12)
- transformers, langgraph, langchain-core

### Other
- arrow, tzdata, and various jupyter/lab dependencies

## Notes
- **CPU-only / no GPU** — PyTorch installed from CPU index URL
- No heavy training frameworks (no CUDA, no training loops by default)
- Use `nova-light` for light work to avoid spawning heavy processes
- To add packages: edit `requirements-box-cpu.txt` and re-run:
  ```bash
  /home/box/case-studies/venv/bin/pip install -r /home/box/case-studies/requirements-box-cpu.txt
  ```