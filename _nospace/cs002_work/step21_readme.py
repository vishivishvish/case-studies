#!/usr/bin/env python3
"""CS002 step 21: write README.md into the real 002 directory."""
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

REAL = "/home/box/case-studies/002 - EEG Motor Imagery"
WORK = "/home/box/case-studies/_nospace/cs002_work"
PROC = os.path.join(WORK, "data/processed")


def load(name, default=None):
    path = os.path.join(PROC, name)
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path) as f:
        return json.load(f)


def main():
    configs = load("config_scores.json", {})
    stage_a = load("stage_a_result.json", {})
    stage_b = load("stage_b_result.json", {})
    tabpfn = load("tabpfn_result.json", {})
    rf = load("rf_result.json", {})
    xgb = load("xgb_result.json", {})
    eegnet = load("eegnet_result.json", {})

    default_score = {"mean": 0.0, "std": 0.0}
    if configs:
        best_name, best = max(configs.items(), key=lambda x: x[1].get("mean", 0))
    else:
        best_name, best = "N/A", default_score

    c3c4_key = "Subject-specific C3/C4 sensorimotor mu-beta"
    c3c4 = configs.get(c3c4_key, default_score)

    lap_key = None
    for k in configs:
        if "Laplacian" in k or "laplacian" in k:
            lap_key = k
            break
    lap = configs.get(lap_key, default_score) if lap_key else default_score

    if tabpfn.get("status") == "success":
        tabpfn_cell = str(tabpfn.get("accuracy", "N/A"))
    else:
        tabpfn_cell = "skipped (needs TABPFN_TOKEN)"

    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")

    rf_acc = rf.get("mean_cv") or rf.get("accuracy") or rf.get("mean")
    xgb_acc = xgb.get("mean_cv") or xgb.get("accuracy") or xgb.get("mean")
    eeg_acc = eegnet.get("accuracy") or eegnet.get("mean")

    lines = [
        "# CS002: EEG Motor Imagery Classification",
        "",
        "## Status: Pipeline complete (auto-generated)",
        f"*Last updated: {now}*",
        "",
        "## Dataset",
        "- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset",
        "- **Subject:** 1 (single-subject pilot)",
        "- **Runs:** 4, 8, 12 (motor imagery: left/right hand)",
        "- **Channels:** 64 EEG, 160 Hz",
        "- **Trials used:** 45 imagery epochs after processing",
        "",
        "## Key results",
        "",
        "### Best electrode configuration",
        f"**{best_name}** — CV accuracy: **{best.get('mean', 0):.3f} ± {best.get('std', 0):.3f}**",
        "",
        "### Model / config snapshot",
        "| Item | Score |",
        "|------|-------|",
        f"| C3/C4 mu-beta RF | {c3c4.get('mean', 0):.3f} |",
        f"| {lap_key or 'Laplacian config'} | {lap.get('mean', 0):.3f} |",
        f"| Random Forest (full PSD) | {rf_acc if rf_acc is not None else 'see notebook'} |",
        f"| XGBoost | {xgb_acc if xgb_acc is not None else 'see notebook'} |",
        f"| EEGNetLite | {eeg_acc if eeg_acc is not None else 'see notebook'} |",
        f"| TabPFN | {tabpfn_cell} |",
        f"| Stage B enriched | {stage_b.get('accuracy', 'N/A')} |",
        "",
        "### Notes",
        "- PSD features use log10 + StandardScaler for RF/XGB/SHAP/config scoring",
        "- Bandpass highcut capped below Nyquist (~40 Hz) for 160 Hz data",
        "- TabPFN skipped without Prior Labs token",
        "- Agentic layer is a schema stub (not a full agent run)",
        "- Single-subject slice — treat accuracies as exploratory",
        "",
        "### Stage A signal quality",
        f"- Flagged trials: {stage_a.get('flagged_count', '?')} / {stage_a.get('total', '?')}",
        "",
        "## Artifacts",
        "- `eeg_motor_imagery_executed.ipynb`",
        "- `plots/` (SHAP beeswarm, band importance, topomaps, alpha C3/C4)",
        "- `data/processed/` (features, models, scores)",
        "",
        "## Reproduce",
        "```bash",
        "cd /home/box/case-studies/_nospace/cs002_work",
        "/home/box/case-studies/venv/bin/python cs002_pipeline.py --list",
        "/home/box/case-studies/venv/bin/python cs002_pipeline.py --step N",
        "```",
        "",
    ]
    os.makedirs(REAL, exist_ok=True)
    out = os.path.join(REAL, "README.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
