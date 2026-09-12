#!/usr/bin/env python3
"""Expand CS013 notebook stub with House Stack / pipeline outline cells."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

WORK = Path("/home/box/case-studies/_nospace/cs013_work")
REAL = Path("/home/box/case-studies/013 - Self-Driving Car Sim")
nb_path = REAL / "self_driving_car_sim.ipynb"
proc = WORK / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)

sections = [
    ("# CS013: Self-Driving Car Sim\n\n"
     "**Pitch origin:** CS032 (polymath-dream).  \n"
     "**Env:** Gymnasium `CarRacing-v3` (Box2D), CPU-first.  \n"
     "**Work tree:** `_nospace/cs013_work/` gated pipeline.\n\n"
     f"*Stub expanded: {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M IST')}*\n"),
    ("## 0. Setup\n\nImports, headless SDL, paths, seeds."),
    ("## 1. Environment smoke\n\nCarRacing-v3 reset/step; sample frames (pipeline step 2)."),
    ("## 2. L1 — PID / pure-pursuit expert\n\nHand-tuned baseline; collect expert rollouts."),
    ("## 3. L1 — Behavioral cloning\n\nImitate expert; compare to PID on metrics ladder."),
    ("## 4. L1 unsupervised — maneuver clusters\n\nStraight / turn / recovery segments."),
    ("## 5. L3 — Light DL policy\n\nSmall MLP/CNN on state or pixels (CPU, time-boxed)."),
    ("## 6. L3/L2 — RL (optional time-box) + XAI\n\nPPO/SAC short run; SHAP/saliency on BC or linear probe."),
    ("## 7. L4 — Foundation model\n\n**Honest skip** on CPU (Decision Transformer / driving FM later)."),
    ("## 8. L5/L6 — GenAI tracks & features\n\nProcedural/synthetic curvature profiles; conditioning ablation."),
    ("## 9. L7 — Agentic stub + hybrid note\n\nCurriculum/safety schema; tiny reward search."),
    ("## 10. Metrics ladder & conclusion\n\nLane deviation, speed, jerk, interventions, laps — PID → BC → RL."),
]

cells = []
for i, md in enumerate(sections):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in md.split("\n")]})
    # add a small code placeholder after Setup and smoke
    if i == 1:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import os\n",
                "os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')\n",
                "from pathlib import Path\n",
                "ROOT = Path('/home/box/case-studies/_nospace/cs013_work')\n",
                "REAL = Path('/home/box/case-studies/013 - Self-Driving Car Sim')\n",
                "print('CS013 paths OK', ROOT.exists(), REAL.exists())\n",
            ],
        })
    if i == 2:
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Smoke frames land in REAL/plots/smoke_t*.png via pipeline step 2\n",
                "frames = sorted((REAL / 'plots').glob('smoke_t*.png'))\n",
                "print('smoke frames:', [p.name for p in frames])\n",
            ],
        })

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
nb_path.write_text(json.dumps(nb, indent=1))
# copy to work tree for OpenClaude-safe path
work_nb = WORK / "self_driving_car_sim.ipynb"
work_nb.write_text(nb_path.read_text())

summary = {
    "notebook": str(nb_path),
    "work_copy": str(work_nb),
    "n_cells": len(cells),
    "n_markdown": sum(1 for c in cells if c["cell_type"] == "markdown"),
    "n_code": sum(1 for c in cells if c["cell_type"] == "code"),
}
(proc / "extract_stub_notebook.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
