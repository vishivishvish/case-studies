#!/usr/bin/env python3
"""
CS7/CS8 Retrofit Orchestrator

Sequentially executes and validates CS7 (QKD Eavesdropper Detection) and CS8 (Sparse Financial Panel)
notebooks on CPU-only hardware, documenting honest blockers for GPU-dependent stages.

Stages 5-8 (GenAI, synthetic data, LangGraph, autonomous agent) require NVIDIA NIM API access.
On this CPU-only box, these stages fall back to documented proxies.
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

PROGRESS_DIR = Path("/home/box/case-studies/nova-cs78-progress")
PROGRESS_DIR.mkdir(exist_ok=True)

def log_message(msg: str):
    """Log to run.log with IST timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    with open(PROGRESS_DIR / "run.log", "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(msg)

def update_status(stage: str, state: str, detail: str):
    """Update status.json with current progress."""
    status = {
        "stage": stage,
        "state": state,
        "detail": detail,
        "updated_at_ist": datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')
    }
    with open(PROGRESS_DIR / "status.json", "w") as f:
        json.dump(status, f, indent=2)

def execute_notebook(notebook_path: Path, timeout: int = 3600) -> dict:
    """Execute a notebook and return results."""
    log_message(f"Executing {notebook_path.name}...")
    update_status(f"executing:{notebook_path.stem}", "running", "Notebook execution started")

    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name='cs78-venv')

    try:
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        log_message(f"Successfully executed {notebook_path.name}")
        update_status(f"executing:{notebook_path.stem}", "success", "Notebook executed without errors")
        return {"success": True, "notebook": nb, "error": None}
    except Exception as e:
        log_message(f"Error executing {notebook_path.name}: {e}")
        update_status(f"executing:{notebook_path.stem}", "failed", str(e))
        return {"success": False, "notebook": nb, "error": str(e)}

def check_notebook_stages(notebook_path: Path) -> dict:
    """Analyze notebook for 8-stage completeness."""
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Check for stage markers in markdown cells
    stages_found = {i: False for i in range(1, 9)}
    stage_keywords = {
        1: ["stage 1", "primary technique", "classical ml", "arima", "content-based"],
        2: ["stage 2", "light deep learning", "lstm", "collaborative filtering"],
        3: ["stage 3", "foundation model", "chronos", "tabpfn", "soft-impute"],
        4: ["stage 4", "explainable", "shap", "permutation importance", "law rediscovery"],
        5: ["stage 5", "genai", "feature engineering", "nvidia", "nim"],
        6: ["stage 6", "synthetic data", "augmentation"],
        7: ["stage 7", "langgraph", "agentic", "pipeline"],
        8: ["stage 8", "autonomous agent", "planning", "tool use", "memory"]
    }

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            content = cell.source.lower()
            for stage, keywords in stage_keywords.items():
                if any(kw in content for kw in keywords):
                    stages_found[stage] = True

    return stages_found

def main():
    log_message("=" * 60)
    log_message("CS7/CS8 Retrofit Orchestrator Starting")
    log_message("=" * 60)

    # CS7: QKD Eavesdropper Detection
    cs7_path = Path("/home/box/case-studies/007 - Eavesdropper Detection in QKD Channels/qkd_eavesdropper_detection.ipynb")
    # CS8: Sparse Financial Panel Recommendation
    cs8_path = Path("/home/box/case-studies/008 - Sparse Financial Panel Recommendation/financial_panel_recommendation.ipynb")

    # First, analyze current stage coverage
    log_message("Analyzing CS7 stage coverage...")
    cs7_stages = check_notebook_stages(cs7_path)
    log_message(f"CS7 stages found: {cs7_stages}")

    log_message("Analyzing CS8 stage coverage...")
    cs8_stages = check_notebook_stages(cs8_path)
    log_message(f"CS8 stages found: {cs8_stages}")

    # Execute CS7
    log_message("\n" + "=" * 60)
    log_message("EXECUTING CS7: QKD Eavesdropper Detection")
    log_message("=" * 60)
    cs7_result = execute_notebook(cs7_path, timeout=7200)

    # Execute CS8
    log_message("\n" + "=" * 60)
    log_message("EXECUTING CS8: Sparse Financial Panel Recommendation")
    log_message("=" * 60)
    cs8_result = execute_notebook(cs8_path, timeout=7200)

    # Summary
    log_message("\n" + "=" * 60)
    log_message("RETROFIT SUMMARY")
    log_message("=" * 60)
    log_message(f"CS7 execution: {'SUCCESS' if cs7_result['success'] else 'FAILED'}")
    log_message(f"CS8 execution: {'SUCCESS' if cs8_result['success'] else 'FAILED'}")

    if cs7_result['success'] and cs8_result['success']:
        update_status("retrofit_complete", "success", "Both CS7 and CS8 executed successfully")
        log_message("All executions completed successfully!")
        return 0
    else:
        update_status("retrofit_complete", "failed", "One or more notebooks failed")
        log_message("Some executions failed. Check run.log for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())