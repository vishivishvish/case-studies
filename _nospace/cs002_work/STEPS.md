# CS002 EEG Motor Imagery — Pipeline Steps

**Total Steps:** 22  
**Estimated Runtime:** ~15-25 minutes on CPU (16GB RAM)  
**Target:** Bring to CS001-level completeness (executed notebook, plots, README, git PR)

---

## Step List

| # | Step Name | Title | Description | Difficulty | Dependencies |
|---|-----------|-------|-------------|------------|--------------|
| 1 | `env_check` | Environment & Dependency Check | Verify Python venv, key packages (mne, sklearn, xgboost, shap, torch, tabpfn) | Easy | — |
| 2 | `data_check` | Data Availability Check | Verify PhysioNet EEG Motor Imagery data loads via MNE (subject 1, runs 4,8,12) | Easy | 1 |
| 3 | `extract_notebook_code` | Extract Notebook Code | Export all 110 code cells to `extracted_notebook.py` for inspection/debugging | Easy | — |
| 4 | `load_and_filter` | Load & Bandpass Filter | Load raw EDF, apply 1–100 Hz FIR filter, save filtered raws to `data/processed/` | Easy | 2 |
| 5 | `epoching` | Epoching & Baseline Correction | Extract events (T1=left, T2=right), epoch −1 to +3 s, baseline −1 to 0 s | Easy | 4 |
| 6 | `feature_extraction` | PSD Feature Extraction | Compute Welch PSD (5 bands × 64 ch = 320 features) for imagery window 0.5–2.5 s | Medium | 5 |
| 7 | `human_features` | Human-Engineered Features | C3/C4 alpha/beta asymmetry, band totals, broadband variability (5 features) | Easy | 6 |
| 8 | `train_rf` | Train Random Forest | 500 trees, max_depth=10, 5-fold stratified CV, save model + CV scores | Medium | 6 |
| 9 | `train_xgb` | Train XGBoost | 500 estimators, depth=5, lr=0.05, 5-fold CV, save model | Medium | 6 |
| 10 | `shap_analysis` | SHAP Analysis | TreeExplainer on RF → beeswarm plot (top 30), band importance bar chart | Medium | 8 |
| 11 | `topomaps` | Topographic Brain Maps | Class-specific SHAP topomaps (alpha/beta/gamma) + imagery vs movement alpha C3/C4 | Medium | 5, 10 |
| 12 | `dl_light` | Light Deep Learning (EEGNetLite) | Small 1D CNN on 320 PSD features, 50 epochs CPU, save weights | Medium | 6 |
| 13 | `tabpfn` | TabPFN Foundation Model | TabPFNClassifier on same split; gracefully skip if unavailable (CPU) | Easy | 6 |
| 14 | `stage_a` | Stage A: Signal Quality | Flag top 10% broadband variability trials for review; save indices | Easy | 7 |
| 15 | `stage_b` | Stage B: Control-State Classification | RF on enriched features (320 + 5 human) → control-state probability | Medium | 7, 8 |
| 16 | `config_scores` | Configuration Scoring | 4 electrode configs ranked by 5-fold CV (C3/C4, 3×3 Laplacian, full, occipital) | Medium | 6 |
| 17 | `agentic_stub` | Agentic Layer Scaffold | LangGraph state schema + empty memory file for future autonomous agent | Easy | — |
| 18 | `executed_notebook` | Execute Full Notebook | nbconvert --execute → `eeg_motor_imagery_executed.ipynb` with all outputs | Medium | 1,2,4,5,6,8,9,10,11,12,13,14,15,16 |
| 19 | `fill_conclusion` | Inject Results Summary | Append auto-generated results table to notebook Conclusion section | Easy | 18 |
| 20 | `sync_to_real` | Sync to Real Directory | Copy executed notebook, plots/, data/processed/ to `002 - EEG Motor Imagery/` | Easy | 18, 19 |
| 21 | `readme_update` | Create README.md | Generate comprehensive README with results, artifacts, reproduction steps | Easy | 20 |
| 22 | `git_commit` | Git Commit + PR | Add, commit, push feature branch; create PR via `gh` if available | Easy | 21 |

---

## Execution Order (Critical Path)

```
1 → 2 → 4 → 5 → 6 → {7, 8, 9} → 10 → 11
                    ↓
                  12, 13 (independent)
                    ↓
                  14, 15, 16 (depend on 7, 8)
                    ↓
                  17 (independent)
                    ↓
                  18 → 19 → 20 → 21 → 22
```

- Steps 3, 17 can run anytime (no deps)
- Steps 8, 9, 12, 13 can run in parallel after step 6
- Step 18 (notebook execution) depends on most prior steps for data artifacts

---

## Usage

```bash
cd /home/box/case-studies/_nospace/cs002_work

# List all steps
python cs002_pipeline.py --list

# Show status
python cs002_pipeline.py --status

# Run single step (checks dependencies)
python cs002_pipeline.py --step 1
python cs002_pipeline.py --step 4
# ...

# Run all sequentially
python cs002_pipeline.py --run-all
```

---

## Outputs

| Artifact | Location |
|----------|----------|
| Step status log | `step_status.json` |
| Step run log | `steps.log` |
| Processed data | `data/processed/` (X.npy, y.npy, models, SHAP, configs) |
| Plots | `plots/` (shap_beeswarm.png, shap_band_importance.png, shap_topomaps_class_band.png) |
| Executed notebook | `eeg_motor_imagery_executed.ipynb` |
| Real dir sync | `/home/box/case-studies/002 - EEG Motor Imagery/` |
| README | `/home/box/case-studies/002 - EEG Motor Imagery/README.md` |

---

## Notes

- **CPU-only**: All DL uses CPU (torch.device="cpu", TabPFN device="cpu")
- **Memory**: ~2-4 GB peak (epochs + PSD + SHAP)
- **Idempotency**: Steps overwrite files in `data/processed/` and `plots/`; safe to re-run
- **TabPFN**: Optional; step records `{"status": "skipped", "reason": "..."}` if import fails
- **Git PR**: Requires `gh` CLI authenticated; otherwise just creates local branch + push