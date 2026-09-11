# CS013 Self-Driving Car Sim — Pipeline Steps

**Total Steps:** 22  
**Case id:** 013 (pitch CS032)  
**Default env:** Gymnasium CarRacing-v3  

| # | Step Name | Title | Difficulty | Dependencies |
|---|-----------|-------|------------|--------------|
| 1 | `env_check` | Environment & dependency check | Easy | — |
| 2 | `sim_smoke` | Env smoke + sample frames | Easy | 1 |
| 3 | `extract_stub_notebook` | Notebook stub / outline | Easy | — |
| 4 | `pid_baseline` | PID / pure-pursuit baseline | Easy | 2 |
| 5 | `collect_expert` | Expert (PID) rollouts | Easy | 4 |
| 6 | `bc_train` | Behavioral cloning | Medium | 5 |
| 7 | `bc_eval` | BC vs PID metrics | Easy | 6 |
| 8 | `rl_ppo` | Time-boxed PPO/SAC | Medium | 2 |
| 9 | `rl_eval` | Ladder eval PID/BC/RL | Easy | 8 |
| 10 | `xai_policy` | SHAP / saliency | Medium | 6 |
| 11 | `feature_clusters` | Maneuver clustering | Easy | 5 |
| 12 | `dl_light` | Small CNN/MLP policy | Medium | 5 |
| 13 | `foundation_skip` | FM honest skip | Easy | — |
| 14 | `genai_tracks_stub` | Synthetic track profiles | Medium | 2 |
| 15 | `genai_features` | Track-type conditioning | Medium | 14,6 |
| 16 | `agentic_stub` | Curriculum + safety schema | Easy | 9 |
| 17 | `hybrid_endgame_note` | Tiny reward search / hybrid note | Medium | 16 |
| 18 | `executed_notebook` | Execute / assemble notebook | Medium | many |
| 19 | `fill_conclusion` | Inject results | Easy | 18 |
| 20 | `sync_to_real` | Sync to `013 - …/` | Easy | 18,19 |
| 21 | `readme_update` | README with results | Easy | 20 |
| 22 | `git_commit` | commit-cursor + PR | Easy | 21 |
