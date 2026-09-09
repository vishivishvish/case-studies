# Case Studies

A series of AI/ML case studies, designed for citizen AI engineers and data scientists who need to work across various domains and to gain consultant-level expertise in each one.

## The core idea: law rediscovery

Each case study is structured so that an explainability method (SHAP, LIME, permutation importance, or partial dependence, depending on the case) surfaces a scientific or engineering law that took experts years to establish, without the model ever being told that law in advance. The model finds the Bethe-Weizsacker binding energy formula, the Boids flocking rules, Denning's statistical intrusion-detection model, or the Population Stability Index threshold on its own, from data alone. The explanation is the proof.

## Domain coverage

Case studies span a deliberately wide range of fields: nuclear medicine, neuroscience and brain-machine interfaces, robotics, autonomous vehicles, drone swarms, cognitive psychology, quantum key distribution, quantitative finance, rocket engineering, network security, data engineering, and speech recognition, with more domains added over time, until every domain on the project's target list has a case study.

## The technical stack per case study

Every case study is intended to build toward the same eight-stage pipeline, though not all have reached it yet (see the per-case-study status column below):

1. A primary technique category: classical ML, unsupervised clustering, network analytics, or time series
2. A light deep learning model trained on the same data
3. A pretrained foundation model benchmark, where the domain allows one (for example, TabPFN for tabular tasks)
4. Explainable AI, with the method chosen to fit the model and the domain rather than defaulting to one method everywhere
5. GenAI-assisted feature engineering, where a free NVIDIA-hosted language model proposes candidate features, and an ablation confirms that human-engineered and AI-generated features together outperform either alone
6. GenAI-guided synthetic data augmentation, checked for fidelity by comparing model performance on original-only, original-plus-synthetic, and synthetic-only data against the same held-out test set
7. An agentic layer built with LangGraph, calling a free NVIDIA-hosted language model to turn model output into a natural-language recommendation
8. An autonomous agent variant of that same layer, with self-directed planning, real tool use, and memory of past cases, built to show what separates a scripted agentic pipeline from genuine agent autonomy

Note: the free LLM backend for stages 5, 7, and 8 switched from Hugging Face to NVIDIA NIM (NVIDIA Inference Microservices, serving nvidia/nemotron-3-ultra-550b-a55b, the model confirmed to work reliably with this account's API key) after Hugging Face's free tier proved too rate-limited for repeated agentic experimentation.

## Case studies

| # | Case study | Domain | Predicts | Law rediscovered | 8-stage status |
|---|---|---|---|---|---|
| 1 | [Targeted Alpha Therapy](001%20-%20Targeted%20Alpha%20Therapy/targeted_alpha_therapy.ipynb) | Nuclear medicine | Nuclear binding energy per nucleon from proton/neutron counts (AME2020, ~3,554 nuclides) | The Semi-Empirical Mass Formula (Bethe-Weizsacker, 1935), found independently via SHAP | Full 8 stages |
| 2 | [EEG Motor Imagery](002%20-%20EEG%20Motor%20Imagery/eeg_motor_imagery.ipynb) | Neuroscience / brain-machine interfaces | Left vs. right hand motor imagery from 64-channel EEG (PhysioNet, 109 subjects) | Pfurtscheller's event-related desynchronization and Penfield's motor homunculus (C3/C4 lateralization) | Executing... |
| 3 | [Humanoid Robot Grasp Prediction](003%20-%20Humanoid%20Robot%20Grasp%20Prediction/grasp_prediction.ipynb) | Robotics | Grasp rectangle for household objects from RGB images (synthetic Cornell Grasping replica) | Napier's Power/Precision Grip dichotomy (1956) and Gibson's affordances (1979) | Full 8 stages |
| 4 | [Autonomous Vehicle Fleet Road Intelligence](004%20-%20Autonomous%20Vehicle%20Fleet%20Road%20Intelligence/av_fleet_network_intelligence.ipynb) | Autonomous vehicles / network science | Taxi knowledge-sharing network structure from synthetic GPS traces (500 taxis, 7 days) | Milgram's small-world phenomenon (the Barabasi-Albert scale-free law was tested but not confirmed on this data; Granovetter's Strength of Weak Ties was tested and dropped after failing to hold even under a purpose-built diffusion mechanism) | Full 8 stages |
| 5 | [Autonomous Drone Swarm Behavioral Clustering](005%20-%20Autonomous%20Drone%20Swarm%20Behavioral%20Clustering/drone_swarm_clustering.ipynb) | Drones / multi-agent systems | Behavioral regime of a simulated drone swarm from kinematic features alone | Reynolds' Boids rules, Separation/Alignment/Cohesion (SIGGRAPH 1987), found via K-Means after DBSCAN was tested and found not to separate this data into 3 clusters | Full 8 stages |
| 6 | [Memory Decay and Reconstructive Narrative Regression](006%20-%20Memory%20Decay%20and%20Reconstructive%20Narrative%20Regression/memory_decay_regression.ipynb) | Cognitive psychology | Days since a recalled event from linguistic features of its retelling (Hippocorpus is gated behind a manual login with no programmatic access, like TabPFN in CS4; synthetic fallback text is generated through an explicit, time-scaled decay mechanism rather than a hardcoded formula) | Bartlett's reconstructive memory (1932) and Ebbinghaus's forgetting curve (1885), with Brown and Kulik's rehearsal effect (1977) tested by an explicit per-group slope regression rather than an eyeballed plot | Full 8 stages |
| 7 | [Eavesdropper Detection in QKD Channels](007%20-%20Eavesdropper%20Detection%20in%20QKD%20Channels/qkd_eavesdropper_detection.ipynb) | Quantum cryptography | Eavesdropper presence in quantum key distribution from QBER time series | Page's CUSUM (1954) and the No-Cloning Theorem (Wootters and Zurek, 1982) | Full 8 stages |
| 8 | [Sparse Financial Panel Recommendation](008%20-%20Sparse%20Financial%20Panel%20Recommendation/financial_panel_recommendation.ipynb) | Quantitative finance | Missing and next-day stock returns from a sparse 500-stock by 1,260-day panel | Fama-French / Ross Arbitrage Pricing Theory (1976) via SVD singular vectors | Full 8 stages |
| 9 | [Rocket Engine Performance Prediction](009%20-%20Rocket%20Engineering%20Performance%20Prediction/rocket_engine_performance.ipynb) | Rocket engineering | Specific impulse from propellant and nozzle-geometry parameters | Isp proportional to the square root of chamber temperature over exhaust molecular weight | Full 8 stages |
| 10 | [Network Intrusion Detection](010%20-%20Network%20Intrusion%20Detection/network_intrusion_detection.ipynb) | Network security | Benign vs. SYN-flood/port-scan/brute-force network flows | Denning's statistical intrusion-detection model (1987) | Full 8 stages |
| 11 | [Data Quality Drift Detection](011%20-%20Data%20Quality%20Drift%20Detection/data_quality_drift_detection.ipynb) | Data engineering | Stable vs. drifted incoming data batches from batch-level statistics | The Population Stability Index / Kolmogorov-Smirnov industry drift threshold | Full 8 stages |
| 12 | [Spoken Word Recognition](012%20-%20Spoken%20Word%20Recognition/spoken_word_recognition.ipynb) | Speech recognition / audio machine learning | One of twenty spoken words (the digits zero through nine and ten command words) from MFCC spectrograms (Google Speech Commands v0.02, roughly 77,000 clips from roughly 2,580 speakers, split speaker-disjoint) | The Haskins Laboratories formant-transition finding (Delattre, Liberman, and Cooper, 1955), that word identity is carried by spectral change over time, which a time-averaged spectrum discards | Full 8 stages |

"Full 8 stages" means: a primary technique (classical ML, clustering, network analytics, or time series), a light deep learning model, a pretrained foundation model benchmark where the domain allows one, explainable AI, GenAI-assisted feature engineering, GenAI-guided synthetic data augmentation, a fixed LangGraph agentic pipeline, and an autonomous agent variant with real planning, tool use, and memory.

## Status

Case studies are added one at a time, each requiring explicit approval before it is built. Of the 12 case studies registered so far, CS1, CS2, CS3, CS4, CS5, CS6, CS7, CS8, CS9, CS10, CS11, and CS12 are complete at all 8 stages.

CS6's retrofit finished first: stages 1, 3, and 4 already existed and were repaired (a DeadKernelError, and a scientifically bogus synthetic-data fallback replaced with a real, mechanism-driven text-decay generator, Hippocorpus being gated with no programmatic access), then stages 2, 5, 6, 7, and 8 were built and executed end to end at 59 code cells with zero errors.

CS7 and CS8 were then migrated to the full 8-stage pipeline sequentially, each receiving a narrative-quality pass bringing their history and science sections up to CS1's standard of named historical figures, dated external citations, and a built cause-and-effect story rather than a reference table. Both notebooks now execute end-to-end on CPU-only hardware with honest blockers noted where GPU-dependent stages (Chronos, TabPFN, NVIDIA NIM-backed GenAI stages) fall back to documented proxies.

CS9-11 already had all 8 stages structurally and were migrated from their Hugging Face LLM backend to NVIDIA NIM for consistency with CS1-2.

CS12 adds speech and audio to the project's field list, an approved extension of scope rather than a substitution for any field already queued. All 8 stages are complete: the notebook has been executed end-to-end (27 code cells, zero errors), all `TODO(numbers)` markers have been filled from the notebook's own executed outputs, the Rediscovery Standard verdict has been resolved from permutation importance results, and sections 18.1–18.2 have been written from the measured results. Stage 3 uses wav2vec 2.0 in place of TabPFN (which expects tabular structure and does not fit an audio domain), and stage 4 uses permutation importance in place of SHAP or LIME (because it folds back onto the 40 by 32 spectrogram grid in a way neither of those does). Stages 5, 6, 7, and 8 call NVIDIA NIM through the shared rate limiter, each with a documented fallback so the notebook runs without credentials.

Two results from CS6 are worth recording, because the notebook reports both as computed rather than as hoped for. First, the frozen sentence-encoder probe (stage 3) **beats** the seven theory-grounded features (stage 1) on held-out RMSE, 0.6529 against 0.7015. The theory features earn their place by being interpretable, not by being the most accurate: SHAP on the seven can name which psychological construct drove a prediction, and a 384-dimensional embedding cannot. Second, the autonomous agent (stage 8) was caught fabricating its own tool output on the first run, inventing a predicted age, a stated age, and three past case ids no tool had returned. Rather than patch the prompt and move on, the loop now discards anything the model writes after its own action, and a grounding check re-derives the true figures and reports any number in the closing note the tools never produced. That check prints its verdict either way, and it is the clearest available illustration of what separates a fixed agentic pipeline from an autonomous one: the LangGraph graph in stage 7 cannot go off-script, and the agent in stage 8 can.

The shared NVIDIA rate limiter (`nvidia_rate_limited_call.py`, copied identically into CS4, CS5, CS6, and CS12) now enforces a hard wall-clock ceiling on every API call. A stalled TLS read can outlive the `requests` read timeout entirely, because CPython blocks inside the SSL layer and the socket deadline never fires; this wedged one CS6 run for 25 minutes on a single call. A wedged socket now costs one retry instead of the whole run.

## Authorship

Although ideated by me (vishivishvish), the co-authors of this repository include Claude, Codex, Cursor, Gemini, Grok, Qwen and other agents who contribute by working across research, narrative, implementation, validation, and review. But each change remains subject to human approval before merge or publication.
