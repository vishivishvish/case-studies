# AI Case Studies

A series of AI/ML case studies, designed for citizen AI engineers and data scientists who need to work across various domains and to gain consultant-level expertise in each one.

## The core idea: law rediscovery

Each case study is structured so that an explainability method (SHAP, LIME, permutation importance, or partial dependence, depending on the case) surfaces a scientific or engineering law that took an expert years to establish, without the model ever being told that law in advance. The model finds the Bethe-Weizsacker binding energy formula, the Boids flocking rules, Denning's statistical intrusion-detection model, or the Population Stability Index threshold on its own, from data alone. The explanation is the proof.

## Domain coverage

Case studies span a deliberately wide range of fields: nuclear medicine, neuroscience and brain-machine interfaces, robotics, autonomous vehicles, drone swarms, cognitive psychology, quantum key distribution, quantitative finance, rocket engineering, network security, data engineering, and speech recognition, with more domains added over time until every domain on the project's target list has a case study.

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

Note: the free LLM backend for stages 5, 7, and 8 switched from Hugging Face to NVIDIA NIM (NVIDIA Inference Microservices, serving nvidia/nemotron-3-ultra-550b-a55b, the model confirmed to work reliably across this account's API key) after Hugging Face's free tier proved too rate-limited for repeated agentic experimentation.

## Case studies

| # | Case study | Domain | Predicts | Law rediscovered | 8-stage status |
|---|---|---|---|---|---|
| 1 | [Targeted Alpha Therapy](001%20-%20Targeted%20Alpha%20Therapy/targeted_alpha_therapy.ipynb) | Nuclear medicine | Nuclear binding energy per nucleon from proton/neutron counts (AME2020, ~3,554 nuclides) | The Semi-Empirical Mass Formula (Bethe-Weizsacker, 1935), found independently via SHAP | Full 8 stages |
| 2 | [EEG Motor Imagery](002%20-%20EEG%20Motor%20Imagery/eeg_motor_imagery.ipynb) | Neuroscience / brain-machine interfaces | Left vs. right hand motor imagery from 64-channel EEG (PhysioNet, 109 subjects) | Pfurtscheller's event-related desynchronization and Penfield's motor homunculus (C3/C4 lateralization) | Full 8 stages |
| 3 | [Humanoid Robot Grasp Prediction](003%20-%20Humanoid%20Robot%20Grasp%20Prediction/grasp_prediction.ipynb) | Robotics | Grasp rectangle for household objects from RGB images (synthetic Cornell Grasping replica) | Napier's Power/Precision Grip dichotomy (1956) and Gibson's affordances (1979) | Full 8 stages |
| 4 | [Autonomous Vehicle Fleet Road Intelligence](004%20-%20Autonomous%20Vehicle%20Fleet%20Road%20Intelligence/av_fleet_network_intelligence.ipynb) | Autonomous vehicles / network science | Taxi knowledge-sharing network structure from synthetic GPS traces (500 taxis, 7 days) | Milgram's small-world phenomenon (the Barabasi-Albert scale-free law was tested but not confirmed on this data; Granovetter's Strength of Weak Ties was tested and dropped after failing to hold even under a purpose-built diffusion mechanism) | Full 8 stages |
| 5 | [Autonomous Drone Swarm Behavioral Clustering](005%20-%20Autonomous%20Drone%20Swarm%20Behavioral%20Clustering/drone_swarm_clustering.ipynb) | Drones / multi-agent systems | Behavioral regime of a simulated drone swarm from kinematic features alone | Reynolds' Boids rules, Separation/Alignment/Cohesion (SIGGRAPH 1987), found via K-Means after DBSCAN was tested and found not to separate this data into 3 clusters | Full 8 stages |
| 6 | [Memory Decay and Reconstructive Narrative Regression](006%20-%20Memory%20Decay%20and%20Reconstructive%20Narrative%20Regression/memory_decay_regression.ipynb) | Cognitive psychology | Days since a recalled event from linguistic features of its retelling (Hippocorpus is gated behind a manual login with no programmatic access, like TabPFN in CS4; synthetic fallback text is generated through an explicit, time-scaled decay mechanism rather than a hardcoded formula) | Bartlett's reconstructive memory (1932) and Ebbinghaus's forgetting curve (1885), with Brown and Kulik's rehearsal effect (1977) tested by an explicit per-group slope regression rather than an eyeballed plot | Full 8 stages |
| 7 | [Eavesdropper Detection in QKD Channels](007%20-%20Eavesdropper%20Detection%20in%20QKD%20Channels/qkd_eavesdropper_detection.ipynb) | Quantum cryptography | Eavesdropper presence in quantum key distribution from QBER time series | Page's CUSUM (1954) and the No-Cloning Theorem (Wootters and Zurek, 1982) | Pending retrofit |
| 8 | [Sparse Financial Panel Recommendation](008%20-%20Sparse%20Financial%20Panel%20Recommendation/financial_panel_recommendation.ipynb) | Quantitative finance | Missing and next-day stock returns from a sparse 500-stock by 1,260-day panel | Fama-French / Ross Arbitrage Pricing Theory (1976) via SVD singular vectors | Pending retrofit |
| 9 | [Rocket Engine Performance Prediction](009%20-%20Rocket%20Engineering%20Performance%20Prediction/rocket_engine_performance.ipynb) | Rocket engineering | Specific impulse from propellant and nozzle-geometry parameters | Isp proportional to the square root of chamber temperature over exhaust molecular weight | Full 8 stages |
| 10 | [Network Intrusion Detection](010%20-%20Network%20Intrusion%20Detection/network_intrusion_detection.ipynb) | Network security | Benign vs. SYN-flood/port-scan/brute-force network flows | Denning's statistical intrusion-detection model (1987) | Full 8 stages |
| 11 | [Data Quality Drift Detection](011%20-%20Data%20Quality%20Drift%20Detection/data_quality_drift_detection.ipynb) | Data engineering | Stable vs. drifted incoming data batches from batch-level statistics | The Population Stability Index / Kolmogorov-Smirnov industry drift threshold | Full 8 stages |
| 12 | [Spoken Word Recognition](012%20-%20Spoken%20Word%20Recognition/spoken_word_recognition.ipynb) | Speech recognition / audio machine learning | One of twenty spoken words (the digits zero through nine and ten command words) from MFCC spectrograms (Google Speech Commands v0.02, roughly 77,000 clips from roughly 2,580 speakers, split speaker-disjoint) | The Haskins Laboratories formant-transition finding (Delattre, Liberman, and Cooper, 1955), that word identity is carried by spectral change over time, which a time-averaged spectrum discards. TODO(numbers): Section 15 confirms or drops it once the leak-free evaluation lands, on the precedent CS4 set with Granovetter | All 8 stages built, none executed. See the CS12 handoff section below for how to finish it |

"Full 8 stages" means: a primary technique (classical ML, clustering, network analytics, or time series), a light deep learning model, a pretrained foundation model benchmark where the domain allows one, explainable AI, GenAI-assisted feature engineering, GenAI-guided synthetic data augmentation, a fixed LangGraph agentic pipeline, and an autonomous agent variant with real planning, tool use, and memory.

## Status

Case studies are added one at a time, each requiring explicit approval before it is built. Of the 12 case studies registered so far, CS1, CS2, CS3, CS4, CS5, CS6, CS9, CS10, and CS11 are complete at all 8 stages. CS6's retrofit is now finished: stages 1, 3, and 4 already existed and were repaired first (a DeadKernelError, and a scientifically bogus synthetic-data fallback replaced with a real, mechanism-driven text-decay generator, Hippocorpus being gated with no programmatic access), and stages 2, 5, 6, 7, and 8 have since been built and executed end to end at 59 code cells with zero errors. CS7-8 will be migrated to the full 8-stage pipeline next, one at a time and in numeric order, along with a narrative-quality pass bringing their history and science sections up to CS1's standard of named historical figures, dated external citations, and a built cause-and-effect story rather than a reference table. CS9-11 already have all 8 stages structurally but are being migrated from their Hugging Face LLM backend to NVIDIA NIM for consistency with CS1-2.

Two results from CS6 are worth recording, because the notebook reports both as computed rather than as hoped for. First, the frozen sentence-encoder probe (stage 3) **beats** the seven theory-grounded features (stage 1) on held-out RMSE, 0.6529 against 0.7015. The theory features earn their place by being interpretable, not by being the most accurate: SHAP on the seven can name which psychological construct drove a prediction, and a 384-dimensional embedding cannot. Second, the autonomous agent (stage 8) was caught fabricating its own tool output on the first run, inventing a predicted age, a stated age, and three past case ids no tool had returned. Rather than patch the prompt and move on, the loop now discards anything the model writes after its own action, and a grounding check re-derives the true figures and reports any number in the closing note the tools never produced. That check prints its verdict either way, and it is the clearest available illustration of what separates a fixed agentic pipeline from an autonomous one: the LangGraph graph in stage 7 cannot go off-script, and the agent in stage 8 can.

The shared NVIDIA rate limiter (`nvidia_rate_limited_call.py`, copied identically into CS4, CS5, CS6, and CS12) now enforces a hard wall-clock ceiling on every API call. A stalled TLS read can outlive the `requests` read timeout entirely, because CPython blocks inside the SSL layer and the socket deadline never fires; this wedged one CS6 run for 25 minutes on a single call. A wedged socket now costs one retry instead of the whole run.

CS12 adds speech and audio to the project's field list, an approved extension of scope rather than a substitution for any field already queued. All 8 stages are now built in code, and none has been executed. Its notebook carries the full 18-section house structure, the history and science sections, and the Rediscovery Standard, with every measured result marked `TODO(numbers)`.

Two design substitutions apply. Stage 3 uses wav2vec 2.0 in place of TabPFN, which expects tabular structure and does not fit an audio domain, and stage 4 uses permutation importance in place of SHAP or LIME, because it folds back onto the 40 by 32 spectrogram grid in a way neither of those does. Stages 5, 6, 7, and 8 call NVIDIA NIM through the same shared rate limiter CS3 through CS6 use, each with a documented fallback so the notebook runs without credentials.

The case study is built around a controlled comparison. Every model runs under two MFCC feature representations, a mean-pooled 40-feature vector and the full flattened 1,280-feature spectrogram, which differ in exactly one thing, the time axis, so the accuracy gap between them is attributable to nothing else. The split is speaker-disjoint, since MFCCs encode the vocal tract as much as the word and a random split would let the highest-capacity model score well by recognising voices. No performance metric appears anywhere in the notebook yet, as it has not been executed.

### CS12 handoff: what remains and how to finish it

The notebook holds 148 cells, 27 of them code, with all 8 stages written and 61 `TODO(numbers)` markers outstanding. Work through the following in order.

1. Execute the notebook end to end through nbconvert, then verify by parsing the `.ipynb` JSON directly for zero cells with `output_type == "error"` and a sequential `execution_count`. State the cell count in the commit message, as the other case studies do.
2. Fill every `TODO(numbers)` marker from the notebook's own executed outputs. They cluster in sections 7.4, 8.6, 9.3, 10.3, 11.2, 11.3, 12.2, 13.3, 14.2, 14.3, 15, and 16.
3. Resolve the section 15 law verdict from the section 14 output, following the rule already written into section 3.2.
4. Write sections 18.1 and 18.2 last, from the results.

Prerequisites are in place on the development machine. `NVIDIA_API_KEY` sits in `~/.config/secrets/.env`, torch reports MPS available, and transformers is installed. Budget several hours for a full run. Extracting wav2vec 2.0 embeddings over roughly 77,000 clips is the longest step and should use MPS. Fitting XGBoost on the 1,280-feature representation is the second longest, and it dominates CPU time by a wide margin.

**The law verdict is genuinely open, and one of its two checks is the deciding one.** Check 1 asks whether the full spectrogram beats the mean-pooled representation by more than the extra parameters alone would explain. Independent measurement supports it: the full representation won for every model tested, and the margin widened as model capacity rose. Check 2 asks whether permutation importance over the grid concentrates on particular time frames rather than spreading evenly across them. Nobody has computed check 2 yet. Section 14 produces it during execution, so the verdict resolves on that run. If either check fails, drop the law and say so, on the precedent CS4 set when Granovetter's Strength of Weak Ties failed its test and was removed rather than softened.

**Reference measurements from an independent implementation.** A separate scikit-learn and Keras implementation, built outside this repository and not sharing code with CS12's PyTorch stage 12, measured the figures below on a three-way speaker-disjoint split of 57,601 training, 8,413 validation, and 11,440 test clips over the same 20 words. Treat these as a sanity check on CS12's own executed output and nothing more. Do not paste them into the notebook. CS12 must report what its own cells produce.

| Representation | Decision Tree | Random Forest | XGBoost | Feedforward net |
|---|---|---|---|---|
| Mean-pooled, 40 features | 15.1 | 31.1 | 31.8 | 35.0 |
| Frame-subsampled, 640 features | 35.6 | 67.9 | 75.2 | 81.0 |
| Full spectrogram, 1,280 features | 35.4 | 69.2 | 75.7 | not completed |

Read those rows with two caveats. The mean-pooled row came from an earlier harness that early-stopped the neural network on the test partition, which flatters that column slightly, so it is not strictly comparable to the rows below it. The 640-feature row is the only one measured cleanly end to end, with early stopping on a validation partition and the test partition scored once per model. On that row the feedforward network sat 5.8 points above XGBoost and 13.2 points above Random Forest, and its three seeds landed at 81.4, 81.1, and 80.7. XGBoost consumed 6,042 CPU seconds against roughly 500 for the network, so the network won on cost as well as accuracy. XGBoost used 587 of its 600 rounds without early stopping triggering, so it had not fully plateaued and a larger budget would earn it a little more.

The corpus downloads inside the notebook. A local copy of Google Speech Commands v0.02 may still sit at `/Users/vishnusubramanian/Documents/audio_cs_work/data/speech_commands`, holding 105,829 wav files across 35 words, alongside cached MFCC and log-mel arrays under `audio_cs_work/cache`. That directory belongs to a separate piece of work and may be deleted at any time, so treat it as a convenience and not a dependency.

## Authorship

Although ideated by me (vishivishvish), the co-authors of this repository include Claude, Codex, Cursor, Gemini, Qwen and other agents who contribute by working across research, narrative, implementation, validation, and review. But each change remains subject to human approval before merge or publication.
