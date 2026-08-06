# GL Case Studies

A series of machine learning case studies built for Great Learning, designed for citizen AI engineers: data scientists who work across domains without being domain experts in each one.

## The core idea: law rediscovery

Each case study is structured so that an explainability method (SHAP, LIME, permutation importance, or partial dependence, depending on the case) surfaces a scientific or engineering law that took an expert years to establish, without the model ever being told that law in advance. The model finds the Bethe-Weizsacker binding energy formula, the Boids flocking rules, Denning's statistical intrusion-detection model, or the Population Stability Index threshold on its own, from data alone. The explanation is the proof.

## Domain coverage

Case studies span a deliberately wide range of fields: nuclear medicine, neuroscience and brain-machine interfaces, robotics, autonomous vehicles, drone swarms, cognitive psychology, quantum key distribution, quantitative finance, rocket engineering, network security, and data engineering, with more domains added over time until every field on the project's target list has a case study.

## The technical stack per case study

Every case study builds toward the same eight-stage pipeline:

1. A primary technique category: classical ML, unsupervised clustering, network analytics, or time series
2. A light deep learning model trained on the same data
3. A pretrained foundation model benchmark, where the domain allows one (for example, TabPFN for tabular tasks)
4. Explainable AI, with the method chosen to fit the model and the domain rather than defaulting to one method everywhere
5. GenAI-assisted feature engineering, where a free Hugging Face-hosted language model proposes candidate features, and an ablation confirms that human-engineered and AI-generated features together outperform either alone
6. GenAI-guided synthetic data augmentation, checked for fidelity by comparing model performance on original-only, original-plus-synthetic, and synthetic-only data against the same held-out test set
7. An agentic layer built with LangGraph, calling a free Hugging Face-hosted language model to turn model output into a natural-language recommendation
8. An autonomous agent variant of that same layer, with self-directed planning, real tool use, and memory of past cases, built to show what separates a scripted agentic pipeline from genuine agent autonomy

## Status

Case studies are added one at a time, each requiring explicit approval before it is built. Earlier case studies are being retrofitted with the stages above as the standard evolves.
