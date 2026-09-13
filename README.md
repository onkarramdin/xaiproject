# Interpretable AI for Diabetic Retinopathy Detection

Verifying post-hoc saliency explanations (Grad-CAM++, Integrated Gradients) against expert-annotated clinical ground truth, using a chance-baselined verification protocol.

MSc Data Science thesis project — Onkar Ramdin, Liverpool John Moores University / IIIT Bengaluru.

## Overview

Deep learning models can now grade diabetic retinopathy (DR) at expert level, but they operate as black boxes. Explainable AI (XAI) methods like Grad-CAM++ and Integrated Gradients promise visual heatmaps to justify predictions — but these heatmaps are typically accepted at face value, with no quantitative check against real disease lesions and no baseline for what a *random* attention map would score.

This project builds an EfficientNet-B0 classifier for DR severity grading on the [IDRiD](https://idrid.grand-challenge.org/Data/) dataset, generates saliency maps with two different attribution methods, and verifies them against pixel-level expert lesion annotations using an empirical Monte Carlo chance floor. The resulting **Attribution Lift** metric (real overlap ÷ chance-floor overlap) answers three research questions:

- **RQ1 — Do saliency maps find the lesions?** Both methods beat chance, but only modestly (Grad-CAM++ 1.46×, Integrated Gradients 1.39×) — coarse regional cues, not pixel-precise segmentation.
- **RQ2 — Gaze drift.** Grad-CAM++ attends **1.69× more strongly** to the healthy optic disc than to actual lesions (2.70× chance vs 1.60× chance) — a shortcut-learning signature that would be invisible without the chance-baselined comparison.
- **RQ3 — Can preprocessing fix it?** CLAHE contrast enhancement significantly reduces drift (p < 0.0001) with no cost to lesion alignment. Directly masking the optic disc backfires — drift *increases* to 2.46× because the synthetic mask edge creates a new high-contrast artifact.

## Repository Structure

```
xaiproject/
├── notebooks/
│   └── xaiproject.ipynb    # End-to-end pipeline: preprocessing, training, XAI, verification, ablation
├── src/
│   ├── __init__.py
│   └── dataset.py          # Scaffold for factoring out the dataset/loader classes (WIP)
├── data/                   # IDRiD dataset — not tracked in git, see Data Setup below
├── requirements.txt
└── README.md
```

All pipeline logic currently lives in `notebooks/xaiproject.ipynb`, structured as seven self-contained cells (each rebuilds its own state, so any cell can be re-run independently after a kernel restart):

| Cell | Contents |
|---|---|
| 1–2 | Environment setup, dependency install |
| 3 | Main pipeline — data ingestion, EfficientNet-B0 training, Grad-CAM++/Integrated Gradients generation, chance-baselined IoU/DSC verification (RQ1) |
| 4 | Statistical significance testing (Wilcoxon signed-rank) for the RQ1/gaze-drift results |
| 5 | RQ3 ablation — three preprocessing arms (CLAHE+FOV / FOV-only / +optic-disc-masking), trained and verified independently |
| 6 | Paired statistical tests across the three arms + visual confirmation of the Arm C artifact |
| 7 | Arm C mechanism diagnostic — isolates whether the drift increase comes from the masked region itself or residual uncovered tissue |

## Data Setup

Download the IDRiD dataset from the [official source](https://idrid.grand-challenge.org/Data/) (or the [Kaggle mirror](https://www.kaggle.com/datasets/onkarramdin/idrid-dataset) used for this project) and place it under `data/` so the structure looks like:

```
data/
├── A. Segmentation/A. Segmentation/...
├── B. Disease Grading/B. Disease Grading/...
└── C. Localization/C. Localization/...   (not used by this pipeline)
```

The notebook reads two configurable paths, both overridable via environment variables:

```bash
export IDRID_ROOT=data       # defaults to "data/idrid-dataset" — set this if your data sits directly under data/, as above
export OUTPUT_DIR=outputs    # checkpoints, CSVs, and figures are written here (created automatically)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter notebook notebooks/xaiproject.ipynb
```

A CUDA-capable GPU is strongly recommended for training (the original runs used a Kaggle T4); the notebook falls back to CPU automatically if none is available, but training and Integrated Gradients generation will be slow.

## Results Summary

| Metric | Value |
|---|---|
| Macro ROC-AUC | 0.8463 |
| Exact 5-class accuracy | 47.4% |
| Macro F1-score | 0.409 |
| Grad-CAM++ Attribution Lift (lesions) | 1.46× |
| Integrated Gradients Attribution Lift (lesions) | 1.39× |
| Grad-CAM++ Attribution Lift (optic disc) | 2.70× |
| Gaze drift (optic disc lift ÷ lesion lift) | 1.69× |
| CLAHE effect on drift (Arm A vs Arm B) | Significant reduction, p < 0.0001 |
| Optic-disc masking (Arm C) | Backfires — drift rises to 2.46× |

The modest exact-accuracy figure reflects severe class imbalance in IDRiD (Mild NPDR: n = 25 of 516 images) rather than poor ranking ability — the ROC-AUC indicates the classifier discriminates severity well.

## Key Contributions

- **Methodological** — a reusable, chance-baselined verification protocol (20-sample Monte Carlo floor → Attribution Lift) that makes raw IoU/DSC scores on sparse medical lesions interpretable, applicable beyond diabetic retinopathy.
- **Empirical** — chance-baselined quantitative evidence of gaze drift in a DR classifier, giving a concrete 1.69× figure rather than a qualitative concern.
- **Practical** — discovery of a synthetic edge halo artifact: naively masking a confounding region can introduce a new, stronger one.

## Limitations

- Single-center dataset (IDRiD): 516 grading / 81 segmentation images, with real Stage 1 (Mild NPDR) sparsity (n = 25).
- Concept Bottleneck Models were excluded from training due to label sparsity, not evaluated empirically.
- Fixed 75th-percentile binarisation threshold and Grad-CAM++'s inherent 16×16 feature-grid resolution limit spatial precision.

See the notebook's final cells and the accompanying thesis for the full discussion, future work, and statistical detail.

## Author

**Onkar Ramdin** — MSc Data Science, Liverpool John Moores University / IIIT Bengaluru
