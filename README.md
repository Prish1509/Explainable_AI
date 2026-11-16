# UML-CAM: Uncertainty-Aware Multi-Layer Grad-CAM

Course project — **Explainable AI, Track 2 (Extension of a Research Paper)**.

We extend **Grad-CAM** (Selvaraju et al., ICCV 2017) with two inference-only
additions and test rigorously whether they help:

1. **Multi-layer fusion** — combine Grad-CAM maps from three convolutional
   depths, weighted by each layer's measured faithfulness.
2. **TTA uncertainty** — a per-pixel uncertainty map from test-time
   augmentation, marking where the explanation is unstable.

The backbone is **never trained or fine-tuned**. Everything is post-hoc on
pretrained ImageNet ResNet-50 (and VGG-16), evaluated on PASCAL VOC 2007 test.

## Research questions

- **H1** — Does faithfulness-weighted multi-layer fusion produce a *more
  faithful* explanation than single-layer Grad-CAM and other CAM baselines?
- **H2** — Does the TTA uncertainty map carry a meaningful reliability signal,
  i.e. are high-uncertainty regions less faithful?

## Main findings

**Faithfulness (100 images, ROAD, higher is better).** Weighted fusion ranks
first on all three faithfulness metrics, but the margin over single-layer
Grad-CAM is not significant:

| Method | ROAD | 95% CI |
|---|---|---|
| UML-CAM (weighted) | **0.1069** | 0.0932 – 0.1207 |
| Grad-CAM | 0.1064 | 0.0934 – 0.1204 |
| Score-CAM | 0.0995 | 0.0867 – 0.1125 |
| Layer-CAM | 0.0987 | 0.0857 – 0.1130 |
| Grad-CAM++ | 0.0983 | 0.0848 – 0.1120 |
| UML-CAM (uniform) | 0.0973 | 0.0838 – 0.1112 |

- vs Grad-CAM: Wilcoxon **p = 0.83** → statistically tied.
- vs uniform fusion: Holm-adjusted **p = 0.00002** → the **weighting is what
  makes fusion viable**; naive averaging is the worst method tested.
- Same ordering on Deletion and Insertion, and on a second architecture (VGG-16).

**Why fusion cannot beat the deepest layer.** The layer sweep shows shallow
CAMs are nearly uninformative (ROAD 0.022 for layer2, 0.039 for layer3, vs
0.098 for layer4), and on localization they barely exceed a random-map
baseline. There is little for fusion to add.

**Localization (84 images).** Single-layer Grad-CAM localizes best
(EBPG 0.608 vs 0.488 for weighted fusion); fusion keeps the peak on the object
(Pointing Game 0.845, joint best) but disperses the surrounding energy. There
is **no faithfulness/localization trade-off** — the deep layer leads on both.

**H2 — supported at image level, confounded at pixel level.** Mean uncertainty
predicts faithfulness across images (Spearman **ρ = −0.434, p = 0.0005, N=60**).
At pixel level, removing the *most* uncertain pixels damages the explanation
most, because uncertainty concentrates on the salient object. An earlier null
result (ρ = −0.11, p = 0.55) was an artefact of a degenerate augmentation set
that repeated views.

**Sanity check passes.** Cascading randomization collapses map correlation from
1.0 to ≈0.18, so the explanation genuinely depends on the learned weights.

## Setup

```bash
pip install -r requirements.txt
```

Runs on a single GPU (developed on a free Kaggle T4). VOC 2007 test is detected
from a Kaggle-mounted copy if present, otherwise downloaded.

## Reproducing the experiments

```bash
python src/run_validation.py     # correctness gate (run this first)
python src/run_benchmark.py      # 100-image benchmark, CIs, Holm tests, sweeps
python src/run_localization.py   # Pointing Game / IoU / EBPG + qualitative grid
python src/run_vgg16.py          # second architecture
python src/run_uncertainty.py    # TTA uncertainty, sparsification, H2
python src/run_sanity.py         # cascading randomization
```

Outputs are written to `results/` (override with `UMLCAM_OUT=<dir>`). Seeds are
fixed (`SEED = 0`). Fusion weights are calibrated on images 100–119 and all
results are reported on images 0–99, so weights are never tuned on the
evaluation set.

## Repository structure

- **`week1/`** — our initial from-scratch implementation and its early results.
  Kept as a record of the first milestone; it uses different preprocessing and
  is **superseded by `src/`**.
- **`src/`** — the final pipeline. **All reported results come from here.**
- **`results/`** — every CSV and figure, each regenerable by a script in `src/`.
- **`notebooks/`** — thin Kaggle runners that clone the repo, run `src/`, and
  push results back.

## Validation

`src/run_validation.py` checks the from-scratch code before any result is used:

- Our Grad-CAM vs the reference `pytorch-grad-cam` implementation:
  Pearson **r = 1.0000**.
- Our Deletion metric ranks a real saliency map above a random one
  (mean AUC 0.0826 vs 0.0889). Note the per-image win rate is only 0.50,
  which is itself evidence that Deletion is weakly discriminative — one reason
  we treat ROAD as the primary metric.

## Division of work

- **Prisha** — Grad-CAM implementation, multi-layer fusion, CAM baselines,
  main benchmark and sweeps, localization, VGG-16.
- **Parvathy** — Deletion and ROAD metrics, TTA uncertainty, sparsification and
  H2 experiments, cascading randomization sanity check.

Both members reviewed each other's code and results.

## LLM-use disclosure

We used an LLM assistant (Claude) throughout this project: to scaffold starter
code, explain methods and metrics, suggest experimental designs, debug runtime
errors, help interpret results, and draft this report and README. The research
questions, the choice of experiments to run, and the conclusions drawn are ours.

How the work was verified:

1. **Reference check** — our from-scratch Grad-CAM was validated against
   `pytorch-grad-cam` (r = 1.0000) and our Deletion metric against a
   random-saliency control, via `src/run_validation.py`.
2. **Every number is ours** — all reported values were produced by running the
   committed code in our own Kaggle sessions, and `src/` reproduces the values
   we first obtained interactively (ROAD 0.1068 → 0.1069; H2 ρ −0.428 → −0.434).
3. **Errors were found and corrected** — a code review showed the initial TTA
   set repeated views (T = 8 gave only 6 distinct augmentations); replacing it
   with 16 distinct augmentations changed H2 from non-significant to
   significant. We report both outcomes.

## References

- Selvaraju et al., *Grad-CAM*, ICCV 2017
- Chattopadhyay et al., *Grad-CAM++*, WACV 2018
- Jiang et al., *Layer-CAM*, IEEE TIP 2021
- Wang et al., *Score-CAM*, CVPR Workshops 2020
- Petsiuk et al., *RISE* (Deletion/Insertion), BMVC 2018
- Rong et al., *ROAD*, ICML 2022
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018
- Binder et al., *Shortcomings of Top-Down Randomization-Based Sanity Checks*,
  CVPR 2023
