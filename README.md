
Course project — **Explainable AI, Track 2 (Extension of a Research Paper)**.

We extend **Grad-CAM** (Selvaraju et al., ICCV 2017) with two inference-only
additions and test whether they actually help:

1. **Multi-layer fusion** — combine Grad-CAM maps from three convolutional
   depths, weighted by each layer's measured faithfulness.
2. **TTA uncertainty** — a per-pixel uncertainty map from test-time
   augmentation, showing where the explanation is unstable.

The backbone is **never trained or fine-tuned** — everything is post-hoc on a
pretrained ImageNet ResNet-50, evaluated on PASCAL VOC 2007.

## Research questions

- **H1:** Does faithfulness-weighted multi-layer fusion produce a *more
  faithful* explanation than single-layer Grad-CAM and other CAM baselines?
- **H2:** Does the TTA uncertainty map carry a meaningful reliability signal —
  are high-uncertainty regions less faithful?

## Main findings (honest summary)

- **H1 — not supported.** Over 100 VOC test images, single-layer Grad-CAM
  (ROAD 0.1027) is nominally best and our weighted fusion (0.0997) does not
  beat it; a paired Wilcoxon test gives **p = 0.35**, i.e. the methods are
  statistically comparable. All 95% confidence intervals overlap.
- **Why.** The layer sweep explains it: the deepest layer alone is by far the
  most faithful (ROAD 0.096) while shallow layers are weak (layer3 0.060,
  layer2 0.039), so fusing them in only *dilutes* the strong map.
- **Weighting does help relative to naive averaging:** weighted fusion (0.0997)
  > uniform fusion (0.0963).
- **The two faithfulness metrics disagree** — fusion ranks best on Deletion but
  not on ROAD, a caution against single-metric evaluation.
- **H2 — not supported.** The uncertainty map is spatially structured (it
  concentrates on the object), but the image-level correlation between
  uncertainty and faithfulness is not significant (Spearman r = −0.11,
  p = 0.55, N = 30).

A negative result, but a measured and explained one.

## Setup

```bash
pip install -r requirements.txt
```

Runs on a single GPU (developed on a free Kaggle T4). PASCAL VOC 2007 `test` is
downloaded automatically, or picked up from a Kaggle-mounted copy if present.

## Reproducing the experiments

```bash
python week1/run_experiments.py validate    # Week-1 correctness gate
python week1/run_experiments.py benchmark   # main 100-image benchmark -> results_large.csv, benchmark_plot.png
python week1/run_experiments.py sweep       # layer-set sensitivity -> results_sweep.csv
python week1/run_experiments.py h2          # both uncertainty experiments
python week1/run_experiments.py all         # everything
```

Random seeds are fixed (`SEED = 0`). Fusion weights are calibrated on images
100–119 and all reported results use images 0–99, so the weights are never
tuned on the evaluation set.

## Repository structure

```
week1/
  method.py            Grad-CAM implemented from scratch + model loading
  metrics.py           Deletion faithfulness metric from scratch
  road_metric.py       ROAD metric (primary, debiased faithfulness measure)
  fusion.py            Multi-layer CAMs, uniform and faithfulness-weighted fusion
  tta.py               Test-time-augmentation mean explanation + uncertainty map
  baselines.py         Grad-CAM++, Layer-CAM, Score-CAM baselines
  run_experiments.py   Reproduces every reported experiment
  results.csv          30-image benchmark (ROAD + Deletion, incl. Score-CAM)
  results_large.csv    100-image benchmark (ROAD, with CIs)
  benchmark_plot.png   Faithfulness by method with 95% confidence intervals
requirements.txt
```

## Validation

Before any experiment, the harness was checked for correctness:

- Our from-scratch Grad-CAM matches the reference `pytorch-grad-cam`
  implementation (Pearson correlation **1.000**).
- Our Deletion metric ranks a real saliency map above a random one
  (mean AUC 0.096 vs 0.102; the real map wins on 70% of images).

## LLM-use disclosure

We used an LLM assistant (Claude) to scaffold starter code, explain methods and
metrics, and help draft the report. All code was read, executed and verified by
the team before any result was trusted — specifically, our from-scratch Grad-CAM
was validated against the `pytorch-grad-cam` library and our Deletion metric
against a random-saliency control (see **Validation** above). Every number
reported in this repository and in the report comes from our own runs on our own
hardware. The experimental design decisions, the analysis and the conclusions
are our own, and we are responsible for them.

## References

- Selvaraju et al., *Grad-CAM*, ICCV 2017
- Chattopadhyay et al., *Grad-CAM++*, WACV 2018
- Jiang et al., *Layer-CAM*, IEEE TIP 2021
- Wang et al., *Score-CAM*, CVPR Workshops 2020
- Petsiuk et al., *RISE* (Deletion/Insertion), BMVC 2018
- Rong et al., *ROAD*, ICML 2022
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018
