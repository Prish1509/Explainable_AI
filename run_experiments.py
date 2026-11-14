"""
run_experiments.py -- reproduces every experiment reported in the project.

Usage (from the repository root):
    python week1/run_experiments.py validate    # Week-1 correctness gate
    python week1/run_experiments.py benchmark   # main 100-image benchmark + plot
    python week1/run_experiments.py sweep       # layer-set sensitivity analysis
    python week1/run_experiments.py h2          # uncertainty experiments (H2)
    python week1/run_experiments.py all

Everything is inference-only: the ImageNet-pretrained backbone is never trained
or fine-tuned. Results are written next to this file as CSV/PNG.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from method import load_model, grad_cam                      # noqa: E402
from metrics import deletion_auc                             # noqa: E402
from road_metric import road_score                           # noqa: E402
from fusion import multi_layer_cams, fuse_uniform, fuse_weighted  # noqa: E402
from baselines import cam_method                             # noqa: E402
from tta import tta_mean_uncertainty                         # noqa: E402
from torchvision.datasets import VOCDetection                # noqa: E402

SEED = 0
# Image index ranges. Calibration and evaluation sets are disjoint so the
# fusion weights are never tuned on the images we report results for.
CALIB_IDX = range(100, 120)
EVAL_N = 100
SWEEP_N = 20
H2_N = 20
H2_CORR_N = 30


# ----------------------------------------------------------------- utilities
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_voc():
    """Load VOC2007 test: prefer a Kaggle-mounted copy, else download."""
    hits = glob.glob("/kaggle/input/**/VOCdevkit", recursive=True)
    if hits:
        return VOCDetection(os.path.dirname(hits[0]), year="2007",
                            image_set="test", download=False)
    return VOCDetection(os.path.join(HERE, "..", "data"), year="2007",
                        image_set="test", download=True)


def load_image(voc, preprocess, i, device):
    img, _ = voc[i]
    return preprocess(img.convert("RGB")).unsqueeze(0).to(device)


def setup():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    model, preprocess, layers = load_model(device)
    return model, preprocess, layers, get_voc(), device


def calibrate_weights(model, layers, voc, preprocess, device):
    """Per-layer faithfulness weights: w_l ∝ 1 / DeletionAUC_l, on CALIB_IDX."""
    per_layer = [[], [], []]
    for i in tqdm(CALIB_IDX, desc="calibrating weights"):
        x = load_image(voc, preprocess, i, device)
        cams, cls = multi_layer_cams(model, layers, x)
        for l in range(3):
            per_layer[l].append(deletion_auc(model, x, cams[l], cls, steps=20)[0])
    w = 1.0 / np.array([np.mean(d) for d in per_layer])
    return w / w.sum()


# ---------------------------------------------------------------- experiments
def run_validate():
    """Week-1 gate: our Grad-CAM must match the reference library, and our
    Deletion metric must rank a real map above a random one."""
    model, preprocess, layers, voc, device = setup()
    x = load_image(voc, preprocess, 0, device)

    ours, cls = grad_cam(model, layers[-1], x)
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    with GradCAM(model=model, target_layers=[layers[-1]]) as cam:
        ref = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    r = pearsonr(ours.ravel(), ref.ravel())[0]
    print(f"[Grad-CAM]  corr(ours, library) = {r:.4f}   (want > 0.95)")

    rng = np.random.default_rng(SEED)
    good, rand = [], []
    for i in tqdm(range(20), desc="deletion check"):
        xi = load_image(voc, preprocess, i, device)
        h, c = grad_cam(model, layers[-1], xi)
        good.append(deletion_auc(model, xi, h, c)[0])
        rand.append(deletion_auc(model, xi, rng.random(h.shape), c)[0])
    print(f"[Deletion]  good {np.mean(good):.4f} vs random {np.mean(rand):.4f} "
          f"(want good < random); win-rate "
          f"{np.mean(np.array(good) < np.array(rand)):.2f}")
    print("GATE PASSED" if r > 0.95 and np.mean(good) < np.mean(rand) else "GATE FAILED")


def run_benchmark():
    """Main benchmark: all methods on ROAD over EVAL_N images, with 95% CIs
    and a paired significance test. Writes results_large.csv + benchmark_plot.png."""
    model, preprocess, layers, voc, device = setup()
    w = calibrate_weights(model, layers, voc, preprocess, device)
    print("fusion weights [layer2, layer3, layer4]:", np.round(w, 3))

    rows = []
    for i in tqdm(range(EVAL_N), desc="benchmark"):
        x = load_image(voc, preprocess, i, device)
        cams, cls = multi_layer_cams(model, layers, x)
        maps = {
            "gradcam":      cams[-1],
            "gradcam++":    cam_method("gradcam++", model, layers[-1], x, cls)[0],
            "layercam":     cam_method("layercam",  model, layers[-1], x, cls)[0],
            "uml_uniform":  fuse_uniform(cams),
            "uml_weighted": fuse_weighted(cams, w),
        }
        for name, m in maps.items():
            rows.append({"img": i, "method": name,
                         "road": road_score(model, x, m, cls)})

    df = pd.DataFrame(rows)
    g = df.groupby("method")["road"]
    summ = g.agg(["mean", "std", "count"])
    summ["ci95"] = 1.96 * summ["std"] / np.sqrt(summ["count"])
    summ = summ.sort_values("mean", ascending=False)
    print("\n", summ.round(4))

    df.to_csv(os.path.join(HERE, "results_large.csv"), index=False)

    pivot = df.pivot(index="img", columns="method", values="road")
    best = summ.index[0]
    if best != "uml_weighted":
        p = wilcoxon(pivot[best], pivot["uml_weighted"]).pvalue
        print(f"\n{best} vs uml_weighted: Wilcoxon p={p:.3f} "
              f"({'significant' if p < 0.05 else 'NOT significant'})")

    order = summ.index
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(order)), summ["mean"][order], yerr=summ["ci95"][order],
            capsize=4, color="steelblue")
    plt.xticks(range(len(order)), order, rotation=45, ha="right")
    plt.ylabel("ROAD (higher = better)")
    plt.title(f"Faithfulness by method ({EVAL_N} images, 95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "benchmark_plot.png"), dpi=120)
    print("wrote results_large.csv and benchmark_plot.png")


def run_sweep():
    """Sensitivity analysis: which layers should be fused?"""
    model, preprocess, layers, voc, device = setup()
    subsets = {"layer4 only": [2], "layer3+4": [1, 2], "layer2+3+4": [0, 1, 2],
               "layer3 only": [1], "layer2 only": [0]}
    res = {k: [] for k in subsets}
    for i in tqdm(range(SWEEP_N), desc="layer sweep"):
        x = load_image(voc, preprocess, i, device)
        cams, cls = multi_layer_cams(model, layers, x)
        for k, idx in subsets.items():
            res[k].append(road_score(model, x, fuse_uniform(cams[np.array(idx)]), cls))
    print("\nROAD by layer set (higher = better):")
    out = []
    for k in subsets:
        print(f"  {k:12s}: {np.mean(res[k]):.4f}")
        out.append({"layer_set": k, "road": np.mean(res[k])})
    pd.DataFrame(out).to_csv(os.path.join(HERE, "results_sweep.csv"), index=False)


def run_h2():
    """H2: is the TTA uncertainty map a meaningful reliability signal?"""
    model, preprocess, layers, voc, device = setup()

    # (a) pixel-level sparsification
    def prune(mean, var, mode, frac=0.3, seed=SEED):
        m = mean.copy().ravel(); v = var.ravel(); k = int(len(v) * frac)
        if mode == "most_uncertain":
            idx = np.argsort(v)[::-1][:k]
        elif mode == "least_uncertain":
            idx = np.argsort(v)[:k]
        else:
            idx = np.random.default_rng(seed).choice(len(v), k, replace=False)
        m[idx] = 0.0
        return m.reshape(mean.shape)

    res = {"most_uncertain": [], "random": [], "least_uncertain": []}
    for i in tqdm(range(H2_N), desc="H2 sparsification"):
        x = load_image(voc, preprocess, i, device)
        mean, var, cls = tta_mean_uncertainty(model, layers[-1], x, T=8)
        for mode in res:
            res[mode].append(road_score(model, x, prune(mean, var, mode), cls))
    print("\nROAD after removing 30% of pixels (higher = stayed faithful):")
    for mode in ["most_uncertain", "random", "least_uncertain"]:
        print(f"  removed {mode:16s}: {np.mean(res[mode]):.4f}")

    # (b) image-level correlation
    unc, faith = [], []
    for i in tqdm(range(H2_CORR_N), desc="H2 correlation"):
        x = load_image(voc, preprocess, i, device)
        mean, var, cls = tta_mean_uncertainty(model, layers[-1], x, T=8)
        unc.append(float(var.mean()))
        faith.append(road_score(model, x, mean, cls))
    r, p = spearmanr(unc, faith)
    print(f"\ncorr(image uncertainty, faithfulness) = {r:.3f} (p={p:.3f}, "
          f"N={H2_CORR_N})")
    print("Not significant at p>0.05 => no reliable evidence that TTA "
          "uncertainty predicts faithfulness.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment",
                    choices=["validate", "benchmark", "sweep", "h2", "all"])
    args = ap.parse_args()
    jobs = {"validate": run_validate, "benchmark": run_benchmark,
            "sweep": run_sweep, "h2": run_h2}
    if args.experiment == "all":
        for name, fn in jobs.items():
            print(f"\n{'='*60}\n{name}\n{'='*60}")
            fn()
    else:
        jobs[args.experiment]()


if __name__ == "__main__":
    main()
