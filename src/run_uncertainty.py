"""
run_uncertainty.py -- test-time-augmentation uncertainty and hypothesis H2.
(Parvathy / Member 2)

The augmentation set is 16 DISTINCT views (flips x rotations x brightness).
An earlier version cycled t%2 and t%3, which repeats after 6 views, so T=8 was
really only 6 distinct augmentations; that degenerate set was too weak to
detect the H2 effect.

Produces:
    sparsification_curves.csv / .png   pixel-level H2 (10-90% pruning) + AUSE
    h2_correlation.csv                 image-level H2 (Spearman, N=60)
    tta_count_sweep.csv                how many augmentations are needed

Run:  python src/run_uncertainty.py
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from common import (init, load_sample, grad_cam, road_score,
                    IMAGENET_MEAN, IMAGENET_STD, OUTPUT_DIR, SEED)

N_SPARSE = 15
N_CORRELATION = 60
N_TTA_SWEEP = 10


def brightness_adjust_normalized(x, factor):
    """Brightness scaling applied in normalized space."""
    mean = IMAGENET_MEAN.to(x.device, x.dtype).unsqueeze(0)
    std = IMAGENET_STD.to(x.device, x.dtype).unsqueeze(0)
    return factor * x + (factor - 1.0) * mean / std


# 16 distinct augmentations, ordered mild -> strong.
TTA_OPS = [
    {"flip": False, "angle": 0,   "bright": 1.00},
    {"flip": True,  "angle": 0,   "bright": 1.00},
    {"flip": False, "angle": 0,   "bright": 0.90},
    {"flip": False, "angle": 0,   "bright": 1.10},
    {"flip": False, "angle": 5,   "bright": 1.00},
    {"flip": False, "angle": -5,  "bright": 1.00},
    {"flip": True,  "angle": 0,   "bright": 0.90},
    {"flip": True,  "angle": 0,   "bright": 1.10},
    {"flip": True,  "angle": 5,   "bright": 1.00},
    {"flip": True,  "angle": -5,  "bright": 1.00},
    {"flip": False, "angle": 10,  "bright": 1.00},
    {"flip": False, "angle": -10, "bright": 1.00},
    {"flip": False, "angle": 0,   "bright": 0.95},
    {"flip": False, "angle": 0,   "bright": 1.05},
    {"flip": True,  "angle": 10,  "bright": 1.00},
    {"flip": True,  "angle": -10, "bright": 1.00},
]


def tta_mean_uncertainty(model, layer, x, n_aug=8, class_idx=None):
    """Mean map = explanation, per-pixel variance = uncertainty.

    Geometric transforms are inverted on the heat-map before aggregating, so
    all maps are pixel-aligned and the variance measures disagreement, not
    misalignment.
    """
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x)[0].argmax().item())
    if n_aug > len(TTA_OPS):
        raise ValueError(f"max {len(TTA_OPS)} distinct augmentations")
    maps = []
    for op in TTA_OPS[:n_aug]:
        xi = (brightness_adjust_normalized(x, op["bright"])
              if op["bright"] != 1.0 else x.clone())
        if op["angle"]:
            xi = TF.rotate(xi, float(op["angle"]))
        if op["flip"]:
            xi = torch.flip(xi, dims=[3])
        heat, _ = grad_cam(model, layer, xi, class_idx)
        if op["flip"]:
            heat = heat[:, ::-1]
        if op["angle"]:
            t = torch.from_numpy(np.ascontiguousarray(heat)).float()[None, None]
            heat = TF.rotate(t, -float(op["angle"]))[0, 0].numpy()
        maps.append(np.ascontiguousarray(heat))
    maps = np.stack(maps, 0)
    return maps.mean(0), maps.var(0), class_idx


def prune(mean_map, var, mode, frac, seed=SEED):
    """Zero out a fraction of the explanation, chosen by uncertainty."""
    flat = mean_map.reshape(-1).copy()
    v = var.reshape(-1)
    k = max(1, int(len(v) * frac))
    if mode == "most_uncertain":
        idx = np.argsort(v)[::-1][:k]
    elif mode == "least_uncertain":
        idx = np.argsort(v)[:k]
    else:
        idx = np.random.default_rng(seed).choice(len(v), k, replace=False)
    flat[idx] = 0.0
    return flat.reshape(mean_map.shape)


def main():
    model, layers, _, voc = init("resnet50")
    layer = layers[-1]

    # ---- pixel-level H2: sparsification curves ----
    fractions = np.arange(0.1, 0.91, 0.1)
    rows = []
    for i in range(N_SPARSE):
        _, _, x = load_sample(voc, i)
        mean_map, var, cls = tta_mean_uncertainty(model, layer, x, n_aug=8)
        for f in fractions:
            for mode in ["most_uncertain", "random", "least_uncertain"]:
                rows.append({"image": i, "fraction": float(f), "mode": mode,
                             "road": road_score(
                                 model, x, prune(mean_map, var, mode, f), cls)})
    sp = pd.DataFrame(rows)
    sp.to_csv(OUTPUT_DIR / "sparsification_curves.csv", index=False)
    curve = sp.groupby(["mode", "fraction"]).road.mean().unstack(0)
    print(curve.round(4))

    # Area between the curves. H2 predicts removing the MOST uncertain pixels
    # preserves faithfulness best, i.e. most_uncertain above least_uncertain.
    gap = float(np.trapz(curve["most_uncertain"], fractions)
                - np.trapz(curve["least_uncertain"], fractions))
    print(f"\narea(most - least) = {gap:.4f}")
    print("positive => high-uncertainty pixels are the removable ones (H2);")
    print("negative => high-uncertainty pixels are the IMPORTANT ones "
          "(uncertainty tracks saliency).")

    plt.figure(figsize=(6, 4))
    for mode in ["most_uncertain", "random", "least_uncertain"]:
        plt.plot(fractions, curve[mode], marker="o", label=mode)
    plt.xlabel("fraction of pixels pruned")
    plt.ylabel("ROAD (higher = better remaining attribution)")
    plt.title("Sparsification curves")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sparsification_curves.png", dpi=150)

    # ---- image-level H2: does uncertainty predict faithfulness? ----
    unc, faith = [], []
    for i in range(N_CORRELATION):
        _, _, x = load_sample(voc, i)
        mean_map, var, cls = tta_mean_uncertainty(model, layer, x, n_aug=8)
        unc.append(float(var.mean()))
        faith.append(road_score(model, x, mean_map, cls))
    rho, p = spearmanr(unc, faith)
    pd.DataFrame({"uncertainty": unc, "road": faith}).to_csv(
        OUTPUT_DIR / "h2_correlation.csv", index=False)
    print(f"\nSpearman(mean uncertainty, ROAD) rho={rho:.3f} p={p:.4f} "
          f"N={N_CORRELATION} ->",
          "significant" if p < 0.05 else "not significant")

    # ---- how many augmentations are needed? ----
    rows = []
    for i in range(N_TTA_SWEEP):
        _, _, x = load_sample(voc, i)
        with torch.no_grad():
            cls = int(model(x)[0].argmax().item())
        for n_aug in [2, 4, 8, 16]:
            m, v, _ = tta_mean_uncertainty(model, layer, x, n_aug=n_aug,
                                           class_idx=cls)
            rows.append({"image": i, "n_aug": n_aug,
                         "road": road_score(model, x, m, cls),
                         "mean_var": float(v.mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUTPUT_DIR / "tta_count_sweep.csv", index=False)
    print("\n", t.groupby("n_aug")[["road", "mean_var"]].mean().round(4))


if __name__ == "__main__":
    main()
