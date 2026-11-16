"""
run_sanity.py -- cascading model-parameter randomization (Adebayo et al., 2018).

Randomizes the network top-down, one block at a time, and measures how much the
Grad-CAM map changes against the pretrained map. A method that genuinely
depends on the learned weights should show the correlation collapse. We treat
this as a pass/fail control, not a way to rank methods -- Binder et al. (CVPR
2023) show top-down randomization is a weak ranking criterion.

A note on degenerate maps: once weights are randomized, the CAM for an image
sometimes comes out completely flat (ReLU zeroes everything). Rank correlation
is undefined against a constant array, so those images are excluded from the
mean and counted separately rather than silently dropped. A flat map is itself
evidence the explanation collapsed, so the count is reported alongside.

Produces: cascading_randomization.csv

Run:  python src/run_sanity.py
"""
import copy

import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import spearmanr

from common import init, load_sample, grad_cam, OUTPUT_DIR

N_IMAGES = 5
BLOCKS = ["fc", "layer4", "layer3", "layer2", "layer1"]


def safe_spearman(a, b):
    """Rank correlation, or None if either map is constant (undefined)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    rho = spearmanr(a, b)[0]
    return None if np.isnan(rho) else float(rho)


def main():
    model, layers, _, voc = init("resnet50")

    originals = []
    for i in range(N_IMAGES):
        _, _, x = load_sample(voc, i)
        heat, cls = grad_cam(model, layers[-1], x)
        originals.append((x, heat, cls))

    randomized = copy.deepcopy(model).eval()
    rows = []
    for block in BLOCKS:
        for module in getattr(randomized, block).modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.reset_parameters()

        valid, degenerate = [], 0
        for x, heat0, cls in originals:
            heat, _ = grad_cam(randomized, randomized.layer4[-1], x, cls)
            rho = safe_spearman(heat0.ravel(), heat.ravel())
            if rho is None:
                degenerate += 1
            else:
                valid.append(rho)

        mean_rho = float(np.mean(valid)) if valid else float("nan")
        rows.append({"randomized_upto": block,
                     "mean_spearman": mean_rho,
                     "n_valid": len(valid),
                     "n_degenerate": degenerate,
                     "n_images": N_IMAGES})
        note = f" ({degenerate} flat map(s) excluded)" if degenerate else ""
        print(f"{block:8s} mean Spearman vs original: {mean_rho:.4f}  "
              f"[over {len(valid)} of {N_IMAGES} images]{note}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "cascading_randomization.csv", index=False)

    total_degenerate = int(out.n_degenerate.sum())
    print("\nA large drop from 1.0 means the explanation depends on the "
          "learned weights (sanity check passes).")
    if total_degenerate:
        print(f"{total_degenerate} map(s) were completely flat after "
              f"randomization and are excluded from the means above; a flat "
              f"map is itself a collapsed explanation.")


if __name__ == "__main__":
    main()
