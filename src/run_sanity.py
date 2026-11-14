"""
run_sanity.py -- cascading model-parameter randomization (Adebayo et al., 2018).
(Parvathy / Member 2)

Randomizes the network top-down, one block at a time, and measures how much the
Grad-CAM map changes against the pretrained map. A method that genuinely depends
on the learned weights should show the correlation collapse. We treat this as a
pass/fail control, not a way to rank methods -- Binder et al. (CVPR 2023) show
top-down randomization is a weak ranking criterion.

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
        cors = []
        for x, heat0, cls in originals:
            heat, _ = grad_cam(randomized, randomized.layer4[-1], x, cls)
            cors.append(spearmanr(heat0.ravel(), heat.ravel())[0])
        rows.append({"randomized_upto": block,
                     "mean_spearman": float(np.nanmean(cors))})
        print(f"{block:8s} mean Spearman vs original: {np.nanmean(cors):.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "cascading_randomization.csv", index=False)
    print("\nA large drop from 1.0 means the explanation depends on the "
          "learned weights (sanity check passes).")


if __name__ == "__main__":
    main()
