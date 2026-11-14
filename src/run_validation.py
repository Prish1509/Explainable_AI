"""
run_validation.py -- correctness gate for the from-scratch implementations.

Two checks, both of which must pass before any result is trusted:

  1. Our Grad-CAM (written from scratch in common.py) must match the reference
     pytorch-grad-cam implementation on the same layer and class.
  2. Our Deletion metric must rank a real saliency map above a random one --
     if it cannot tell signal from noise, it cannot rank explanation methods.

Produces: validation.csv

Run:  python src/run_validation.py
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import (init, load_sample, grad_cam, perturbation_auc,
                    OUTPUT_DIR, SEED)

N_IMAGES = 20


def main():
    model, layers, _, voc = init("resnet50")
    layer = layers[-1]

    # ---- 1. our Grad-CAM vs the reference library ----
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    _, _, x0 = load_sample(voc, 0)
    ours, class_idx = grad_cam(model, layer, x0)
    with GradCAM(model=model, target_layers=[layer]) as cam:
        reference = cam(input_tensor=x0,
                        targets=[ClassifierOutputTarget(class_idx)])[0]
    correlation = float(pearsonr(ours.ravel(), reference.ravel())[0])
    print(f"Grad-CAM vs library, Pearson correlation: {correlation:.4f} "
          f"(want > 0.95)")

    # ---- 2. our Deletion metric: real map vs random map ----
    rng = np.random.default_rng(SEED)
    real, random_ = [], []
    for i in range(N_IMAGES):
        _, _, x = load_sample(voc, i)
        heat, cls = grad_cam(model, layer, x)
        real.append(perturbation_auc(model, x, heat, cls,
                                     "deletion", "blur", 20)[0])
        random_.append(perturbation_auc(model, x, rng.random(heat.shape), cls,
                                        "deletion", "blur", 20)[0])
    real_mean, random_mean = float(np.mean(real)), float(np.mean(random_))
    win_rate = float(np.mean(np.asarray(real) < np.asarray(random_)))
    print(f"Deletion AUC (lower = more faithful): real={real_mean:.4f}, "
          f"random={random_mean:.4f}, real-map win rate={win_rate:.2f}")

    pd.DataFrame({"image": np.arange(N_IMAGES),
                  "real_deletion": real,
                  "random_deletion": random_}).to_csv(
        OUTPUT_DIR / "validation.csv", index=False)

    passed = correlation > 0.95 and real_mean < random_mean
    print("VALIDATION PASSED" if passed else "VALIDATION FAILED")


if __name__ == "__main__":
    main()
