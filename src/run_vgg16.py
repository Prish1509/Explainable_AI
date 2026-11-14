"""
run_vgg16.py -- repeat the faithfulness comparison on a second architecture.
(Prisha / Member 1)

The proposal names VGG-16 as a second backbone. This checks whether the
conclusions drawn on ResNet-50 are architecture-specific. Fusion weights are
recalibrated on the same disjoint calibration indices.

Produces: vgg16_results.csv

Run:  python src/run_vgg16.py
"""
import numpy as np
import pandas as pd
import torch

from common import (init, load_sample, multi_layer_cams, fuse_uniform,
                    fuse_weighted, library_cam, road_score,
                    calibrate_fusion_weights, OUTPUT_DIR)

N_IMAGES = 20


def main():
    model, layers, _, voc = init("vgg16")
    weights, per_layer = calibrate_fusion_weights(model, layers, voc)
    print("per-layer deletion:", np.round(per_layer, 4))
    print("fusion weights:", np.round(weights, 4))

    rows = []
    for i in range(N_IMAGES):
        _, _, x = load_sample(voc, i)
        cams, cls = multi_layer_cams(model, layers, x)
        maps = {
            "gradcam": cams[-1],
            "gradcam++": library_cam("gradcam++", model, layers[-1], x, cls),
            "layercam": library_cam("layercam", model, layers[-1], x, cls),
            "uml_uniform": fuse_uniform(cams),
            "uml_weighted": fuse_weighted(cams, weights),
        }
        for name, heat in maps.items():
            rows.append({"image": i, "architecture": "vgg16", "method": name,
                         "road": road_score(model, x, heat, cls)})

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "vgg16_results.csv", index=False)
    print(out.groupby("method").road.mean()
          .sort_values(ascending=False).round(4))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
