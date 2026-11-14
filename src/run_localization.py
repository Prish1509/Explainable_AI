"""
run_localization.py -- localization (Pointing Game, IoU, Energy-Based Pointing
Game) for every CAM method, plus the qualitative figure grid.
(Prisha / Member 1)

VOC's 20 classes do not map one-to-one onto ImageNet's 1000. We use an explicit
crosswalk of direct category aliases: for each image we explain the
highest-scoring mapped ImageNet class present, and score only that class's VOC
boxes. Images with no mapped class are skipped, so this is a restricted
localization check rather than a full VOC benchmark.

Produces:
    localization_by_method.csv
    qualitative_grid.png

Run:  python src/run_localization.py
"""
import re

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (init, load_sample, multi_layer_cams, fuse_uniform,
                    fuse_weighted, library_cam, calibrate_fusion_weights,
                    OUTPUT_DIR, N_LOCALIZATION)

VOC_TO_IMAGENET_HINTS = {
    "aeroplane": ["airliner", "warplane"],
    "bicycle": ["mountain bike", "bicycle-built-for-two"],
    "boat": ["speedboat", "canoe", "lifeboat", "gondola", "schooner"],
    "bottle": ["beer bottle", "water bottle", "wine bottle", "pop bottle"],
    "bus": ["minibus", "school bus", "trolleybus"],
    "car": ["sports car", "racer", "convertible", "jeep", "limousine", "cab"],
    "cat": ["tabby", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat"],
    "chair": ["rocking chair", "folding chair", "barber chair", "throne"],
    "cow": ["ox", "water buffalo", "bison"],
    "diningtable": ["dining table"],
    "dog": ["golden retriever", "Labrador retriever", "German shepherd",
            "poodle", "beagle", "boxer"],
    "horse": ["sorrel"],
    "motorbike": ["moped", "motor scooter"],
    "pottedplant": ["flowerpot"],
    "sheep": ["ram"],
    "sofa": ["studio couch"],
    "train": ["bullet train", "electric locomotive", "steam locomotive",
              "freight car"],
    "tvmonitor": ["television", "monitor", "screen"],
}


def norm_label(v):
    return re.sub(r"[^a-z0-9]+", " ", v.lower()).strip()


def imagenet_alias_indices(categories, hints):
    out = {}
    for voc_name, aliases in hints.items():
        want = {norm_label(a) for a in aliases}
        hit = [i for i, c in enumerate(categories)
               if any(norm_label(p) in want for p in c.split(","))]
        if hit:
            out[voc_name] = sorted(set(hit))
    return out


def voc_objects(target):
    objs = target.get("annotation", {}).get("object", [])
    if isinstance(objs, dict):
        objs = [objs]
    return [{"name": o["name"].strip().lower(),
             "box": [float(o["bndbox"][k])
                     for k in ("xmin", "ymin", "xmax", "ymax")]}
            for o in objs if "bndbox" in o]


def box_mask(boxes, original_size, out=224):
    """Map VOC boxes into the 224x224 frame (PREPROCESS is a direct resize)."""
    W, H = original_size
    mask = np.zeros((out, out), dtype=bool)
    sx, sy = out / W, out / H
    for x1, y1, x2, y2 in boxes:
        l = int(np.clip(np.floor((x1 - 1) * sx), 0, out - 1))
        t = int(np.clip(np.floor((y1 - 1) * sy), 0, out - 1))
        r = int(np.clip(np.ceil(x2 * sx), 1, out))
        b = int(np.clip(np.ceil(y2 * sy), 1, out))
        mask[t:b, l:r] = True
    return mask


def main():
    model, layers, categories, voc = init("resnet50")
    weights, _ = calibrate_fusion_weights(model, layers, voc)
    print("fusion weights:", np.round(weights, 4))

    class_indices = imagenet_alias_indices(categories, VOC_TO_IMAGENET_HINTS)
    rows = []
    for i in range(min(N_LOCALIZATION, len(voc))):
        image, target, x = load_sample(voc, i)
        choices = {}
        for o in voc_objects(target):
            if o["name"] in class_indices:
                choices.setdefault(o["name"], []).append(o["box"])
        cand = [(n, ii, b) for n, b in choices.items() for ii in class_indices[n]]
        if not cand:
            continue
        with torch.no_grad():
            logits = model(x)[0]
        voc_name, class_idx, boxes = max(
            cand, key=lambda t: float(logits[t[1]].item()))
        gt = box_mask(boxes, image.size)
        cams, _ = multi_layer_cams(model, layers, x, class_idx)
        maps = {
            "layer2": cams[0], "layer3": cams[1], "gradcam": cams[-1],
            "gradcam++": library_cam("gradcam++", model, layers[-1], x, class_idx),
            "layercam": library_cam("layercam", model, layers[-1], x, class_idx),
            "uml_uniform": fuse_uniform(cams),
            "uml_weighted": fuse_weighted(cams, weights),
        }
        for name, heat in maps.items():
            py, px = np.unravel_index(np.argmax(heat), heat.shape)
            pred = heat >= np.quantile(heat, 0.80)
            union = np.logical_or(pred, gt).sum()
            s = np.maximum(heat, 0)
            tot = s.sum()
            rows.append({
                "image": i, "voc_class": voc_name, "method": name,
                "pointing_game": int(bool(gt[py, px])),
                "iou": float(np.logical_and(pred, gt).sum() / union) if union else 0.0,
                "ebpg": float(s[gt].sum() / tot) if tot > 0 else np.nan,
                "box_area": float(gt.mean()),
            })

    loc = pd.DataFrame(rows)
    loc.to_csv(OUTPUT_DIR / "localization_by_method.csv", index=False)
    print("eligible images:", loc.image.nunique())
    print(loc.groupby("method")[["pointing_game", "iou", "ebpg"]]
          .mean().round(4).sort_values("ebpg", ascending=False))
    print("\nrandom-map EBPG baseline (mean box area):",
          round(loc.box_area.mean(), 4))

    # ---- qualitative grid ----
    idxs = [0, 1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(idxs), 6, figsize=(16, 2.7 * len(idxs)))
    for r, i in enumerate(idxs):
        image, _, x = load_sample(voc, i)
        cams, _ = multi_layer_cams(model, layers, x)
        rgb = np.asarray(image.resize((224, 224)))
        panels = [("image", None), ("layer2", cams[0]), ("layer3", cams[1]),
                  ("layer4 / Grad-CAM", cams[2]),
                  ("uniform fusion", fuse_uniform(cams)),
                  ("weighted fusion", fuse_weighted(cams, weights))]
        for c, (title, heat) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(rgb)
            if heat is not None:
                ax.imshow(heat, cmap="jet", alpha=0.5)
            if r == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "qualitative_grid.png", dpi=140)
    print("wrote qualitative_grid.png")


if __name__ == "__main__":
    main()
