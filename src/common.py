"""
common.py -- shared setup for every UML-CAM experiment.

Model loading, VOC access, Grad-CAM (implemented from scratch), multi-layer
fusion, and the faithfulness metrics. The ImageNet backbone is never trained or
fine-tuned: everything here is post-hoc and inference-only.

Import from the experiment scripts:
    from common import init, grad_cam, multi_layer_cams, road_score, ...
"""
from __future__ import annotations

import glob
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
)

SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# np.trapz was renamed np.trapezoid in NumPy 2.0 and removed in later versions.
# Bind whichever exists so the code runs on both old and new NumPy.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Repository root, resolved from this file, so the defaults below work anywhere
# (a laptop, a server, Kaggle) without any environment setup.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Results go into <repo>/results by default; override with UMLCAM_OUT=<dir>.
OUTPUT_DIR = Path(os.environ.get("UMLCAM_OUT", REPO_ROOT / "results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Evaluation sizes. Calibration indices (100-119) are disjoint from the
# evaluation set (0-99) so fusion weights are never tuned on test images.
N_MAIN = 100
N_AUC = 15
N_CALIBRATION = 20
N_SWEEP = 20
N_LOCALIZATION = 100
CALIB_RANGE = range(100, 100 + N_CALIBRATION)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN.flatten().tolist(),
                         IMAGENET_STD.flatten().tolist()),
])


def set_seeds(seed: int = SEED) -> None:
    """Fix every source of randomness so a run is repeatable.

    cuDNN's benchmark mode picks convolution algorithms by timing them at
    runtime, so it can select different (and sometimes non-deterministic)
    kernels on different runs or machines. We turn it off and request
    deterministic kernels instead: slightly slower, but repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def discover_voc_root():
    """Find VOC2007, or say where to download it.

    Returns (root, needs_download). A Kaggle-mounted copy is preferred and is
    never downloaded into, because /kaggle/input is read-only. Otherwise we
    look for an existing copy next to the repository, and failing that we
    download into <repo>/data.
    """
    for p in glob.glob("/kaggle/input/**/VOCdevkit", recursive=True):
        if os.path.isdir(os.path.join(p, "VOC2007", "JPEGImages")):
            return os.path.dirname(p), False

    local = REPO_ROOT / "data"
    if (local / "VOCdevkit" / "VOC2007" / "JPEGImages").is_dir():
        return str(local), False
    return str(local), True


def load_model(architecture: str = "resnet50"):
    """Return (model, cam_layers, imagenet_category_names). Never trained."""
    if architecture == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        layers = [model.layer2[-1], model.layer3[-1], model.layer4[-1]]
    elif architecture == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights)
        layers = [model.features[15], model.features[22], model.features[29]]
    else:
        raise ValueError("architecture must be 'resnet50' or 'vgg16'")
    return model.to(DEVICE).eval(), layers, list(weights.meta["categories"])


def init(architecture: str = "resnet50"):
    """Set seeds, load VOC2007 test and the backbone."""
    set_seeds()
    root, need_download = discover_voc_root()
    voc = VOCDetection(root=root, year="2007", image_set="test",
                       download=need_download)
    model, layers, categories = load_model(architecture)
    print(f"device={DEVICE} | voc_root={root} | images={len(voc)}")
    return model, layers, categories, voc


def load_sample(voc, index):
    image, target = voc[index]
    image = image.convert("RGB")
    return image, target, PREPROCESS(image).unsqueeze(0).to(DEVICE)


def normalize_map(heat):
    heat = np.maximum(np.asarray(heat, dtype=np.float32), 0)
    heat = heat - float(heat.min())
    peak = float(heat.max())
    return heat / peak if peak > 0 else heat


# ----------------------------------------------------------------- Grad-CAM
def grad_cam(model, layer, x, class_idx=None):
    """Grad-CAM from scratch: channel weights = spatial mean of the gradients."""
    acts, grads = {}, {}

    def save(_module, _inputs, output):
        acts["v"] = output
        output.register_hook(lambda g: grads.update(v=g))

    handle = layer.register_forward_hook(save)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            if class_idx is None:
                class_idx = int(logits[0].argmax().item())
            logits[0, class_idx].backward()
    finally:
        handle.remove()

    a, g = acts["v"].detach(), grads["v"].detach()
    heat = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True))
    heat = F.interpolate(heat, size=x.shape[-2:], mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()
    return normalize_map(heat), class_idx


def multi_layer_cams(model, target_layers, x, class_idx=None):
    """One Grad-CAM per layer, all explaining the SAME class."""
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x)[0].argmax().item())
    cams = [grad_cam(model, l, x, class_idx)[0] for l in target_layers]
    return np.stack(cams), class_idx


def fuse_uniform(cams):
    return normalize_map(np.mean(cams, axis=0))


def fuse_weighted(cams, weights):
    w = np.asarray(weights, dtype=np.float64)
    if len(w) != len(cams) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be nonnegative, positive-sum, one per CAM")
    w = w / w.sum()
    return normalize_map(np.tensordot(w, cams, axes=(0, 0)))


def library_cam(name, model, layer, x, class_idx):
    """Reference baselines from pytorch-grad-cam."""
    from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    cls = {"gradcam++": GradCAMPlusPlus, "layercam": LayerCAM,
           "scorecam": ScoreCAM}[name]
    with cls(model=model, target_layers=[layer]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    return normalize_map(heat)


# ------------------------------------------------------------------ metrics
def gaussian_blur(x, k=11, sigma=5.0):
    c = x.shape[1]
    co = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-(co ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    ker = (g[:, None] * g[None, :]).expand(c, 1, k, k)
    return F.conv2d(x, ker, padding=k // 2, groups=c)


def make_baseline(x, kind):
    if kind == "blur":
        return gaussian_blur(x)
    if kind == "mean":
        return torch.zeros_like(x)          # zero in normalized space = dataset mean
    if kind == "black":
        mean = IMAGENET_MEAN.to(x.device, x.dtype)
        std = IMAGENET_STD.to(x.device, x.dtype)
        return (torch.zeros_like(x) - mean) / std
    raise ValueError("baseline must be blur|mean|black")


@torch.no_grad()
def perturbation_auc(model, x, saliency, class_idx, mode="deletion",
                     baseline="blur", steps=20):
    """Deletion (lower better) / Insertion (higher better) AUC, from scratch."""
    base = make_baseline(x, baseline)
    h, w = saliency.shape
    order = np.argsort(saliency.reshape(-1))[::-1]
    fractions = np.linspace(0.0, 1.0, steps + 1)
    scores = []
    for f in fractions:
        n = int(round(f * len(order)))
        current = base.clone() if mode == "insertion" else x.clone()
        if n:
            ys, xs = np.unravel_index(order[:n], (h, w))
            source = x if mode == "insertion" else base
            current[0, :, ys, xs] = source[0, :, ys, xs]
        scores.append(float(F.softmax(model(current), dim=1)[0, class_idx].item()))
    return float(trapezoid(scores, fractions)), np.asarray(scores)


def road_score(model, x, heat, class_idx):
    """ROAD (Rong et al., ICML 2022) -- primary debiased metric. Higher better."""
    from pytorch_grad_cam.metrics.road import ROADCombined
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
    s = ROADCombined(percentiles=[20, 40, 60, 80])(
        x, np.asarray(heat, dtype=np.float32)[None, ...],
        [ClassifierOutputSoftmaxTarget(class_idx)], model)
    return float(s.item() if torch.is_tensor(s) else s)


def calibrate_fusion_weights(model, target_layers, voc, indices=CALIB_RANGE):
    """w_l proportional to 1 / DeletionAUC_l, measured on held-out images."""
    scores = [[] for _ in target_layers]
    for i in indices:
        _, _, x = load_sample(voc, i)
        cams, cls = multi_layer_cams(model, target_layers, x)
        for j, heat in enumerate(cams):
            scores[j].append(
                perturbation_auc(model, x, heat, cls, "deletion", "blur", 20)[0])
    means = np.asarray([np.mean(s) for s in scores])
    inv = 1.0 / np.maximum(means, 1e-8)
    return inv / inv.sum(), means
