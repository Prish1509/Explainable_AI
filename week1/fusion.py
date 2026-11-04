import numpy as np, torch
from method import grad_cam

def multi_layer_cams(model, layers, x, class_idx=None):
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x).argmax(1))
    cams = [grad_cam(model, L, x, class_idx)[0] for L in layers]
    return np.stack(cams, 0), class_idx

def _norm(m):
    m = np.maximum(m, 0); m = m - m.min()
    return m / m.max() if m.max() > 0 else m

def fuse_uniform(cams):
    return _norm(cams.mean(0))

def fuse_weighted(cams, weights):
    w = np.asarray(weights, float); w = w / w.sum()
    return _norm((w[:, None, None] * cams).sum(0))
