import torch, numpy as np
import torch.nn.functional as F

def blur(x, k=11, s=5.0):
    c = x.shape[1]
    co = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-(co**2) / (2 * s * s))
    g = g / g.sum()
    ker = (g[:, None] * g[None, :]).expand(c, 1, k, k)
    return F.conv2d(x, ker, padding=k // 2, groups=c)

@torch.no_grad()
def deletion_auc(model, x, sal, cls, steps=50):
    base = blur(x)
    H, W = sal.shape
    order = np.argsort(sal.ravel())[::-1]
    n = H * W
    chunk = max(1, n // steps)
    cur = x.clone()
    probs = []

    for s in range(0, n + 1, chunk):
        probs.append(F.softmax(model(cur), 1)[0, cls].item())
        if s >= n:
            break
        ys, xs = np.unravel_index(order[s:s + chunk], (H, W))
        cur[0, :, ys, xs] = base[0, :, ys, xs]

    return float(np.mean(probs)), np.array(probs)
