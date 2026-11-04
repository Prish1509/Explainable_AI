import numpy as np, torch
from method import grad_cam

def tta_mean_uncertainty(model, layer, x, T=8):
    with torch.no_grad():
        cls = int(model(x).argmax(1))
    maps = []
    for t in range(T):
        xi = x.clone()
        flip = (t % 2 == 1)
        if flip: xi = torch.flip(xi, dims=[3])
        xi = xi * (1.0 + 0.1*((t % 3) - 1))
        heat, _ = grad_cam(model, layer, xi, cls)
        if flip: heat = heat[:, ::-1]
        maps.append(np.ascontiguousarray(heat))
    maps = np.stack(maps, 0)
    return maps.mean(0), maps.var(0), cls
