import torch, numpy as np
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

def load_model(device="cpu"):
    w = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=w).eval().to(device)
    return model, w.transforms(), [model.layer2[-1], model.layer3[-1], model.layer4[-1]]

def grad_cam(model, layer, x, class_idx=None):
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m,i,o: acts.update(v=o.detach()))
    h2 = layer.register_full_backward_hook(lambda m,gi,go: grads.update(v=go[0].detach()))
    model.zero_grad(set_to_none=True)
    out = model(x)
    if class_idx is None: class_idx = int(out.argmax(1))
    out[0, class_idx].backward(); h1.remove(); h2.remove()
    A, G = acts['v'], grads['v']
    weights = G.mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights*A).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    cam = np.maximum(cam,0); cam -= cam.min()
    if cam.max()>0: cam /= cam.max()
    return cam, class_idx
