import numpy as np, torch
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def cam_method(name, model, layer, x, class_idx=None):
    if class_idx is None:
        with torch.no_grad():
            class_idx = int(model(x).argmax(1))
    M = {"gradcam++": GradCAMPlusPlus, "layercam": LayerCAM, "scorecam": ScoreCAM}[name]
    with M(model=model, target_layers=[layer]) as cam:
        g = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    return g, class_idx
