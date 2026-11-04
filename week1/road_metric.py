import numpy as np
from pytorch_grad_cam.metrics.road import ROADCombined
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

def road_score(model, x, heat, cls):
    metric = ROADCombined(percentiles=[20, 40, 60, 80])
    cam = np.expand_dims(heat.astype("float32"), 0)   # (1, H, W)
    return float(metric(x, cam, [ClassifierOutputSoftmaxTarget(cls)], model))
