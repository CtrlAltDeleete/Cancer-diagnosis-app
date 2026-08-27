"""
Core model-loading and inference logic for the Oral Histopathology
(Normal vs OSCC) demo.
Mirrors oral_cancer_complete_run_8_models.ipynb exactly:
  - Same 8 architectures (Cell 3.1 / build_model)
  - Same preprocessing (Cell 2.1: eval_tf)
  - Same class order (Cell 1.1: CLASS_NAMES = ["Normal", "OSCC"])
This dataset has no lesion masks, so there is intentionally no segmentation
path here — classification only.
Every checkpoint is a raw state_dict (torch.save(model.state_dict())),
so the architecture built here MUST match the notebook exactly or
load_state_dict() will fail (or silently mismatch).
"""

import os
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----------------------------------------------------------------------
# Constants — must match the notebook exactly (Cell 1.1)
# ----------------------------------------------------------------------
CLASS_NAMES = ["Normal", "OSCC"]
DISPLAY_NAMES = {"Normal": "Normal", "OSCC": "OSCC (Oral Squamous Cell Carcinoma)"}
NUM_CLASSES = len(CLASS_NAMES)

CLF_IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Preprocessing — identical to eval_tf (Cell 2.1)
# ----------------------------------------------------------------------
clf_eval_tf = A.Compose([
    A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ----------------------------------------------------------------------
# Classifier architecture builders (Cell 3.1)
# ----------------------------------------------------------------------
def _build_efficientnet_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4),
                                  nn.Linear(m.classifier[1].in_features, NUM_CLASSES))
    return m

def _build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m

def _build_swin_small():
    import timm
    return timm.create_model("swin_small_patch4_window7_224", pretrained=False,
                              num_classes=NUM_CLASSES, drop_rate=0.3)

def _build_densenet121():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4),
                                  nn.Linear(m.classifier.in_features, NUM_CLASSES))
    return m

def _build_densenet201():
    m = models.densenet201(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4),
                                  nn.Linear(m.classifier.in_features, NUM_CLASSES))
    return m

def _build_inception_v3():
    import timm
    return timm.create_model("inception_v3", pretrained=False,
                              num_classes=NUM_CLASSES, drop_rate=0.4)

def _build_xception():
    import timm
    return timm.create_model("xception", pretrained=False,
                              num_classes=NUM_CLASSES, drop_rate=0.4)

def _build_convnext_tiny():
    import timm
    return timm.create_model("convnext_tiny", pretrained=False,
                              num_classes=NUM_CLASSES, drop_rate=0.4)


# ----------------------------------------------------------------------
# Registry: checkpoint stem (from "best_<stem>.pth") -> metadata
# ----------------------------------------------------------------------
CLASSIFIER_REGISTRY = {
    "efficientnet_b0": ("EfficientNetB0", _build_efficientnet_b0),
    "resnet50":        ("ResNet50", _build_resnet50),
    "swin_small":      ("Swin-S", _build_swin_small),
    "densenet121":     ("DenseNet121", _build_densenet121),
    "densenet201":     ("DenseNet201", _build_densenet201),
    "inception_v3":    ("InceptionV3", _build_inception_v3),
    "xception":        ("Xception", _build_xception),
    "convnext_tiny":   ("ConvNeXt-Tiny", _build_convnext_tiny),
}

# No segmentation task for this scan type (dataset has no lesion masks).
SEGMENTER_REGISTRY = {}


def discover_checkpoints(models_dir):
    """Scan models_dir for best_<stem>.pth files and match against the registry."""
    found = {"classifier": {}, "segmenter": {}}
    if not os.path.isdir(models_dir):
        return found
    for fname in sorted(os.listdir(models_dir)):
        if not (fname.startswith("best_") and fname.endswith(".pth")):
            continue
        stem = fname[len("best_"):-len(".pth")]
        path = os.path.join(models_dir, fname)
        if stem in CLASSIFIER_REGISTRY:
            display, _ = CLASSIFIER_REGISTRY[stem]
            found["classifier"][stem] = (display, path)
    return found


def load_classifier(stem, path, device=DEVICE):
    _, builder = CLASSIFIER_REGISTRY[stem]
    model = builder()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
@torch.no_grad()
def classify_image(model, rgb_uint8, device=DEVICE):
    """rgb_uint8: HxWx3 uint8 numpy array. Returns dict of class -> probability."""
    x = clf_eval_tf(image=rgb_uint8)["image"].unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}
