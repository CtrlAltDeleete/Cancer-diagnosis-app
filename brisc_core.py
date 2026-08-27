"""
Core model-loading and inference logic for the BRISC brain-tumor demo.
Mirrors BRISC_COMPLETE_PROCESSED.ipynb exactly:
  - Same architectures (Part 3 / Part 4)
  - Same preprocessing (Cell 12: clf_eval_tf / seg_eval_tf)
  - Same class order (Cell 3: CLASS_NAMES)
Every checkpoint is a raw state_dict (torch.save(model.state_dict())),
so the architecture built here MUST match the notebook exactly or
load_state_dict() will fail (or silently mismatch).
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------
# Constants — must match the notebook exactly
# ----------------------------------------------------------------------
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
DISPLAY_NAMES = {"glioma": "Glioma", "meningioma": "Meningioma",
                  "no_tumor": "No Tumor", "pituitary": "Pituitary"}
NUM_CLASSES = len(CLASS_NAMES)

CLF_IMG_SIZE = 224
SEG_IMG_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------
# Preprocessing — identical to clf_eval_tf / seg_eval_tf (Cell 12)
# ----------------------------------------------------------------------
clf_eval_tf = A.Compose([
    A.Resize(CLF_IMG_SIZE, CLF_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
seg_eval_tf = A.Compose([
    A.Resize(SEG_IMG_SIZE, SEG_IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


# ----------------------------------------------------------------------
# Classifier architecture builders (Part 3, Cell 19)
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
# Segmenter architecture builders (Part 4, Cell 26) — encoder resnet34
# ----------------------------------------------------------------------
ENC = "resnet34"

def _build_unet():
    return smp.Unet(encoder_name=ENC, encoder_weights=None, in_channels=3, classes=1)

def _build_attention_unet():
    return smp.Unet(encoder_name=ENC, encoder_weights=None, in_channels=3,
                     classes=1, decoder_attention_type="scse")

def _build_unetpp():
    return smp.UnetPlusPlus(encoder_name=ENC, encoder_weights=None, in_channels=3, classes=1)

def _build_swin_unet():
    try:
        m = smp.Unet(encoder_name="tu-swin_tiny_patch4_window7_224",
                     encoder_weights=None, in_channels=3, classes=1)
        with torch.no_grad():
            _ = m(torch.zeros(1, 3, SEG_IMG_SIZE, SEG_IMG_SIZE))
        return m
    except Exception:
        return smp.Unet(encoder_name=ENC, encoder_weights=None, in_channels=3, classes=1)

def _build_fpn():
    return smp.FPN(encoder_name=ENC, encoder_weights=None, in_channels=3, classes=1)

def _build_deeplabv3plus():
    return smp.DeepLabV3Plus(encoder_name=ENC, encoder_weights=None, in_channels=3, classes=1)


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

SEGMENTER_REGISTRY = {
    "unet":           ("U-Net", _build_unet),
    "attention_unet": ("Attention U-Net", _build_attention_unet),
    "unetpp":         ("U-Net++", _build_unetpp),
    "swin_unet":      ("Swin-UNet", _build_swin_unet),
    "fpn":            ("FPN", _build_fpn),
    "deeplabv3plus":  ("DeepLabV3+", _build_deeplabv3plus),
}


def discover_checkpoints(models_dir):
    """Scan models_dir for best_<stem>.pth files and match against registries."""
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
        elif stem in SEGMENTER_REGISTRY:
            display, _ = SEGMENTER_REGISTRY[stem]
            found["segmenter"][stem] = (display, path)
    return found


def load_classifier(stem, path, device=DEVICE):
    _, builder = CLASSIFIER_REGISTRY[stem]
    model = builder()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_segmenter(stem, path, device=DEVICE):
    _, builder = SEGMENTER_REGISTRY[stem]
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


@torch.no_grad()
def segment_image(model, rgb_uint8, device=DEVICE, threshold=0.5):
    """rgb_uint8: HxWx3 uint8 numpy array.
    Returns a binary mask (uint8, values 0/1) resized back to the original H x W.
    """
    orig_h, orig_w = rgb_uint8.shape[:2]
    x = seg_eval_tf(image=rgb_uint8)["image"].unsqueeze(0).to(device)
    logits = model(x)
    prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
    bin_mask = (prob_mask > threshold).astype(np.uint8)
    import cv2
    bin_mask_full = cv2.resize(bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return bin_mask_full


def overlay_mask(rgb_uint8, mask_uint8, color=(46, 204, 113), alpha=0.45):
    """Blend a binary mask onto the RGB image as a translucent color overlay."""
    import cv2
    overlay = rgb_uint8.copy()
    color_layer = np.zeros_like(rgb_uint8)
    color_layer[:, :] = color
    mask_bool = mask_uint8.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * rgb_uint8[mask_bool] + alpha * color_layer[mask_bool]
    ).astype(np.uint8)
    # draw a contour outline for clarity
    contours, _ = cv2.findContours(mask_uint8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay
