"""
Radiology AI Console
Classification + Segmentation across multiple imaging modalities
(brain MRI tumor typing, breast ultrasound lesion typing).
Run with:  streamlit run app.py
"""
import os
import numpy as np
from PIL import Image
import streamlit as st
import brisc_core
import busi_core
import oral_core

BASE_DIR = os.path.dirname(__file__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Radiology AI Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DESIGN TOKENS + CSS
# Visual identity: a radiology reading-room instrument panel — near-black
# viewport, hairline dividers, monospace data readouts, corner-bracket
# scan frames (borrowed from PACS viewers), one slow scanline sweep as the
# single animated signature.
# =============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    :root {
        --void: #05070a;
        --panel: #0c0f15;
        --panel-raised: #11151d;
        --panel-hover: #151a24;
        --hairline: rgba(255,255,255,0.07);
        --hairline-strong: rgba(255,255,255,0.14);
        --ink-100: #e9edf3;
        --ink-70: #9aa4b5;
        --ink-45: #5b6474;
        --cyan: #22d3ee;
        --cyan-dim: rgba(34,211,238,0.14);
        --amber: #f0a83c;
        --green: #34d399;
        --red: #f7576a;
        --glioma: #f97316;
        --meningioma: #a78bfa;
        --no-tumor: #34d399;
        --pituitary: #f472b6;
        --benign: #34d399;
        --malignant: #f7576a;
        --normal: #60a5fa;
        --oral-normal: #2dd4bf;
        --oral-oscc: #fb7185;
        --font-display: 'Space Grotesk', sans-serif;
        --font-body: 'Inter', -apple-system, sans-serif;
        --font-mono: 'IBM Plex Mono', monospace;
    }

    html { font-size: 19px; }

    .stApp { background: var(--void) !important; font-family: var(--font-body) !important; }
    .main > div { padding-top: 0 !important; }

    /* Keep Streamlit's native menu (hamburger / three-dot) + header visible
       so users can still reach light/dark mode, print, settings, etc. */
    #MainMenu { visibility: visible !important; }
    header[data-testid="stHeader"], header {
        visibility: visible !important;
        background: var(--void) !important;
        border-bottom: 1px solid var(--hairline);
    }
    [data-testid="stToolbar"] { visibility: visible !important; }
    footer {visibility: hidden;}

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--void); }
    ::-webkit-scrollbar-thumb { background: var(--hairline-strong); border-radius: 3px; }

    h1, h2, h3 { font-family: var(--font-display) !important; }

    /* ---------------------------------------------------------------- */
    /* Animation library                                                 */
    /* ---------------------------------------------------------------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.96); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes growX {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 4px currentColor; opacity: 1; }
        50% { box-shadow: 0 0 12px currentColor; opacity: 0.6; }
    }
    @keyframes shimmerMove {
        0% { transform: translateX(-120%); }
        100% { transform: translateX(220%); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }
    }

    /* ---------------------------------------------------------------- */
    /* Top bar                                                           */
    /* ---------------------------------------------------------------- */
    .topbar {
        position: relative;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 6px;
        padding: 1.4rem 1.8rem;
        margin: 1rem 0 1.6rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        overflow: hidden;
        animation: fadeInUp 0.6s ease both;
        transition: border-color 0.3s ease;
    }
    .topbar:hover { border-color: var(--hairline-strong); }

    .topbar::after {
        content: '';
        position: absolute;
        top: 0; left: -30%;
        width: 30%; height: 100%;
        background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
        animation: sweep 7s linear infinite;
        pointer-events: none;
    }

    @keyframes sweep {
        0% { left: -30%; }
        100% { left: 100%; }
    }

    @media (prefers-reduced-motion: reduce) {
        .topbar::after { animation: none; display: none; }
    }

    .brand-row { display: flex; align-items: center; gap: 0.9rem; z-index: 1; }

    .brand-mark {
        width: 40px; height: 40px;
        border: 1.5px solid var(--cyan);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-family: var(--font-mono);
        font-size: 1.1rem;
        color: var(--cyan);
        flex-shrink: 0;
        background: var(--cyan-dim);
        transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.4s ease;
    }
    .topbar:hover .brand-mark {
        transform: rotate(-8deg) scale(1.08);
        box-shadow: 0 0 16px var(--cyan-dim);
    }

    .brand-name {
        font-family: var(--font-display);
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--ink-100);
        letter-spacing: 0.01em;
        line-height: 1.1;
    }

    .brand-sub {
        font-size: 0.8rem;
        color: var(--ink-70);
        margin-top: 0.15rem;
        letter-spacing: 0.01em;
    }

    .topbar-status {
        z-index: 1;
        text-align: right;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--ink-70);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .topbar-status .dot {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--green);
        margin-right: 0.4rem;
        color: var(--green);
        animation: pulseGlow 2.2s ease-in-out infinite;
    }

    .header-desc {
        color: var(--ink-70);
        font-size: 0.88rem;
        line-height: 1.55;
        max-width: 640px;
        margin: -0.6rem 0 1.6rem 0.1rem;
        animation: fadeInUp 0.6s ease 0.1s both;
    }

    /* ---------------------------------------------------------------- */
    /* Section labels / eyebrows                                        */
    /* ---------------------------------------------------------------- */
    .eyebrow {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--ink-45);
        margin-bottom: 0.7rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .eyebrow::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--hairline);
    }

    /* ---------------------------------------------------------------- */
    /* Result card — primary diagnosis                                  */
    /* ---------------------------------------------------------------- */
    .dx-card {
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-left: 2px solid var(--cyan);
        border-radius: 6px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.4rem;
        animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease;
    }
    .dx-card:hover {
        transform: translateY(-3px);
        border-color: var(--hairline-strong);
        box-shadow: 0 10px 30px -12px rgba(34,211,238,0.25);
    }

    .dx-model {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--ink-45);
        margin-bottom: 0.9rem;
    }

    .dx-name {
        font-family: var(--font-display);
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--ink-100);
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.6rem;
    }

    .dx-confidence {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
        font-family: var(--font-mono);
    }

    .dx-confidence .value {
        font-size: 2.1rem;
        font-weight: 600;
        color: var(--cyan);
        display: inline-block;
        animation: scaleIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both;
    }

    .dx-confidence .label {
        font-size: 0.78rem;
        color: var(--ink-45);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Class indicator dot */
    .dot { display: inline-block; width: 11px; height: 11px; border-radius: 3px; flex-shrink: 0; }
    .dot-glioma { background: var(--glioma); }
    .dot-meningioma { background: var(--meningioma); }
    .dot-no_tumor { background: var(--no-tumor); }
    .dot-pituitary { background: var(--pituitary); }
    .dot-benign { background: var(--benign); }
    .dot-malignant { background: var(--malignant); }
    .dot-normal { background: var(--normal); }
    .dot-oral-normal { background: var(--oral-normal); }
    .dot-oral-oscc { background: var(--oral-oscc); }

    /* ---------------------------------------------------------------- */
    /* Confidence breakdown bars                                        */
    /* ---------------------------------------------------------------- */
    .conf-row {
        margin-bottom: 0.95rem;
        animation: fadeInUp 0.5s ease both;
        transition: opacity 0.3s ease;
    }
    .conf-row:hover { opacity: 1 !important; }

    .conf-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }

    .conf-name {
        font-size: 0.84rem;
        font-weight: 500;
        color: var(--ink-70);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .conf-percent {
        font-family: var(--font-mono);
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--ink-100);
    }

    .conf-track {
        background: rgba(255,255,255,0.04);
        border-radius: 2px;
        height: 6px;
        overflow: hidden;
    }

    .conf-fill {
        height: 100%;
        border-radius: 2px;
        transform-origin: left center;
        animation: growX 0.9s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both;
    }
    .fill-glioma { background: var(--glioma); }
    .fill-meningioma { background: var(--meningioma); }
    .fill-no_tumor { background: var(--no-tumor); }
    .fill-pituitary { background: var(--pituitary); }
    .fill-benign { background: var(--benign); }
    .fill-malignant { background: var(--malignant); }
    .fill-normal { background: var(--normal); }
    .fill-oral-normal { background: var(--oral-normal); }
    .fill-oral-oscc { background: var(--oral-oscc); }

    /* ---------------------------------------------------------------- */
    /* Viewport — scan frame with corner brackets (signature element)   */
    /* ---------------------------------------------------------------- */
    .viewport {
        position: relative;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 4px;
        padding: 1.4rem 0.9rem 0.9rem 0.9rem;
        animation: scaleIn 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: border-color 0.35s ease, transform 0.35s ease, box-shadow 0.35s ease;
    }
    .viewport:hover {
        transform: translateY(-2px);
        border-color: rgba(34,211,238,0.3);
        box-shadow: 0 12px 28px -14px rgba(34,211,238,0.3);
    }
    .viewport:hover .vp-corner { opacity: 1; width: 18px; height: 18px; }

    .vp-corner {
        position: absolute;
        width: 14px; height: 14px;
        border-color: var(--cyan);
        opacity: 0.55;
        z-index: 2;
        transition: all 0.3s ease;
    }
    .vp-tl { top: 8px; left: 8px; border-top: 1.5px solid; border-left: 1.5px solid; }
    .vp-tr { top: 8px; right: 8px; border-top: 1.5px solid; border-right: 1.5px solid; }
    .vp-bl { bottom: 8px; left: 8px; border-bottom: 1.5px solid; border-left: 1.5px solid; }
    .vp-br { bottom: 8px; right: 8px; border-bottom: 1.5px solid; border-right: 1.5px solid; }

    .vp-label {
        position: absolute;
        top: 10px; left: 24px;
        font-family: var(--font-mono);
        font-size: 0.6rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-45);
        z-index: 2;
    }

    .vp-caption {
        text-align: center;
        padding-top: 0.6rem;
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--ink-70);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ---------------------------------------------------------------- */
    /* Segmentation status + metrics                                    */
    /* ---------------------------------------------------------------- */
    .seg-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 2.2rem 0 1.2rem 0;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid var(--hairline);
        animation: fadeInUp 0.5s ease both;
    }

    .seg-title {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--ink-100);
    }

    .seg-title .model-tag {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        font-weight: 500;
        color: var(--ink-45);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-left: 0.6rem;
    }

    .seg-status {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        animation: fadeIn 0.6s ease 0.15s both;
    }

    .readout-row {
        display: flex;
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-radius: 6px;
        margin-bottom: 1.4rem;
        overflow: hidden;
        animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
    }

    .readout-cell {
        flex: 1;
        padding: 0.9rem 1.4rem;
        text-align: center;
        border-right: 1px solid var(--hairline);
        transition: background 0.3s ease;
    }
    .readout-cell:hover { background: var(--panel-hover); }
    .readout-cell:last-child { border-right: none; }

    .readout-value {
        font-family: var(--font-mono);
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--cyan);
    }

    .readout-label {
        font-family: var(--font-mono);
        font-size: 0.64rem;
        color: var(--ink-45);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }

    /* ---------------------------------------------------------------- */
    /* Sidebar — instrument panel                                       */
    /* ---------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--hairline);
    }

    .panel-title {
        font-family: var(--font-mono);
        font-size: 0.66rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--ink-45);
        margin-bottom: 0.9rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--hairline);
        animation: fadeIn 0.5s ease both;
    }

    .panel-title-center {
        text-align: center;
    }

    .device-label-center {
        text-align: center;
    }

    .device-readout {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        font-family: var(--font-mono);
        font-weight: 600;
        color: var(--ink-100);
        font-size: 0.92rem;
        animation: fadeInUp 0.5s ease both;
    }

    .device-readout .dot-status {
        width: 7px; height: 7px; border-radius: 50%;
        animation: pulseGlow 2.2s ease-in-out infinite;
    }

    .info-block {
        background: var(--cyan-dim);
        border: 1px solid rgba(34,211,238,0.2);
        border-radius: 6px;
        padding: 0.7rem 0.9rem;
        margin-top: 0.9rem;
        font-size: 0.76rem;
        color: var(--ink-70);
        line-height: 1.5;
    }

    .pipeline-block {
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--ink-45);
        line-height: 1.8;
    }

    .threshold-scale {
        display: flex;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.66rem;
        color: var(--ink-45);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: -0.4rem;
    }

    .threshold-scale .cur { font-weight: 700; color: var(--cyan); }

    .stSlider > div > div > div { background: var(--cyan) !important; }

    .stAlert {
        background: var(--panel-raised) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 6px !important;
        color: var(--ink-100) !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] p { color: var(--ink-100) !important; }

    hr { border-color: var(--hairline) !important; }

    /* ---------------------------------------------------------------- */
    /* Empty state                                                       */
    /* ---------------------------------------------------------------- */
    .empty-state {
        text-align: center;
        padding: 4.5rem 2rem;
        border: 1px dashed var(--hairline-strong);
        border-radius: 8px;
        background: var(--panel);
        animation: fadeInUp 0.6s ease both;
        transition: border-color 0.3s ease;
    }
    .empty-state:hover { border-color: var(--cyan); }

    .empty-state .glyph {
        font-family: var(--font-mono);
        font-size: 2.2rem;
        color: var(--cyan);
        opacity: 0.7;
        margin-bottom: 1rem;
        display: inline-block;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }

    .empty-state .title {
        font-family: var(--font-display);
        font-size: 1.15rem;
        font-weight: 600;
        color: var(--ink-100);
        margin-bottom: 0.4rem;
    }

    .empty-state .hint {
        font-size: 0.85rem;
        color: var(--ink-45);
    }

    /* ---------------------------------------------------------------- */
    /* No-tumor / segmentation-skipped panel                            */
    /* ---------------------------------------------------------------- */
    .skip-card {
        background: var(--panel);
        border: 1px solid var(--hairline);
        border-left: 2px solid var(--green);
        border-radius: 6px;
        padding: 1.6rem 1.8rem;
        margin: 2.2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: fadeInUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .skip-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px -14px rgba(52,211,153,0.3);
    }
    .skip-icon {
        width: 34px; height: 34px;
        border-radius: 50%;
        border: 1.5px solid var(--green);
        display: flex; align-items: center; justify-content: center;
        font-family: var(--font-mono);
        color: var(--green);
        font-size: 1rem;
        flex-shrink: 0;
        animation: pulseGlow 2.4s ease-in-out infinite;
    }
    .skip-title {
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--ink-100);
    }
    .skip-sub {
        font-size: 0.85rem;
        color: var(--ink-70);
        margin-top: 0.2rem;
    }

    /* ---------------------------------------------------------------- */
    /* Footer                                                            */
    /* ---------------------------------------------------------------- */
    .app-footer {
        text-align: center;
        padding: 2rem;
        color: var(--ink-45);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        border-top: 1px solid var(--hairline);
        margin-top: 3rem;
    }

    @media (max-width: 768px) {
        .brand-name { font-size: 1.1rem; }
        .dx-name { font-size: 1.4rem; }
        .dx-confidence .value { font-size: 1.6rem; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SCAN TYPE CONFIGURATION
# Each scan type binds together: which core module (architectures + class
# names) to use, where its checkpoints live, its class colors, its published
# accuracy metrics, and the copy shown in the console.
# =============================================================================
SCAN_CONFIGS = {
    "brain_mri": {
        "label": "Brain MRI",
        "core": brisc_core,
        "has_segmentation": True,
        "models_dir": os.path.join(BASE_DIR, "models"),
        "no_lesion_class": "no_tumor",
        "class_dot": {
            "glioma": "dot-glioma",
            "meningioma": "dot-meningioma",
            "no_tumor": "dot-no_tumor",
            "pituitary": "dot-pituitary",
        },
        "class_bar": {
            "glioma": "fill-glioma",
            "meningioma": "fill-meningioma",
            "no_tumor": "fill-no_tumor",
            "pituitary": "fill-pituitary",
        },
        "clf_accuracy": {
            "efficientnet_b0": 99.2,
            "resnet50":        99.4,
            "swin_small":      99.2,
            "densenet121":     99.4,
            "densenet201":     98.8,
            "inception_v3":    98.4,
            "xception":        99.2,
            "convnext_tiny":   99.1,
        },
        "seg_pixel_acc": {
            "unet":           0.9953,
            "attention_unet": 0.9954,
            "unetpp":         0.9956,
            "swin_unet":      0.9956,
            "fpn":            0.9953,
            "deeplabv3plus":  0.9954,
        },
        "default_clf": "resnet50",
        "default_seg": "unetpp",
        "brand_name": "BRISC",
        "brand_sub": "Neuro-Oncology Imaging Console",
        "header_desc": (
            "Deep-learning classification and segmentation for contrast-enhanced "
            "T1-weighted brain MRI slices. Upload a scan to identify tumor subtype "
            "and localize the affected region."
        ),
        "empty_title": "Upload an MRI Scan to Begin Analysis",
        "empty_hint": "Supported formats: JPG, PNG, BMP, TIF — contrast-enhanced T1-weighted recommended",
        "image_caption": "Original · T1-CE",
        "skip_title": "No Tumor Detected — Segmentation Skipped",
        "skip_sub": (
            "The classifier found no evidence of a tumor in this scan, so the segmentation "
            "model was not run. Segmenting a tumor-free image can otherwise produce spurious mask regions."
        ),
        "footer": "Radiology AI Console · Brain MRI Module · Built with Streamlit · Models trained on the BRISC pipeline",
        "lesion_noun": "Tumor",
    },
    "breast_us": {
        "label": "Breast Ultrasound",
        "core": busi_core,
        "has_segmentation": True,
        "models_dir": os.path.join(BASE_DIR, "models_busi"),
        "no_lesion_class": "normal",
        "class_dot": {
            "benign": "dot-benign",
            "malignant": "dot-malignant",
            "normal": "dot-normal",
        },
        "class_bar": {
            "benign": "fill-benign",
            "malignant": "fill-malignant",
            "normal": "fill-normal",
        },
        "clf_accuracy": {
            "efficientnet_b0": 87.18,
            "resnet50":        77.78,
            "swin_small":      88.89,
            "densenet121":     89.74,
            "densenet201":     88.03,
            "inception_v3":    79.49,
            "xception":        88.03,
            "convnext_tiny":   17.09,
        },
        "seg_pixel_acc": {
            "unet":           0.9466,
            "attention_unet": 0.9514,
            "unetpp":         0.9519,
            "swin_unet":      0.9524,
            "fpn":            0.9465,
            "deeplabv3plus":  0.9467,
        },
        "default_clf": "densenet121",
        "default_seg": "swin_unet",
        "brand_name": "BUSI",
        "brand_sub": "Breast Sonography Imaging Console",
        "header_desc": (
            "Deep-learning classification and segmentation for B-mode breast "
            "ultrasound images. Upload a scan to identify lesion type (benign, "
            "malignant, or normal) and localize the affected region."
        ),
        "empty_title": "Upload an Ultrasound Scan to Begin Analysis",
        "empty_hint": "Supported formats: JPG, PNG, BMP, TIF — grayscale B-mode breast ultrasound recommended",
        "image_caption": "Original · B-Mode US",
        "skip_title": "No Lesion Detected — Segmentation Skipped",
        "skip_sub": (
            "The classifier found no evidence of a lesion in this scan, so the segmentation "
            "model was not run. Segmenting a lesion-free image can otherwise produce spurious mask regions."
        ),
        "footer": "Radiology AI Console · Breast Ultrasound Module · Built with Streamlit · Models trained on the BUSI pipeline",
        "lesion_noun": "Lesion",
    },
    "oral_histo": {
        "label": "Oral Histopathology",
        "core": oral_core,
        "has_segmentation": False,
        "models_dir": os.path.join(BASE_DIR, "models_oral"),
        "no_lesion_class": "Normal",
        "class_dot": {
            "Normal": "dot-oral-normal",
            "OSCC": "dot-oral-oscc",
        },
        "class_bar": {
            "Normal": "fill-oral-normal",
            "OSCC": "fill-oral-oscc",
        },
        "clf_accuracy": {
            "efficientnet_b0": 97.69,
            "resnet50":        97.95,
            "swin_small":      97.56,
            "densenet121":     97.43,
            "densenet201":     97.95,
            "inception_v3":    97.43,
            "xception":        97.56,
            "convnext_tiny":   97.18,
        },
        "seg_pixel_acc": {},
        "default_clf": "resnet50",
        "default_seg": None,
        "brand_name": "ORAL-HISTO",
        "brand_sub": "Oral Histopathology Imaging Console",
        "header_desc": (
            "Deep-learning classification for H&E-stained oral histopathology slide "
            "images. Upload an image to identify whether the tissue is Normal or shows "
            "OSCC (Oral Squamous Cell Carcinoma)."
        ),
        "empty_title": "Upload a Histopathology Image to Begin Analysis",
        "empty_hint": "Supported formats: JPG, PNG, BMP, TIF — H&E-stained oral histopathology slide images recommended",
        "image_caption": "Original · H&E Histopathology",
        "skip_title": "",
        "skip_sub": "",
        "footer": "Radiology AI Console · Oral Histopathology Module · Built with Streamlit · Models trained on the oral histopathology pipeline · Classification only — no segmentation",
        "lesion_noun": "Lesion",
    },
}

# =============================================================================
# TOP BAR (scan-type selector renders further below, in the sidebar — the
# brand block here reflects whichever scan type is currently selected)
# =============================================================================
scan_type_key = st.session_state.get("scan_type_key", "brain_mri")
scan_cfg = SCAN_CONFIGS[scan_type_key]
core = scan_cfg["core"]
MODELS_DIR = scan_cfg["models_dir"]
CLASS_DOT = scan_cfg["class_dot"]
CLASS_BAR = scan_cfg["class_bar"]
CLF_ACCURACY = scan_cfg["clf_accuracy"]
SEG_PIXEL_ACC = scan_cfg["seg_pixel_acc"]

st.markdown(f"""
<div class="topbar">
    <div class="brand-row">
        <div class="brand-mark">◈</div>
        <div>
            <div class="brand-name">{scan_cfg['brand_name']}</div>
            <div class="brand-sub">{scan_cfg['brand_sub']}</div>
        </div>
    </div>
    <div class="topbar-status"><span class="dot"></span>System Ready</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="header-desc">
    {scan_cfg['header_desc']}
</div>
""", unsafe_allow_html=True)

# =============================================================================
# DISCOVER MODELS
# =============================================================================
found = core.discover_checkpoints(MODELS_DIR)
clf_options = found["classifier"]
seg_options = found["segmenter"]

HAS_SEGMENTATION = scan_cfg.get("has_segmentation", True)

if not clf_options and not seg_options:
    models_subdir = os.path.basename(MODELS_DIR)
    expected = ["- `best_resnet50.pth` (classifier)"]
    if HAS_SEGMENTATION:
        expected.append("- `best_unetpp.pth` (segmenter)")
    st.error(f"""
    ### No Model Checkpoints Found

    Expected files in `{models_subdir}/`:
    {chr(10).join(expected)}

    Please add your trained model weights and refresh.
    """)
    st.stop()

# =============================================================================
# SIDEBAR — INSTRUMENT PANEL
# =============================================================================
with st.sidebar:
    st.markdown('<div class="panel-title panel-title-center">Configuration</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-title">Scan Type</div>', unsafe_allow_html=True)
    scan_labels = {k: v["label"] for k, v in SCAN_CONFIGS.items()}
    scan_keys = list(SCAN_CONFIGS.keys())
    selected_scan_key = st.selectbox(
        "Scan type",
        scan_keys,
        format_func=lambda k: scan_labels[k],
        index=scan_keys.index(scan_type_key),
        label_visibility="collapsed",
        key="scan_type_selector",
    )
    if selected_scan_key != scan_type_key:
        st.session_state["scan_type_key"] = selected_scan_key
        st.rerun()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    device_color = "var(--green)" if "cuda" in str(core.DEVICE) else "var(--amber)"
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div class="device-label-center" style="font-family: var(--font-mono); font-size: 0.68rem; color: var(--ink-45);
                    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.35rem;">
            Compute Device
        </div>
        <div class="device-readout">
            <span class="dot-status" style="background: {device_color}; color: {device_color};"></span>
            {str(core.DEVICE).upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="panel-title">Classification Model</div>', unsafe_allow_html=True)
    if clf_options:
        clf_stems = list(clf_options.keys())
        default_clf_stem = scan_cfg["default_clf"]
        default_clf = clf_stems.index(default_clf_stem) if default_clf_stem in clf_stems else 0
        clf_choice = st.selectbox(
            "Select classifier",
            clf_stems,
            format_func=lambda s: f"{clf_options[s][0]}   ·   {CLF_ACCURACY.get(s, '—')}%",
            index=default_clf,
            label_visibility="collapsed",
            key=f"clf_choice_{scan_type_key}",
        )
    else:
        clf_choice = None
        st.warning("No classifier checkpoints found.")

    if HAS_SEGMENTATION:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-title">Segmentation Model</div>', unsafe_allow_html=True)
        if seg_options:
            seg_stems = list(seg_options.keys())
            default_seg_stem = scan_cfg["default_seg"]
            default_seg = seg_stems.index(default_seg_stem) if default_seg_stem in seg_stems else 0
            seg_choice = st.selectbox(
                "Select segmenter",
                seg_stems,
                format_func=lambda s: f"{seg_options[s][0]}   ·   {SEG_PIXEL_ACC.get(s, 0)*100:.2f}%",
                index=default_seg,
                label_visibility="collapsed",
                key=f"seg_choice_{scan_type_key}",
            )
        else:
            seg_choice = None
            st.warning("No segmenter checkpoints found.")

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-title">Segmentation Threshold</div>', unsafe_allow_html=True)
        threshold = st.slider("", 0.1, 0.9, 0.5, 0.05, label_visibility="collapsed")
        st.markdown(f"""
        <div class="threshold-scale">
            <span>Sensitive</span>
            <span class="cur">{threshold:.2f}</span>
            <span>Strict</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-block">
            <strong>About threshold</strong><br>
            Lower values flag more potential tumor pixels (higher sensitivity).
            Higher values require stronger model confidence (higher specificity).
            Recommended range: 0.40–0.50.
        </div>
        """, unsafe_allow_html=True)
    else:
        seg_choice = None
        threshold = None

# =============================================================================
# CACHED MODEL LOADERS
# Keyed by scan type + stem so switching scan types can't return a model
# built with the wrong architecture/class count from the cache.
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_classifier(scan_key, stem, path):
    return SCAN_CONFIGS[scan_key]["core"].load_classifier(stem, path, device=core.DEVICE)

@st.cache_resource(show_spinner=False)
def get_segmenter(scan_key, stem, path):
    return SCAN_CONFIGS[scan_key]["core"].load_segmenter(stem, path, device=core.DEVICE)

# =============================================================================
# MAIN CONTENT
# =============================================================================
if not clf_options and not seg_options:
    st.stop()

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(f"""
    <div class="empty-state">
        <div class="glyph">◈</div>
        <div class="title">{scan_cfg['empty_title']}</div>
        <div class="hint">{scan_cfg['empty_hint']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Process uploaded image
pil_img = Image.open(uploaded).convert("RGB")
rgb = np.array(pil_img, dtype=np.uint8)

# =============================================================================
# CLASSIFICATION
# =============================================================================
col_img, col_results = st.columns([1, 1.2], gap="large")

with col_img:
    st.image(pil_img, use_container_width=True)
    st.markdown(
        f'<div style="text-align:center; padding-top:0.6rem; font-family:var(--font-mono); '
        f'font-size:0.68rem; color:var(--ink-70); text-transform:uppercase; letter-spacing:0.08em;">'
        f'{rgb.shape[1]} × {rgb.shape[0]} px</div>',
        unsafe_allow_html=True
    )

top_class = None  # tracked outside the column so segmentation logic can gate on it

with col_results:
    if clf_choice:
        clf_display, clf_path = clf_options[clf_choice]

        with st.spinner("Analyzing scan..."):
            model_c = get_classifier(scan_type_key, clf_choice, clf_path)
            probs = core.classify_image(model_c, rgb, device=core.DEVICE)

        top_class = max(probs, key=probs.get)
        top_conf = probs[top_class]
        top_display = core.DISPLAY_NAMES[top_class]

        st.markdown(f"""
        <div class="dx-card">
            <div class="dx-model">Primary Diagnosis — {clf_display}</div>
            <div class="dx-name"><span class="dot {CLASS_DOT[top_class]}"></span>{top_display}</div>
            <div class="dx-confidence">
                <span class="value">{top_conf*100:.1f}%</span>
                <span class="label">confidence</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="eyebrow">Confidence Breakdown</div>', unsafe_allow_html=True)

        for i, cname in enumerate(core.CLASS_NAMES):
            p = probs[cname]
            is_top = (cname == top_class)
            opacity = "1" if is_top else "0.55"
            row_delay = i * 0.07
            fill_delay = 0.15 + i * 0.07
            st.markdown(f"""
            <div class="conf-row" style="opacity: {opacity}; animation-delay: {row_delay:.2f}s;">
                <div class="conf-header">
                    <span class="conf-name"><span class="dot {CLASS_DOT[cname]}"></span>{core.DISPLAY_NAMES[cname]}</span>
                    <span class="conf-percent">{p*100:.1f}%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill {CLASS_BAR[cname]}" style="width: {p*100:.1f}%; animation-delay: {fill_delay:.2f}s;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No classification model configured.")

# =============================================================================
# SEGMENTATION
# A lesion-free classification is treated as authoritative: running the
# segmenter on a scan the classifier already read as lesion-free tends to
# produce spurious mask fragments, so it's skipped rather than shown.
# =============================================================================
no_lesion_confirmed = (clf_choice is not None) and (top_class == scan_cfg["no_lesion_class"])

if seg_choice and no_lesion_confirmed:
    st.markdown(f"""
    <div class="skip-card">
        <div class="skip-icon">✓</div>
        <div>
            <div class="skip-title">{scan_cfg['skip_title']}</div>
            <div class="skip-sub">{scan_cfg['skip_sub']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif seg_choice:
    seg_display, seg_path = seg_options[seg_choice]

    with st.spinner("Generating segmentation mask..."):
        model_s = get_segmenter(scan_type_key, seg_choice, seg_path)
        mask = core.segment_image(model_s, rgb, device=core.DEVICE, threshold=threshold)
        overlay = core.overlay_mask(rgb, mask)

    tumor_pixels = int(mask.sum())
    lesion_noun = scan_cfg["lesion_noun"]

    if tumor_pixels == 0:
        status_color, status_text = "var(--amber)", f"No {lesion_noun} Detected"
    else:
        status_color, status_text = "var(--green)", f"{lesion_noun} Region Detected"

    pct = 100 * tumor_pixels / mask.size if mask.size > 0 else 0

    st.markdown(f"""
    <div class="seg-header">
        <div class="seg-title">Segmentation Analysis<span class="model-tag">{seg_display}</span></div>
        <div class="seg-status" style="color: {status_color};">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    if tumor_pixels > 0:
        st.markdown(f"""
        <div class="readout-row">
            <div class="readout-cell">
                <div class="readout-value">{tumor_pixels:,}</div>
                <div class="readout-label">Flagged Pixels</div>
            </div>
            <div class="readout-cell">
                <div class="readout-value">{pct:.2f}%</div>
                <div class="readout-label">Of Total Image</div>
            </div>
            <div class="readout-cell">
                <div class="readout-value">{threshold:.2f}</div>
                <div class="readout-label">Threshold Applied</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No {lesion_noun.lower()} region exceeded the confidence threshold. Try lowering the threshold in the sidebar if you believe a {lesion_noun.lower()} is present.")

    c1, c2, c3 = st.columns(3)
    frames = [
        (c1, rgb, scan_cfg["image_caption"], None),
        (c2, mask * 255, "Predicted Mask", "clamp"),
        (c3, overlay, "Overlay View", None),
    ]
    for i, (col, img, caption, clamp) in enumerate(frames):
        with col:
            if clamp:
                st.image(img, use_container_width=True, clamp=True)
            else:
                st.image(img, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center; padding-top:0.6rem; font-family:var(--font-mono); '
                f'font-size:0.68rem; color:var(--ink-70); text-transform:uppercase; letter-spacing:0.08em;">'
                f'{caption}</div>',
                unsafe_allow_html=True
            )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(f"""
<div class="app-footer">
    {scan_cfg['footer']}
</div>
""", unsafe_allow_html=True)
