"""Rendering helpers for the Clinical Cobalt Streamlit presentation."""
from __future__ import annotations

from html import escape
from typing import Mapping

from src.inference import PredictionResult

MODEL_LABELS = {"convnext": "ConvNeXt-Tiny", "efficientnet": "EfficientNetV2-S", "multimodal": "Multimodal DINOv2"}
VERSION_LABELS = {"v1_frozen": "V1", "v2_adaptive": "V2"}
VERSION_DISPLAY_NAMES = {"v1_frozen": "V1 Frozen", "v2_adaptive": "V2 Adaptive"}
DDI_URL = "https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images"
DDI_LINK = f'<a href="{DDI_URL}" target="_blank" rel="noopener noreferrer">DDI</a>'
PROJECT_METRICS = {"development": ("0.922 ROC-AUC", "0.848 Macro-F1", "92.5% malignant sensitivity"), "v1_ddi_images": f"656 images from an independent external dataset not used in training or validation ({DDI_LINK})", "v2_ddi": "57.3% → 68.4% malignant sensitivity · +11.1 percentage points · 19 additional malignant lesions detected"}
VERSION_COMPARISON = """#### V1 — Frozen
- Original frozen research ensemble
- ConvNeXt-Tiny + EfficientNetV2-S + Multimodal DINOv2
- Equal ensemble weighting
- Uses historical missing-metadata handling
- Configuration used for the original evaluation on an independent external dataset not used in training or validation ([DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images))
- Showed a more balanced historical sensitivity/specificity profile on that external evaluation

#### V2 — Adaptive
- Newer metadata-aware deployment version
- Uses Multimodal DINOv2 when supported metadata is provided
- Uses the saved adaptive three-model policy when metadata is available
- Uses ConvNeXt + EfficientNet when no metadata is supplied
- In the later post-hoc no-metadata comparison on the same independent external dataset ([DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images)), detected more malignant lesions / showed higher malignant sensitivity

> V1 is the original externally evaluated configuration with a more balanced historical external-test profile. V2 is a newer metadata-aware deployment configuration that showed higher malignant sensitivity in the later post-hoc no-metadata comparison while making more positive predictions overall."""


def render_hero(st) -> None:
    st.markdown("""<section class="hero"><h1>Skin Lesion Classification Model</h1><p>Deep-learning research demo for benign vs malignant lesion classification</p><p class="creator">Created by Samuel Dankner</p></section>""", unsafe_allow_html=True)


def render_version_comparison(st) -> None:
    """Keep deployment tradeoffs available without expanding the main layout."""
    with st.popover("Compare V1 and V2", use_container_width=True):
        st.markdown(VERSION_COMPARISON)


def render_usage_guide(st) -> None:
    with st.expander("How to use this demo", expanded=False):
        st.markdown("""#### What to upload
Upload a clear photograph containing a **focal skin lesion** you want classified, such as a mole-like, suspicious, or other localized lesion similar to images used during development. Center the lesion where practical, avoid extreme blur or severe obstruction, and avoid images with no identifiable focal lesion.

#### What the model predicts
This is a **benign vs malignant lesion classifier**. It reports a benign or malignant classification and a **malignant probability**.

#### What not to upload
It is not designed for normal or blank skin, acne without a focal lesion, diffuse rashes, cuts, bruises, burns, wounds, non-skin photographs, or multiple unrelated regions without one clear focal lesion. It is not a general skin-condition classifier, lesion detector, infection classifier, healthy-skin classifier, or general dermatology diagnosis system.

#### Important distinction
**The classifier assumes that an appropriate focal skin lesion is already present in the image. It does not first determine whether a lesion exists; it classifies an uploaded focal lesion as benign or malignant.**

#### Optional metadata
You may provide age, sex, and anatomical site. For v2, at least one supplied field activates the multimodal DINOv2 component. Providing all available supported metadata gives the multimodal model the most complete input, but does not guarantee accuracy. v1 always passes metadata through its historical persisted missing-value preprocessing.

Research and educational demonstration only. This system is not a medical device and is not intended for diagnosis or treatment decisions.""")


def render_result(st, result: PredictionResult, variant: str) -> None:
    classification = escape(result.predicted_class.title())
    css_class = "result-malignant" if result.predicted_class == "malignant" else "result-benign"
    ensemble = "Final Ensemble" if variant == "v1_frozen" else "Adaptive Ensemble"
    st.markdown(f'''<section class="clinical-card"><div class="eyebrow">Analysis result</div><div class="muted">Model classification</div><div class="result-label {css_class}">{classification}</div><div class="probability">{result.malignant_probability:.1%}</div><div class="muted">Malignant probability</div><p class="fineprint">Decision threshold: {result.threshold:.0%} · {ensemble}</p><p class="fineprint">Analyzed with {VERSION_DISPLAY_NAMES[variant]}</p></section>''', unsafe_allow_html=True)


def render_model_outputs(st, result: PredictionResult, variant: str) -> None:
    ensemble_label = "Final Ensemble" if variant == "v1_frozen" else "Adaptive Ensemble"
    st.markdown('<section class="clinical-card"><div class="eyebrow">Model outputs</div><p class="muted">Each value is a malignant probability.</p>', unsafe_allow_html=True)
    values: Mapping[str, float] = {**result.individual_models, "ensemble": result.malignant_probability}
    labels = {**MODEL_LABELS, "ensemble": ensemble_label}
    for name in ("convnext", "efficientnet", "multimodal", "ensemble"):
        if name in result.inactive_models:
            st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>N/A</span></div><p class="muted">Not used — no metadata provided</p>', unsafe_allow_html=True)
        else:
            value = float(values[name])
            st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>{value:.1%}</span></div>', unsafe_allow_html=True)
            st.progress(value)
    st.markdown("</section>", unsafe_allow_html=True)


def render_about(st) -> None:
    with st.expander("About This Model", expanded=False):
        st.markdown(f'''<section class="clinical-card"><p>A PyTorch skin-lesion classification system combining convolutional and transformer-based vision models with optional structured metadata.</p><p class="muted">ConvNeXt-Tiny · EfficientNetV2-S · Multimodal DINOv2 · metadata fusion · leakage-aware development · validation-based selection · ensemble inference · original v1 evaluation on an independent external dataset not used in training or validation ({DDI_LINK}) · adaptive v2 deployment policy · Streamlit interface · Docker-ready deployment</p></section>''', unsafe_allow_html=True)

    with st.expander("Project Highlights", expanded=False):
        cards = [*PROJECT_METRICS["development"], f"v1: {PROJECT_METRICS['v1_ddi_images']}", f"v2 post-hoc comparison on the same independent external dataset: {PROJECT_METRICS['v2_ddi']}"]
        columns = st.columns(2)
        for index, text in enumerate(cards):
            with columns[index % 2]: st.markdown(f'<section class="highlight-card">{text}</section>', unsafe_allow_html=True)
        st.caption("v1 is the independent external evaluation; the v2 comparison on the same external dataset is post-hoc.")
        with st.expander("Research & evaluation details", expanded=False):
            st.markdown("v1 was frozen before its original evaluation on an independent external dataset not used in training or validation ([DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images)). v2 was developed later; its comparison on the same external dataset is post-hoc, and its weighting search used validation data only.")


def render_next_steps(st) -> None:
    with st.expander("Next Steps", expanded=False):
        items = (("New external validation", "Evaluate v2 on a new untouched external dataset."), ("Broader training data", "Expand coverage across environments, skin tones, presentations, and cameras."), ("Calibration", "Research probabilities that better correspond to observed risk."), ("Explainability", "Add Grad-CAM and transformer interpretability visualizations."), ("Metadata robustness", "Improve multimodal behavior with partial metadata."), ("Prospective testing", "Validate the full workflow on newly collected data."))
        columns = st.columns(2)
        for index, (title, body) in enumerate(items):
            with columns[index % 2]: st.markdown(f'<section class="next-card"><strong>{title}</strong><p>{body}</p></section>', unsafe_allow_html=True)


def render_disclaimer(st) -> None:
    st.markdown("""<div class="disclaimer">Research and educational demonstration only. This system is not a medical device and must not be used for diagnosis, treatment decisions, or other clinical decision-making. Consult a qualified healthcare professional for medical concerns.</div>""", unsafe_allow_html=True)
