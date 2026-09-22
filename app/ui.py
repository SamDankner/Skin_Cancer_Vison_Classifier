"""Small rendering helpers that keep visual code out of the app entrypoint."""
from __future__ import annotations

from html import escape
from typing import Mapping

from src.inference import PredictionResult


MODEL_LABELS = {
    "convnext": "ConvNeXt-Tiny",
    "efficientnet": "EfficientNetV2-S",
    "multimodal": "Multimodal DINOv2",
}


def render_hero(st) -> None:
    st.markdown("""<section class="hero"><h1>Skin Lesion AI</h1><p>Deep-learning research demo for skin-lesion classification from ordinary clinical photographs.</p></section>""", unsafe_allow_html=True)


def render_result(st, result: PredictionResult) -> None:
    classification = escape(result.predicted_class.title())
    css_class = "result-malignant" if result.predicted_class == "malignant" else "result-benign"
    st.markdown(
        f'''<section class="clinical-card"><div class="eyebrow">Analysis result</div>
        <div class="muted">Model classification</div><div class="result-label {css_class}">{classification}</div>
        <div class="probability">{result.malignant_probability:.1%}</div><div class="muted">Malignant probability</div>
        <p class="fineprint">Decision threshold: {result.threshold:.0%}</p>
        <p class="fineprint">Adaptive ensemble combines available model outputs using a validation-selected consensus strategy.</p></section>''',
        unsafe_allow_html=True,
    )


def render_model_outputs(st, result: PredictionResult) -> None:
    st.markdown('<section class="clinical-card"><div class="eyebrow">Model outputs</div><p class="muted">Each value is a malignant probability.</p>', unsafe_allow_html=True)
    values: Mapping[str, float] = {**result.individual_models, "ensemble": result.malignant_probability}
    labels = {**MODEL_LABELS, "ensemble": "Adaptive Ensemble"}
    for name in ("convnext", "efficientnet", "multimodal", "ensemble"):
        if name == "multimodal" and name in result.inactive_models:
            st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>N/A</span></div><p class="muted">Not used — no metadata provided</p>', unsafe_allow_html=True)
            continue
        value = float(values[name])
        st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>{value:.1%}</span></div>', unsafe_allow_html=True)
        st.progress(value)
    st.markdown("</section>", unsafe_allow_html=True)


def render_about(st) -> None:
    with st.expander("About this model"):
        st.markdown("""This PyTorch research demo combines ConvNeXt-Tiny, EfficientNetV2-S, and multimodal DINOv2. The multimodal member accepts optional age, sex, and anatomical-site metadata; it is used when at least one of these fields is supplied.

The final ensemble uses the multimodal DINOv2 model. A separate standalone DINOv2 development model exists in the repository but is not part of the final ensemble and was not evaluated on DDI.

**Held-out development test:** ROC-AUC 0.922 · Macro-F1 0.848 · Balanced accuracy 0.846 · Malignant sensitivity 92.5%.

**Independent DDI external test (v1 frozen ensemble):** N = 656 · ROC-AUC 0.721 · Macro-F1 0.651 · Sensitivity 57.3% · Specificity 75.9%. These results apply to the historical equal-weight v1 system, not the post-DDI v2 experimental deployment.""")

    with st.expander("Future Development"):
        st.markdown("Additional independent external validation, broader training-domain diversity, adaptive-ensemble evaluation, calibration research, explainability/Grad-CAM, robustness testing, metadata robustness, and prospective evaluation are planned areas of investigation.")


def render_disclaimer(st) -> None:
    st.markdown("""<div class="disclaimer">Research and educational demonstration only. This system is not a medical device and must not be used for diagnosis, treatment decisions, or other clinical decision-making. Consult a qualified healthcare professional for medical concerns.</div>""", unsafe_allow_html=True)
