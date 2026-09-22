"""Rendering helpers for the Clinical Cobalt Streamlit presentation."""
from __future__ import annotations

import logging
from html import escape
from pathlib import Path
from typing import Mapping

from src.inference import PredictionResult

LOGGER = logging.getLogger(__name__)

ATTRIBUTION_MODELS = {
    "ConvNeXt-Tiny": "convnext",
    "EfficientNetV2-S": "efficientnet",
    "DINOv2": "multimodal",
}

MODEL_LABELS = {"convnext": "ConvNeXt-Tiny", "efficientnet": "EfficientNetV2-S", "multimodal": "Multimodal DINOv2"}
VERSION_LABELS = {"v1_frozen": "V1", "v2_adaptive": "V2"}
VERSION_DISPLAY_NAMES = {"v1_frozen": "V1 Frozen", "v2_adaptive": "V2 Adaptive"}
DDI_URL = "https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images"
DDI_LINK = f'<a href="{DDI_URL}" target="_blank" rel="noopener noreferrer">DDI</a>'
CLOSE_UP_EXAMPLE_PATH = Path(__file__).with_name("assets") / "pad_ufes_20_close_up_examples.jpg"
PROJECT_METRICS = {
    "development": ("0.922 ROC-AUC", "0.848 Macro-F1", "92.5% malignant sensitivity"),
    "architecture": "Three-model CNN + transformer ensemble with optional age, sex, and lesion-location information",
    "methodology": "Leakage-aware patient, lesion, and image grouping with validation-only model and threshold selection",
    "v2_ddi": "+19.4% relative increase in malignant-lesion sensitivity in the post-hoc V2 comparison on previously inspected DDI · 19 additional malignant lesions detected",
}
VERSION_COMPARISON = """#### V1 — Frozen
- Original frozen research ensemble
- ConvNeXt-Tiny + EfficientNetV2-S + Multimodal DINOv2
- Equal ensemble weighting
- Uses historical missing-information handling
- Configuration used for the original evaluation on an independent external dataset not used in training or validation ([DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images))
- Showed a more balanced historical sensitivity/specificity profile on that external evaluation

#### V2 — Adaptive
- Newer information-aware deployment version
- Uses Multimodal DINOv2 when optional information about you is provided
- Uses the saved adaptive three-model policy when that information is available
- Uses ConvNeXt + EfficientNet when no optional information is supplied
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

#### Best photo inputs
Use one clearly visible focal skin lesion, reasonably centered and relatively close-up so it occupies a meaningful part of a well-lit, sharp image. Where practical, minimize obstruction from hair, clothing, fingers, rulers, and glare. Avoid extremely distant images, many unrelated lesions without a clear target, blank or normal skin, and unrelated dermatologic conditions. These characteristics do not guarantee a correct prediction.
""")
        st.image(CLOSE_UP_EXAMPLE_PATH, width=460)
        st.caption("Published close-up clinical-image examples from the PAD-UFES-20 development dataset. A close-up with one focal lesion is generally most useful; this example does not guarantee a correct classification. Source: PAD-UFES-20, CC BY 4.0.")
        st.markdown("""#### Optional information about you
If you choose to share it, you may add your age, sex, and lesion location. For V2, adding any one of these activates the multimodal DINOv2 component. Adding all available supported information gives the multimodal model the most complete input, but does not guarantee accuracy. V1 always passes this information through its historical persisted missing-value preprocessing.

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
            st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>N/A</span></div><p class="muted">Not used — no optional information provided</p>', unsafe_allow_html=True)
        else:
            value = float(values[name])
            st.markdown(f'<div class="model-row"><span>{labels[name]}</span><span>{value:.1%}</span></div>', unsafe_allow_html=True)
            st.progress(value)
    st.markdown("</section>", unsafe_allow_html=True)


def render_attribution(st, uploaded, result: PredictionResult | None, variant: str, metadata: dict) -> None:
    """Render the optional, on-demand attribution controls below model outputs."""
    from src.explainability import overlay_heatmap
    from src.inference import validate_uploaded_image
    from app.streamlit_app import load_predictor

    st.markdown('<section class="clinical-card"><div class="eyebrow">Model Attribution</div><p class="muted">Model attribution highlights image regions that had greater influence on an individual model\'s malignant output. It is an interpretability tool, not a medical explanation of why a lesion is benign or malignant.</p>', unsafe_allow_html=True)
    st.caption("Choose an individual model to visualize which image regions most influenced its output.")
    label = st.selectbox("Model", tuple(ATTRIBUTION_MODELS), key="attribution_model")
    model_name = ATTRIBUTION_MODELS[label]
    active = bool(result and model_name in result.active_models)
    if result and model_name == "multimodal" and model_name in result.inactive_models:
        st.info("DINOv2 was not active for this prediction because no metadata was provided.")
    if not result:
        st.caption("Analyze an image first to generate model attribution.")
    if st.button("Generate Attribution", type="primary", disabled=not active, key="generate_attribution"):
        try:
            image = validate_uploaded_image(uploaded.getvalue(), uploaded.name)
            with st.spinner("Generating model attribution..."):
                attribution = load_predictor(variant).attribute(image, model_name, **metadata)
            st.session_state["attribution"] = {"model": model_name, "method": attribution.method, "heatmap": attribution.heatmap}
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.exception("Streamlit attribution failed")
            st.error(f"Unable to generate attribution: {exc}")
    saved = st.session_state.get("attribution")
    if saved and saved.get("model") == model_name and uploaded:
        opacity = st.slider("Overlay opacity", min_value=0.1, max_value=0.9, value=0.45, step=0.05, key="attribution_opacity")
        st.caption("Overlay opacity controls how strongly the attribution heatmap is drawn over the original image. Lower values show more of the original photo; higher values emphasize the highlighted attribution regions.")
        image = validate_uploaded_image(uploaded.getvalue(), uploaded.name)
        original_col, map_col = st.columns(2)
        with original_col:
            st.markdown("**Original Image**")
            st.image(image, use_container_width=True)
        with map_col:
            st.markdown("**Attribution**")
            st.image(overlay_heatmap(image, saved["heatmap"], opacity), use_container_width=True)
        st.caption(f"Method: {saved['method']}")
        st.caption("Highlighted regions had greater influence on this model's malignant output.")
        st.caption("Attribution visualizations are interpretability tools and do not establish why a lesion is benign or malignant.")
        if saved["model"] == "multimodal":
            st.caption("This visualization reflects image-based attribution only; optional metadata contributions are not represented.")
    st.markdown("</section>", unsafe_allow_html=True)


def render_about(st) -> None:
    with st.expander("About This Model", expanded=False):
        st.markdown(f'''<section class="clinical-card"><p>A PyTorch skin-lesion classification system combining convolutional and transformer-based vision models with optional structured information about the person.</p><p class="muted">ConvNeXt-Tiny · EfficientNetV2-S · Multimodal DINOv2 · age/sex/location fusion · leakage-aware development · validation-based selection · ensemble inference · original V1 evaluation on an independent external dataset not used in training or validation ({DDI_LINK}) · adaptive V2 deployment policy · Streamlit interface · Docker-ready deployment</p><p><strong>Training &amp; development datasets</strong></p><p class="muted">MILK10k · PAD-UFES-20</p></section>''', unsafe_allow_html=True)

    with st.expander("Project Highlights", expanded=False):
        cards = [
            *PROJECT_METRICS["development"],
            PROJECT_METRICS["architecture"],
            PROJECT_METRICS["methodology"],
            PROJECT_METRICS["v2_ddi"],
        ]
        columns = st.columns(2)
        for index, text in enumerate(cards):
            with columns[index % 2]: st.markdown(f'<section class="highlight-card">{text}</section>', unsafe_allow_html=True)
        st.caption("The V1 evaluation is independent and external; the V2 comparison on the same external dataset is post-hoc.")
        with st.expander("Research & evaluation details", expanded=False):
            st.markdown("V1 was frozen before its original evaluation on an independent external dataset not used in training or validation ([DDI](https://aimi.stanford.edu/datasets/ddi-diverse-dermatology-images)). V2 was developed later; its comparison on the same external dataset is post-hoc, and its weighting search used validation data only.")


def render_next_steps(st) -> None:
    with st.expander("Next Steps", expanded=False):
        items = (("Safety-focused workflow research", "Study uncertainty, explanation, and human-review experiences before any clinical use."), ("Broader training data", "Expand coverage across environments, skin tones, presentations, and cameras."), ("Calibration", "Research probabilities that better correspond to observed risk."), ("Explainability", "Add Grad-CAM and transformer interpretability visualizations."), ("Information robustness", "Improve multimodal behavior when only some optional information is shared."), ("Prospective testing", "Validate the full workflow on newly collected data."))
        columns = st.columns(2)
        for index, (title, body) in enumerate(items):
            with columns[index % 2]: st.markdown(f'<section class="next-card"><strong>{title}</strong><p>{body}</p></section>', unsafe_allow_html=True)


def render_disclaimer(st) -> None:
    st.markdown("""<div class="disclaimer">Research and educational demonstration only. This system is not a medical device and must not be used for diagnosis, treatment decisions, or other clinical decision-making. Consult a qualified healthcare professional for medical concerns.</div>""", unsafe_allow_html=True)
