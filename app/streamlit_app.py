"""Clinical Cobalt Streamlit UI for versioned deployment ensembles."""
from __future__ import annotations

import hashlib
import logging

from app.styles import CLINICAL_COBALT_CSS
from app.ui import VERSION_LABELS, render_about, render_attribution, render_disclaimer, render_hero, render_model_outputs, render_next_steps, render_result, render_usage_guide, render_version_comparison
from src.inference import DEPLOYMENT_VARIANTS, SkinCancerPredictor, validate_uploaded_image

LOGGER = logging.getLogger(__name__)
SEX_OPTIONS = ("Not provided", "Female", "Male")
SITE_OPTIONS = ("Not provided", "Head / neck", "Upper extremity", "Lower extremity", "Trunk", "Hand / foot")
SITE_VALUES = {"Head / neck": "head_neck", "Upper extremity": "upper_extremity", "Lower extremity": "lower_extremity", "Trunk": "trunk", "Hand / foot": "hand_foot"}

try:
    import streamlit as st
except ImportError:
    st = None

if st is not None:
    @st.cache_resource(show_spinner=False)
    def load_predictor(variant: str = "v2_adaptive") -> SkinCancerPredictor:
        """Lazily cache each immutable deployment bundle under its variant key."""
        if variant not in DEPLOYMENT_VARIANTS:
            raise ValueError(f"Unsupported deployment variant: {variant}")
        return SkinCancerPredictor.from_frozen_config(variant=variant)
else:
    def load_predictor(variant: str = "v2_adaptive") -> SkinCancerPredictor:
        raise RuntimeError("Streamlit is not installed. Install requirements-app.txt before launching the demo.")


def _metadata(age, sex: str, site: str) -> dict:
    return {"age": age, "sex": None if sex == "Not provided" else sex.lower(), "anatomical_site": None if site == "Not provided" else SITE_VALUES[site]}


def _input_signature(uploaded, metadata: dict, variant: str):
    upload = (uploaded.name, uploaded.size, hashlib.sha256(uploaded.getvalue()).hexdigest()) if uploaded else None
    return upload, tuple(metadata.items()), variant


def _invalidate_if_inputs_changed(signature) -> None:
    if st.session_state.get("analysis_signature") != signature:
        st.session_state["analysis_signature"] = signature
        st.session_state.pop("prediction", None)
        st.session_state.pop("prediction_variant", None)
        st.session_state.pop("attribution", None)


def main() -> None:
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements-app.txt before launching the demo.")
    st.set_page_config(page_title="Skin Lesion Classification Model", page_icon="◈", layout="wide")
    st.markdown(CLINICAL_COBALT_CSS, unsafe_allow_html=True)
    render_hero(st)
    render_usage_guide(st)
    st.markdown('<div class="selector-label">Model Version</div>', unsafe_allow_html=True)
    selector_col, comparison_col = st.columns((2, 1), vertical_alignment="bottom")
    with selector_col:
        variant = st.radio("Model Version", DEPLOYMENT_VARIANTS, index=1, format_func=VERSION_LABELS.get, horizontal=True, label_visibility="collapsed")
    with comparison_col:
        render_version_comparison(st)
    uploaded = st.file_uploader("Upload a focal skin-lesion photograph", type=["png", "jpg", "jpeg"], help="PNG, JPG, or JPEG up to 10 MB. Images are processed in memory.")
    st.markdown('<section class="metadata-card"><div class="eyebrow">Optional Information About You</div><p class="muted">If you choose to share it, your age, sex, and lesion location can provide additional context to the multimodal model.</p>', unsafe_allow_html=True)
    age_col, sex_col, site_col = st.columns(3)
    with age_col:
        age = st.number_input("Age", min_value=0, max_value=120, value=None)
        st.caption("Enter age in years")
    with sex_col:
        sex = st.selectbox("Sex", SEX_OPTIONS)
        st.caption("Select the available sex value")
    with site_col:
        site = st.selectbox("Anatomical Site", SITE_OPTIONS)
        st.caption("Body location of the lesion")
    metadata = _metadata(age, sex, site)
    st.caption("Sharing information about you is optional. For V2, adding your age, sex, or lesion location activates the multimodal DINOv2 component; V1 retains its historical missing-value preprocessing.")
    _invalidate_if_inputs_changed(_input_signature(uploaded, metadata, variant))
    analyze = st.button("Analyze image", type="primary", disabled=uploaded is None)
    st.markdown("</section>", unsafe_allow_html=True)
    if analyze and uploaded:
        try:
            image = validate_uploaded_image(uploaded.getvalue(), uploaded.name)
            with st.spinner("Loading selected model version..."): predictor = load_predictor(variant)
            with st.spinner("Analyzing image..."): result = predictor.predict(image, **metadata)
            st.session_state.prediction, st.session_state.prediction_variant = result, variant
            st.rerun()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.exception("Streamlit inference failed")
            st.session_state.pop("prediction", None)
            st.error(f"Unable to analyze this image: {exc}")
    preview_col, result_col = st.columns((1, 1), gap="large")
    with preview_col:
        st.markdown('<section class="clinical-card"><div class="eyebrow">Uploaded image</div>', unsafe_allow_html=True)
        if uploaded: st.image(uploaded, use_container_width=True)
        else: st.markdown('<p class="muted">Upload a focal skin-lesion photograph to run the classification ensemble.</p>', unsafe_allow_html=True)
        st.markdown("</section>", unsafe_allow_html=True)
    with result_col:
        result = st.session_state.get("prediction")
        if result: render_result(st, result, variant)
        else: st.markdown('<section class="clinical-card"><div class="eyebrow">Analysis result</div><p class="muted">Results will appear here after you upload an image and select Analyze image.</p></section>', unsafe_allow_html=True)
    result = st.session_state.get("prediction")
    if result:
        if "multimodal" in result.inactive_models: st.info("No optional information was provided. Multimodal DINOv2 was not used.")
        elif variant == "v1_frozen": st.info("V1 uses the multimodal model with its persisted missing-value preprocessing.")
        else: st.info("Multimodal inference is active. Missing optional information is handled by the saved preprocessing pipeline.")
        render_model_outputs(st, result, variant)
    render_attribution(st, uploaded, result, variant, metadata)
    render_about(st)
    render_next_steps(st)
    render_disclaimer(st)


if __name__ == "__main__":
    main()
