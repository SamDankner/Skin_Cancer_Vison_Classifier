"""Clinical Cobalt Streamlit UI for the versioned deployment ensembles."""
from __future__ import annotations

import logging

from app.styles import CLINICAL_COBALT_CSS
from app.ui import render_about, render_disclaimer, render_hero, render_model_outputs, render_result
from src.inference import SkinCancerPredictor, validate_uploaded_image

LOGGER = logging.getLogger(__name__)
SEX_OPTIONS = ("Not provided", "Female", "Male")
SITE_OPTIONS = ("Not provided", "Head / neck", "Upper extremity", "Lower extremity", "Trunk", "Hand / foot")
SITE_VALUES = {"Head / neck": "head_neck", "Upper extremity": "upper_extremity", "Lower extremity": "lower_extremity", "Trunk": "trunk", "Hand / foot": "hand_foot"}

try:
    import streamlit as st
except ImportError:  # permits lightweight source-level smoke tests
    st = None


if st is not None:
    @st.cache_resource(show_spinner=False)
    def load_predictor() -> SkinCancerPredictor:
        """Initialize the selected versioned model bundle once per process."""
        return SkinCancerPredictor.from_frozen_config()
else:
    def load_predictor() -> SkinCancerPredictor:
        raise RuntimeError("Streamlit is not installed. Install requirements-app.txt before launching the demo.")


def _metadata(age, sex: str, site: str) -> dict:
    return {"age": age, "sex": None if sex == "Not provided" else sex.lower(), "anatomical_site": None if site == "Not provided" else SITE_VALUES[site]}


def main() -> None:
    """Render the app without retaining uploads or stale predictions."""
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install requirements-app.txt before launching the demo.")
    st.set_page_config(page_title="Skin Lesion AI | Research Demo", page_icon="◈", layout="wide")
    st.markdown(CLINICAL_COBALT_CSS, unsafe_allow_html=True)
    render_hero(st)
    uploaded = st.file_uploader("Upload a focal skin-lesion photograph", type=["png", "jpg", "jpeg"], help="PNG, JPG, or JPEG up to 10 MB. Images are processed in memory.")
    signature = (uploaded.name, uploaded.size) if uploaded else None
    if st.session_state.get("upload_signature") != signature:
        st.session_state.upload_signature = signature
        st.session_state.pop("prediction", None)

    preview_col, result_col = st.columns((1, 1), gap="large")
    with preview_col:
        st.markdown('<section class="clinical-card"><div class="eyebrow">Uploaded image</div>', unsafe_allow_html=True)
        if uploaded:
            st.image(uploaded, use_container_width=True)
        else:
            st.markdown('<p class="muted">Upload a focal skin-lesion photograph to run the classification ensemble.</p>', unsafe_allow_html=True)
        st.markdown("</section>", unsafe_allow_html=True)
    with result_col:
        result = st.session_state.get("prediction")
        if result:
            render_result(st, result)
        else:
            st.markdown('<section class="clinical-card"><div class="eyebrow">Analysis result</div><p class="muted">Results will appear here after you upload an image and select Analyze image.</p></section>', unsafe_allow_html=True)

    st.markdown('<section class="metadata-card"><div class="eyebrow">Optional metadata</div>', unsafe_allow_html=True)
    age_col, sex_col, site_col = st.columns(3)
    with age_col:
        age = st.number_input("Age (optional)", min_value=0, max_value=120, value=None, placeholder="Not provided")
    with sex_col:
        sex = st.selectbox("Sex (optional)", SEX_OPTIONS)
    with site_col:
        site = st.selectbox("Anatomical site (optional)", SITE_OPTIONS)
    st.caption("Providing all available metadata gives the multimodal model the most complete input. If no metadata is provided, the multimodal model will not be used.")
    analyze = st.button("Analyze image", type="primary", disabled=uploaded is None)
    st.markdown("</section>", unsafe_allow_html=True)

    if analyze and uploaded:
        try:
            image = validate_uploaded_image(uploaded.getvalue(), uploaded.name)
            with st.spinner("Loading models..."):
                predictor = load_predictor()
            with st.spinner("Analyzing image..."):
                st.session_state.prediction = predictor.predict(image, **_metadata(age, sex, site))
            st.rerun()
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            LOGGER.exception("Streamlit inference failed")
            st.session_state.pop("prediction", None)
            st.error(f"Unable to analyze this image: {exc}")

    result = st.session_state.get("prediction")
    if result:
        if "multimodal" in result.inactive_models:
            st.info("No metadata provided. Multimodal DINOv2 was not used.")
        elif all(result.metadata_used.values()):
            st.info("All supported metadata provided.")
        else:
            st.info("Multimodal inference is active. Missing metadata fields are handled by the saved preprocessing pipeline.")
        render_model_outputs(st, result)
    render_about(st)
    render_disclaimer(st)


if __name__ == "__main__":
    main()
