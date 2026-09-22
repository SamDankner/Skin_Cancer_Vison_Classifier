"""Smoke tests keep the Streamlit entrypoint separate from model loading."""
from __future__ import annotations


def test_streamlit_entrypoint_imports_without_loading_models():
    import app.streamlit_app as streamlit_app

    assert callable(streamlit_app.main)
    assert callable(streamlit_app.load_predictor)
    assert streamlit_app._metadata(None, "Not provided", "Not provided") == {
        "age": None, "sex": None, "anatomical_site": None,
    }
