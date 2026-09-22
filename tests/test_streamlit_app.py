"""Smoke tests keep the Streamlit entrypoint separate from model loading."""
from __future__ import annotations

from types import SimpleNamespace


def test_streamlit_entrypoint_imports_without_loading_models():
    import app.streamlit_app as streamlit_app

    assert callable(streamlit_app.main)
    assert callable(streamlit_app.load_predictor)
    assert streamlit_app._metadata(None, "Not provided", "Not provided") == {
        "age": None, "sex": None, "anatomical_site": None,
    }


def test_ui_copy_documents_focal_lesion_task_and_versioned_demo():
    from app import ui

    guide_source = ui.render_usage_guide.__code__.co_consts
    guide = " ".join(value for value in guide_source if isinstance(value, str))
    assert "focal skin lesion" in guide
    assert "benign vs malignant lesion classifier" in guide
    assert "not a general skin-condition classifier" in guide
    assert "does not first determine whether a lesion exists" in guide
    assert "blank skin" in guide
    assert "Skin Lesion Classification Lab" in " ".join(value for value in ui.render_hero.__code__.co_consts if isinstance(value, str))
    assert "Created by Samuel Dankner" in " ".join(value for value in ui.render_hero.__code__.co_consts if isinstance(value, str))
    assert ui.VERSION_LABELS == {"v1_frozen": "v1 Frozen", "v2_adaptive": "v2 Adaptive"}
    assert "57.3% → 68.4%" in ui.PROJECT_METRICS["v2_ddi"]
    assert "+11.1 percentage points" in ui.PROJECT_METRICS["v2_ddi"]
    assert "19 additional malignant lesions" in ui.PROJECT_METRICS["v2_ddi"]


def test_input_signature_change_invalidates_stale_prediction(monkeypatch):
    import app.streamlit_app as streamlit_app

    fake_streamlit = SimpleNamespace(session_state={"analysis_signature": ("old",), "prediction": object(), "prediction_variant": "v1_frozen"})
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)
    streamlit_app._invalidate_if_inputs_changed(("new",))
    assert fake_streamlit.session_state["analysis_signature"] == ("new",)
    assert "prediction" not in fake_streamlit.session_state
    assert "prediction_variant" not in fake_streamlit.session_state
