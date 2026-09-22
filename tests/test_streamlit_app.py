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


def test_ui_copy_documents_focal_lesion_task_and_refined_versioned_demo():
    from app import ui

    guide_source = ui.render_usage_guide.__code__.co_consts
    guide = " ".join(value for value in guide_source if isinstance(value, str))
    assert "focal skin lesion" in guide
    assert "benign vs malignant lesion classifier" in guide
    assert "not a general skin-condition classifier" in guide
    assert "does not first determine whether a lesion exists" in guide
    assert "blank skin" in guide
    hero = " ".join(value for value in ui.render_hero.__code__.co_consts if isinstance(value, str))
    assert "Skin Lesion Classification Model" in hero
    assert "Skin Lesion Classification Lab" not in hero
    assert "Created by Samuel Dankner" in " ".join(value for value in ui.render_hero.__code__.co_consts if isinstance(value, str))
    assert ui.VERSION_LABELS == {"v1_frozen": "v1 Frozen", "v2_adaptive": "v2 Adaptive"}
    assert "57.3% → 68.4%" in ui.PROJECT_METRICS["v2_ddi"]
    assert "+11.1 percentage points" in ui.PROJECT_METRICS["v2_ddi"]
    assert "19 additional malignant lesions" in ui.PROJECT_METRICS["v2_ddi"]


def test_refinement_uses_visible_controls_and_collapsed_information_sections():
    from app import streamlit_app, ui

    source = streamlit_app.main.__code__.co_consts
    page_copy = " ".join(value for value in source if isinstance(value, str))
    assert "Model Version" in page_copy
    assert "Optional Metadata" in page_copy
    assert "Supported metadata can provide additional context" in page_copy
    assert "Age" in page_copy and "Sex" in page_copy and "Anatomical Site" in page_copy
    assert "Compare model versions" in " ".join(value for value in ui.render_version_comparison.__code__.co_consts if isinstance(value, str))
    assert "original frozen research ensemble" in ui.VERSION_COMPARISON
    assert "post-hoc" in ui.VERSION_COMPARISON
    assert "better overall" not in ui.VERSION_COMPARISON
    assert "more accurate overall" not in ui.VERSION_COMPARISON
    assert ui.VERSION_LABELS == {"v1_frozen": "v1 Frozen", "v2_adaptive": "v2 Adaptive"}
    assert "Original frozen research ensemble" in ui.VERSION_SUMMARIES["v1_frozen"]
    assert "Equal weighting · all three models · original external DDI evaluation" in ui.VERSION_SUMMARIES["v1_frozen"]
    assert "Newer metadata-aware deployment version" in ui.VERSION_SUMMARIES["v2_adaptive"]
    assert "Adaptive three-model behavior with metadata · two-model behavior without metadata" in ui.VERSION_SUMMARIES["v2_adaptive"]
    assert "v1: Original externally evaluated three-model ensemble" in ui.VERSION_TRADEOFF
    assert "v2: Newer metadata-aware version" in ui.VERSION_TRADEOFF
    assert "better overall" not in ui.VERSION_TRADEOFF
    assert "more accurate overall" not in ui.VERSION_TRADEOFF

    field_copy = " ".join(value for value in source if isinstance(value, str))
    assert "Age in years" in field_copy
    assert "Enter age in years" in field_copy
    assert "Select the available sex value" in field_copy
    assert "Body location of the lesion" in field_copy

    class FakeStreamlit:
        def __init__(self): self.expanders = []
        def expander(self, label, expanded=False):
            self.expanders.append((label, expanded))
            return _NullContext()
        def markdown(self, *_args, **_kwargs): pass
        def columns(self, count): return [_NullContext() for _ in range(count)]
        def caption(self, *_args, **_kwargs): pass
    class _NullContext:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def markdown(self, *_args, **_kwargs): pass

    fake = FakeStreamlit()
    ui.render_about(fake)
    ui.render_next_steps(fake)
    assert ("About This Model", False) in fake.expanders
    assert ("Project Highlights", False) in fake.expanders
    assert ("Next Steps", False) in fake.expanders


def test_version_overview_renders_visible_summaries_and_tradeoff():
    from app import ui

    class FakeStreamlit:
        def __init__(self): self.markdowns = []
        def columns(self, count): return [_NullContext(self) for _ in range(count)]
        def markdown(self, text, **_kwargs): self.markdowns.append(text)
    class _NullContext:
        def __init__(self, parent): self.parent = parent
        def __enter__(self): return self.parent
        def __exit__(self, *_args): return False

    fake = FakeStreamlit()
    ui.render_version_overview(fake)
    rendered = " ".join(fake.markdowns)
    assert "v1 Frozen" in rendered
    assert "v2 Adaptive" in rendered
    assert "Original frozen research ensemble" in rendered
    assert "Newer metadata-aware deployment version" in rendered
    assert "later post-hoc no-metadata comparison" in rendered


def test_input_signature_change_invalidates_stale_prediction(monkeypatch):
    import app.streamlit_app as streamlit_app

    fake_streamlit = SimpleNamespace(session_state={"analysis_signature": ("old",), "prediction": object(), "prediction_variant": "v1_frozen"})
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)
    streamlit_app._invalidate_if_inputs_changed(("new",))
    assert fake_streamlit.session_state["analysis_signature"] == ("new",)
    assert "prediction" not in fake_streamlit.session_state
    assert "prediction_variant" not in fake_streamlit.session_state
