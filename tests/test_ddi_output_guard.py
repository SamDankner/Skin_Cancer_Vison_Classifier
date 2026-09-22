from __future__ import annotations

from pathlib import Path

import pytest

from run_ddi_final_evaluation import _guard_output_lifecycle


def test_fresh_and_same_invocation_preflight_directories_are_allowed(tmp_path):
    fresh = tmp_path / "fresh"
    _guard_output_lifecycle(fresh)
    fresh.mkdir()
    (fresh / "ddi_dataset_report.json").write_text("{}", encoding="utf-8")
    (fresh / "ddi_overlap_audit.json").write_text("{}", encoding="utf-8")
    _guard_output_lifecycle(fresh)


def test_completed_or_unowned_ddi_artifacts_still_block_rerun(tmp_path):
    output = tmp_path / "ddi"; output.mkdir()
    (output / "ddi_dataset_report.json").write_text("{}", encoding="utf-8")
    (output / "ddi_report.json").write_text("completed", encoding="utf-8")
    with pytest.raises(FileExistsError, match="completed or unowned"):
        _guard_output_lifecycle(output)


def test_frozen_configuration_is_not_written_by_output_guard(tmp_path):
    frozen = tmp_path / "final_model.yaml"; frozen.write_text("threshold: 0.51\n", encoding="utf-8")
    before = frozen.read_bytes()
    _guard_output_lifecycle(tmp_path / "ddi")
    assert frozen.read_bytes() == before
