from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import assert_canonical_callable_path


@pytest.mark.unit
def test_guardrail_fails_on_alternate_backend_path() -> None:
    with pytest.raises(ValueError):
        assert_canonical_callable_path("asr_jetson.postprocessing.some_other_backend.run")
