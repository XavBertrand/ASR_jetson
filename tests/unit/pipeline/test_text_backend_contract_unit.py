from __future__ import annotations

import pytest

from asr_jetson.pipeline.text_backend_adapter import (
    CANONICAL_BACKEND_CALLABLE,
    assert_canonical_callable_path,
)
from asr_jetson.pipeline.text_backend_contract import (
    CANONICAL_WARNING_CODE,
    CANONICAL_WARNING_LEVEL,
    CANONICAL_WARNING_MESSAGE,
    TextAnonymizationRequest,
    canonical_fallback_warning,
)


@pytest.mark.unit
def test_canonical_warning_values_are_stable() -> None:
    warning = canonical_fallback_warning()
    assert warning.warning_code == CANONICAL_WARNING_CODE
    assert warning.warning_level == CANONICAL_WARNING_LEVEL
    assert warning.warning_message == CANONICAL_WARNING_MESSAGE


@pytest.mark.unit
def test_request_defaults_are_stable() -> None:
    request = TextAnonymizationRequest(text="hello")
    assert request.preserve_dates is True
    assert request.model_name


@pytest.mark.unit
def test_assert_canonical_callable_path_guard() -> None:
    assert_canonical_callable_path(CANONICAL_BACKEND_CALLABLE)
    with pytest.raises(ValueError):
        assert_canonical_callable_path("asr_jetson.postprocessing.alt_backend.run")
