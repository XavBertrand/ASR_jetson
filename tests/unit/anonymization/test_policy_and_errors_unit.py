from pathlib import Path

import pytest

from asr_jetson.anonymization.core.errors import PolicyValidationError
from asr_jetson.anonymization.core.policy import load_policy


@pytest.mark.unit
def test_load_policy_strict_offline_default() -> None:
    policy = load_policy("strict_offline", Path("configs/anonymization_profiles.yaml"))
    assert policy.policy_id == "strict_offline"
    assert policy.allow_network is False
    assert policy.enable_regex is True


@pytest.mark.unit
def test_unknown_policy_raises_sanitized_error() -> None:
    with pytest.raises(PolicyValidationError) as exc:
        load_policy("missing-profile", Path("configs/anonymization_profiles.yaml"))
    assert exc.value.code == "POLICY_VALIDATION_ERROR"
    assert "missing-profile" in exc.value.message_safe
