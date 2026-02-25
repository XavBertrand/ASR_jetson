"""Explicit no-network-by-default guard utilities."""

from __future__ import annotations

from typing import Callable, TypeVar

from asr_jetson.anonymization.core.errors import SecurityPolicyError
from asr_jetson.anonymization.core.models import Policy

T = TypeVar("T")


def assert_network_allowed(policy: Policy, reason: str = "network_call") -> None:
    if not policy.allow_network:
        raise SecurityPolicyError(f"Outbound network disabled by policy ({reason})")


def guarded_call(policy: Policy, fn: Callable[[], T], reason: str = "network_call") -> T:
    assert_network_allowed(policy, reason=reason)
    return fn()
