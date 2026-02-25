"""Anonymization-specific safe error models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnonymizationError(Exception):
    """Base sanitized exception carrying a stable code."""

    code: str
    message_safe: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message_safe}"


class PolicyValidationError(AnonymizationError):
    def __init__(self, message_safe: str) -> None:
        super().__init__("POLICY_VALIDATION_ERROR", message_safe)


class InputValidationError(AnonymizationError):
    def __init__(self, message_safe: str) -> None:
        super().__init__("INPUT_VALIDATION_ERROR", message_safe)


class ProcessingError(AnonymizationError):
    def __init__(self, message_safe: str) -> None:
        super().__init__("PROCESSING_ERROR", message_safe)


class SecurityPolicyError(AnonymizationError):
    def __init__(self, message_safe: str) -> None:
        super().__init__("SECURITY_POLICY_ERROR", message_safe)
