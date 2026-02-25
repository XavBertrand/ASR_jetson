"""Key provider abstraction for mapping encryption keys."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass

from asr_jetson.anonymization.core.errors import SecurityPolicyError


@dataclass(frozen=True)
class KeyMaterial:
    key_id: str
    key_bytes: bytes


class KeyProvider:
    def resolve(self) -> KeyMaterial:
        raise NotImplementedError


class EnvKeyProvider(KeyProvider):
    """Read key id and bytes from environment variables."""

    def __init__(self, key_id_env: str = "ANON_KEY_ID", key_value_env: str = "ANON_MAPPING_KEY") -> None:
        self.key_id_env = key_id_env
        self.key_value_env = key_value_env

    def resolve(self) -> KeyMaterial:
        key_id = os.environ.get(self.key_id_env, "").strip()
        raw_key = os.environ.get(self.key_value_env, "").strip()
        if not key_id:
            raise SecurityPolicyError("Mapping key id unavailable")
        if not raw_key:
            raise SecurityPolicyError("Mapping key unavailable")

        try:
            key_bytes = base64.b64decode(raw_key, validate=True)
        except (ValueError, binascii.Error):
            key_bytes = raw_key.encode("utf-8")

        if len(key_bytes) not in (16, 24, 32):
            raise SecurityPolicyError("Mapping key length invalid")
        return KeyMaterial(key_id=key_id, key_bytes=key_bytes)


class KeystoreKeyProvider(KeyProvider):
    """Placeholder keystore provider interface.

    In this repository we keep a safe explicit failure path until a concrete
    keystore integration is wired.
    """

    def resolve(self) -> KeyMaterial:
        raise SecurityPolicyError("Keystore provider not configured")


def resolve_key_provider() -> KeyProvider:
    provider_name = os.environ.get("ANON_KEY_PROVIDER", "env").strip().lower()
    if provider_name == "env":
        return EnvKeyProvider()
    if provider_name == "keystore":
        return KeystoreKeyProvider()
    raise SecurityPolicyError("Unknown key provider")
