"""Encrypted reversible mapping persistence with AEAD."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from asr_jetson.anonymization.core.errors import SecurityPolicyError
from asr_jetson.anonymization.storage.fs_security import ensure_dir_permissions, ensure_file_permissions
from asr_jetson.anonymization.storage.key_provider import KeyProvider, resolve_key_provider


def _aad(case_id: str, schema_version: str, key_id: str, document_id: str) -> bytes:
    return f"{case_id}|{schema_version}|{key_id}|{document_id}".encode("utf-8")


def _mapping_filename(document_id: str) -> str:
    return f"{document_id}.enc.json"


class MappingStore:
    def __init__(self, key_provider: KeyProvider | None = None) -> None:
        self._provider = key_provider or resolve_key_provider()

    def write_mapping(
        self,
        *,
        case_id: str,
        document_id: str,
        schema_version: str,
        mapping: dict[str, str],
        output_root: Path,
        file_mode: int = 0o600,
        dir_mode: int = 0o700,
    ) -> Path:
        key = self._provider.resolve()
        aad = _aad(case_id, schema_version, key.key_id, document_id)

        clear_payload = {
            "case_id": case_id,
            "document_id": document_id,
            "schema_version": schema_version,
            "mapping": mapping,
        }
        plaintext = json.dumps(clear_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        nonce = os.urandom(12)
        cipher = AESGCM(key.key_bytes)
        ciphertext = cipher.encrypt(nonce, plaintext, aad)

        artifact = {
            "schema_version": schema_version,
            "encryption_alg": "AES-GCM",
            "key_id": key.key_id,
            "nonce_b64": base64.b64encode(nonce).decode("ascii"),
            "aad_digest": hashlib.sha256(aad).hexdigest(),
            "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
        }

        mappings_dir = output_root / "mappings"
        ensure_dir_permissions(mappings_dir, dir_mode)
        target = mappings_dir / _mapping_filename(document_id)
        target.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        ensure_file_permissions(target, file_mode)
        return target

    def read_mapping(self, *, case_id: str, document_id: str, mapping_path: Path) -> dict[str, str]:
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SecurityPolicyError("Encrypted mapping unreadable") from exc

        schema_version = str(payload.get("schema_version", ""))
        key_id = str(payload.get("key_id", ""))
        nonce_b64 = str(payload.get("nonce_b64", ""))
        ciphertext_b64 = str(payload.get("ciphertext_b64", ""))
        if not schema_version or not key_id or not nonce_b64 or not ciphertext_b64:
            raise SecurityPolicyError("Encrypted mapping metadata invalid")

        key = self._provider.resolve()
        if key.key_id != key_id:
            raise SecurityPolicyError("Mapping key mismatch")

        aad = _aad(case_id, schema_version, key_id, document_id)
        try:
            nonce = base64.b64decode(nonce_b64)
            ciphertext = base64.b64decode(ciphertext_b64)
            plaintext = AESGCM(key.key_bytes).decrypt(nonce, ciphertext, aad)
            clear_payload = json.loads(plaintext.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise SecurityPolicyError("Encrypted mapping verification failed") from exc

        if clear_payload.get("case_id") != case_id or clear_payload.get("document_id") != document_id:
            raise SecurityPolicyError("Encrypted mapping scope mismatch")
        mapping = clear_payload.get("mapping")
        if not isinstance(mapping, dict):
            raise SecurityPolicyError("Encrypted mapping payload invalid")
        return {str(k): str(v) for k, v in mapping.items()}
