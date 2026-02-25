"""CLI command for batch document anonymization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from asr_jetson.anonymization.core.errors import (
    AnonymizationError,
    InputValidationError,
    SecurityPolicyError,
)
from asr_jetson.anonymization.core.models import BatchRequest
from asr_jetson.anonymization.core.policy import load_policy
from asr_jetson.anonymization.core.service import DocumentAnonymizer
from asr_jetson.anonymization.storage.mapping_store import MappingStore


def _discover_inputs(input_arg: Path, patterns: tuple[str, ...]) -> list[Path]:
    if input_arg.is_file():
        return [input_arg]
    if not input_arg.exists():
        raise InputValidationError(f"Input path does not exist: {input_arg}")

    files: list[Path] = []
    for path in sorted(input_arg.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in patterns:
            files.append(path)
    if not files:
        raise InputValidationError("No supported input files found")
    return files


def _require_batch_args(args: argparse.Namespace) -> None:
    missing: list[str] = []
    for field in ("input", "output", "case_id", "policy", "report"):
        value = getattr(args, field)
        if value is None or str(value).strip() == "":
            missing.append(field)
    if missing:
        raise InputValidationError(f"Missing required batch arguments: {', '.join(missing)}")


def _authorize_mapping_resolution(provided_token: str | None) -> None:
    expected = os.environ.get("ANON_INTERNAL_API_KEY", "").strip()
    provided = (provided_token or "").strip()
    if not expected or not provided or provided != expected:
        raise SecurityPolicyError("Mapping access denied")


def _handle_mapping_resolution(args: argparse.Namespace) -> int:
    if not args.case_id:
        raise InputValidationError("--case-id is required for mapping resolution")
    if not args.resolve_document_id:
        raise InputValidationError("--resolve-document-id is required for mapping resolution")

    _authorize_mapping_resolution(args.internal_api_key)
    store = MappingStore()
    mapping = store.read_mapping(
        case_id=args.case_id,
        document_id=args.resolve_document_id,
        mapping_path=Path(args.resolve_mapping).expanduser().resolve(),
    )
    serialized = json.dumps({"mapping": mapping}, indent=2, ensure_ascii=False)
    if args.resolve_output:
        target = Path(args.resolve_output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(serialized, encoding="utf-8")
        print(f"Mapping resolved: {target}")
    else:
        print(serialized)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch anonymize documents")

    parser.add_argument("--input", required=False, help="Path to a file or directory")
    parser.add_argument("--output", required=False, help="Output root directory")
    parser.add_argument("--case-id", required=False, help="Deterministic case scope")
    parser.add_argument("--policy", required=False, help="Policy profile name")
    parser.add_argument("--report", required=False, help="Report JSON destination")

    parser.add_argument(
        "--mapping",
        default="auto",
        choices=("auto", "always", "never"),
        help="Mapping emission mode",
    )
    parser.add_argument("--config", default="configs/anonymization_profiles.yaml", help="Policy config path")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first document failure")

    parser.add_argument("--resolve-mapping", default=None, help="Decrypt a mapping artifact path")
    parser.add_argument("--resolve-document-id", default=None, help="Document id bound to mapping artifact")
    parser.add_argument("--resolve-output", default=None, help="Optional output path for decrypted mapping")
    parser.add_argument(
        "--internal-api-key",
        default=None,
        help="Internal authorization token (equivalent to X-Internal-API-Key)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.resolve_mapping:
            return _handle_mapping_resolution(args)

        _require_batch_args(args)
        input_path = Path(args.input).expanduser().resolve()
        output_root = Path(args.output).expanduser().resolve()
        report_path = Path(args.report).expanduser().resolve()
        policy = load_policy(args.policy, Path(args.config).expanduser().resolve())

        inputs = _discover_inputs(input_path, (".pdf", ".docx", ".xlsx", ".txt"))
        if len(inputs) > policy.max_documents_per_batch:
            raise InputValidationError("Document batch exceeds max_documents_per_batch")

        service = DocumentAnonymizer()
        result = service.anonymize_batch(
            BatchRequest(
                case_id=args.case_id,
                policy_name=args.policy,
                policy=policy,
                input_paths=inputs,
                output_root=output_root,
                report_path=report_path,
                mapping_mode=args.mapping,
                continue_on_error=not args.fail_fast,
            )
        )

        print(f"Anonymization completed: {result.status}")
        print(f"Report: {result.report_path}")
        return 0 if result.totals["failed"] == 0 else 10
    except AnonymizationError as exc:
        print(f"ERROR {exc.code}: {exc.message_safe}")
        return 40 if exc.code == "SECURITY_POLICY_ERROR" else 20


if __name__ == "__main__":
    raise SystemExit(main())
