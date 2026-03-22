"""Top-level ASR CLI multiplexer."""

from __future__ import annotations

import sys

from asr_jetson.pipeline.cli import main as pipeline_main


_HELP = """usage: asr {pipeline} ...

ASR Jetson command suite

Commands:
  pipeline    Run audio ASR pipeline (legacy default behavior)
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_HELP.rstrip())
        return 0

    command = args[0]
    if command in {"pipeline", "asr-pipeline"}:
        pipeline_main(args[1:])
        return 0
    if command.startswith("-"):
        # Backward-compatible mode: `asr --audio ...` behaves like legacy pipeline CLI.
        pipeline_main(args)
        return 0

    print(f"Unknown command: {command}")
    print(_HELP.rstrip())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
