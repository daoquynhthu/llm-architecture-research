#!/usr/bin/env python3
"""Lightweight manifest checker for frontier-architectures.yaml.

This script intentionally avoids third-party dependencies so it can run in a
fresh Python environment. It performs structural checks only; it is not a full
YAML validator.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "frontier-architectures.yaml"

REQUIRED_TOKENS = [
    "schema_version:",
    "repository_license:",
    "families:",
    "status:",
    "upstream:",
]


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing manifest: {MANIFEST}", file=sys.stderr)
        return 1

    text = MANIFEST.read_text(encoding="utf-8")
    missing = [token for token in REQUIRED_TOKENS if token not in text]
    if missing:
        print("manifest is missing required tokens:", file=sys.stderr)
        for token in missing:
            print(f"  - {token}", file=sys.stderr)
        return 1

    if "model_weights: do_not_mirror_by_default" not in text:
        print("manifest must preserve the no-default-weight-mirroring policy", file=sys.stderr)
        return 1

    print("manifest structural check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
