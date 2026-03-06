#!/usr/bin/env python3
"""Bootstrap TapDB templates for the file registry.

This ensures the TapDB templates used by `daylily_ursa.file_registry.FileRegistry` exist
in the configured TapDB namespace (strict namespace mode).

Usage:
    python scripts/bootstrap_file_registry_templates.py [--profile PROFILE] [--region REGION]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap TapDB templates for FileRegistry")
    parser.add_argument(
        "--region", default=os.environ.get("AWS_REGION", "us-west-2"), help="AWS region (optional)"
    )
    parser.add_argument(
        "--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile (optional)"
    )
    args = parser.parse_args()

    if args.profile:
        os.environ.setdefault("AWS_PROFILE", args.profile)
    if args.region:
        os.environ.setdefault("AWS_REGION", args.region)

    try:
        from daylily_ursa.file_registry import FileRegistry
    except ImportError as exc:
        print(f"Error importing FileRegistry: {exc}")
        return 1

    try:
        registry = FileRegistry()
        registry.bootstrap()
    except Exception as exc:
        print(f"Error bootstrapping templates: {exc}")
        return 1

    print("✓ TapDB templates bootstrapped for FileRegistry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
