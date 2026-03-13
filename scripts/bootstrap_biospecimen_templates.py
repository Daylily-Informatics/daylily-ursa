#!/usr/bin/env python3
"""Bootstrap TapDB templates for the biospecimen registry.

This ensures the TapDB templates used by `daylib_ursa.biospecimen.BiospecimenRegistry`
exist in the configured TapDB namespace (strict namespace mode).

Usage:
    python scripts/bootstrap_biospecimen_templates.py [--profile PROFILE] [--region REGION]
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap TapDB templates for BiospecimenRegistry")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"), help="AWS region (optional)")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"), help="AWS profile (optional)")
    args = parser.parse_args()

    if args.profile:
        os.environ.setdefault("AWS_PROFILE", args.profile)
    if args.region:
        os.environ.setdefault("AWS_REGION", args.region)

    biospecimen_registry = None
    last_import_error: Exception | None = None
    for module_name in ("daylib_ursa.biospecimen", "daylily_ursa.biospecimen"):
        try:
            module = __import__(module_name, fromlist=["BiospecimenRegistry"])
            biospecimen_registry = getattr(module, "BiospecimenRegistry", None)
            if biospecimen_registry is not None:
                break
        except ImportError as exc:
            last_import_error = exc

    if biospecimen_registry is None:
        print(
            "○ BiospecimenRegistry is not present in this checkout; "
            "skipping biospecimen template bootstrap"
        )
        if last_import_error is not None:
            print(f"  Last import error: {last_import_error}")
        return 0

    try:
        registry = biospecimen_registry()
        registry.bootstrap()
    except Exception as exc:
        print(f"Error bootstrapping templates: {exc}")
        return 1

    print("✓ TapDB templates bootstrapped for BiospecimenRegistry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
