#!/usr/bin/env python3
"""TapDB teardown helper.

Ursa no longer provisions/removes per-service NoSQL tables. Use migration/DBA tooling
for destructive actions against the TapDB database.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Print TapDB teardown guidance")
    parser.add_argument("--force", action="store_true", help="Acknowledge manual teardown flow")
    args = parser.parse_args()

    if not args.force:
        print("Refusing destructive operation without --force")
        print("Use: python scripts/drop_daylily_tables.py --force")
        return 1

    print("Manual teardown required for TapDB-backed Ursa.")
    print("1. Stop API/worker/monitor.")
    print("2. Snapshot PostgreSQL database.")
    print("3. Apply DBA-approved SQL cleanup/migration.")
    print("4. Re-run bootstrap: `ursa aws setup`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
