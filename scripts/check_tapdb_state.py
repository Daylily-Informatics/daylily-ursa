#!/usr/bin/env python3
"""Inspect Ursa TapDB-backed workset records."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from daylib.workset_state_db import WorksetState, WorksetStateDB


def _json_default(obj: Any):
    return str(obj)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect TapDB-backed workset records")
    parser.add_argument("--table-name", default="tapdb-worksets", help="Logical table name label")
    parser.add_argument("--region", default="us-west-2", help="AWS region for auxiliary clients")
    parser.add_argument("--profile", default=None, help="AWS profile")
    parser.add_argument("--state", default=None, help="Optional state filter")
    parser.add_argument("-n", "--num-records", type=int, default=10, help="Max records to print")
    args = parser.parse_args()

    db = WorksetStateDB(
        table_name=args.table_name,
        region=args.region,
        profile=args.profile,
    )

    if args.state:
        rows = db.list_worksets_by_state(WorksetState(args.state), limit=args.num_records)
    else:
        rows = db._list_all_worksets(limit=args.num_records)  # compatibility utility

    print(json.dumps(rows, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
