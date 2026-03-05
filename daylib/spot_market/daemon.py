from __future__ import annotations

import signal
import time
from datetime import datetime, timezone

from daylib.spot_market import runner as sm_runner


_STOP = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _handle_stop(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    from daylib.config import get_settings
    from daylib.ursa_config import get_ursa_config

    settings = get_settings()
    ursa_config = get_ursa_config()

    if ursa_config.is_configured:
        allowed_regions = ursa_config.get_allowed_regions()
        aws_profile = ursa_config.aws_profile or settings.aws_profile
    else:
        allowed_regions = settings.get_allowed_regions()
        aws_profile = settings.aws_profile

    if not allowed_regions:
        print("[spot-market] No regions configured. Create ~/.ursa/ursa-config.yaml")
        return 2

    print(f"[spot-market] started at {_now_iso()} profile={aws_profile} allowed_regions={allowed_regions}")

    while not _STOP:
        cfg = sm_runner.load_config(default_regions=allowed_regions, create_if_missing=True)
        regions = [str(r) for r in (cfg.get("regions") or []) if str(r)]
        interval_seconds = int(cfg.get("interval_seconds") or 0) or 6 * 60 * 60

        print(f"[spot-market] tick {_now_iso()} regions={regions} interval_seconds={interval_seconds}")

        for region in regions[:3]:
            if _STOP:
                break
            if region not in allowed_regions:
                print(f"[spot-market] skipping unconfigured region: {region}")
                continue
            try:
                p = sm_runner.compute_and_store_snapshot(region=region, aws_profile=aws_profile, cfg=cfg)
                print(f"[spot-market] snapshot region={region} path={p}")
            except Exception as e:
                print(f"[spot-market] snapshot FAILED region={region}: {type(e).__name__}: {e}")

        # Sleep in small increments so stop signals are honored quickly.
        slept = 0
        while not _STOP and slept < interval_seconds:
            time.sleep(1)
            slept += 1

    print(f"[spot-market] stopped at {_now_iso()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

