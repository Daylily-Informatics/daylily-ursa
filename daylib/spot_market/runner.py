from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import yaml

from daylib.ephemeral_cluster.runner import resolve_daylily_ec

_DEFAULT_INTERVAL_SECONDS = 6 * 60 * 60
_INTERVAL_LABEL_TO_SECONDS: Dict[str, int] = {
    "6h": 6 * 60 * 60,
    "12h": 12 * 60 * 60,
    "1d": 24 * 60 * 60,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_dir() -> Path:
    return Path.home() / ".ursa" / "spot-market"


def _config_path() -> Path:
    return _base_dir() / "config.json"


def _jobs_dir() -> Path:
    return _base_dir() / "jobs"


def _logs_dir() -> Path:
    return _base_dir() / "logs"


def _snapshots_dir() -> Path:
    return _base_dir() / "snapshots"


def _default_config(*, default_regions: List[str]) -> Dict[str, Any]:
    # Mirrors defaults used by daylily-ephemeral-cluster cost tooling:
    # daylily-ephemeral-cluster: bin/calc_daylily_aws_cost_estimates.py
    return {
        "regions": default_regions[:3],
        "interval_seconds": _DEFAULT_INTERVAL_SECONDS,
        "partition": "i192",
        "zones_suffixes": ["a", "b", "c", "d"],
        "cost_model": "harmonic",
        "x_coverage": 30.0,
        "align": 307.2,
        "snvcall": 684.0,
        "svcall": 19.0,
        "other": 0.021,
    }


def _coerce_regions(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for r in value:
        if isinstance(r, str) and r.strip():
            out.append(r.strip())
    return out


def load_config(*, default_regions: List[str], create_if_missing: bool = False) -> Dict[str, Any]:
    """Load spot market tracker config from disk (or return defaults).

    Args:
        default_regions: list of allowed regions from Ursa config/settings.
        create_if_missing: if True, writes the merged config to disk when absent.
    """
    defaults = _default_config(default_regions=default_regions)
    path = _config_path()

    loaded: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = _read_json(path)
        except Exception:
            loaded = {}

    cfg: Dict[str, Any] = dict(defaults)

    regions = _coerce_regions(loaded.get("regions"))
    if regions:
        cfg["regions"] = regions

    interval_seconds = loaded.get("interval_seconds")
    if isinstance(interval_seconds, int) and interval_seconds > 0:
        cfg["interval_seconds"] = interval_seconds

    for k in ("partition", "cost_model"):
        v = loaded.get(k)
        if isinstance(v, str) and v.strip():
            cfg[k] = v.strip()

    zones_suffixes = loaded.get("zones_suffixes")
    if isinstance(zones_suffixes, list) and zones_suffixes:
        suffixes: List[str] = []
        for s in zones_suffixes:
            if isinstance(s, str) and s.strip():
                suffixes.append(s.strip())
        if suffixes:
            cfg["zones_suffixes"] = suffixes

    for k in ("x_coverage", "align", "snvcall", "svcall", "other"):
        v = loaded.get(k)
        if isinstance(v, (int, float)):
            cfg[k] = float(v)

    if create_if_missing and not path.is_file():
        save_config(cfg)

    return cfg


def save_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _atomic_write_json(_config_path(), cfg)
    return cfg


def interval_label_to_seconds(label: str) -> int:
    if label not in _INTERVAL_LABEL_TO_SECONDS:
        raise ValueError(f"Invalid interval: {label} (expected one of {sorted(_INTERVAL_LABEL_TO_SECONDS)})")
    return _INTERVAL_LABEL_TO_SECONDS[label]


def interval_seconds_to_label(seconds: int) -> str:
    for label, sec in _INTERVAL_LABEL_TO_SECONDS.items():
        if sec == seconds:
            return label
    if seconds >= 24 * 60 * 60:
        return "1d"
    if seconds >= 12 * 60 * 60:
        return "12h"
    return "6h"


def get_daylily_ec_resources_dir() -> Path:
    daylily_ec = resolve_daylily_ec()
    proc = subprocess.run(
        [str(daylily_ec), "resources-dir"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"Failed to locate daylily-ec resources-dir (rc={proc.returncode}): {stderr}")
    p = Path((proc.stdout or "").strip()).expanduser()
    if not p.is_dir():
        raise RuntimeError(f"daylily-ec resources-dir returned a non-directory path: {p}")
    return p


def _load_prod_cluster_yaml(*, resources_dir: Path) -> Dict[str, Any]:
    cfg_path = resources_dir / "config" / "day_cluster" / "prod_cluster.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"daylily-ec prod_cluster.yaml not found under resources dir: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {cfg_path}")
    return data


def _extract_instances(*, cluster_cfg: Dict[str, Any], partition: str) -> List[str]:
    scheduling = cluster_cfg.get("Scheduling") or {}
    queues = scheduling.get("SlurmQueues") or []
    if not isinstance(queues, list):
        queues = []

    for queue in queues:
        if not isinstance(queue, dict):
            continue
        if str(queue.get("Name") or "") != partition:
            continue
        instances: List[str] = []
        for resource in queue.get("ComputeResources") or []:
            if not isinstance(resource, dict):
                continue
            for inst in resource.get("Instances") or []:
                if not isinstance(inst, dict):
                    continue
                it = inst.get("InstanceType")
                if isinstance(it, str) and it:
                    instances.append(it)
        # Deduplicate while keeping order.
        seen = set()
        out: List[str] = []
        for it in instances:
            if it in seen:
                continue
            seen.add(it)
            out.append(it)
        return out

    raise ValueError(f"No instances found for partition: {partition}")


def _parse_n_vcpus(partition: str) -> int:
    # i192 -> 192
    digits = ""
    for ch in reversed(partition):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    try:
        return int(digits) if digits else 192
    except ValueError:
        return 192


def _harmonic_mean(data: List[float]) -> float:
    vals = [d for d in data if not math.isnan(d) and d > 0]
    if not vals or any(v == 0 for v in vals):
        return float("nan")
    return len(vals) / math.fsum(1 / v for v in vals)


def _calculate_vcpu_mins(*, x_coverage: float, align: float, snvcall: float, svcall: float, other: float) -> float:
    return x_coverage * (align + snvcall + svcall + other)


def _list_zones(ec2, *, region: str, suffixes: List[str]) -> List[str]:
    try:
        resp = ec2.describe_availability_zones(AllAvailabilityZones=False)
    except Exception:
        return [f"{region}{s}" for s in suffixes]

    zones: List[str] = []
    for z in resp.get("AvailabilityZones", []) or []:
        name = z.get("ZoneName")
        state = z.get("State")
        if not isinstance(name, str) or not name:
            continue
        if state not in (None, "", "available"):
            continue
        if not name.startswith(region):
            continue
        if suffixes and name[-1] not in suffixes:
            continue
        zones.append(name)

    # Keep deterministic order.
    return sorted(set(zones))


def _fetch_spot_price(ec2, *, instance_type: str, zone: str) -> float:
    try:
        resp = ec2.describe_spot_price_history(
            InstanceTypes=[instance_type],
            AvailabilityZone=zone,
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=1,
        )
        rows = resp.get("SpotPriceHistory") or []
        if not rows:
            return float("nan")
        price = rows[0].get("SpotPrice")
        return float(price) if price is not None else float("nan")
    except Exception:
        return float("nan")


def compute_spot_snapshot(
    *,
    region: str,
    aws_profile: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute spot market statistics for a region.

    This mirrors the "zone stats" model from daylily-ephemeral-cluster cost tooling,
    but fetches spot prices via boto3 instead of AWS CLI subprocess calls.
    """
    resources_dir = get_daylily_ec_resources_dir()
    cluster_cfg = _load_prod_cluster_yaml(resources_dir=resources_dir)

    partition = str(cfg.get("partition") or "i192")
    instance_types = _extract_instances(cluster_cfg=cluster_cfg, partition=partition)
    n_vcpus = _parse_n_vcpus(partition)

    zones_suffixes = cfg.get("zones_suffixes") or ["a", "b", "c", "d"]
    suffixes = [str(s) for s in zones_suffixes if str(s)]

    sess = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    ec2 = sess.client("ec2", region_name=region)
    zones = _list_zones(ec2, region=region, suffixes=suffixes)

    spot_data: Dict[str, Dict[str, float]] = {}
    for it in instance_types:
        spot_data[it] = {}
        for zone in zones:
            spot_data[it][zone] = _fetch_spot_price(ec2, instance_type=it, zone=zone)

    cost_model = str(cfg.get("cost_model") or "harmonic")
    if cost_model not in {"min", "max", "median", "harmonic"}:
        cost_model = "harmonic"

    vcpu_mins = _calculate_vcpu_mins(
        x_coverage=float(cfg.get("x_coverage") or 30.0),
        align=float(cfg.get("align") or 307.2),
        snvcall=float(cfg.get("snvcall") or 684.0),
        svcall=float(cfg.get("svcall") or 19.0),
        other=float(cfg.get("other") or 0.021),
    )

    zone_rows: List[Dict[str, Any]] = []
    for zone in zones:
        prices = []
        for it in instance_types:
            p = spot_data.get(it, {}).get(zone, float("nan"))
            if math.isnan(p):
                continue
            prices.append(p)

        row: Dict[str, Any] = {
            "zone": zone,
            "instances": 0,
            "median_price": None,
            "min_price": None,
            "max_price": None,
            "harmonic_price": None,
            "stability": None,
            "est_cost": None,
        }

        if prices:
            row["instances"] = len(prices)
            median = statistics.median(prices)
            mn = min(prices)
            mx = max(prices)
            harmonic = _harmonic_mean(prices)
            stability = mx - mn

            row["median_price"] = float(median)
            row["min_price"] = float(mn)
            row["max_price"] = float(mx)
            row["harmonic_price"] = float(harmonic) if not math.isnan(harmonic) else None
            row["stability"] = float(stability)

            cost_per_vcpu_min: Dict[str, float] = {
                "min": mn / n_vcpus / 60,
                "max": mx / n_vcpus / 60,
                "median": median / n_vcpus / 60,
                "harmonic": (harmonic / n_vcpus / 60) if not math.isnan(harmonic) else float("nan"),
            }
            c = cost_per_vcpu_min.get(cost_model, float("nan"))
            if not math.isnan(c):
                row["est_cost"] = float(vcpu_mins * c)

        zone_rows.append(row)

    best_zone: Optional[Dict[str, Any]] = None
    best_cost = None
    for r in zone_rows:
        cost = r.get("est_cost")
        if cost is None:
            continue
        if best_cost is None or float(cost) < float(best_cost):
            best_cost = float(cost)
            best_zone = dict(r)

    return {
        "timestamp": _now_iso(),
        "region": region,
        "profile": aws_profile,
        "partition": partition,
        "n_vcpus": n_vcpus,
        "vcpu_mins": vcpu_mins,
        "cost_model": cost_model,
        "instance_types": instance_types,
        "zones": zone_rows,
        "best_zone": best_zone,
        "resources_dir": str(resources_dir),
    }


def write_snapshot(*, region: str, snapshot: Dict[str, Any]) -> Path:
    out_dir = _snapshots_dir() / region
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_timestamp_compact()}.json"
    _atomic_write_json(path, snapshot)
    return path


def compute_and_store_snapshot(
    *,
    region: str,
    aws_profile: Optional[str],
    cfg: Dict[str, Any],
) -> Path:
    snap = compute_spot_snapshot(region=region, aws_profile=aws_profile, cfg=cfg)
    return write_snapshot(region=region, snapshot=snap)


def start_poll_job(
    *,
    region: str,
    aws_profile: Optional[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Start a background poll job for a single region."""
    job_id = f"sm_{_timestamp_compact()}_{uuid.uuid4().hex[:8]}"
    jobs_dir = _jobs_dir()
    logs_dir = _logs_dir()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    job_path = jobs_dir / f"{job_id}.json"
    log_path = logs_dir / f"{job_id}.log"

    env_overrides: Dict[str, str] = {"PYTHONUNBUFFERED": "1"}
    if aws_profile:
        env_overrides["AWS_PROFILE"] = aws_profile

    job_doc: Dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "return_code": None,
        "error": None,
        "region": region,
        "aws_profile": aws_profile,
        "log_path": str(log_path),
        "snapshot_path": None,
        "cfg": cfg,
        "env_overrides": env_overrides,
        "runner_pid": None,
    }
    _atomic_write_json(job_path, job_doc)

    proc = subprocess.Popen(
        [sys.executable, "-m", "daylib.spot_market.job_runner", "--job-file", str(job_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd=Path.cwd(),
        env={**os.environ.copy(), **env_overrides},
    )

    job_doc["runner_pid"] = proc.pid
    job_doc["status"] = "running"
    job_doc["started_at"] = _now_iso()
    _atomic_write_json(job_path, job_doc)

    return {
        "job_id": job_id,
        "status": "running",
        "region": region,
        "aws_profile": aws_profile,
        "job_path": str(job_path),
        "log_path": str(log_path),
    }


def list_jobs(*, limit: int = 20) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    d = _jobs_dir()
    if not d.exists():
        return []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            jobs.append(_read_json(p))
        except Exception:
            continue
        if len(jobs) >= limit:
            break
    return jobs


def read_job(job_id: str) -> Dict[str, Any]:
    job_path = _jobs_dir() / f"{job_id}.json"
    if not job_path.is_file():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return _read_json(job_path)


def _tail_file(path: Path, *, lines: int) -> str:
    if lines <= 0:
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(data.splitlines()[-lines:])
    except Exception:
        return ""


def tail_job_log(job_id: str, *, lines: int = 200) -> str:
    job = read_job(job_id)
    log_path = Path(str(job.get("log_path") or "")).expanduser()
    if not log_path.is_file():
        return ""
    return _tail_file(log_path, lines=lines)


def list_snapshots(*, region: str, limit: int = 200) -> List[Dict[str, Any]]:
    d = _snapshots_dir() / region
    if not d.exists():
        return []
    snaps: List[Dict[str, Any]] = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            snaps.append(_read_json(p))
        except Exception:
            continue
        if len(snaps) >= limit:
            break
    return snaps


def get_last_snapshot_timestamps(*, regions: List[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for r in regions:
        out[r] = None
        d = _snapshots_dir() / r
        if not d.exists():
            continue
        latest = None
        for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            latest = p
            break
        if not latest:
            continue
        try:
            doc = _read_json(latest)
            ts = doc.get("timestamp")
            if isinstance(ts, str) and ts:
                out[r] = ts
        except Exception:
            continue
    return out
