"""Spot market tracker routes (admin-only).

This module provides a lightweight mechanism for collecting and visualizing
spot market data over time, stored on disk under `~/.ursa/spot-market/`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from daylily_ursa.config import Settings

LOGGER = logging.getLogger("daylily.routes.spot_market")


class SpotMarketConfigUpdate(BaseModel):
    regions: List[str] = Field(..., description="Regions to track (max 3 recommended)")
    interval: str = Field(..., description="Polling interval: 6h, 12h, 1d")


class SpotMarketPollRequest(BaseModel):
    regions: Optional[List[str]] = Field(None, description="Optional override regions for this poll")
    mode: str = Field("now", description="Currently only 'now' is supported")


class SpotMarketDependencies:
    def __init__(self, settings: Settings, get_current_user):
        self.settings = settings
        self.get_current_user = get_current_user


def create_spot_market_router(deps: SpotMarketDependencies) -> APIRouter:
    router = APIRouter(tags=["spot-market"])
    settings = deps.settings
    get_current_user = deps.get_current_user

    def _get_allowed_regions_and_profile() -> tuple[List[str], Optional[str]]:
        from daylily_ursa.ursa_config import get_ursa_config

        ursa_config = get_ursa_config()
        if ursa_config.is_configured:
            allowed = ursa_config.get_allowed_regions()
            profile = ursa_config.aws_profile or settings.aws_profile
        else:
            allowed = settings.get_allowed_regions()
            profile = settings.aws_profile
        return (allowed or []), profile

    @router.get("/api/spot-market/status")
    async def get_spot_market_status(
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="view spot market status")

        allowed_regions, aws_profile = _get_allowed_regions_and_profile()
        if not allowed_regions:
            raise HTTPException(
                status_code=400,
                detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
            )

        cfg = sm_runner.load_config(default_regions=allowed_regions, create_if_missing=False)
        last_samples = sm_runner.get_last_snapshot_timestamps(regions=cfg.get("regions") or [])
        running_jobs = sm_runner.list_running_jobs(limit=20)
        running_job_ids = [str(j.get("job_id") or "") for j in running_jobs if j.get("job_id")]

        return {
            "allowed_regions": allowed_regions,
            "aws_profile": aws_profile,
            "config": {
                **cfg,
                "interval": sm_runner.interval_seconds_to_label(int(cfg.get("interval_seconds") or 0)),
            },
            "last_samples": last_samples,
            "running_jobs": running_jobs,
            "running_job_ids": running_job_ids,
            "has_running_jobs": bool(running_job_ids),
        }

    @router.post("/api/spot-market/config")
    async def update_spot_market_config(
        update: SpotMarketConfigUpdate,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="update spot market config")

        allowed_regions, _aws_profile = _get_allowed_regions_and_profile()
        if not allowed_regions:
            raise HTTPException(
                status_code=400,
                detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
            )

        regions = [r.strip() for r in update.regions if isinstance(r, str) and r.strip()]
        if not regions:
            raise HTTPException(status_code=400, detail="regions must contain at least one region")
        for r in regions:
            if r not in allowed_regions:
                raise HTTPException(status_code=400, detail=f"Region '{r}' is not configured for this deployment")

        try:
            interval_seconds = sm_runner.interval_label_to_seconds(update.interval)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        cfg = sm_runner.load_config(default_regions=allowed_regions, create_if_missing=True)
        cfg["regions"] = regions[:3]
        cfg["interval_seconds"] = interval_seconds
        sm_runner.save_config(cfg)

        return {
            "config": {
                **cfg,
                "interval": sm_runner.interval_seconds_to_label(int(cfg.get("interval_seconds") or 0)),
            }
        }

    @router.post("/api/spot-market/poll")
    async def poll_spot_market(
        poll: SpotMarketPollRequest,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="poll spot market")

        allowed_regions, aws_profile = _get_allowed_regions_and_profile()
        if not allowed_regions:
            raise HTTPException(
                status_code=400,
                detail="No regions configured. Create ~/.ursa/ursa-config.yaml with region definitions.",
            )

        cfg = sm_runner.load_config(default_regions=allowed_regions, create_if_missing=True)
        req_regions = poll.regions if poll.regions is not None else cfg.get("regions")
        regions = [r.strip() for r in (req_regions or []) if isinstance(r, str) and r.strip()]
        if not regions:
            raise HTTPException(status_code=400, detail="No regions configured to poll")
        for r in regions:
            if r not in allowed_regions:
                raise HTTPException(status_code=400, detail=f"Region '{r}' is not configured for this deployment")

        running_jobs = sm_runner.list_running_jobs(limit=20)
        if running_jobs:
            running_ids = [str(j.get("job_id") or "") for j in running_jobs if j.get("job_id")]
            suffix = f" ({', '.join(running_ids)})" if running_ids else ""
            raise HTTPException(
                status_code=409,
                detail=f"A spot-market poll is already running{suffix}. Wait for completion before starting another.",
            )

        jobs = []
        for r in regions[:3]:
            try:
                jobs.append(sm_runner.start_poll_job(region=r, aws_profile=aws_profile, cfg=cfg))
            except FileNotFoundError as e:
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                LOGGER.exception("Failed to start spot-market poll job for %s: %s", r, e)
                raise HTTPException(status_code=500, detail=f"Failed to start spot-market poll for {r}: {e}")

        return {"jobs": jobs, "job_ids": [j.get("job_id") for j in jobs]}

    @router.get("/api/spot-market/jobs")
    async def list_spot_market_jobs(
        limit: int = Query(20, ge=1, le=100),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="list spot market jobs")
        return {"jobs": sm_runner.list_jobs(limit=limit)}

    @router.get("/api/spot-market/jobs/{job_id}")
    async def get_spot_market_job(
        job_id: str,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="read spot market jobs")
        try:
            return sm_runner.read_job(job_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/api/spot-market/jobs/{job_id}/logs")
    async def get_spot_market_job_logs(
        job_id: str,
        lines: int = Query(200, ge=1, le=2000),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="read spot market job logs")
        try:
            log_text = sm_runner.tail_job_log(job_id, lines=lines)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"job_id": job_id, "lines": lines, "log": log_text}

    @router.get("/api/spot-market/snapshots")
    async def list_spot_market_snapshots(
        region: str = Query(..., description="AWS region to list snapshots for"),
        limit: int = Query(200, ge=1, le=1000),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        from daylily_ursa.file_api import _enforce_admin_only
        from daylily_ursa.spot_market import runner as sm_runner

        _enforce_admin_only(current_user, operation="list spot market snapshots")

        allowed_regions, _aws_profile = _get_allowed_regions_and_profile()
        if region not in allowed_regions:
            raise HTTPException(status_code=400, detail=f"Region '{region}' is not configured for this deployment")

        return {"region": region, "snapshots": sm_runner.list_snapshots(region=region, limit=limit)}

    return router
