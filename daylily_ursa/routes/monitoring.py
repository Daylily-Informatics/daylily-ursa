"""Monitoring and admin routes for Daylily API.

Contains routes for admin workset monitoring:
- GET /api/v2/admin/worksets/{workset_id}/command-log

Note: Queue stats, scheduler stats, and worksets/next are in routes/worksets.py.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import boto3

from fastapi import APIRouter, HTTPException, Query, Request, status

from daylily_ursa.config import Settings
from daylily_ursa.security import sanitize_for_log
from daylily_ursa.workset_state_db import WorksetStateDB

LOGGER = logging.getLogger("daylily.routes.monitoring")


class MonitoringDependencies:
    """Container for monitoring route dependencies."""

    def __init__(
        self,
        state_db: WorksetStateDB,
        settings: Settings,
    ):
        self.state_db = state_db
        self.settings = settings


def create_monitoring_router(deps: MonitoringDependencies) -> APIRouter:
    """Create monitoring router with injected dependencies."""
    router = APIRouter()
    state_db = deps.state_db

    @router.get("/api/v2/admin/worksets/{workset_id}/command-log", tags=["admin"])
    async def get_workset_command_log(
        request: Request,
        workset_id: str,
        grep: Optional[str] = Query(None, description="Filter log entries containing this text"),
        label: Optional[str] = Query(None, description="Filter by command label"),
        limit: int = Query(1000, ge=1, le=10000, description="Maximum lines to return"),
    ):
        """Admin-only endpoint to view command logs for a workset."""
        is_admin = getattr(request, "session", {}).get("is_admin", False)
        if not is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to view command logs",
            )

        workset = state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Workset {workset_id} not found")

        bucket = workset.get("bucket")
        prefix = workset.get("prefix", "").rstrip("/") + "/"
        if not bucket or not prefix:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Workset missing bucket or prefix")

        command_log_key = f"{prefix}workset_command.log"

        try:
            app_settings = request.app.state.settings
            session_kwargs = {"region_name": app_settings.get_effective_region()}
            if app_settings.aws_profile:
                session_kwargs["profile_name"] = app_settings.aws_profile
            session = boto3.Session(**session_kwargs)
            s3_client = session.client("s3")

            try:
                response = s3_client.get_object(Bucket=bucket, Key=command_log_key)
                log_content = response["Body"].read().decode("utf-8")
            except s3_client.exceptions.NoSuchKey:
                return {
                    "workset_id": workset_id, "log_available": False,
                    "message": "No command log found for this workset",
                    "entries": [], "entry_count": 0,
                }
        except Exception as e:
            LOGGER.warning("Failed to fetch command log for %s: %s", sanitize_for_log(workset_id), str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch command log: {str(e)}")

        # Parse and filter log entries
        lines_list = log_content.split("\n")
        entries: List[str] = []
        current_entry: List[str] = []
        entry_count = 0

        for line in lines_list:
            if line.startswith("[") and "=====" in line:
                if current_entry:
                    entry_text = "\n".join(current_entry)
                    include_entry = True
                    if grep and grep.lower() not in entry_text.lower():
                        include_entry = False
                    if label:
                        label_match = f"LABEL: {label}"
                        if label_match.lower() not in entry_text.lower():
                            include_entry = False
                    if include_entry:
                        entries.append(entry_text)
                        entry_count += 1
                current_entry = [line]
            else:
                current_entry.append(line)

        # Don't forget the last entry
        if current_entry:
            entry_text = "\n".join(current_entry)
            include_entry = True
            if grep and grep.lower() not in entry_text.lower():
                include_entry = False
            if label:
                label_match = f"LABEL: {label}"
                if label_match.lower() not in entry_text.lower():
                    include_entry = False
            if include_entry:
                entries.append(entry_text)
                entry_count += 1

        # Apply limit
        total_lines = sum(len(e.split("\n")) for e in entries)
        if total_lines > limit:
            limited_entries = []
            line_count = 0
            for entry in entries:
                entry_lines = len(entry.split("\n"))
                if line_count + entry_lines <= limit:
                    limited_entries.append(entry)
                    line_count += entry_lines
                else:
                    break
            entries = limited_entries

        return {
            "workset_id": workset_id, "log_available": True, "bucket": bucket,
            "key": command_log_key, "entry_count": entry_count,
            "entries_returned": len(entries),
            "filters_applied": {"grep": grep, "label": label, "limit": limit},
            "entries": entries,
        }

    return router

