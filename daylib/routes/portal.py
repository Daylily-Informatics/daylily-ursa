"""Portal routes for Daylily customer web interface.

Contains HTML template-based routes for:
- Authentication (login, logout, registration, password management)
- Dashboard
- Worksets management
- Files management
- Usage/billing
- Biospecimen management
- Account settings
- Documentation and support
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import boto3
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from daylib.config import Settings
from daylib.file_registry import detect_file_format as _detect_file_format
from daylib.routes.dependencies import (
    convert_customer_for_template,
    format_file_size,
    get_file_icon,
    PortalFileAutoRegisterRequest,
    PortalFileAutoRegisterResponse,
)
from daylib.workset_state_db import WorksetStateDB, WorksetState

if TYPE_CHECKING:
    from daylib.workset_auth import CognitoAuth
    from daylib.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.portal")


class PortalDependencies:
    """Container for portal route dependencies.

    Encapsulates all external dependencies needed by portal routes,
    making them easier to inject and test.
    """

    def __init__(
        self,
        state_db: WorksetStateDB,
        templates: Jinja2Templates,
        settings: Settings,
        enable_auth: bool = False,
        cognito_auth: Optional["CognitoAuth"] = None,
        customer_manager: Optional["CustomerManager"] = None,
        file_registry: Optional[Any] = None,
        linked_bucket_manager: Optional[Any] = None,
        biospecimen_registry: Optional[Any] = None,
    ):
        self.state_db = state_db
        self.templates = templates
        self.settings = settings
        self.enable_auth = enable_auth
        self.cognito_auth = cognito_auth
        self.customer_manager = customer_manager
        self.file_registry = file_registry
        self.linked_bucket_manager = linked_bucket_manager
        self.biospecimen_registry = biospecimen_registry
        self.region = settings.get_effective_region()
        self.profile = settings.aws_profile


def _get_customer_for_session(request: Request, deps: PortalDependencies):
    """Get the customer for the currently logged-in user.

    Looks up the customer by the user's email from the session.
    Returns (customer, customer_config) tuple or (None, None) if not found.
    """
    if not deps.customer_manager:
        return None, None

    user_email = None
    if hasattr(request, "session"):
        user_email = request.session.get("user_email")

    if not user_email:
        return None, None

    customer_config = deps.customer_manager.get_customer_by_email(user_email)
    if customer_config:
        return convert_customer_for_template(customer_config), customer_config

    return None, None


def _get_template_context(request: Request, deps: PortalDependencies, **kwargs) -> Dict[str, Any]:
    """Build common template context."""
    cache_bust = str(int(datetime.now().timestamp()))
    context = {
        "request": request,
        "auth_enabled": deps.enable_auth,
        "current_year": datetime.now().year,
        "cache_bust": cache_bust,
        **kwargs,
    }
    if "customer" in kwargs and kwargs["customer"]:
        context["customer_id"] = kwargs["customer"].customer_id
    if hasattr(request, "session") and request.session.get("user_email"):
        context["user_email"] = request.session.get("user_email")
        context["user_authenticated"] = True
        context["is_admin"] = request.session.get("is_admin", False)
    return context


def _require_portal_auth(request: Request) -> Optional[RedirectResponse]:
    """Check if user is authenticated for portal access.

    Returns RedirectResponse to login if not authenticated, None if authenticated.
    """
    if not hasattr(request, "session"):
        return RedirectResponse(url="/portal/login", status_code=302)
    if not request.session.get("user_email"):
        return RedirectResponse(url="/portal/login", status_code=302)
    if not request.session.get("customer_id"):
        LOGGER.warning(f"Session missing customer_id for user {request.session.get('user_email')}")
        return RedirectResponse(url="/portal/login", status_code=302)
    return None


def _get_first_customer_converted(deps: PortalDependencies):
    """Helper to get first customer converted for template use."""
    if deps.customer_manager:
        customers = deps.customer_manager.list_customers()
        if customers:
            return convert_customer_for_template(customers[0])
    return None


def _format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_index = 0
    size = float(size_bytes)
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    return f"{size:.1f} {units[unit_index]}"


def _extract_workset_metrics(workset: Dict[str, Any]) -> Dict[str, Any]:
    """Extract resource usage metrics from workset performance_metrics.

    Populates:
    - storage_gb, storage_bytes, storage_human from post_export_metrics or pre_export_metrics
    - vcpu_hours: Total vCPU seconds from benchmark_data converted to hours
    - memory_gb_hours: Total memory usage (max_rss * duration) from benchmark_data
    - duration_formatted: Pipeline duration as "Xh Ym" from duration_info
    - storage_cost_daily: Daily S3 storage cost ($0.023/GB/month)
    - storage_cost_total: Cumulative storage cost based on days stored
    - storage_days: Number of days data has been stored

    Args:
        workset: Workset dict to extract and enrich with metrics

    Returns:
        The workset dict with resource usage fields added
    """
    from datetime import datetime, timezone

    pm = workset.get("performance_metrics", {})

    # Try post_export_metrics first (new format), fall back to pre_export_metrics (legacy)
    export_metrics = {}
    if pm and isinstance(pm, dict):
        export_metrics = pm.get("post_export_metrics", {})
        if not export_metrics:
            # Backwards compatibility: try pre_export_metrics
            export_metrics = pm.get("pre_export_metrics", {})

    # Extract storage size from export metrics
    size_bytes = 0
    size_human = ""
    if export_metrics and isinstance(export_metrics, dict):
        size_bytes = int(export_metrics.get("analysis_directory_size_bytes", 0) or 0)
        size_human = export_metrics.get("analysis_directory_size_human", "")

    # Populate storage fields
    workset["storage_bytes"] = size_bytes
    workset["storage_gb"] = round(size_bytes / (1024**3), 2) if size_bytes > 0 else 0
    workset["storage_human"] = size_human if size_human else (_format_bytes(size_bytes) if size_bytes > 0 else "")
    workset["storage_available"] = size_bytes > 0

    # Calculate storage costs (S3 Standard: $0.023/GB/month)
    S3_STANDARD_RATE_PER_GB_MONTH = 0.023
    storage_gb = workset["storage_gb"]
    daily_rate = S3_STANDARD_RATE_PER_GB_MONTH / 30.0  # ~$0.000767/GB/day

    workset["storage_cost_daily"] = round(storage_gb * daily_rate, 4) if storage_gb > 0 else 0

    # Calculate days stored since completion
    storage_days = 0
    completed_at = workset.get("completed_at") or workset.get("updated_at")
    if completed_at and workset.get("state") in ("complete", "error", "archived"):
        try:
            if isinstance(completed_at, str):
                # Parse ISO format timestamp
                if completed_at.endswith("Z"):
                    completed_at = completed_at[:-1] + "+00:00"
                completed_dt = datetime.fromisoformat(completed_at)
            else:
                completed_dt = completed_at
            now = datetime.now(timezone.utc)
            storage_days = max(1, (now - completed_dt).days)  # At least 1 day
        except (ValueError, TypeError):
            storage_days = 1

    workset["storage_days"] = storage_days
    workset["storage_cost_total"] = round(workset["storage_cost_daily"] * storage_days, 2) if storage_days > 0 else 0
    workset["storage_class"] = "S3 Standard"  # Default for recent exports

    # Calculate transfer costs based on storage size
    # AWS data transfer pricing (approximate, varies by region)
    TRANSFER_INTERNET_PER_GB = 0.09  # Data transfer out to internet
    TRANSFER_CROSS_REGION_PER_GB = 0.02  # Data transfer to different AWS region
    TRANSFER_SAME_REGION_PER_GB = 0.01  # Data transfer within same region

    workset["transfer_cost_internet"] = round(storage_gb * TRANSFER_INTERNET_PER_GB, 4) if storage_gb > 0 else 0
    workset["transfer_cost_cross_region"] = round(storage_gb * TRANSFER_CROSS_REGION_PER_GB, 4) if storage_gb > 0 else 0
    workset["transfer_cost_same_region"] = round(storage_gb * TRANSFER_SAME_REGION_PER_GB, 4) if storage_gb > 0 else 0

    # Extract vCPU hours and memory GB-hours from benchmark_data
    vcpu_hours = 0.0
    memory_gb_hours = 0.0
    if pm and isinstance(pm, dict):
        benchmark_data = pm.get("benchmark_data", [])
        if benchmark_data and isinstance(benchmark_data, list):
            for entry in benchmark_data:
                if not isinstance(entry, dict):
                    continue
                # cpu_time is in seconds - sum all cpu_time and convert to hours
                try:
                    cpu_time_s = float(entry.get("cpu_time", 0) or 0)
                    vcpu_hours += cpu_time_s / 3600.0
                except (ValueError, TypeError):
                    pass
                # memory: max_rss (in MB) * duration (s) / 3600 / 1024 = GB-hours
                try:
                    max_rss_mb = float(entry.get("max_rss", 0) or 0)
                    duration_s = float(entry.get("s", 0) or 0)
                    memory_gb_hours += (max_rss_mb / 1024.0) * (duration_s / 3600.0)
                except (ValueError, TypeError):
                    pass

    workset["vcpu_hours"] = round(vcpu_hours, 2)
    workset["memory_gb_hours"] = round(memory_gb_hours, 2)

    # Extract duration from duration_info (parsed from Snakemake logs)
    duration_info = pm.get("duration_info", {}) if pm and isinstance(pm, dict) else {}
    duration_seconds = duration_info.get("duration_seconds", 0) if duration_info else 0

    if duration_seconds > 0:
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        if hours > 0:
            workset["duration_formatted"] = f"{hours}h {minutes}m"
        else:
            workset["duration_formatted"] = f"{minutes}m"
        workset["duration_seconds"] = duration_seconds
    else:
        workset["duration_formatted"] = None
        workset["duration_seconds"] = 0

    return workset


# Backwards compatibility alias
def _extract_workset_storage(workset: Dict[str, Any]) -> Dict[str, Any]:
    """Alias for _extract_workset_metrics for backwards compatibility."""
    return _extract_workset_metrics(workset)


def create_portal_router(deps: PortalDependencies) -> APIRouter:
    """Create portal router with injected dependencies.

    Args:
        deps: PortalDependencies container with all required dependencies

    Returns:
        Configured APIRouter with portal routes
    """
    router = APIRouter(tags=["portal"])

    # ========== Dashboard ==========

    @router.get("/portal", response_class=HTMLResponse)
    async def portal_dashboard(request: Request):
        """Customer portal dashboard."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect

        customer = None
        customer_id = None
        worksets = []
        stats = {
            "active_worksets": 0,
            "completed_worksets": 0,
            "storage_used_gb": 0,
            "storage_percent": 0,
            "max_storage_gb": 500,
            "cost_this_month": 0,
            "in_progress_worksets": 0,
            "ready_worksets": 0,
            "error_worksets": 0,
            "registered_files": 0,
            "total_file_size_gb": 0,
            "total_file_size_bytes": 0,
        }

        if deps.customer_manager:
            customer, customer_config = _get_customer_for_session(request, deps)
            if customer:
                customer_id = customer.customer_id
            elif deps.settings.demo_mode:
                customers = deps.customer_manager.list_customers()
                if customers:
                    customer_raw = customers[0]
                    customer = convert_customer_for_template(customer_raw)
                    customer_id = customer_raw.customer_id

            if customer_id:
                all_worksets = []
                for ws_state in WorksetState:
                    batch = deps.state_db.list_worksets_by_state(ws_state, limit=100)
                    all_worksets.extend(batch)
                worksets = all_worksets[:10]

                stats["in_progress_worksets"] = len([w for w in all_worksets if w.get("state") == "in_progress"])
                stats["ready_worksets"] = len([w for w in all_worksets if w.get("state") == "ready"])
                stats["completed_worksets"] = len([w for w in all_worksets if w.get("state") == "complete"])
                stats["error_worksets"] = len([w for w in all_worksets if w.get("state") == "error"])
                stats["active_worksets"] = stats["in_progress_worksets"]

                # Calculate actual costs and storage from completed worksets with performance metrics
                # Cost calculation constants
                S3_STORAGE_COST_PER_GB_MONTH = 0.023  # S3 Standard storage
                TRANSFER_COST_PER_GB = 0.10  # Placeholder transfer cost

                total_actual_cost = 0.0
                total_workset_storage_bytes = 0
                completed = [w for w in all_worksets if w.get("state") == "complete"]
                for ws in completed:
                    pm = ws.get("performance_metrics", {})
                    if pm and isinstance(pm, dict):
                        cost_summary = pm.get("cost_summary", {})
                        if cost_summary and isinstance(cost_summary, dict):
                            total_actual_cost += float(cost_summary.get("total_cost", 0))
                        # Aggregate workset storage (try post_export_metrics first, fall back to pre_export_metrics)
                        export_metrics = pm.get("post_export_metrics", {}) or pm.get("pre_export_metrics", {})
                        if export_metrics and isinstance(export_metrics, dict):
                            total_workset_storage_bytes += int(export_metrics.get("analysis_directory_size_bytes", 0) or 0)
                    else:
                        # Fall back to estimated cost if no performance metrics
                        total_actual_cost += float(ws.get("cost_usd", 0) or ws.get("metadata", {}).get("cost_usd", 0) or 0)

                # Calculate cost breakdown
                workset_storage_gb = total_workset_storage_bytes / (1024**3)
                storage_cost = workset_storage_gb * S3_STORAGE_COST_PER_GB_MONTH
                transfer_cost = workset_storage_gb * TRANSFER_COST_PER_GB
                total_cost = total_actual_cost + storage_cost + transfer_cost

                stats["compute_cost_usd"] = round(total_actual_cost, 2)
                stats["storage_cost_usd"] = round(storage_cost, 4)
                stats["transfer_cost_usd"] = round(transfer_cost, 4)
                stats["cost_this_month"] = round(total_cost, 2)
                stats["actual_cost_total"] = round(total_actual_cost, 2)
                stats["workset_storage_bytes"] = total_workset_storage_bytes
                stats["workset_storage_gb"] = round(workset_storage_gb, 2)
                stats["workset_storage_human"] = _format_bytes(total_workset_storage_bytes)

                if deps.file_registry:
                    try:
                        customer_files = deps.file_registry.list_customer_files(customer_id, limit=1000)
                        stats["registered_files"] = len(customer_files)
                        total_bytes = sum(f.file_metadata.file_size_bytes for f in customer_files if f.file_metadata)
                        stats["total_file_size_bytes"] = total_bytes
                        stats["total_file_size_gb"] = round(total_bytes / (1024**3), 2)
                    except Exception as e:
                        LOGGER.warning(f"Failed to get file statistics: {e}")

        return deps.templates.TemplateResponse(
            request,
            "dashboard.html",
            _get_template_context(request, deps, customer=customer, worksets=worksets, stats=stats, active_page="dashboard"),
        )

    # ========== Authentication Routes ==========

    @router.get("/portal/login", response_class=HTMLResponse)
    async def portal_login(request: Request, error: Optional[str] = None, success: Optional[str] = None):
        """Login page."""
        return deps.templates.TemplateResponse(
            request,
            "auth/login.html",
            _get_template_context(request, deps, error=error, success=success),
        )

    @router.post("/portal/login")
    async def portal_login_submit(request: Request, email: str = Form(...), password: str = Form(...)):
        """Handle login form submission with proper authentication."""
        LOGGER.debug(f"portal_login_submit: Login attempt for email: {email}")

        # Validate email domain against whitelist
        domain_valid, domain_error = deps.settings.validate_email_domain(email)
        if not domain_valid:
            LOGGER.warning(f"Login blocked for {email}: {domain_error}")
            return RedirectResponse(url=f"/portal/login?error={domain_error.replace(' ', '+')}", status_code=302)

        if not deps.customer_manager:
            if not deps.enable_auth:
                LOGGER.warning("portal_login_submit: No customer manager; creating demo session for %s", email)
                request.session["user_email"] = email
                request.session["user_authenticated"] = True
                request.session["customer_id"] = "demo-customer"
                request.session["is_admin"] = True
                return RedirectResponse(url="/portal/", status_code=302)
            LOGGER.error("portal_login_submit: Customer manager not configured")
            return RedirectResponse(url="/portal/login?error=Authentication+not+configured", status_code=302)

        customer = deps.customer_manager.get_customer_by_email(email)
        if not customer:
            LOGGER.warning(f"portal_login_submit: Login attempt for non-existent customer: {email}")
            return RedirectResponse(url="/portal/login?error=Invalid+email+or+password", status_code=302)

        if deps.cognito_auth:
            try:
                LOGGER.debug(f"portal_login_submit: Authenticating with Cognito for: {email}")
                auth_result = deps.cognito_auth.authenticate(email, password)

                if "challenge" in auth_result:
                    challenge_name = auth_result["challenge"]
                    LOGGER.info(f"portal_login_submit: Challenge required for {email}: {challenge_name}")
                    if challenge_name == "NEW_PASSWORD_REQUIRED":
                        request.session["challenge_session"] = auth_result["session"]
                        request.session["challenge_email"] = email
                        return RedirectResponse(url="/portal/change-password?reason=temporary", status_code=302)
                    else:
                        LOGGER.error(f"portal_login_submit: Unsupported challenge: {challenge_name}")
                        return RedirectResponse(url=f"/portal/login?error=Authentication+challenge+required:+{challenge_name}", status_code=302)

                request.session["access_token"] = auth_result["access_token"]
                request.session["id_token"] = auth_result["id_token"]
                LOGGER.info(f"portal_login_submit: Cognito authentication successful for: {email}")
            except ValueError as e:
                LOGGER.warning(f"portal_login_submit: Cognito authentication failed for {email}: {e}")
                return RedirectResponse(url="/portal/login?error=Invalid+email+or+password", status_code=302)
            except Exception as e:
                LOGGER.error(f"portal_login_submit: Cognito authentication error for {email}: {e}")
                return RedirectResponse(url="/portal/login?error=Authentication+service+error", status_code=302)
        else:
            LOGGER.warning("portal_login_submit: Cognito not configured - allowing login for registered customer %s WITHOUT password validation", email)

        LOGGER.debug(f"portal_login_submit: Setting session for authenticated user: {email}")
        request.session["user_email"] = email
        request.session["user_authenticated"] = True
        request.session["customer_id"] = customer.customer_id
        request.session["is_admin"] = customer.is_admin
        LOGGER.info(f"portal_login_submit: Login successful for customer {customer.customer_id} ({email})")
        return RedirectResponse(url="/portal/", status_code=302)

    @router.get("/portal/logout", response_class=RedirectResponse)
    async def portal_logout(request: Request):
        """Logout and redirect to login page.

        Thoroughly clears all session data to prevent stale state.
        """
        user_email = request.session.get("user_email", "unknown")
        LOGGER.info(f"Logging out user: {user_email}")

        # Explicitly clear all session keys (more thorough than just .clear())
        for key in list(request.session.keys()):
            del request.session[key]

        # Also call clear() to ensure complete cleanup
        request.session.clear()

        LOGGER.info(f"Session cleared for user: {user_email}")
        return RedirectResponse(url="/portal/login?success=You+have+been+logged+out", status_code=302)

    @router.get("/portal/forgot-password", response_class=HTMLResponse)
    async def portal_forgot_password(request: Request, error: Optional[str] = None, success: Optional[str] = None):
        """Forgot password page."""
        return deps.templates.TemplateResponse(
            request,
            "auth/forgot_password.html",
            _get_template_context(request, deps, error=error, success=success),
        )

    @router.post("/portal/forgot-password")
    async def portal_forgot_password_submit(request: Request, email: str = Form(...)):
        """Handle forgot password form submission."""
        if not deps.cognito_auth:
            return RedirectResponse(url="/portal/forgot-password?error=Password+reset+not+available", status_code=302)
        try:
            deps.cognito_auth.forgot_password(email)
            LOGGER.info(f"Password reset initiated for {email}")
            return RedirectResponse(url="/portal/reset-password?email=" + email, status_code=302)
        except ValueError as e:
            LOGGER.warning(f"Forgot password error for {email}: {e}")
            return RedirectResponse(url=f"/portal/forgot-password?error={str(e)}", status_code=302)
        except Exception as e:
            LOGGER.error(f"Forgot password error for {email}: {e}")
            return RedirectResponse(url="/portal/forgot-password?error=Password+reset+failed", status_code=302)

    @router.get("/portal/reset-password", response_class=HTMLResponse)
    async def portal_reset_password(request: Request, email: Optional[str] = None, error: Optional[str] = None, success: Optional[str] = None):
        """Reset password page."""
        return deps.templates.TemplateResponse(
            request,
            "auth/reset_password.html",
            _get_template_context(request, deps, email=email, error=error, success=success),
        )

    @router.post("/portal/reset-password")
    async def portal_reset_password_submit(request: Request, email: str = Form(...), code: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
        """Handle reset password form submission."""
        if not deps.cognito_auth:
            return RedirectResponse(url="/portal/reset-password?error=Password+reset+not+available", status_code=302)
        if password != confirm_password:
            return RedirectResponse(url=f"/portal/reset-password?email={email}&error=Passwords+do+not+match", status_code=302)
        try:
            deps.cognito_auth.confirm_forgot_password(email, code, password)
            LOGGER.info(f"Password reset successful for {email}")
            return RedirectResponse(url="/portal/login?success=Password+reset+successful.+Please+log+in", status_code=302)
        except ValueError as e:
            LOGGER.warning(f"Reset password error for {email}: {e}")
            return RedirectResponse(url=f"/portal/reset-password?email={email}&error={str(e)}", status_code=302)
        except Exception as e:
            LOGGER.error(f"Reset password error for {email}: {e}")
            return RedirectResponse(url=f"/portal/reset-password?email={email}&error=Password+reset+failed", status_code=302)

    @router.get("/portal/change-password", response_class=HTMLResponse)
    async def portal_change_password(request: Request, reason: Optional[str] = None, error: Optional[str] = None, success: Optional[str] = None):
        """Change password page (for NEW_PASSWORD_REQUIRED challenge)."""
        if not request.session.get("challenge_session"):
            return RedirectResponse(url="/portal/login?error=Session+expired.+Please+log+in+again", status_code=302)
        email = request.session.get("challenge_email", "")
        return deps.templates.TemplateResponse(
            request,
            "auth/change_password.html",
            _get_template_context(request, deps, email=email, reason=reason, error=error, success=success),
        )

    @router.post("/portal/change-password")
    async def portal_change_password_submit(request: Request, new_password: str = Form(...), confirm_password: str = Form(...)):
        """Handle change password form submission (NEW_PASSWORD_REQUIRED challenge)."""
        if not deps.cognito_auth:
            return RedirectResponse(url="/portal/login?error=Authentication+not+available", status_code=302)
        session = request.session.get("challenge_session")
        email = request.session.get("challenge_email")
        if not session or not email:
            return RedirectResponse(url="/portal/login?error=Session+expired.+Please+log+in+again", status_code=302)
        if new_password != confirm_password:
            return RedirectResponse(url="/portal/change-password?error=Passwords+do+not+match", status_code=302)
        try:
            tokens = deps.cognito_auth.respond_to_new_password_challenge(email, new_password, session)
            request.session.pop("challenge_session", None)
            request.session.pop("challenge_email", None)
            customer = deps.customer_manager.get_customer_by_email(email) if deps.customer_manager else None
            if not customer:
                LOGGER.error(f"Customer not found for {email} after password change")
                return RedirectResponse(url="/portal/login?error=Account+not+found", status_code=302)
            request.session["access_token"] = tokens["access_token"]
            request.session["id_token"] = tokens["id_token"]
            request.session["user_email"] = email
            request.session["user_authenticated"] = True
            request.session["customer_id"] = customer.customer_id
            request.session["is_admin"] = customer.is_admin
            LOGGER.info(f"Password changed successfully for {email}, user logged in")
            return RedirectResponse(url="/portal/?success=Password+changed+successfully", status_code=302)
        except ValueError as e:
            LOGGER.warning(f"Password change error for {email}: {e}")
            return RedirectResponse(url=f"/portal/change-password?error={str(e)}", status_code=302)
        except Exception as e:
            LOGGER.error(f"Password change error for {email}: {e}")
            return RedirectResponse(url="/portal/change-password?error=Password+change+failed", status_code=302)

    @router.get("/portal/register", response_class=HTMLResponse)
    async def portal_register(request: Request, error: Optional[str] = None, success: Optional[str] = None):
        """Registration page."""
        # Get allowed regions for bucket provisioning
        allowed_regions = deps.settings.get_allowed_regions() if deps.settings else ["us-west-2"]
        default_region = deps.settings.aws_default_region if deps.settings else "us-west-2"
        return deps.templates.TemplateResponse(
            request,
            "auth/register.html",
            _get_template_context(
                request, deps,
                error=error,
                success=success,
                allowed_regions=allowed_regions,
                default_region=default_region,
            ),
        )

    @router.post("/portal/register", response_class=HTMLResponse)
    async def portal_register_submit(
        request: Request,
        customer_name: str = Form(...),
        email: str = Form(...),
        max_concurrent_worksets: int = Form(10),
        max_storage_gb: int = Form(500),
        billing_account_id: Optional[str] = Form(None),
        cost_center: Optional[str] = Form(None),
        s3_option: str = Form("auto"),
        custom_s3_bucket: Optional[str] = Form(None),
        bucket_region: Optional[str] = Form(None),
    ):
        """Handle registration form submission."""
        if not deps.customer_manager:
            return deps.templates.TemplateResponse(
                request, "auth/register.html",
                _get_template_context(request, deps, error="Customer management not configured"),
            )

        # Validate email domain against whitelist
        domain_valid, domain_error = deps.settings.validate_email_domain(email)
        if not domain_valid:
            LOGGER.warning(f"Registration blocked for {email}: {domain_error}")
            return deps.templates.TemplateResponse(
                request, "auth/register.html",
                _get_template_context(request, deps, error=domain_error),
            )

        try:
            custom_bucket = None
            effective_bucket_region = bucket_region if s3_option == "auto" else None
            if s3_option == "byob" and custom_s3_bucket:
                custom_bucket = custom_s3_bucket.strip()
                if custom_bucket:
                    from daylib.s3_bucket_validator import S3BucketValidator
                    validator = S3BucketValidator(region=deps.region, profile=deps.profile)
                    result = validator.validate_bucket(custom_bucket)
                    if not result.is_valid:
                        error_msg = f"S3 bucket validation failed: {'; '.join(result.errors)}"
                        return deps.templates.TemplateResponse(
                            request, "auth/register.html",
                            _get_template_context(request, deps, error=error_msg),
                        )
            config = deps.customer_manager.onboard_customer(
                customer_name=customer_name,
                email=email,
                max_concurrent_worksets=max_concurrent_worksets,
                max_storage_gb=max_storage_gb,
                billing_account_id=billing_account_id,
                cost_center=cost_center,
                custom_s3_bucket=custom_bucket,
                bucket_region=effective_bucket_region,
            )
            LOGGER.info(f"Registration: enable_auth={deps.enable_auth}, cognito_auth={'SET' if deps.cognito_auth else 'NONE'}")
            if deps.enable_auth and deps.cognito_auth:
                try:
                    LOGGER.info(f"Creating Cognito user for {email} (customer_id: {config.customer_id})")
                    deps.cognito_auth.create_customer_user(email=email, customer_id=config.customer_id, temporary_password=None)
                    LOGGER.info(f"Cognito user created successfully for {email}")
                except ValueError as e:
                    LOGGER.warning(f"Cognito user creation skipped: {e}")
                except Exception as e:
                    LOGGER.error(f"Failed to create Cognito user for {email}: {e}")
                    import traceback
                    LOGGER.error(f"Traceback: {traceback.format_exc()}")
            else:
                LOGGER.warning(f"Skipping Cognito user creation: enable_auth={deps.enable_auth}, cognito_auth={'SET' if deps.cognito_auth else 'NONE'}")
            bucket_info = f" Your S3 bucket: {config.s3_bucket}." if config.s3_bucket else ""
            if not deps.enable_auth:
                LOGGER.info(f"Auto-logging in new customer {config.customer_id} ({email}) in no-auth mode")
                request.session["user_email"] = email
                request.session["user_authenticated"] = True
                request.session["customer_id"] = config.customer_id
                request.session["is_admin"] = False
                success_msg = f"✅ Account created! Customer ID: {config.customer_id}.{bucket_info} Welcome!"
                return RedirectResponse(url=f"/portal/?success={success_msg}", status_code=302)
            else:
                success_msg = (
                    f"✅ Account created! Customer ID: {config.customer_id}.{bucket_info} "
                    f"📧 CHECK YOUR EMAIL (including spam folder) for your temporary password from no-reply@verificationemail.com. "
                    f"Use it to log in below."
                )
                return RedirectResponse(url=f"/portal/login?success={success_msg}", status_code=302)
        except Exception as e:
            LOGGER.error(f"Registration failed for {email}: {e}")
            return deps.templates.TemplateResponse(
                request, "auth/register.html",
                _get_template_context(request, deps, error=str(e)),
            )

    # ========== Worksets Routes ==========

    def _get_s3_sentinel_status(bucket: str, prefix: str) -> str:
        """Get workset status from S3 sentinel files.

        Priority order:
        1. daylily.complete -> complete
        2. daylily.error -> error
        3. daylily.in_progress -> in_progress
        4. daylily.ignore -> ignored
        5. daylily.ready -> ready
        6. No sentinels -> unknown
        """
        import boto3
        session_kwargs = {"region_name": deps.region}
        if deps.profile:
            session_kwargs["profile_name"] = deps.profile
        s3 = boto3.Session(**session_kwargs).client("s3")

        # Sentinel file names
        sentinels = {
            "daylily.complete": "complete",
            "daylily.error": "error",
            "daylily.in_progress": "in_progress",
            "daylily.ignore": "ignored",
            "daylily.ready": "ready",
        }

        # Normalize prefix
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=100)
            found_sentinels = set()
            for obj in response.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if filename in sentinels:
                    found_sentinels.add(filename)

            # Return in priority order
            for sentinel_file, status in sentinels.items():
                if sentinel_file in found_sentinels:
                    return status
            return "unknown"
        except Exception:
            return "error"

    @router.get("/portal/worksets", response_class=HTMLResponse)
    async def portal_worksets(
        request: Request,
        page: int = 1,
        status: str = "",
        type: str = "",
        search: str = "",
        sort: str = "created_desc",
    ):
        """Worksets list page (excludes archived and deleted worksets).

        Supports filtering by:
        - status: Filter by workset state (ready, queued, in_progress, complete, error, etc.)
        - type: Filter by workset type (clinical, ruo, lsmc)
        - search: Free text search across workset name and metadata
        - sort: Sort order (created_desc, created_asc, status)
        """
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        worksets = []

        # Determine which states to query based on status filter
        excluded_states = {WorksetState.ARCHIVED, WorksetState.DELETED}
        if status:
            # Filter by specific status
            try:
                target_state = WorksetState(status)
                if target_state not in excluded_states:
                    batch = deps.state_db.list_worksets_by_state(target_state, limit=500)
                    worksets.extend(batch)
            except ValueError:
                # Invalid status, return empty
                pass
        else:
            # Get all non-excluded states
            for ws_state in WorksetState:
                if ws_state in excluded_states:
                    continue
                batch = deps.state_db.list_worksets_by_state(ws_state, limit=100)
                worksets.extend(batch)

        # Process worksets and apply additional filters
        filtered_worksets = []
        search_lower = search.lower().strip() if search else ""
        type_filter = type.lower().strip() if type else ""

        for ws in worksets:
            metadata = ws.get("metadata", {})
            if isinstance(metadata, dict):
                ws["sample_count"] = metadata.get("sample_count", 0)
                ws["pipeline_type"] = metadata.get("pipeline_type", "germline")
                ws["workset_type"] = metadata.get("workset_type", ws.get("workset_type", "ruo"))
            else:
                ws["sample_count"] = 0
                ws["pipeline_type"] = "germline"
                ws["workset_type"] = ws.get("workset_type", "ruo")

            # Apply type filter
            if type_filter:
                ws_type = (ws.get("workset_type") or "ruo").lower()
                if ws_type != type_filter:
                    continue

            # Apply search filter
            if search_lower:
                searchable = " ".join([
                    ws.get("workset_id", ""),
                    ws.get("name", ""),
                    ws.get("workset_type", ""),
                    ws.get("pipeline_type", ""),
                    str(metadata.get("workset_name", "")),
                    str(metadata.get("notification_email", "")),
                ]).lower()
                if search_lower not in searchable:
                    continue

            # Add S3 sentinel status
            bucket = ws.get("bucket", "")
            prefix = ws.get("prefix", "")
            if bucket and prefix:
                ws["s3_status"] = _get_s3_sentinel_status(bucket, prefix)
            else:
                ws["s3_status"] = "unknown"

            # Extract actual compute cost from performance_metrics if available
            compute_cost = 0.0
            is_actual_cost = False
            pm = ws.get("performance_metrics", {})
            if pm and isinstance(pm, dict):
                cost_summary = pm.get("cost_summary", {})
                if cost_summary and isinstance(cost_summary, dict):
                    actual_cost = cost_summary.get("total_cost")
                    if actual_cost is not None and actual_cost > 0:
                        compute_cost = float(actual_cost)
                        is_actual_cost = True

            # Fall back to estimated cost
            if compute_cost == 0:
                compute_cost = float(ws.get("cost_usd", 0) or 0)
                if compute_cost == 0 and isinstance(metadata, dict):
                    compute_cost = float(metadata.get("cost_usd", 0) or metadata.get("estimated_cost_usd", 0) or 0)

            ws["compute_cost"] = compute_cost
            ws["is_actual_cost"] = is_actual_cost

            # Extract storage size from performance_metrics
            _extract_workset_storage(ws)

            filtered_worksets.append(ws)

        # Sort worksets
        if sort == "created_asc":
            filtered_worksets.sort(key=lambda w: w.get("created_at", ""), reverse=False)
        elif sort == "status":
            filtered_worksets.sort(key=lambda w: w.get("state", ""))
        else:  # created_desc (default)
            filtered_worksets.sort(key=lambda w: w.get("created_at", ""), reverse=True)

        # Paginate
        per_page = 20
        total_count = len(filtered_worksets)
        total_pages = max(1, (total_count + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_worksets = filtered_worksets[start_idx:end_idx]

        return deps.templates.TemplateResponse(
            request,
            "worksets/list.html",
            _get_template_context(
                request, deps,
                customer=customer,
                worksets=paginated_worksets,
                current_page=page,
                total_pages=total_pages,
                total_count=total_count,
                filter_status=status,
                filter_type=type,
                filter_search=search,
                filter_sort=sort,
                active_page="worksets",
            ),
        )

    @router.get("/portal/worksets/new", response_class=HTMLResponse)
    async def portal_worksets_new(request: Request):
        """New workset submission page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        return deps.templates.TemplateResponse(
            request,
            "worksets/new.html",
            _get_template_context(request, deps, customer=customer, active_page="worksets"),
        )

    @router.get("/portal/worksets/archived", response_class=HTMLResponse)
    async def portal_worksets_archived(request: Request):
        """Archived worksets page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer = None
        archived_worksets = []
        if deps.customer_manager:
            customers = deps.customer_manager.list_customers()
            if customers:
                customer = convert_customer_for_template(customers[0])
                all_archived = deps.state_db.list_archived_worksets(limit=500)
                customer_config = deps.customer_manager.get_customer_config(customers[0].customer_id)
                if customer_config:
                    archived_worksets = [w for w in all_archived if w.get("bucket") == customer_config.s3_bucket]
        # Ensure workset_type is available for each archived workset
        for ws in archived_worksets:
            metadata = ws.get("metadata", {})
            if isinstance(metadata, dict):
                ws["workset_type"] = metadata.get("workset_type", ws.get("workset_type", "ruo"))
                ws["pipeline_type"] = metadata.get("pipeline_type", "germline")
                ws["sample_count"] = metadata.get("sample_count", 0)
            else:
                ws["workset_type"] = ws.get("workset_type", "ruo")
                ws["pipeline_type"] = "germline"
                ws["sample_count"] = 0
        return deps.templates.TemplateResponse(
            request,
            "worksets/archived.html",
            _get_template_context(request, deps, customer=customer, worksets=archived_worksets, active_page="worksets"),
        )

    @router.get("/portal/worksets/{workset_id}", response_class=HTMLResponse)
    async def portal_workset_detail(request: Request, workset_id: str):
        """Workset detail page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        workset = deps.state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(status_code=404, detail="Workset not found")
        customer, _ = _get_customer_for_session(request, deps)
        metadata = workset.get("metadata", {})
        if metadata:
            if "samples" in metadata and "samples" not in workset:
                workset["samples"] = metadata["samples"]
            for field in ["workset_name", "pipeline_type", "reference_genome", "notification_email", "enable_qc", "archive_results", "sample_count"]:
                if field in metadata and field not in workset:
                    workset[field] = metadata[field]

        # Extract storage size from performance_metrics
        _extract_workset_storage(workset)

        return deps.templates.TemplateResponse(
            request,
            "worksets/detail.html",
            _get_template_context(request, deps, customer=customer, workset=workset, active_page="worksets"),
        )

    @router.get("/portal/worksets/{workset_id}/download")
    async def portal_workset_download(request: Request, workset_id: str):
        """Download workset results as a presigned URL redirect."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        workset = deps.state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(status_code=404, detail="Workset not found")
        if workset.get("state") != "complete":
            raise HTTPException(status_code=400, detail=f"Workset is not complete (current state: {workset.get('state')})")
        bucket = workset.get("bucket")
        prefix = workset.get("prefix", "").rstrip("/")
        results_prefix = f"{prefix}/results/" if prefix else "results/"
        try:
            session_kwargs = {"region_name": deps.region}
            if deps.profile:
                session_kwargs["profile_name"] = deps.profile
            session = boto3.Session(**session_kwargs)
            s3 = session.client("s3")
            response = s3.list_objects_v2(Bucket=bucket, Prefix=results_prefix, MaxKeys=100)
            result_files = []
            for obj in response.get("Contents", []):
                key = obj["Key"]
                filename = key.split("/")[-1]
                if filename and not filename.startswith("."):
                    result_files.append({"key": key, "filename": filename, "size": obj.get("Size", 0)})
            if not result_files:
                raise HTTPException(status_code=404, detail="No result files found for this workset")
            main_results = [f for f in result_files if f["filename"].endswith((".vcf", ".vcf.gz", ".bam", ".cram", ".html"))]
            download_file = main_results[0] if main_results else result_files[0]
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": download_file["key"], "ResponseContentDisposition": f'attachment; filename="{download_file["filename"]}"'},
                ExpiresIn=3600,
            )
            return RedirectResponse(url=presigned_url, status_code=302)
        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error(f"Error generating download for workset {workset_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate download: {str(e)}")

    @router.get("/portal/worksets/{workset_id}/results/browse", response_class=HTMLResponse)
    async def portal_workset_results_browse(request: Request, workset_id: str, prefix: str = ""):
        """Browse workset results in S3."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        workset = deps.state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(status_code=404, detail="Workset not found")

        # Get results S3 URI
        results_uri = workset.get("results_s3_uri")
        if not results_uri:
            raise HTTPException(status_code=400, detail="No results location available for this workset")

        # Parse S3 URI: s3://bucket/prefix/path
        if not results_uri.startswith("s3://"):
            raise HTTPException(status_code=400, detail="Invalid results URI format")
        uri_parts = results_uri[5:].split("/", 1)
        bucket = uri_parts[0]
        base_prefix = uri_parts[1].rstrip("/") + "/" if len(uri_parts) > 1 else ""

        # Combine base prefix with browsed prefix
        current_prefix = base_prefix + prefix if prefix else base_prefix

        items = []
        breadcrumbs = []
        parent_prefix = None

        # Calculate parent prefix for navigation
        if prefix:
            parts = prefix.rstrip("/").split("/")
            if len(parts) > 1:
                parent_prefix = "/".join(parts[:-1]) + "/"
            else:
                parent_prefix = ""

        try:
            session_kwargs = {"region_name": deps.region}
            if deps.profile:
                session_kwargs["profile_name"] = deps.profile
            session = boto3.Session(**session_kwargs)
            s3 = session.client("s3")

            # List objects with delimiter to get folders
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=current_prefix, Delimiter="/"):
                # Add folders (common prefixes)
                for cp in page.get("CommonPrefixes", []):
                    folder_key = cp["Prefix"]
                    # Get relative name from base_prefix
                    relative_key = folder_key[len(base_prefix):] if folder_key.startswith(base_prefix) else folder_key
                    folder_name = relative_key.rstrip("/").split("/")[-1]
                    if folder_name and not folder_name.startswith("."):
                        items.append({
                            "name": folder_name,
                            "key": relative_key,
                            "is_folder": True,
                            "size_bytes": None,
                            "last_modified": None,
                            "file_format": None,
                        })

                # Add files
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # Skip the prefix itself
                    if key == current_prefix:
                        continue
                    relative_key = key[len(base_prefix):] if key.startswith(base_prefix) else key
                    filename = key.split("/")[-1]
                    if filename and not filename.startswith("."):
                        # Determine file format
                        file_format = None
                        if filename.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz")):
                            file_format = "fastq"
                        elif filename.endswith((".bam", ".bam.bai")):
                            file_format = "bam"
                        elif filename.endswith((".vcf", ".vcf.gz", ".vcf.gz.tbi")):
                            file_format = "vcf"
                        elif filename.endswith((".cram", ".cram.crai")):
                            file_format = "cram"
                        elif filename.endswith(".html"):
                            file_format = "html"
                        elif filename.endswith(".pdf"):
                            file_format = "pdf"
                        elif filename.endswith((".tsv", ".csv")):
                            file_format = "table"

                        items.append({
                            "name": filename,
                            "key": relative_key,
                            "full_key": key,
                            "is_folder": False,
                            "size_bytes": obj.get("Size", 0),
                            "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                            "file_format": file_format,
                        })
        except Exception as e:
            LOGGER.warning(f"Failed to browse results for {workset_id}: {e}")

        # Build breadcrumbs from prefix
        if prefix:
            parts = prefix.rstrip("/").split("/")
            for i, part in enumerate(parts):
                breadcrumbs.append({"name": part, "prefix": "/".join(parts[:i + 1]) + "/"})

        return deps.templates.TemplateResponse(
            request,
            "worksets/results_browse.html",
            _get_template_context(
                request, deps,
                customer=customer,
                workset=workset,
                items=items,
                breadcrumbs=breadcrumbs,
                current_prefix=prefix,
                parent_prefix=parent_prefix,
                results_uri=results_uri,
                active_page="worksets",
            ),
        )

    @router.get("/portal/worksets/{workset_id}/results/download")
    async def portal_workset_results_download(request: Request, workset_id: str, key: str = Query(...)):
        """Download a file from workset results."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        workset = deps.state_db.get_workset(workset_id)
        if not workset:
            raise HTTPException(status_code=404, detail="Workset not found")

        results_uri = workset.get("results_s3_uri")
        if not results_uri:
            raise HTTPException(status_code=400, detail="No results location available")

        # Parse S3 URI
        if not results_uri.startswith("s3://"):
            raise HTTPException(status_code=400, detail="Invalid results URI format")
        uri_parts = results_uri[5:].split("/", 1)
        bucket = uri_parts[0]
        base_prefix = uri_parts[1].rstrip("/") + "/" if len(uri_parts) > 1 else ""

        # Full S3 key
        full_key = base_prefix + key

        try:
            session_kwargs = {"region_name": deps.region}
            if deps.profile:
                session_kwargs["profile_name"] = deps.profile
            session = boto3.Session(**session_kwargs)
            s3 = session.client("s3")

            filename = key.split("/")[-1]
            presigned_url = s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": bucket,
                    "Key": full_key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"',
                },
                ExpiresIn=3600,
            )
            return RedirectResponse(url=presigned_url, status_code=302)
        except Exception as e:
            LOGGER.error(f"Error generating download URL for {workset_id}/{key}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate download: {str(e)}")

    # ========== Manifest Generator Routes ==========

    @router.get("/portal/yaml-generator", response_class=HTMLResponse)
    async def portal_yaml_generator(request: Request):
        """Redirect to manifest-generator."""
        return RedirectResponse(url="/portal/manifest-generator", status_code=302)

    @router.get("/portal/manifest-generator", response_class=HTMLResponse)
    async def portal_manifest_generator(request: Request):
        """Manifest/YAML generator page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        return deps.templates.TemplateResponse(
            request,
            "manifest_generator.html",
            _get_template_context(request, deps, customer=customer, active_page="manifest"),
        )

    # ========== Files Routes ==========

    @router.get("/portal/files", response_class=HTMLResponse)
    async def portal_files(request: Request, prefix: str = "", subject_id: str = "", biosample_id: str = ""):
        """File registry page - main file management interface."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        files = []
        stats = {"total_files": 0, "total_size": 0, "unique_subjects": 0, "unique_biosamples": 0}
        buckets = []
        if deps.file_registry and customer:
            customer_id = customer.customer_id
            try:
                if subject_id:
                    files = deps.file_registry.search_files_by_tag(customer_id, f"subject:{subject_id}")
                elif biosample_id:
                    files = deps.file_registry.search_files_by_tag(customer_id, f"biosample:{biosample_id}")
                else:
                    # Fetch all files (use high limit to get complete list for display/filtering)
                    file_registrations = deps.file_registry.list_customer_files(customer_id, limit=10000)

                    def parse_s3_uri(s3_uri):
                        """Parse s3://bucket/key into (bucket, key)."""
                        if s3_uri and s3_uri.startswith("s3://"):
                            parts = s3_uri[5:].split("/", 1)
                            return parts[0], parts[1] if len(parts) > 1 else ""
                        return "", ""

                    files = []
                    for f in file_registrations:
                        s3_uri = f.file_metadata.s3_uri if f.file_metadata else ""
                        bucket, key = parse_s3_uri(s3_uri)
                        registered_at = f.registered_at
                        if isinstance(registered_at, str):
                            registered_at_str = registered_at
                        elif hasattr(registered_at, 'isoformat'):
                            registered_at_str = registered_at.isoformat()
                        else:
                            registered_at_str = None
                        files.append({
                            "file_id": f.file_id,
                            "customer_id": f.customer_id,
                            "s3_bucket": bucket,
                            "s3_key": key,
                            "s3_uri": s3_uri,
                            "filename": key.split("/")[-1] if key else "",
                            "file_size_bytes": f.file_metadata.file_size_bytes if f.file_metadata else 0,
                            "size": f.file_metadata.file_size_bytes if f.file_metadata else 0,
                            "size_formatted": format_file_size(f.file_metadata.file_size_bytes) if f.file_metadata else "0 B",
                            "file_format": f.file_metadata.file_format if f.file_metadata else "unknown",
                            "file_type": f.file_metadata.file_format if f.file_metadata else "unknown",
                            "subject_id": f.biosample_metadata.subject_id if f.biosample_metadata else None,
                            "biosample_id": f.biosample_metadata.biosample_id if f.biosample_metadata else None,
                            "sample_type": f.biosample_metadata.sample_type if f.biosample_metadata else None,
                            "sequencing_platform": f.sequencing_metadata.platform if f.sequencing_metadata else None,
                            "tags": f.tags if f.tags else [],
                            "registered_at": registered_at_str,
                            "workset_count": 0,  # TODO: Add workset count lookup
                        })
                stats["total_files"] = len(files)
                stats["total_size"] = sum(f.get("size", 0) for f in files)
                stats["unique_subjects"] = len(set(f.get("subject_id") for f in files if f.get("subject_id")))
                stats["unique_biosamples"] = len(set(f.get("biosample_id") for f in files if f.get("biosample_id")))
            except Exception as e:
                LOGGER.warning(f"Failed to load files: {e}")
        if deps.linked_bucket_manager and customer:
            try:
                linked_buckets = deps.linked_bucket_manager.list_customer_buckets(customer.customer_id)
                buckets = [{"bucket_id": b.bucket_id, "bucket_name": b.bucket_name, "display_name": b.display_name} for b in linked_buckets]
            except Exception as e:
                LOGGER.warning(f"Failed to load buckets: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/index.html",
            _get_template_context(request, deps, customer=customer, files=files, stats=stats, buckets=buckets, prefix=prefix, subject_id=subject_id, biosample_id=biosample_id, active_page="files"),
        )

    @router.get("/portal/files/buckets", response_class=HTMLResponse)
    async def portal_files_buckets(request: Request):
        """Linked buckets management page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        buckets = []
        if deps.linked_bucket_manager and customer:
            try:
                linked_buckets = deps.linked_bucket_manager.list_customer_buckets(customer.customer_id)
                buckets = [
                    {"bucket_id": b.bucket_id, "bucket_name": b.bucket_name, "bucket_type": b.bucket_type, "display_name": b.display_name,
                     "description": b.description, "is_validated": b.is_validated, "can_read": b.can_read, "can_write": b.can_write,
                     "can_list": b.can_list, "read_only": b.read_only, "region": b.region,
                     "prefix_restriction": b.prefix_restriction,
                     "linked_at": b.linked_at.isoformat() if hasattr(b.linked_at, 'isoformat') else b.linked_at}
                    for b in linked_buckets
                ]
            except Exception as e:
                LOGGER.warning(f"Failed to load buckets: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/buckets.html",
            _get_template_context(request, deps, customer=customer, buckets=buckets, active_page="files"),
        )

    @router.get("/portal/files/browse/{bucket_id}", response_class=HTMLResponse)
    async def portal_files_browse(request: Request, bucket_id: str, prefix: str = ""):
        """Browse files in a linked bucket."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        if not customer:
            if deps.settings.demo_mode and deps.customer_manager:
                customers = deps.customer_manager.list_customers()
                if customers:
                    customer = convert_customer_for_template(customers[0])
        if not customer:
            return RedirectResponse(url="/portal/login?error=No+customer+account+found", status_code=302)
        bucket = None
        items = []
        breadcrumbs = [{"name": "/", "prefix": ""}]  # Root breadcrumb
        current_prefix = prefix.rstrip("/") + "/" if prefix else ""
        # Calculate parent prefix - None at root, empty string for first level, path for deeper
        if not current_prefix:
            parent_prefix = None  # At root, no parent
        else:
            parent_parts = current_prefix.rstrip("/").split("/")[:-1]
            parent_prefix = "/".join(parent_parts) + "/" if parent_parts else ""
        if deps.linked_bucket_manager:
            bucket_obj = deps.linked_bucket_manager.get_bucket(bucket_id)
            if bucket_obj:
                LOGGER.debug(f"Bucket browse: bucket.customer_id={bucket_obj.customer_id}, user.customer_id={customer.customer_id}")
                if bucket_obj.customer_id != customer.customer_id:
                    LOGGER.warning(f"Access denied: bucket {bucket_id} belongs to {bucket_obj.customer_id}, not {customer.customer_id}")
                    return RedirectResponse(url="/portal/files/buckets?error=Access+denied+to+this+bucket", status_code=302)
                bucket = {"bucket_id": bucket_obj.bucket_id, "bucket_name": bucket_obj.bucket_name,
                          "display_name": bucket_obj.display_name or bucket_obj.bucket_name,
                          "read_only": bucket_obj.read_only, "can_write": bucket_obj.can_write,
                          "can_list": bucket_obj.can_list, "can_read": bucket_obj.can_read}
                try:
                    session_kwargs = {"region_name": deps.region}
                    if deps.profile:
                        session_kwargs["profile_name"] = deps.profile
                    session = boto3.Session(**session_kwargs)
                    s3 = session.client("s3")
                    response = s3.list_objects_v2(Bucket=bucket_obj.bucket_name, Prefix=current_prefix, Delimiter="/", MaxKeys=500)
                    for cp in response.get("CommonPrefixes", []):
                        folder_path = cp["Prefix"]
                        folder_name = folder_path.rstrip("/").split("/")[-1]
                        items.append({"name": folder_name, "is_folder": True, "key": folder_path,
                                      "size_bytes": None, "last_modified": None, "file_format": None, "is_registered": False})
                    for obj in response.get("Contents", []):
                        if obj["Key"] == current_prefix:
                            continue
                        filename = obj["Key"].split("/")[-1]
                        if filename:
                            file_format = _detect_file_format(filename)
                            items.append({"name": filename, "is_folder": False, "key": obj["Key"],
                                          "size_bytes": obj.get("Size", 0), "file_format": file_format,
                                          "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None,
                                          "is_registered": False})
                except Exception as e:
                    LOGGER.warning(f"Failed to browse bucket {bucket_id}: {e}")
        if current_prefix:
            parts = current_prefix.rstrip("/").split("/")
            for i, part in enumerate(parts):
                breadcrumbs.append({"name": part, "prefix": "/".join(parts[: i + 1]) + "/"})
        return deps.templates.TemplateResponse(
            request,
            "files/browse.html",
            _get_template_context(request, deps, customer=customer, bucket=bucket, items=items, breadcrumbs=breadcrumbs,
                                  current_prefix=current_prefix, parent_prefix=parent_prefix, active_page="files"),
        )

    @router.get("/portal/files/register", response_class=HTMLResponse)
    async def portal_files_register(request: Request):
        """File registration page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        buckets = []
        if deps.linked_bucket_manager and customer:
            try:
                LOGGER.info(f"Loading buckets for customer: {customer.customer_id}")
                linked_buckets = deps.linked_bucket_manager.list_customer_buckets(customer.customer_id)
                LOGGER.info(f"Found {len(linked_buckets)} linked buckets for {customer.customer_id}")
                buckets = [{"bucket_id": b.bucket_id, "bucket_name": b.bucket_name, "display_name": b.display_name,
                            "is_validated": b.is_validated, "can_read": b.can_read, "can_write": b.can_write, "can_list": b.can_list}
                           for b in linked_buckets]
                LOGGER.info(f"Buckets for dropdown: {buckets}")
            except Exception as e:
                LOGGER.warning(f"Failed to load buckets: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/register.html",
            _get_template_context(request, deps, customer=customer, buckets=buckets, active_page="files"),
        )

    @router.post("/portal/files/register", response_model=PortalFileAutoRegisterResponse)
    async def portal_files_register_submit(request: Request, payload: PortalFileAutoRegisterRequest):
        """Register selected discovered files from a linked bucket."""
        user_email = request.session.get("user_email")
        LOGGER.debug(f"portal_files_register_submit: Session user_email: '{user_email}'")
        if not user_email:
            raise HTTPException(status_code=401, detail="Not authenticated")
        customer, customer_config = _get_customer_for_session(request, deps)
        if not customer:
            raise HTTPException(status_code=401, detail="No customer account found")
        customer_id = customer.customer_id
        if not deps.file_registry or not deps.linked_bucket_manager:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="File management not configured")
        bucket_name = None
        if payload.bucket_id:
            bucket_obj = deps.linked_bucket_manager.get_bucket(payload.bucket_id)
            if not bucket_obj:
                raise HTTPException(status_code=404, detail=f"Bucket ID {payload.bucket_id} not found")
            if bucket_obj.customer_id != customer_id:
                raise HTTPException(status_code=403, detail="Access denied to this bucket")
            bucket_name = bucket_obj.bucket_name
        elif payload.bucket_name:
            bucket_name = payload.bucket_name
        else:
            raise HTTPException(status_code=400, detail="bucket_id or bucket_name required")

        # Discover files first
        from daylib.file_registry import BucketFileDiscovery
        discovery = BucketFileDiscovery(region=deps.settings.get_effective_region())
        formats = payload.file_formats or ["fastq", "bam", "vcf", "cram"]
        discovered_files = discovery.discover_files(
            bucket_name=bucket_name,
            prefix=payload.prefix,
            file_formats=formats,
            max_files=payload.max_files,
        )
        LOGGER.info(f"Discovered {len(discovered_files)} files for auto-registration")

        # Filter to selected keys if provided
        if payload.selected_keys:
            selected_set = set(payload.selected_keys)
            LOGGER.debug(f"Selected keys from request ({len(selected_set)}): {list(selected_set)[:5]}...")
            discovered_keys = {f.key for f in discovered_files}
            LOGGER.debug(f"Discovered keys ({len(discovered_keys)}): {list(discovered_keys)[:5]}...")
            matching_keys = selected_set & discovered_keys
            LOGGER.debug(f"Matching keys ({len(matching_keys)}): {list(matching_keys)[:5]}...")
            discovered_files = [f for f in discovered_files if f.key in selected_set]
            LOGGER.info(f"Filtered to {len(discovered_files)} selected files (requested {len(payload.selected_keys)})")

        # Check registration status
        discovered_files = discovery.check_registration_status(
            discovered_files=discovered_files,
            registry=deps.file_registry,
            customer_id=customer_id,
        )

        # Auto-register unregistered files
        registered, skipped, errors = discovery.auto_register_files(
            discovered_files=discovered_files,
            registry=deps.file_registry,
            customer_id=customer_id,
            biosample_id=payload.biosample_id,
            subject_id=payload.subject_id,
            sequencing_platform=payload.sequencing_platform,
        )
        LOGGER.info(f"Auto-registration complete: {registered} registered, {skipped} skipped, {len(errors)} errors")

        return PortalFileAutoRegisterResponse(registered_count=registered, skipped_count=skipped, errors=errors)

    @router.get("/portal/files/upload", response_class=HTMLResponse)
    async def portal_files_upload(request: Request):
        """File upload page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        buckets = []
        if deps.linked_bucket_manager and customer:
            try:
                linked_buckets = deps.linked_bucket_manager.list_customer_buckets(customer.customer_id)
                buckets = [{"bucket_id": b.bucket_id, "bucket_name": b.bucket_name, "display_name": b.display_name,
                            "is_validated": b.is_validated, "can_read": b.can_read, "can_write": b.can_write}
                           for b in linked_buckets]
            except Exception as e:
                LOGGER.warning(f"Failed to load buckets: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/upload.html",
            _get_template_context(request, deps, customer=customer, buckets=buckets, active_page="files"),
        )

    @router.post("/portal/files/upload")
    async def portal_files_upload_submit(request: Request, bucket_id: str = Form(...), prefix: str = Form(""), file: UploadFile = File(...)):
        """Handle file upload to S3 bucket."""
        user_email = request.session.get("user_email")
        customer_id = request.session.get("customer_id")
        if not user_email or not customer_id:
            raise HTTPException(status_code=401, detail="Not authenticated")
        if not deps.linked_bucket_manager:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="File management not configured")
        LOGGER.info(f"Upload request from {user_email}: {file.filename} to bucket {bucket_id}")
        bucket_obj = deps.linked_bucket_manager.get_bucket(bucket_id)
        if not bucket_obj:
            raise HTTPException(status_code=404, detail="Bucket not found")
        if bucket_obj.customer_id != customer_id:
            raise HTTPException(status_code=403, detail="Access denied to this bucket")
        if not bucket_obj.can_write:
            raise HTTPException(status_code=403, detail="Write access not enabled for this bucket")
        try:
            session_kwargs = {"region_name": deps.region}
            if deps.profile:
                session_kwargs["profile_name"] = deps.profile
            session = boto3.Session(**session_kwargs)
            s3 = session.client("s3")
            s3_key = f"{prefix.strip('/')}/{file.filename}" if prefix else file.filename
            s3.upload_fileobj(file.file, bucket_obj.bucket_name, s3_key)
            LOGGER.info(f"Uploaded {file.filename} to s3://{bucket_obj.bucket_name}/{s3_key}")
            return {"success": True, "bucket": bucket_obj.bucket_name, "key": s3_key, "filename": file.filename}
        except Exception as e:
            LOGGER.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    @router.get("/portal/files/filesets", response_class=HTMLResponse)
    async def portal_files_filesets(request: Request):
        """File sets management page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        filesets = []
        if deps.file_registry and customer:
            try:
                fileset_objs = deps.file_registry.list_customer_filesets(customer.customer_id)
                filesets = [
                    {"fileset_id": fs.fileset_id, "customer_id": fs.customer_id, "name": fs.name, "description": fs.description,
                     "file_count": len(fs.file_ids), "created_at": fs.created_at.isoformat() if fs.created_at else None}
                    for fs in fileset_objs
                ]
            except Exception as e:
                LOGGER.warning(f"Failed to load filesets: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/filesets.html",
            _get_template_context(request, deps, customer=customer, filesets=filesets, active_page="files"),
        )

    @router.get("/portal/files/filesets/{fileset_id}", response_class=HTMLResponse)
    async def portal_files_fileset_detail(request: Request, fileset_id: str):
        """File set detail page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        fileset = None
        files = []
        if deps.file_registry:
            try:
                fileset = deps.file_registry.get_fileset(fileset_id)
                if fileset:
                    files = deps.file_registry.get_fileset_files(fileset_id)
            except Exception as e:
                LOGGER.warning(f"Failed to load fileset: {e}")
        if not fileset:
            raise HTTPException(status_code=404, detail="File set not found")
        unique_subjects = len(set(f.biosample_metadata.subject_id for f in files if f.biosample_metadata and f.biosample_metadata.subject_id))
        return deps.templates.TemplateResponse(
            request,
            "files/fileset_detail.html",
            _get_template_context(request, deps, customer=customer, fileset=fileset, files=files, unique_subjects=unique_subjects, active_page="files"),
        )

    @router.get("/portal/files/{file_id}", response_class=HTMLResponse)
    async def portal_files_detail(request: Request, file_id: str):
        """File detail page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        file = None
        workset_history = []
        if deps.file_registry:
            try:
                file = deps.file_registry.get_file(file_id)
                if file:
                    workset_history = deps.file_registry.get_file_workset_history(file_id)
            except Exception as e:
                LOGGER.warning(f"Failed to load file: {e}")
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return deps.templates.TemplateResponse(
            request,
            "files/detail.html",
            _get_template_context(request, deps, customer=customer, file=file, workset_history=workset_history, active_page="files"),
        )

    @router.get("/portal/files/{file_id}/edit", response_class=HTMLResponse)
    async def portal_files_edit(request: Request, file_id: str):
        """File edit page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        file = None
        if deps.file_registry:
            try:
                file = deps.file_registry.get_file(file_id)
            except Exception as e:
                LOGGER.warning(f"Failed to load file for edit: {e}")
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return deps.templates.TemplateResponse(
            request,
            "files/edit_file.html",
            _get_template_context(request, deps, customer=customer, file=file, active_page="files"),
        )

    @router.get("/portal/files/browser", response_class=HTMLResponse)
    async def portal_files_browser(request: Request, prefix: str = ""):
        """S3 file browser (legacy, uses customer's primary bucket)."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer = None
        folders: List[Dict[str, Any]] = []
        files: List[Dict[str, Any]] = []
        storage: Dict[str, Any] = {"used_gb": 0, "max_gb": 500, "files": files, "folders": folders}
        if deps.customer_manager:
            customers = deps.customer_manager.list_customers()
            if customers:
                customer_raw = customers[0]
                customer = convert_customer_for_template(customer_raw)
                storage["max_gb"] = customer.max_storage_gb
                try:
                    session_kwargs = {"region_name": deps.region}
                    if deps.profile:
                        session_kwargs["profile_name"] = deps.profile
                    session = boto3.Session(**session_kwargs)
                    s3 = session.client("s3")
                    response = s3.list_objects_v2(Bucket=customer.s3_bucket, Prefix=prefix, Delimiter="/")
                    for cp in response.get("CommonPrefixes", []):
                        folder_path = cp["Prefix"]
                        folder_name = folder_path.rstrip("/").split("/")[-1]
                        folders.append({"name": folder_name, "path": folder_path})
                    for obj in response.get("Contents", []):
                        if obj["Key"] == prefix:
                            continue
                        filename = obj["Key"].split("/")[-1]
                        if filename:
                            files.append({"name": filename, "key": obj["Key"], "size": obj.get("Size", 0),
                                          "size_formatted": format_file_size(obj.get("Size", 0)),
                                          "last_modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else None})
                except Exception as e:
                    LOGGER.warning(f"Failed to browse S3: {e}")
        return deps.templates.TemplateResponse(
            request,
            "files/browser.html",
            _get_template_context(request, deps, customer=customer, storage=storage, prefix=prefix, active_page="files"),
        )

    # ========== Usage Routes ==========

    @router.get("/portal/usage", response_class=HTMLResponse)
    async def portal_usage(request: Request):
        """Usage and billing page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer = None
        # Cost calculation constants
        S3_STORAGE_COST_PER_GB_MONTH = 0.023  # S3 Standard storage
        TRANSFER_COST_PER_GB = 0.10  # Placeholder transfer cost

        usage = {
            "total_cost": 0,
            "compute_cost_usd": 0,
            "storage_cost_usd": 0,
            "transfer_cost_usd": 0,
            "cost_change": 0,
            "vcpu_hours": 0,
            "memory_gb_hours": 0,
            "storage_gb": 0,
            "active_worksets": 0,
        }
        usage_details: List[Dict[str, Any]] = []
        if deps.customer_manager:
            customers = deps.customer_manager.list_customers()
            if customers:
                customer = convert_customer_for_template(customers[0])
                customer_usage = deps.customer_manager.get_customer_usage(customers[0].customer_id)
                if customer_usage:
                    usage.update(customer_usage)

        # Generate usage details from completed worksets with actual costs and storage
        workset_storage_breakdown: List[Dict[str, Any]] = []
        if deps.state_db:
            try:
                completed_worksets = deps.state_db.list_worksets_by_state(WorksetState.COMPLETE, limit=100)
                total_actual_compute = 0.0
                total_workset_storage_bytes = 0
                for ws in completed_worksets:
                    pm = ws.get("performance_metrics", {})
                    cost_summary = pm.get("cost_summary", {}) if pm and isinstance(pm, dict) else {}
                    # Try post_export_metrics first, fall back to pre_export_metrics for backwards compatibility
                    export_metrics = {}
                    if pm and isinstance(pm, dict):
                        export_metrics = pm.get("post_export_metrics", {}) or pm.get("pre_export_metrics", {})
                    actual_cost = float(cost_summary.get("total_cost", 0)) if cost_summary else 0
                    # Fallback to estimated cost
                    if actual_cost == 0:
                        actual_cost = float(ws.get("cost_usd", 0) or ws.get("metadata", {}).get("cost_usd", 0) or 0)
                    if actual_cost > 0:
                        total_actual_compute += actual_cost
                        completed_at = ws.get("updated_at", ws.get("created_at", ""))[:10]
                        usage_details.append({
                            "date": completed_at,
                            "type": "Compute",
                            "workset_id": ws.get("workset_id"),
                            "quantity": cost_summary.get("sample_count", 1) if cost_summary else 1,
                            "unit": "samples",
                            "cost": actual_cost,
                            "is_actual": bool(cost_summary),
                        })
                    # Collect storage info per workset
                    storage_bytes = int(export_metrics.get("analysis_directory_size_bytes", 0) or 0) if export_metrics else 0
                    if storage_bytes > 0:
                        total_workset_storage_bytes += storage_bytes
                        workset_storage_breakdown.append({
                            "workset_id": ws.get("workset_id"),
                            "storage_bytes": storage_bytes,
                            "storage_human": export_metrics.get("analysis_directory_size_human", _format_bytes(storage_bytes)),
                            "storage_gb": round(storage_bytes / (1024**3), 2),
                            "completed_at": ws.get("updated_at", ws.get("created_at", ""))[:10],
                        })
                # Sort usage details by date descending
                usage_details.sort(key=lambda x: x["date"], reverse=True)
                # Sort storage breakdown by size descending
                workset_storage_breakdown.sort(key=lambda x: x["storage_bytes"], reverse=True)

                # Calculate cost breakdown
                workset_storage_gb = total_workset_storage_bytes / (1024**3)
                storage_cost = workset_storage_gb * S3_STORAGE_COST_PER_GB_MONTH
                # Transfer cost placeholder: assume 1 transfer per workset at $0.10/GB
                transfer_cost = workset_storage_gb * TRANSFER_COST_PER_GB

                usage["compute_cost_usd"] = round(total_actual_compute, 2)
                usage["storage_cost_usd"] = round(storage_cost, 4)
                usage["transfer_cost_usd"] = round(transfer_cost, 4)
                usage["total_cost"] = round(total_actual_compute + storage_cost + transfer_cost, 2)
                usage["workset_storage_bytes"] = total_workset_storage_bytes
                usage["workset_storage_gb"] = round(workset_storage_gb, 2)
                usage["workset_storage_human"] = _format_bytes(total_workset_storage_bytes)

                # Add storage cost entry to usage_details if there's storage
                if workset_storage_gb > 0:
                    usage_details.append({
                        "date": "-",
                        "type": "Storage",
                        "workset_id": "All worksets",
                        "quantity": round(workset_storage_gb, 2),
                        "unit": "GB/month",
                        "cost": storage_cost,
                        "is_actual": True,
                    })
                    usage_details.append({
                        "date": "-",
                        "type": "Transfer",
                        "workset_id": "Placeholder",
                        "quantity": round(workset_storage_gb, 2),
                        "unit": "GB",
                        "cost": transfer_cost,
                        "is_actual": False,
                    })
            except Exception as e:
                LOGGER.warning(f"Failed to load usage details: {e}")

        return deps.templates.TemplateResponse(
            request,
            "usage.html",
            _get_template_context(request, deps, customer=customer, usage=usage, usage_details=usage_details, workset_storage=workset_storage_breakdown, active_page="usage"),
        )

    # ========== Biospecimen Routes ==========

    @router.get("/portal/biospecimen", response_class=HTMLResponse)
    @router.get("/portal/biospecimen/subjects", response_class=HTMLResponse)
    async def portal_biospecimen_subjects(request: Request):
        """Subjects management page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        customer_id = customer.customer_id if customer else request.session.get("customer_id", "default-customer")
        subjects: List[Dict[str, Any]] = []
        stats = {"subjects": 0, "biosamples": 0, "libraries": 0}

        # Load subjects from biospecimen registry if available
        if deps.biospecimen_registry:
            try:
                subject_objs = deps.biospecimen_registry.list_subjects(customer_id, limit=500)
                for subj in subject_objs:
                    subj_dict = {
                        "subject_id": subj.subject_id,
                        "identifier": subj.identifier,
                        "display_name": subj.display_name,
                        "sex": subj.sex,
                        "cohort": subj.cohort,
                        "created_at": subj.created_at,
                        "biosample_count": 0,
                    }
                    # Count biosamples for this subject
                    try:
                        biosamples = deps.biospecimen_registry.list_biosamples_for_subject(subj.subject_id)
                        subj_dict["biosample_count"] = len(biosamples)
                    except Exception:
                        pass
                    subjects.append(subj_dict)
                stats["subjects"] = len(subjects)
                # Get total biosamples and libraries counts
                try:
                    all_biosamples = deps.biospecimen_registry.list_biosamples(customer_id, limit=1000)
                    stats["biosamples"] = len(all_biosamples)
                except Exception:
                    pass
                try:
                    all_libraries = deps.biospecimen_registry.list_libraries(customer_id, limit=1000)
                    stats["libraries"] = len(all_libraries)
                except Exception:
                    pass
            except Exception as e:
                LOGGER.warning(f"Failed to load subjects from biospecimen registry: {e}")

        return deps.templates.TemplateResponse(
            request,
            "biospecimen/subjects.html",
            _get_template_context(request, deps, customer=customer, subjects=subjects, stats=stats, active_page="biospecimen"),
        )

    # ========== Account Routes ==========

    @router.get("/portal/account", response_class=HTMLResponse)
    async def portal_account(request: Request):
        """Account settings page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        # Use session-based customer lookup, not first customer in DB
        customer, _ = _get_customer_for_session(request, deps)
        app_settings = deps.settings

        # Debug info for troubleshooting session/customer mismatches
        session_info = {
            "session_user_email": request.session.get("user_email"),
            "session_customer_id": request.session.get("customer_id"),
            "session_logged_in": request.session.get("logged_in"),
        }
        db_customer_id = customer.customer_id if customer else None

        env_vars = {
            "AWS_PROFILE": app_settings.aws_profile,
            "AWS_DEFAULT_REGION": app_settings.aws_default_region,
            "AWS_REGION": app_settings.get_effective_region(),
            "AWS_ACCESS_KEY_ID": "***" if os.getenv("AWS_ACCESS_KEY_ID") else None,
            "AWS_SECRET_ACCESS_KEY": "***" if os.getenv("AWS_SECRET_ACCESS_KEY") else None,
            "AWS_ACCOUNT_ID": app_settings.aws_account_id,
            "DAYLILY_CONTROL_BUCKET": app_settings.daylily_control_bucket,
            "DAYLILY_MONITOR_BUCKET": app_settings.daylily_monitor_bucket,
            "WORKSET_TABLE_NAME": app_settings.workset_table_name,
            "CUSTOMER_TABLE_NAME": app_settings.customer_table_name,
            "COGNITO_USER_POOL_ID": app_settings.cognito_user_pool_id,
            "COGNITO_APP_CLIENT_ID": app_settings.cognito_app_client_id,
            "DAYLILY_PRIMARY_REGION": os.getenv("DAYLILY_PRIMARY_REGION"),
            "DAYLILY_MULTI_REGION": os.getenv("DAYLILY_MULTI_REGION"),
            "APPTAINER_HOME": os.getenv("APPTAINER_HOME"),
            "DAY_BIOME": os.getenv("DAY_BIOME"),
            "DAY_ROOT": os.getenv("DAY_ROOT"),
        }
        return deps.templates.TemplateResponse(
            request,
            "account.html",
            _get_template_context(request, deps, customer=customer, active_page="account", env_vars=env_vars, session_info=session_info, db_customer_id=db_customer_id),
        )

    # ========== Static Pages ==========

    @router.get("/portal/docs", response_class=HTMLResponse)
    async def portal_docs(request: Request):
        """Documentation page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        return deps.templates.TemplateResponse(
            request,
            "docs.html",
            _get_template_context(request, deps, customer=customer, active_page="docs"),
        )

    @router.get("/portal/support", response_class=HTMLResponse)
    async def portal_support(request: Request):
        """Support/Contact page."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)
        return deps.templates.TemplateResponse(
            request,
            "support.html",
            _get_template_context(request, deps, customer=customer, active_page="support"),
        )

    # ========== Clusters Routes ==========

    @router.get("/portal/clusters", response_class=HTMLResponse)
    async def portal_clusters(request: Request):
        """Clusters page - shows ParallelCluster instances across regions."""
        auth_redirect = _require_portal_auth(request)
        if auth_redirect:
            return auth_redirect
        customer, _ = _get_customer_for_session(request, deps)

        # Fetch cluster information
        clusters = []
        regions = []
        error = None
        # Check if user is admin to determine if we should fetch SSH status
        is_admin = customer and customer.is_admin
        try:
            from daylib.cluster_service import ClusterService

            allowed_regions = deps.settings.get_allowed_regions()
            regions = allowed_regions

            if allowed_regions:
                service = ClusterService(
                    regions=allowed_regions,
                    aws_profile=deps.settings.aws_profile,
                    cache_ttl_seconds=300,
                )
                # Only fetch SSH status for admin users
                all_clusters = service.get_all_clusters_with_status(
                    fetch_ssh_status=is_admin,
                )
                clusters = [c.to_dict() for c in all_clusters]
        except Exception as e:
            LOGGER.error(f"Failed to fetch clusters: {e}")
            error = str(e)

        return deps.templates.TemplateResponse(
            request,
            "clusters.html",
            _get_template_context(
                request, deps,
                customer=customer,
                clusters=clusters,
                regions=regions,
                error=error,
                active_page="clusters",
            ),
        )

    return router

