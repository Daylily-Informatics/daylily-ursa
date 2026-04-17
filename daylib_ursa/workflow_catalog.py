from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


DEFAULT_ANALYSIS_REPOSITORY = "daylily-omics-analysis"
_REPOSITORY_MODULES = {
    DEFAULT_ANALYSIS_REPOSITORY: (
        "daylily-omics-analysis",
        "daylily_omics_analysis.workflow_catalog",
    ),
}


class WorkflowCatalogRuntimeError(RuntimeError):
    """Raised when the installed analysis package cannot serve the workflow catalog contract."""


class WorkflowCatalogRequestError(ValueError):
    """Raised when a caller asks for an unsupported repository or invalid workflow request."""


def load_workflow_catalog_snapshot(repository: str = DEFAULT_ANALYSIS_REPOSITORY) -> dict[str, Any]:
    distribution_name, module = _resolve_repository_module(repository)
    loader = getattr(module, "load_workflow_catalog", None)
    if not callable(loader):
        raise WorkflowCatalogRuntimeError(
            f"Installed package '{distribution_name}' does not expose load_workflow_catalog()."
        )
    snapshot = loader()
    if not isinstance(snapshot, dict):
        raise WorkflowCatalogRuntimeError(
            f"Installed package '{distribution_name}' returned an invalid workflow catalog payload."
        )
    result = dict(snapshot)
    result.setdefault("repository", repository)
    result["repository_ref"] = _distribution_version(distribution_name)
    return result


def preview_workflow_command(
    *,
    repository: str,
    workflow_id: str,
    genome_build: str,
    execution_profile: str,
    options: dict[str, Any] | None = None,
    input_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    distribution_name, module = _resolve_repository_module(repository)
    renderer = getattr(module, "render_workflow_command", None)
    if not callable(renderer):
        raise WorkflowCatalogRuntimeError(
            f"Installed package '{distribution_name}' does not expose render_workflow_command()."
        )
    try:
        preview = renderer(
            workflow_id=workflow_id,
            genome_build=genome_build,
            execution_profile=execution_profile,
            options=options or {},
            input_context=input_context or {},
        )
    except Exception as exc:  # noqa: BLE001 - surface package contract failures directly
        if isinstance(exc, ValueError):
            raise WorkflowCatalogRequestError(str(exc)) from exc
        raise WorkflowCatalogRuntimeError(str(exc)) from exc
    if not isinstance(preview, dict):
        raise WorkflowCatalogRuntimeError(
            f"Installed package '{distribution_name}' returned an invalid workflow preview payload."
        )
    result = dict(preview)
    result.setdefault("repository", repository)
    result["repository_ref"] = _distribution_version(distribution_name)
    return result


def _resolve_repository_module(repository: str) -> tuple[str, Any]:
    normalized_repository = str(repository or "").strip()
    if normalized_repository not in _REPOSITORY_MODULES:
        supported = ", ".join(sorted(_REPOSITORY_MODULES))
        raise WorkflowCatalogRequestError(
            f"Unsupported analysis repository '{normalized_repository}'. Supported repositories: {supported}."
        )
    distribution_name, module_name = _REPOSITORY_MODULES[normalized_repository]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise WorkflowCatalogRuntimeError(
            f"Required analysis package '{distribution_name}' is not installed in the active Ursa environment."
        ) from exc
    return distribution_name, module


def _distribution_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError as exc:
        raise WorkflowCatalogRuntimeError(
            f"Required analysis package '{distribution_name}' is not installed in the active Ursa environment."
        ) from exc
