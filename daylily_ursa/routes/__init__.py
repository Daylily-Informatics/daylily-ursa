"""Daylily API route modules.

This package contains FastAPI routers organized by functionality:
- worksets: Core workset CRUD, queue stats, scheduling
- customers: Customer CRUD operations
- customer_worksets: Customer-scoped workset CRUD (list, create, cancel, retry, archive, etc.)
- files: Customer file management (S3 operations)
- manifests: Customer manifest storage and retrieval
- s3: S3 utility endpoints (discover, validate, policies)
- clusters: ParallelCluster management
- monitoring: Admin monitoring endpoints (command logs)
- dashboard: Customer dashboard chart data
- billing: Customer billing summary, invoices
- portal: HTML portal pages and web interface routes
"""

from daylily_ursa.routes.dependencies import (
    # Pydantic models
    WorksetCreate,
    WorksetResponse,
    WorksetStateUpdate,
    QueueStats,
    SchedulingStats,
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    WorksetValidationResponse,
    WorkYamlGenerateRequest,
    ChangePasswordRequest,
    APITokenCreateRequest,
    PortalFileAutoRegisterRequest,
    PortalFileAutoRegisterResponse,
    # Utility functions
    format_file_size,
    get_file_icon,
    calculate_cost_with_efficiency,
    convert_customer_for_template,
    verify_workset_ownership,
)

# Router factories
from daylily_ursa.routes.worksets import create_worksets_router
from daylily_ursa.routes.portal import create_portal_router, PortalDependencies
from daylily_ursa.routes.customer_worksets import (
    create_customer_worksets_router,
    CustomerWorksetDependencies,
)
from daylily_ursa.routes.dashboard import create_dashboard_router, DashboardDependencies
from daylily_ursa.routes.billing import create_billing_router, BillingDependencies
from daylily_ursa.routes.customers import create_customers_router, CustomerDependencies
from daylily_ursa.routes.files import create_files_router, FileDependencies
from daylily_ursa.routes.manifests import create_manifests_router, ManifestDependencies
from daylily_ursa.routes.s3 import create_s3_router, S3Dependencies
from daylily_ursa.routes.clusters import create_clusters_router, ClusterDependencies
from daylily_ursa.routes.monitoring import create_monitoring_router, MonitoringDependencies

__all__ = [
    # Router factories
    "create_worksets_router",
    "create_portal_router",
    "PortalDependencies",
    "create_customer_worksets_router",
    "CustomerWorksetDependencies",
    "create_dashboard_router",
    "DashboardDependencies",
    "create_billing_router",
    "BillingDependencies",
    "create_customers_router",
    "CustomerDependencies",
    "create_files_router",
    "FileDependencies",
    "create_manifests_router",
    "ManifestDependencies",
    "create_s3_router",
    "S3Dependencies",
    "create_clusters_router",
    "ClusterDependencies",
    "create_monitoring_router",
    "MonitoringDependencies",
    # Pydantic models
    "WorksetCreate",
    "WorksetResponse",
    "WorksetStateUpdate",
    "QueueStats",
    "SchedulingStats",
    "CustomerCreate",
    "CustomerUpdate",
    "CustomerResponse",
    "WorksetValidationResponse",
    "WorkYamlGenerateRequest",
    "ChangePasswordRequest",
    "APITokenCreateRequest",
    "PortalFileAutoRegisterRequest",
    "PortalFileAutoRegisterResponse",
    # Utility functions
    "format_file_size",
    "get_file_icon",
    "calculate_cost_with_efficiency",
    "convert_customer_for_template",
    "verify_workset_ownership",
]
