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
- utilities: Legacy S3 discovery, cost estimation
- portal: HTML portal pages and web interface routes
"""

from daylib.routes.dependencies import (
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
from daylib.routes.worksets import create_worksets_router
from daylib.routes.utilities import create_utilities_router
from daylib.routes.portal import create_portal_router, PortalDependencies
from daylib.routes.customer_worksets import (
    create_customer_worksets_router,
    CustomerWorksetDependencies,
)
from daylib.routes.dashboard import create_dashboard_router, DashboardDependencies
from daylib.routes.billing import create_billing_router, BillingDependencies
from daylib.routes.customers import create_customers_router, CustomerDependencies
from daylib.routes.files import create_files_router, FileDependencies
from daylib.routes.manifests import create_manifests_router, ManifestDependencies
from daylib.routes.s3 import create_s3_router, S3Dependencies
from daylib.routes.clusters import create_clusters_router, ClusterDependencies
from daylib.routes.monitoring import create_monitoring_router, MonitoringDependencies

__all__ = [
    # Router factories
    "create_worksets_router",
    "create_utilities_router",
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

