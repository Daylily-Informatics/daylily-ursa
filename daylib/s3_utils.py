"""
S3 Region-Aware Client Utilities for Daylily

Provides a region-aware S3 client that:
- Detects bucket regions via GetBucketLocation
- Caches region per bucket to avoid repeated lookups
- Creates region-specific boto3 clients for each bucket
- Handles cross-region bucket operations correctly

This solves the "301 redirect / auth header mismatch" class of bugs that occur
when operating on buckets in different regions with a single-region client.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, cast

import boto3
from botocore.exceptions import ClientError

LOGGER = logging.getLogger("daylily.s3_utils")


def normalize_bucket_name(bucket: Optional[str]) -> Optional[str]:
    """Normalize bucket name by stripping s3:// prefix.
    
    Args:
        bucket: Bucket name, possibly with s3:// prefix
        
    Returns:
        Normalized bucket name or None if input was None/empty
    """
    if not bucket:
        return None
    bucket = bucket.strip()
    if bucket.startswith("s3://"):
        bucket = bucket[5:]
    # Also strip any trailing slashes
    bucket = bucket.rstrip("/")
    # Split off path if accidentally included
    if "/" in bucket:
        bucket = bucket.split("/")[0]
    return bucket if bucket else None


def _normalize_bucket_name_required(bucket: str) -> str:
    """Normalize a bucket name and require the result to be non-empty."""
    normalized = normalize_bucket_name(bucket)
    if not normalized:
        raise ValueError("Bucket name must be non-empty")
    return normalized


class _S3ExceptionsProxy:
    """Proxy for boto3 S3 client exceptions.

    Allows code like `client.exceptions.NoSuchKey` to work with RegionAwareS3Client.
    """
    def __init__(self, client: Any):
        self._client = client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client.exceptions, name)


class RegionAwareS3Client:
    """S3 client that creates region-specific clients per bucket.

    This class solves the problem of operating on S3 buckets in different regions.
    Instead of using a single boto3 S3 client (which is bound to one region),
    this maintains a cache of region-specific clients and automatically
    selects the correct one based on the bucket's actual region.

    Thread-safe: uses locks for cache access.

    Example:
        client = RegionAwareS3Client(profile="my-profile")

        # Operations automatically use the correct regional client
        client.list_objects_v2(Bucket="us-west-2-bucket", MaxKeys=10)
        client.list_objects_v2(Bucket="eu-central-1-bucket", MaxKeys=10)

        # Can also get raw boto3 client for a bucket
        raw_client = client.get_client_for_bucket("my-bucket")
    """

    def __init__(
        self,
        default_region: str = "us-west-2",
        profile: Optional[str] = None,
    ):
        """Initialize region-aware S3 client.

        Args:
            default_region: Default region for operations where bucket region
                           cannot be determined (e.g., creating new buckets)
            profile: AWS profile name (optional)
        """
        self.default_region = default_region
        self.profile = profile

        # Cache: bucket_name -> region
        self._bucket_regions: Dict[str, str] = {}
        self._bucket_regions_lock = threading.Lock()

        # Cache: region -> boto3 S3 client
        self._clients: Dict[str, Any] = {}
        self._clients_lock = threading.Lock()

        # Create default client for region lookups
        self._default_client = self._create_client(default_region)

        # Expose exceptions through the default client
        self.exceptions = _S3ExceptionsProxy(self._default_client)
    
    def _create_client(self, region: str) -> Any:
        """Create a boto3 S3 client for a specific region."""
        session_kwargs = {"region_name": region}
        if self.profile:
            session_kwargs["profile_name"] = self.profile
        session = boto3.Session(**session_kwargs)
        return session.client("s3")
    
    def _get_client_for_region(self, region: str) -> Any:
        """Get or create a boto3 S3 client for a region."""
        with self._clients_lock:
            if region not in self._clients:
                LOGGER.debug("Creating S3 client for region %s", region)
                self._clients[region] = self._create_client(region)
            return self._clients[region]
    
    def get_bucket_region(self, bucket: str) -> str:
        """Get the region of an S3 bucket.
        
        Uses cache to avoid repeated GetBucketLocation calls.
        
        Args:
            bucket: Bucket name (will be normalized)
            
        Returns:
            AWS region string (e.g., "us-west-2", "eu-central-1")
        """
        bucket_normalized = normalize_bucket_name(bucket)
        if not bucket_normalized:
            return self.default_region
        
        with self._bucket_regions_lock:
            if bucket_normalized in self._bucket_regions:
                return self._bucket_regions[bucket_normalized]
        
        # Look up bucket region
        try:
            response = self._default_client.get_bucket_location(Bucket=bucket_normalized)
            # AWS returns None for us-east-1 (legacy behavior)
            region = response.get("LocationConstraint") or "us-east-1"
            LOGGER.debug("Bucket %s is in region %s", bucket_normalized, region)
        except ClientError as e:
            LOGGER.warning("Could not determine region for bucket %s: %s", bucket_normalized, e)
            region = self.default_region
        
        with self._bucket_regions_lock:
            self._bucket_regions[bucket_normalized] = region
        
        return region

    def get_bucket_location(self, Bucket: str, **kwargs) -> Dict[str, Any]:
        """Call GetBucketLocation for a bucket.

        This uses the *default* client because the bucket's region is what we're
        trying to determine.

        Notes:
        - We intentionally do not swallow exceptions here; callers may want to
          treat unknown-region buckets conservatively.
        - On success, we opportunistically populate the internal bucket->region
          cache.
        """
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        response = cast(
            Dict[str, Any],
            self._default_client.get_bucket_location(Bucket=normalized_bucket, **kwargs),
        )

        region = response.get("LocationConstraint") or "us-east-1"
        with self._bucket_regions_lock:
            self._bucket_regions[normalized_bucket] = str(region)

        return response
    
    def get_client_for_bucket(self, bucket: str) -> Any:
        """Get the region-appropriate S3 client for a bucket.

        Args:
            bucket: Bucket name (will be normalized)

        Returns:
            boto3 S3 client configured for the bucket's region
        """
        region = self.get_bucket_region(bucket)
        return self._get_client_for_region(region)

    def invalidate_bucket_cache(self, bucket: str) -> None:
        """Remove a bucket from the region cache.

        Use this if a bucket's region might have changed (rare) or
        to force a fresh lookup.
        """
        bucket_normalized = normalize_bucket_name(bucket)
        if bucket_normalized:
            with self._bucket_regions_lock:
                self._bucket_regions.pop(bucket_normalized, None)

    def get_paginator(self, operation_name: str) -> "_RegionAwarePaginator":
        """Get a paginator for the given operation.

        Returns a wrapper that will use the correct regional client
        based on the Bucket parameter in paginate() calls.

        Args:
            operation_name: The S3 operation (e.g., "list_objects_v2")

        Returns:
            A RegionAwarePaginator instance
        """
        return _RegionAwarePaginator(self, operation_name)

    # =========================================================================
    # Convenience wrappers for common S3 operations
    # These automatically use the correct regional client
    # =========================================================================

    def list_objects_v2(self, Bucket: str, **kwargs) -> Dict[str, Any]:
        """List objects in a bucket using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(Dict[str, Any], client.list_objects_v2(Bucket=normalized_bucket, **kwargs))

    def head_object(self, Bucket: str, Key: str, **kwargs) -> Dict[str, Any]:
        """Get object metadata using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.head_object(Bucket=normalized_bucket, Key=Key, **kwargs),
        )

    def get_object(self, Bucket: str, Key: str, **kwargs) -> Dict[str, Any]:
        """Get object from S3 using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.get_object(Bucket=normalized_bucket, Key=Key, **kwargs),
        )

    def put_object(self, Bucket: str, Key: str, **kwargs) -> Dict[str, Any]:
        """Put object to S3 using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.put_object(Bucket=normalized_bucket, Key=Key, **kwargs),
        )

    def delete_object(self, Bucket: str, Key: str, **kwargs) -> Dict[str, Any]:
        """Delete object from S3 using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.delete_object(Bucket=normalized_bucket, Key=Key, **kwargs),
        )

    def copy_object(self, CopySource: Dict[str, str], Bucket: str, Key: str, **kwargs) -> Dict[str, Any]:
        """Copy object using the correct regional client for destination bucket."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.copy_object(
                CopySource=CopySource,
                Bucket=normalized_bucket,
                Key=Key,
                **kwargs,
            ),
        )

    def head_bucket(self, Bucket: str, **kwargs) -> Dict[str, Any]:
        """Check if bucket exists and is accessible."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        return cast(
            Dict[str, Any],
            client.head_bucket(Bucket=normalized_bucket, **kwargs),
        )

    def upload_file(self, Filename: str, Bucket: str, Key: str, **kwargs) -> None:
        """Upload a local file to S3 using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        client.upload_file(Filename, normalized_bucket, Key, **kwargs)

    def upload_fileobj(self, Fileobj: Any, Bucket: str, Key: str, **kwargs) -> None:
        """Upload a file-like object to S3 using the correct regional client."""
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self.get_client_for_bucket(normalized_bucket)
        client.upload_fileobj(Fileobj, normalized_bucket, Key, **kwargs)

    def generate_presigned_url(
        self,
        ClientMethod: str,
        Params: Dict[str, Any],
        ExpiresIn: int = 3600,
        **kwargs
    ) -> str:
        """Generate a presigned URL using the correct regional client.

        Args:
            ClientMethod: The S3 client method (e.g., 'get_object', 'put_object')
            Params: Parameters for the method (must include 'Bucket')
            ExpiresIn: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        bucket_value = Params.get("Bucket")
        if not bucket_value or not isinstance(bucket_value, str):
            raise ValueError("Params must include 'Bucket' key")

        normalized_bucket = _normalize_bucket_name_required(bucket_value)
        client = self.get_client_for_bucket(normalized_bucket)
        # Normalize bucket in params
        params = dict(Params)
        params["Bucket"] = normalized_bucket
        return cast(
            str,
            client.generate_presigned_url(
            ClientMethod=ClientMethod,
            Params=params,
            ExpiresIn=ExpiresIn,
            **kwargs
            ),
        )


class _RegionAwarePaginator:
    """Paginator wrapper that uses region-appropriate client based on Bucket parameter.

    This allows code like:
        paginator = region_aware_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket="my-bucket", Prefix="foo/"):
            ...

    The paginator will automatically use the correct regional client
    based on the bucket's actual region.
    """

    def __init__(self, region_aware_client: RegionAwareS3Client, operation_name: str):
        self._client = region_aware_client
        self._operation_name = operation_name

    def paginate(self, Bucket: str, **kwargs) -> Any:
        """Create pagination iterator using the correct regional client.

        Args:
            Bucket: S3 bucket name (will be normalized)
            **kwargs: Additional parameters for the paginator

        Returns:
            Pagination iterator from the correct regional client
        """
        normalized_bucket = _normalize_bucket_name_required(Bucket)
        client = self._client.get_client_for_bucket(normalized_bucket)
        paginator = client.get_paginator(self._operation_name)
        return paginator.paginate(Bucket=normalized_bucket, **kwargs)
