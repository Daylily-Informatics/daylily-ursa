"""S3 utility routes for Daylily API.

Contains routes for S3 operations:
- POST /api/v2/s3/discover-samples
- POST /api/v2/s3/validate-bucket
- GET /api/v2/s3/iam-policy/{bucket_name}
- GET /api/v2/s3/bucket-policy/{bucket_name}
- GET /api/v2/s3/bucket-region/{bucket_name}
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import boto3
import yaml  # type: ignore[import-untyped]

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status

from daylily_ursa.config import Settings

LOGGER = logging.getLogger("daylily.routes.s3")


class S3Dependencies:
    """Container for S3 route dependencies."""

    def __init__(
        self,
        settings: Settings,
        get_current_user,
    ):
        self.settings = settings
        self.get_current_user = get_current_user


def create_s3_router(deps: S3Dependencies) -> APIRouter:
    """Create S3 utility router with injected dependencies."""
    router = APIRouter(tags=["utilities"])
    settings = deps.settings
    get_current_user = deps.get_current_user
    region = settings.get_effective_region()
    profile = settings.aws_profile

    @router.post("/api/v2/s3/discover-samples")
    async def discover_samples_from_s3(
        request: Request,
        bucket: str = Body(..., embed=True),
        prefix: str = Body(..., embed=True),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Discover FASTQ samples from an S3 path.

        Lists files in the given S3 location and automatically pairs R1/R2 files
        into samples. Also attempts to parse daylily_work.yaml if present.
        """
        samples: List[Dict[str, Any]] = []
        yaml_content = None
        files_found: List[Dict[str, Any]] = []
        all_keys_found: List[str] = []

        LOGGER.info("S3 Discovery: Starting discovery for bucket=%s, prefix=%s", bucket, prefix)

        try:
            app_settings = request.app.state.settings
            session_kwargs: Dict[str, str] = {"region_name": app_settings.get_effective_region()}
            if app_settings.aws_profile:
                session_kwargs["profile_name"] = app_settings.aws_profile
            session = boto3.Session(**session_kwargs)
            s3_client = session.client("s3")

            normalized_prefix = prefix.strip()
            if normalized_prefix:
                normalized_prefix = normalized_prefix.lstrip("/")
                if not normalized_prefix.endswith("/"):
                    normalized_prefix += "/"

            LOGGER.info("S3 Discovery: Using normalized prefix: '%s'", normalized_prefix)

            paginator = s3_client.get_paginator("list_objects_v2")
            total_objects = 0

            for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
                for obj in page.get("Contents", []):
                    total_objects += 1
                    key = obj["Key"]
                    filename = key.split("/")[-1]
                    all_keys_found.append(key)

                    if not filename:
                        continue

                    if filename.lower() == "daylily_work.yaml":
                        LOGGER.info("S3 Discovery: Found daylily_work.yaml at %s", key)
                        try:
                            response = s3_client.get_object(Bucket=bucket, Key=key)
                            yaml_content = response["Body"].read().decode("utf-8")
                        except Exception as e:
                            LOGGER.warning("S3 Discovery: Failed to read daylily_work.yaml: %s", str(e))

                    fastq_extensions = [".fastq", ".fq", ".fastq.gz", ".fq.gz", ".fastq.bz2", ".fq.bz2"]
                    if any(filename.lower().endswith(ext) for ext in fastq_extensions):
                        files_found.append({"key": key, "filename": filename, "size": obj.get("Size", 0)})

            LOGGER.info("S3 Discovery: Found %d total objects, %d FASTQ files", total_objects, len(files_found))

            if yaml_content:
                try:
                    yaml_data = yaml.safe_load(yaml_content)
                    if yaml_data and isinstance(yaml_data.get("samples"), list):
                        for sample in yaml_data["samples"]:
                            if isinstance(sample, dict):
                                samples.append({
                                    "sample_id": sample.get("sample_id") or sample.get("id") or sample.get("name", "unknown"),
                                    "r1_file": sample.get("r1_file") or sample.get("r1") or sample.get("fq1", ""),
                                    "r2_file": sample.get("r2_file") or sample.get("r2") or sample.get("fq2", ""),
                                    "status": "pending",
                                })
                except Exception as e:
                    LOGGER.warning("S3 Discovery: Failed to parse daylily_work.yaml: %s", str(e))

            if not samples and files_found:
                samples = _pair_fastq_files(files_found)

            return {
                "samples": samples, "yaml_content": yaml_content, "files_found": len(files_found),
                "bucket": bucket, "prefix": prefix, "normalized_prefix": normalized_prefix,
                "total_objects_scanned": total_objects,
            }
        except Exception as e:
            if "NoSuchBucket" in str(type(e).__name__):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"S3 bucket '{bucket}' not found")
            LOGGER.error("S3 Discovery: Failed to discover samples from S3: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to discover samples: {str(e)}")



    @router.post("/api/v2/s3/validate-bucket")
    async def validate_s3_bucket(
        bucket: str = Body(..., embed=True),
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Validate an S3 bucket for Daylily use."""
        from daylily_ursa.s3_bucket_validator import S3BucketValidator

        try:
            validator_inst = S3BucketValidator(region=region, profile=profile)
            result = validator_inst.validate_bucket(bucket)

            instructions = None
            if not result.is_fully_configured:
                instructions = validator_inst.get_setup_instructions(
                    bucket, result, daylily_account_id="108782052779"
                )

            return {
                "bucket": bucket, "valid": result.is_valid, "fully_configured": result.is_fully_configured,
                "exists": result.exists, "accessible": result.accessible, "can_read": result.can_read,
                "can_write": result.can_write, "can_list": result.can_list, "region": result.region,
                "errors": result.errors, "warnings": result.warnings, "setup_instructions": instructions,
            }
        except Exception as e:
            LOGGER.error("S3 Validation: Failed to validate bucket '%s': %s", bucket, str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to validate bucket: {str(e)}")

    @router.get("/api/v2/s3/iam-policy/{bucket_name}")
    async def get_iam_policy_for_bucket(
        bucket_name: str,
        read_only: bool = False,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Generate IAM policy for accessing a customer S3 bucket."""
        from daylily_ursa.s3_bucket_validator import S3BucketValidator

        validator_inst = S3BucketValidator(region=region, profile=profile)
        policy = validator_inst.generate_iam_policy_for_bucket(bucket_name, read_only=read_only)
        return {"bucket": bucket_name, "read_only": read_only, "policy": policy}

    @router.get("/api/v2/s3/bucket-policy/{bucket_name}")
    async def get_bucket_policy_for_daylily(
        bucket_name: str,
        daylily_account_id: str = "108782052779",
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Generate S3 bucket policy for cross-account Daylily access."""
        from daylily_ursa.s3_bucket_validator import S3BucketValidator

        validator_inst = S3BucketValidator(region=region, profile=profile)
        policy = validator_inst.generate_customer_bucket_policy(bucket_name, daylily_account_id)
        return {
            "bucket": bucket_name, "daylily_account_id": daylily_account_id, "policy": policy,
            "apply_command": f"aws s3api put-bucket-policy --bucket {bucket_name} --policy file://bucket-policy.json",
        }

    @router.get("/api/v2/s3/bucket-region/{bucket_name}")
    async def get_bucket_region(
        bucket_name: str,
        current_user: Optional[Dict] = Depends(get_current_user),
    ):
        """Get the AWS region where an S3 bucket is located."""
        from botocore.exceptions import ClientError

        try:
            session_kwargs: Dict[str, str] = {}
            if profile:
                session_kwargs["profile_name"] = profile
            session = boto3.Session(**session_kwargs)
            s3_client = session.client("s3")

            response = s3_client.get_bucket_location(Bucket=bucket_name)
            location = response.get("LocationConstraint")
            bucket_region = location if location else "us-east-1"
            return {"bucket": bucket_name, "region": bucket_region}
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise HTTPException(status_code=404, detail=f"Bucket '{bucket_name}' not found")
            elif error_code == "AccessDenied":
                raise HTTPException(status_code=403, detail=f"Access denied to bucket '{bucket_name}'")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to get bucket region: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get bucket region: {str(e)}")

    return router


def _pair_fastq_files(files_found: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Pair FASTQ R1/R2 files into samples."""
    r1_patterns = [
        re.compile(r"^(.+?)[._](R1|r1)[._]?.*\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
        re.compile(r"^(.+?)[._]1[._]?.*\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
        re.compile(r"^(.+?)_S\d+_L\d+_R1_\d+\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
    ]
    r2_patterns = [
        re.compile(r"^(.+?)[._](R2|r2)[._]?.*\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
        re.compile(r"^(.+?)[._]2[._]?.*\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
        re.compile(r"^(.+?)_S\d+_L\d+_R2_\d+\.(fastq|fq)(\.gz|\.bz2)?$", re.IGNORECASE),
    ]

    r1_files: Dict[str, str] = {}
    r2_files: Dict[str, str] = {}

    for f in files_found:
        filename = f["filename"]
        matched = False

        for pattern in r1_patterns:
            match = pattern.match(filename)
            if match:
                r1_files[match.group(1)] = f["key"]
                matched = True
                break

        if not matched:
            for pattern in r2_patterns:
                match = pattern.match(filename)
                if match:
                    r2_files[match.group(1)] = f["key"]
                    break

    all_sample_names = set(r1_files.keys()) | set(r2_files.keys())
    LOGGER.info("S3 Discovery: Found %d R1 files, %d R2 files, %d unique sample names",
                len(r1_files), len(r2_files), len(all_sample_names))

    samples = []
    for sample_name in sorted(all_sample_names):
        samples.append({
            "sample_id": sample_name,
            "r1_file": r1_files.get(sample_name, ""),
            "r2_file": r2_files.get(sample_name, ""),
            "status": "pending",
        })
    return samples