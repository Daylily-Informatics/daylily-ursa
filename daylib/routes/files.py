"""File management routes for Daylily API.

Contains routes for customer file operations:
- GET /api/customers/{customer_id}/files
- POST /api/customers/{customer_id}/files/upload
- POST /api/customers/{customer_id}/files/create-folder
- GET /api/customers/{customer_id}/files/{file_key}/preview
- GET /api/customers/{customer_id}/files/{file_key}/download-url
- DELETE /api/customers/{customer_id}/files/{file_key}
"""

from __future__ import annotations

import gzip
import io
import logging
import tarfile
import zipfile
from typing import TYPE_CHECKING

import boto3

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status

from daylib.routes.dependencies import (
    format_file_size as _format_file_size,
    get_file_icon as _get_file_icon,
)

if TYPE_CHECKING:
    from daylib.workset_customer import CustomerManager

LOGGER = logging.getLogger("daylily.routes.files")


class FileDependencies:
    """Container for file route dependencies."""

    def __init__(
        self,
        customer_manager: "CustomerManager",
    ):
        self.customer_manager = customer_manager


def create_files_router(deps: FileDependencies) -> APIRouter:
    """Create file management router with injected dependencies."""
    router = APIRouter(tags=["files"])
    customer_manager = deps.customer_manager

    @router.get("/api/customers/{customer_id}/files")
    async def list_customer_files(customer_id: str, prefix: str = ""):
        """List files in customer's S3 bucket."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            response = s3.list_objects_v2(Bucket=config.s3_bucket, Prefix=prefix, Delimiter="/")

            files = []
            for cp in response.get("CommonPrefixes", []):
                folder_path = cp["Prefix"]
                folder_name = folder_path.rstrip("/").split("/")[-1]
                files.append({
                    "key": folder_path, "name": folder_name, "type": "folder",
                    "size": 0, "size_formatted": "-", "modified": None, "icon": "folder",
                })
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key == prefix:
                    continue
                name = key.split("/")[-1]
                if not name:
                    continue
                size = obj["Size"]
                files.append({
                    "key": key, "name": name, "type": "file", "size": size,
                    "size_formatted": _format_file_size(size),
                    "modified": obj["LastModified"].isoformat() if obj.get("LastModified") else None,
                    "icon": _get_file_icon(name),
                })
            return {"files": files, "prefix": prefix, "bucket": config.s3_bucket}
        except Exception as e:
            LOGGER.error("Failed to list files: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/api/customers/{customer_id}/files/upload")
    async def upload_file(customer_id: str, file: UploadFile = File(...), prefix: str = Form("")):
        """Upload a file to customer's S3 bucket."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            key = f"{prefix}{file.filename}" if prefix else file.filename
            content = await file.read()
            s3.put_object(Bucket=config.s3_bucket, Key=key, Body=content,
                          ContentType=file.content_type or "application/octet-stream")
            LOGGER.info(f"Uploaded {key} to bucket {config.s3_bucket}")
            return {"success": True, "key": key, "bucket": config.s3_bucket}
        except Exception as e:
            LOGGER.error("Failed to upload file: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/api/customers/{customer_id}/files/create-folder")
    async def create_folder(customer_id: str, folder_path: str = Body(..., embed=True)):
        """Create a folder in customer's S3 bucket."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            folder_key = folder_path.rstrip("/") + "/"
            s3.put_object(Bucket=config.s3_bucket, Key=folder_key, Body=b"")
            hold_file_key = folder_key.rstrip("/") + "/.hold"
            s3.put_object(Bucket=config.s3_bucket, Key=hold_file_key, Body=b"")
            LOGGER.info(f"Created folder {folder_key} in bucket {config.s3_bucket} (with .hold file)")
            return {"success": True, "folder": folder_key}
        except Exception as e:
            LOGGER.error("Failed to create folder: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


    @router.get("/api/customers/{customer_id}/files/{file_key:path}/preview")
    async def preview_file(customer_id: str, file_key: str, lines: int = 20):
        """Preview file contents.

        For compressed files (.gz, .tgz, .tar.gz), decompresses and shows first N lines.
        For text files, shows first N lines directly.
        For binary files, returns a message indicating preview is not available.
        """
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            head = s3.head_object(Bucket=config.s3_bucket, Key=file_key)
            file_size = head.get("ContentLength", 0)
            content_type = head.get("ContentType", "application/octet-stream")

            file_lower = file_key.lower()
            is_gzip = file_lower.endswith(".gz") or file_lower.endswith(".gzip")
            is_tar_gz = file_lower.endswith(".tar.gz") or file_lower.endswith(".tgz")
            is_zip = file_lower.endswith(".zip")

            text_extensions = {
                ".txt", ".log", ".csv", ".tsv", ".json", ".xml", ".html", ".htm",
                ".yaml", ".yml", ".md", ".rst", ".py", ".js", ".ts", ".sh", ".bash",
                ".r", ".R", ".pl", ".rb", ".java", ".c", ".cpp", ".h", ".hpp",
                ".fastq", ".fq", ".fasta", ".fa", ".sam", ".vcf", ".bed", ".gff", ".gtf",
            }

            base_name = file_key
            if is_gzip and not is_tar_gz:
                base_name = file_key[:-3] if file_lower.endswith(".gz") else file_key[:-5]

            ext = "." + base_name.split(".")[-1] if "." in base_name else ""
            is_text = ext.lower() in text_extensions or content_type.startswith("text/")

            max_download = 10 * 1024 * 1024
            if file_size > max_download:
                response = s3.get_object(Bucket=config.s3_bucket, Key=file_key, Range=f"bytes=0-{max_download}")
            else:
                response = s3.get_object(Bucket=config.s3_bucket, Key=file_key)

            body = response["Body"].read()
            preview_lines = []
            file_type = "text"

            if is_tar_gz:
                file_type = "tar.gz"
                try:
                    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
                        members = tar.getnames()[:20]
                        preview_lines.append(f"=== Archive contents ({len(tar.getnames())} files) ===")
                        for m in members:
                            preview_lines.append(m)
                        if len(tar.getnames()) > 20:
                            preview_lines.append(f"... and {len(tar.getnames()) - 20} more files")
                except Exception as e:
                    preview_lines = [f"Error reading tar.gz: {str(e)}"]
            elif is_gzip:
                file_type = "gzip"
                try:
                    decompressed = gzip.decompress(body)
                    text = decompressed.decode("utf-8", errors="replace")
                    preview_lines = text.split("\n")[:lines]
                except Exception as e:
                    preview_lines = [f"Error decompressing: {str(e)}"]
            elif is_zip:
                file_type = "zip"
                try:
                    with zipfile.ZipFile(io.BytesIO(body)) as zf:
                        names = zf.namelist()[:20]
                        preview_lines.append(f"=== Archive contents ({len(zf.namelist())} files) ===")
                        for name in names:
                            preview_lines.append(name)
                        if len(zf.namelist()) > 20:
                            preview_lines.append(f"... and {len(zf.namelist()) - 20} more files")
                except Exception as e:
                    preview_lines = [f"Error reading zip: {str(e)}"]
            elif is_text or file_size < 1024 * 1024:
                try:
                    text = body.decode("utf-8", errors="replace")
                    preview_lines = text.split("\n")[:lines]
                except Exception:
                    file_type = "binary"
                    preview_lines = ["[Binary file - preview not available]"]
            else:
                file_type = "binary"
                preview_lines = ["[Binary file - preview not available]"]

            return {
                "filename": file_key.split("/")[-1],
                "file_type": file_type,
                "size": file_size,
                "lines": preview_lines,
                "total_lines": len(preview_lines),
                "truncated": len(preview_lines) >= lines,
            }

        except s3.exceptions.NoSuchKey:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {file_key}")
        except Exception as e:
            LOGGER.error("Failed to preview file: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/api/customers/{customer_id}/files/{file_key:path}/download-url")
    async def get_download_url(customer_id: str, file_key: str):
        """Get presigned URL for file download."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": config.s3_bucket, "Key": file_key},
                ExpiresIn=3600,
            )
            return {"url": url}
        except Exception as e:
            LOGGER.error("Failed to generate download URL: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.delete("/api/customers/{customer_id}/files/{file_key:path}")
    async def delete_file(customer_id: str, file_key: str):
        """Delete a file from customer's S3 bucket."""
        config = customer_manager.get_customer_config(customer_id)
        if not config:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Customer {customer_id} not found")

        try:
            s3 = boto3.client("s3")
            s3.delete_object(Bucket=config.s3_bucket, Key=file_key)
            return {"success": True, "deleted": file_key}
        except Exception as e:
            LOGGER.error("Failed to delete file: %s", str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return router
