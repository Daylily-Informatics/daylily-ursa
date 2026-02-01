"""Tests for daylib/cli/certs.py - SSL certificate management.

Tests cover:
- ensure_certs_dir() - directory creation
- generate_self_signed_cert() - certificate generation with various options
- get_cert_info() - certificate information extraction
- certs_exist() - certificate existence checking
- get_cert_paths() - path resolution
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from daylib.cli.certs import (
    certs_exist,
    ensure_certs_dir,
    generate_self_signed_cert,
    get_cert_info,
    get_cert_paths,
    DEFAULT_CERT_PATH,
    DEFAULT_KEY_PATH,
    DEFAULT_CN,
    DEFAULT_SANS,
    DEFAULT_VALIDITY_DAYS,
)


class TestEnsureCertsDir:
    """Tests for ensure_certs_dir function."""

    def test_creates_directory_if_not_exists(self, tmp_path: Path) -> None:
        """Test that directory is created if it doesn't exist."""
        certs_dir = tmp_path / "certs"
        assert not certs_dir.exists()

        with patch("daylib.cli.certs.CERTS_DIR", certs_dir):
            result = ensure_certs_dir()

        assert certs_dir.exists()
        assert certs_dir.is_dir()
        assert result == certs_dir

    def test_returns_existing_directory(self, tmp_path: Path) -> None:
        """Test that existing directory is returned without error."""
        certs_dir = tmp_path / "certs"
        certs_dir.mkdir(parents=True)
        assert certs_dir.exists()

        with patch("daylib.cli.certs.CERTS_DIR", certs_dir):
            result = ensure_certs_dir()

        assert result == certs_dir


class TestGenerateSelfSignedCert:
    """Tests for generate_self_signed_cert function."""

    def test_generates_cert_and_key_files(self, tmp_path: Path) -> None:
        """Test that certificate and key files are created."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        result_cert, result_key = generate_self_signed_cert(
            cert_path=cert_path,
            key_path=key_path,
        )

        assert result_cert == cert_path
        assert result_key == key_path
        assert cert_path.exists()
        assert key_path.exists()

    def test_cert_file_contains_pem_certificate(self, tmp_path: Path) -> None:
        """Test that certificate file contains valid PEM data."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        cert_content = cert_path.read_text()
        assert "-----BEGIN CERTIFICATE-----" in cert_content
        assert "-----END CERTIFICATE-----" in cert_content

    def test_key_file_contains_pem_private_key(self, tmp_path: Path) -> None:
        """Test that key file contains valid PEM private key."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        key_content = key_path.read_text()
        assert "-----BEGIN RSA PRIVATE KEY-----" in key_content
        assert "-----END RSA PRIVATE KEY-----" in key_content

    def test_key_file_has_restrictive_permissions(self, tmp_path: Path) -> None:
        """Test that key file has 0600 permissions."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        # Check permissions (owner read/write only = 0o600)
        mode = key_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_custom_common_name(self, tmp_path: Path) -> None:
        """Test certificate with custom common name."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(
            cert_path=cert_path,
            key_path=key_path,
            cn="myserver.local",
        )

        info = get_cert_info(cert_path)
        assert info is not None
        assert "myserver.local" in info["subject"]

    def test_custom_sans(self, tmp_path: Path) -> None:
        """Test certificate with custom Subject Alternative Names."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(
            cert_path=cert_path,
            key_path=key_path,
            sans=["example.com", "192.168.1.1"],
        )

        info = get_cert_info(cert_path)
        assert info is not None
        assert "example.com" in info["sans"]
        assert "192.168.1.1" in info["sans"]

    def test_custom_validity_days(self, tmp_path: Path) -> None:
        """Test certificate with custom validity period."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        generate_self_signed_cert(
            cert_path=cert_path,
            key_path=key_path,
            days=30,
        )

        info = get_cert_info(cert_path)
        assert info is not None
        # Should expire in approximately 30 days (allow 1 day tolerance)
        assert 29 <= info["days_until_expiry"] <= 30

    def test_custom_key_size(self, tmp_path: Path) -> None:
        """Test certificate with custom key size."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        # Use 4096-bit key
        generate_self_signed_cert(
            cert_path=cert_path,
            key_path=key_path,
            key_size=4096,
        )

        # Verify files were created (key size is internal)
        assert cert_path.exists()
        assert key_path.exists()


class TestGetCertInfo:
    """Tests for get_cert_info function."""

    def test_returns_none_for_nonexistent_cert(self, tmp_path: Path) -> None:
        """Test that None is returned when certificate doesn't exist."""
        cert_path = tmp_path / "nonexistent.crt"
        result = get_cert_info(cert_path)
        assert result is None

    def test_returns_cert_info_dict(self, tmp_path: Path) -> None:
        """Test that certificate info dictionary is returned."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        info = get_cert_info(cert_path)

        assert info is not None
        assert "path" in info
        assert "subject" in info
        assert "issuer" in info
        assert "serial_number" in info
        assert "not_valid_before" in info
        assert "not_valid_after" in info
        assert "fingerprint_sha256" in info
        assert "sans" in info
        assert "is_expired" in info
        assert "days_until_expiry" in info

    def test_cert_is_not_expired(self, tmp_path: Path) -> None:
        """Test that newly generated certificate is not expired."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        info = get_cert_info(cert_path)

        assert info is not None
        assert info["is_expired"] is False
        assert info["days_until_expiry"] > 0

    def test_fingerprint_format(self, tmp_path: Path) -> None:
        """Test that fingerprint is in correct format (colon-separated hex)."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        info = get_cert_info(cert_path)

        assert info is not None
        fingerprint = info["fingerprint_sha256"]
        # SHA256 fingerprint should be 64 hex chars + 31 colons = 95 chars
        assert len(fingerprint) == 95
        assert ":" in fingerprint

    def test_default_sans_included(self, tmp_path: Path) -> None:
        """Test that default SANs are included in certificate."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        info = get_cert_info(cert_path)

        assert info is not None
        # Default SANs should include localhost and IP addresses
        assert "localhost" in info["sans"]


class TestCertsExist:
    """Tests for certs_exist function."""

    def test_returns_false_when_neither_exists(self, tmp_path: Path) -> None:
        """Test returns False when neither cert nor key exists."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"

        result = certs_exist(cert_path=cert_path, key_path=key_path)

        assert result is False

    def test_returns_false_when_only_cert_exists(self, tmp_path: Path) -> None:
        """Test returns False when only certificate exists."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        cert_path.write_text("dummy cert")

        result = certs_exist(cert_path=cert_path, key_path=key_path)

        assert result is False

    def test_returns_false_when_only_key_exists(self, tmp_path: Path) -> None:
        """Test returns False when only key exists."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        key_path.write_text("dummy key")

        result = certs_exist(cert_path=cert_path, key_path=key_path)

        assert result is False

    def test_returns_true_when_both_exist(self, tmp_path: Path) -> None:
        """Test returns True when both cert and key exist."""
        cert_path = tmp_path / "server.crt"
        key_path = tmp_path / "server.key"
        generate_self_signed_cert(cert_path=cert_path, key_path=key_path)

        result = certs_exist(cert_path=cert_path, key_path=key_path)

        assert result is True


class TestGetCertPaths:
    """Tests for get_cert_paths function."""

    def test_returns_default_paths_when_none_provided(self) -> None:
        """Test that default paths are returned when no paths provided."""
        cert_path, key_path = get_cert_paths()

        assert cert_path == DEFAULT_CERT_PATH
        assert key_path == DEFAULT_KEY_PATH

    def test_returns_custom_paths_when_provided(self, tmp_path: Path) -> None:
        """Test that custom paths are returned when provided."""
        custom_cert = tmp_path / "custom.crt"
        custom_key = tmp_path / "custom.key"

        cert_path, key_path = get_cert_paths(
            cert_path=custom_cert,
            key_path=custom_key,
        )

        assert cert_path == custom_cert
        assert key_path == custom_key

    def test_returns_mixed_paths(self, tmp_path: Path) -> None:
        """Test that mixed custom/default paths work correctly."""
        custom_cert = tmp_path / "custom.crt"

        cert_path, key_path = get_cert_paths(cert_path=custom_cert)

        assert cert_path == custom_cert
        assert key_path == DEFAULT_KEY_PATH


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_validity_days(self) -> None:
        """Test default validity is 365 days."""
        assert DEFAULT_VALIDITY_DAYS == 365

    def test_default_cn(self) -> None:
        """Test default common name is localhost."""
        assert DEFAULT_CN == "localhost"

    def test_default_sans(self) -> None:
        """Test default SANs include localhost and common IPs."""
        assert "localhost" in DEFAULT_SANS
        assert "127.0.0.1" in DEFAULT_SANS
        assert "0.0.0.0" in DEFAULT_SANS

