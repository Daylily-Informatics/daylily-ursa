"""SSL certificate management for Ursa server.

This module provides self-signed certificate generation for HTTPS in development mode.
Certificates are stored in ~/.config/ursa/certs/ directory.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

# Certificate storage location
CONFIG_DIR = Path.home() / ".config" / "ursa"
CERTS_DIR = CONFIG_DIR / "certs"
DEFAULT_CERT_PATH = CERTS_DIR / "server.crt"
DEFAULT_KEY_PATH = CERTS_DIR / "server.key"

# Default certificate settings
DEFAULT_VALIDITY_DAYS = 365
DEFAULT_CN = "localhost"
DEFAULT_SANS = ["localhost", "127.0.0.1", "0.0.0.0"]


def ensure_certs_dir() -> Path:
    """Ensure the certificates directory exists."""
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    return CERTS_DIR


def generate_self_signed_cert(
    cert_path: Optional[Path] = None,
    key_path: Optional[Path] = None,
    cn: str = DEFAULT_CN,
    sans: Optional[list[str]] = None,
    days: int = DEFAULT_VALIDITY_DAYS,
    key_size: int = 2048,
) -> Tuple[Path, Path]:
    """Generate a self-signed certificate and private key.

    Args:
        cert_path: Path to write certificate (default: ~/.config/ursa/certs/server.crt)
        key_path: Path to write private key (default: ~/.config/ursa/certs/server.key)
        cn: Common Name for the certificate (default: localhost)
        sans: Subject Alternative Names (default: localhost, 127.0.0.1, 0.0.0.0)
        days: Certificate validity in days (default: 365)
        key_size: RSA key size in bits (default: 2048)

    Returns:
        Tuple of (cert_path, key_path)
    """
    cert_path = cert_path or DEFAULT_CERT_PATH
    key_path = key_path or DEFAULT_KEY_PATH
    sans = sans or DEFAULT_SANS

    ensure_certs_dir()

    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    # Build subject and issuer (self-signed, so they're the same)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Ursa Development"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])

    # Build Subject Alternative Names
    san_entries: list[x509.GeneralName] = []
    for san in sans:
        # Check if it looks like an IP address
        if san.replace(".", "").isdigit() or san == "0.0.0.0":
            try:
                import ipaddress
                san_entries.append(x509.IPAddress(ipaddress.ip_address(san)))
            except ValueError:
                san_entries.append(x509.DNSName(san))
        else:
            san_entries.append(x509.DNSName(san))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=days))
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write private key (with restrictive permissions)
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    key_path.chmod(0o600)  # Owner read/write only

    # Write certificate
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    cert_path.chmod(0o644)  # Owner read/write, others read

    return cert_path, key_path


def get_cert_info(cert_path: Optional[Path] = None) -> Optional[dict]:
    """Get information about an existing certificate.

    Args:
        cert_path: Path to certificate (default: ~/.config/ursa/certs/server.crt)

    Returns:
        Dictionary with certificate info, or None if cert doesn't exist
    """
    cert_path = cert_path or DEFAULT_CERT_PATH

    if not cert_path.exists():
        return None

    cert_data = cert_path.read_bytes()
    cert = x509.load_pem_x509_certificate(cert_data)

    # Calculate fingerprint
    fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    fingerprint_formatted = ":".join(fingerprint[i:i+2].upper() for i in range(0, len(fingerprint), 2))

    # Extract SANs
    sans: list[str] = []
    try:
        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        for name in san_ext.value:
            if isinstance(name, x509.DNSName):
                sans.append(name.value)
            elif isinstance(name, x509.IPAddress):
                sans.append(str(name.value))
    except x509.ExtensionNotFound:
        pass

    return {
        "path": str(cert_path),
        "subject": cert.subject.rfc4514_string(),
        "issuer": cert.issuer.rfc4514_string(),
        "serial_number": cert.serial_number,
        "not_valid_before": cert.not_valid_before_utc,
        "not_valid_after": cert.not_valid_after_utc,
        "fingerprint_sha256": fingerprint_formatted,
        "sans": sans,
        "is_expired": datetime.now(timezone.utc) > cert.not_valid_after_utc,
        "days_until_expiry": (cert.not_valid_after_utc - datetime.now(timezone.utc)).days,
    }


def certs_exist(
    cert_path: Optional[Path] = None,
    key_path: Optional[Path] = None,
) -> bool:
    """Check if both certificate and key files exist.

    Args:
        cert_path: Path to certificate (default: ~/.config/ursa/certs/server.crt)
        key_path: Path to private key (default: ~/.config/ursa/certs/server.key)

    Returns:
        True if both files exist, False otherwise
    """
    cert_path = cert_path or DEFAULT_CERT_PATH
    key_path = key_path or DEFAULT_KEY_PATH
    return cert_path.exists() and key_path.exists()


def get_cert_paths(
    cert_path: Optional[Path] = None,
    key_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Get the certificate and key paths (using defaults if not specified).

    Args:
        cert_path: Custom certificate path or None for default
        key_path: Custom key path or None for default

    Returns:
        Tuple of (cert_path, key_path)
    """
    return (cert_path or DEFAULT_CERT_PATH, key_path or DEFAULT_KEY_PATH)

