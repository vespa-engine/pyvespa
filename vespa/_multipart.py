"""Minimal multipart/form-data encoder.

Replaces the third-party multipart encoder previously used by the deployment
API, which always serializes to bytes and never streams. Byte-for-byte
compatible with the previous encoder's output for the field shapes used in
vespa.deployment.
"""

from io import BytesIO
from typing import Optional, Tuple
from uuid import uuid4


def encode_multipart(fields: dict, boundary: Optional[str] = None) -> Tuple[bytes, str]:
    """Encode multipart/form-data fields to (body_bytes, content_type).

    Each field value is a (filename, data, content_type) tuple; data may be
    bytes, BytesIO or str. Matches the wire format previously produced by the
    third-party encoder, including ``filename=""`` for empty filenames and the
    trailing ``--boundary--\\r\\n``.
    """
    b = boundary or uuid4().hex  # same default-boundary scheme as the previous encoder
    out = bytearray()
    for name, (filename, data, content_type) in fields.items():
        if isinstance(data, BytesIO):
            data = data.getvalue()
        elif isinstance(data, str):
            data = data.encode("utf-8")
        out += (
            f"--{b}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        out += data + b"\r\n"
    out += f"--{b}--\r\n".encode("utf-8")
    return bytes(out), f"multipart/form-data; boundary={b}"
