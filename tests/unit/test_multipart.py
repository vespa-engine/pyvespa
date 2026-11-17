"""Tests for the internal multipart encoder used by vespa.deployment.

The byte-equivalence tests use requests_toolbelt's MultipartEncoder as the
reference oracle. requests_toolbelt is scheduled to be removed from the
runtime dependencies (see the deps cleanup series); those tests skip
gracefully once it is gone.
"""

import hashlib
import json
import re
import unittest
from base64 import standard_b64encode
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cryptography.hazmat.primitives.asymmetric import ec

from vespa._multipart import encode_multipart
from vespa.deployment import VespaCloud

DEFAULT_BOUNDARY_RE = re.compile(r"^multipart/form-data; boundary=[0-9a-f]{32}$")
FIXED_BOUNDARY = "FIXEDBOUNDARY1234567890"


def reference_encode(fields):
    """Encode fields with the legacy third-party encoder (test oracle)."""
    from requests_toolbelt.multipart.encoder import MultipartEncoder

    encoder = MultipartEncoder(fields=fields, boundary=FIXED_BOUNDARY)
    return encoder.to_string(), encoder.content_type


class TestEncodeMultipart(unittest.TestCase):
    def test_byte_equivalent_to_legacy_encoder_for_prod_fields(self):
        """Field shape used by VespaCloud._start_prod_deployment."""
        fields = {
            "submitOptions": (
                "",
                json.dumps({"projectId": 1, "risk": 0}),
                "application/json",
            ),
            "applicationZip": (
                "application.zip",
                b"PK\x03\x04 application zip payload",
                "application/zip",
            ),
        }
        body, content_type = encode_multipart(fields, boundary=FIXED_BOUNDARY)
        expected_body, expected_content_type = reference_encode(fields)
        self.assertEqual(body, expected_body)
        self.assertEqual(content_type, expected_content_type)

    def test_byte_equivalent_to_legacy_encoder_for_dev_fields(self):
        """Field shape used by VespaCloud._start_deployment (BytesIO value)."""
        fields = {
            "applicationZip": (
                "application.zip",
                BytesIO(b"PK\x03\x04 application zip payload"),
                "application/zip",
            ),
            "deployOptions": (
                "",
                json.dumps({"vespaVersion": "8.123.45"}),
                "application/json",
            ),
        }
        body, content_type = encode_multipart(fields, boundary=FIXED_BOUNDARY)
        expected_body, expected_content_type = reference_encode(fields)
        self.assertEqual(body, expected_body)
        self.assertEqual(content_type, expected_content_type)

    def test_default_boundary_format(self):
        _, content_type = encode_multipart(
            {"field": ("file.txt", b"data", "text/plain")}
        )
        self.assertRegex(content_type, DEFAULT_BOUNDARY_RE)

    def test_empty_filename_emits_empty_filename_attribute(self):
        """submitOptions/deployOptions rely on filename="" for JSON parts."""
        body, _ = encode_multipart(
            {"submitOptions": ("", json.dumps({"projectId": 1}), "application/json")},
            boundary=FIXED_BOUNDARY,
        )
        self.assertIn(
            b'Content-Disposition: form-data; name="submitOptions"; filename=""',
            body,
        )


class TestStartProdDeploymentBody(unittest.TestCase):
    """Regression: _start_prod_deployment must send the same request body and
    sign the same bytes as before the encoder swap."""

    def setUp(self):
        self.application_package = None
        VespaCloud._try_get_access_token = lambda self: "fake_access_token"
        self.vespa_cloud = VespaCloud(
            tenant="test-tenant",
            application="test-app",
            application_package=None,
            application_root="/tmp/pyvespa-multipart-regression",
        )
        # Force the api-key path to exercise the content-hash signing input
        self.vespa_cloud.control_plane_auth_method = "api_key"
        self.vespa_cloud.api_key = ec.generate_private_key(ec.SECP256R1())
        self.vespa_cloud.api_public_key_bytes = b"fake-public-key-bytes"

    @patch("vespa.deployment.VespaCloud._request")
    @patch("vespa.deployment.VespaCloud.read_app_package_from_disk")
    def test_request_body_matches_legacy_encoder_output(
        self, mock_read_zip, mock_request
    ):
        mock_read_zip.return_value = b"PK\x03\x04 application zip payload"
        mock_request.return_value = {"message": "ok", "build": 1, "skipped": False}

        with TemporaryDirectory() as tmp:
            build_no = self.vespa_cloud._start_prod_deployment(application_root=tmp)

        self.assertEqual(build_no, 1)
        self.assertEqual(mock_request.call_count, 1)
        _, kwargs = mock_request.call_args
        body = kwargs["body"]
        headers = kwargs["headers"]

        self.assertIsInstance(body, BytesIO)
        body_bytes = body.getvalue()

        # Content-Type carries the boundary; Content-Length matches the body
        self.assertTrue(
            headers["Content-Type"].startswith("multipart/form-data; boundary=")
        )
        self.assertEqual(headers["Content-Length"], str(len(body_bytes)))

        # The signed content hash must be computed over the exact bytes sent
        self.assertEqual(
            headers["X-Content-Hash"],
            standard_b64encode(hashlib.sha256(body_bytes).digest()).decode("UTF-8"),
        )

        # Same fields + same boundary must reproduce the legacy bytes exactly
        boundary = headers["Content-Type"].split("boundary=")[1]
        from requests_toolbelt.multipart.encoder import MultipartEncoder

        fields = {
            "submitOptions": (
                "",
                json.dumps({"projectId": 1, "risk": 0}),
                "application/json",
            ),
            "applicationZip": (
                "application.zip",
                b"PK\x03\x04 application zip payload",
                "application/zip",
            ),
        }
        expected = MultipartEncoder(fields=fields, boundary=boundary).to_string()
        self.assertEqual(body_bytes, expected)


if __name__ == "__main__":
    unittest.main()
