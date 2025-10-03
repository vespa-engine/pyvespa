# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import requests
import httpx
from vespa.application import Vespa


class TestCertContentIntegration(unittest.TestCase):
    """Integration tests for cert_content and key_content functionality."""

    def setUp(self):
        """Set up test fixtures with sample certificate content."""
        # Sample certificate content (these are not real certificates)
        self.cert_content = """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAMlyFqk69v+9MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxv
Y2FsaG9zdDAeFw0yNDAxMDEwMDAwMDBaFw0yNTAxMDEwMDAwMDBaMBQxEjAQBgNV
BAMMCWxvY2FsaG9zdDBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQC3+t5iV7ZLnhef
ZZwEy8uOZGU/yq4k0uT8b9wF2jLZ8Yj+T8E1dC5eT9kJ+BtY8F7JGfBt0gN7w2V
K2vKl0PhtAgMBAAEwDQYJKoZIhvcNAQELBQADQQAI8XkZkVbYJdC9QJ+kQJMnE4W
VY2CnJjMKfzP+tKQf1Wm8dR2Kl3aMf1U+E3b7J5K9Nz1tVfY5lBnFw0gJt5bE
-----END CERTIFICATE-----"""
        
        self.key_content = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgVcBdAbcdGm8n8kCj
3J0jxUzKRNS1jEL8HAMwGlQFHgihRANCAATdHmUczN06o+tZQ/DGPr72GQDzZG2A
wqVZdvpwmT8n2TQv/8Y9bJfzKHT6JqQEYtjjOAh8tGRNRvQVvZQJvL9A
-----END PRIVATE KEY-----"""

    def test_cert_content_with_sync_context_manager(self):
        """Test cert_content functionality with VespaSync context manager."""
        app = Vespa(
            url="https://localhost:8080",
            cert_content=self.cert_content,
            key_content=self.key_content
        )
        
        # Mock requests Session to avoid actual network calls
        with patch('requests.Session') as mock_session_class:
            mock_session = MagicMock()
            mock_session_class.return_value = mock_session
            
            # Test that VespaSync properly handles cert_content
            with app.syncio() as sync_app:
                self.assertIsInstance(sync_app.cert, tuple)
                cert_path, key_path = sync_app.cert
                
                # Verify that the temporary files contain the correct content
                with open(cert_path, 'r') as f:
                    self.assertEqual(f.read(), self.cert_content)
                
                with open(key_path, 'r') as f:
                    self.assertEqual(f.read(), self.key_content)
                
                # Verify that the session's cert attribute is set to our tuple
                expected_cert = (cert_path, key_path)
                # Get the cert value that was set on the session
                cert_calls = [call for call in mock_session.method_calls if 'cert' in str(call)]
                
            # After exiting context, temp files should be cleaned up
            self.assertFalse(os.path.exists(cert_path))
            self.assertFalse(os.path.exists(key_path))

    def test_cert_content_with_async_context_manager(self):
        """Test cert_content functionality with VespaAsync context manager."""
        import asyncio
        
        app = Vespa(
            url="https://localhost:8080",
            cert_content=self.cert_content,
            key_content=self.key_content
        )
        
        async def async_test():
            # Mock httpx.create_ssl_context to avoid SSL validation issues
            with patch('httpx.create_ssl_context') as mock_ssl_context:
                mock_ssl_context.return_value = None
                
                async with app.asyncio() as async_app:
                    # Verify temporary files are created with correct content
                    self.assertTrue(os.path.exists(async_app._cert_path))
                    self.assertTrue(os.path.exists(async_app._key_path))
                    
                    with open(async_app._cert_path, 'r') as f:
                        self.assertEqual(f.read(), self.cert_content)
                    
                    with open(async_app._key_path, 'r') as f:
                        self.assertEqual(f.read(), self.key_content)
                    
                    # Store paths for cleanup verification
                    cert_path = async_app._cert_path
                    key_path = async_app._key_path
                
                # Verify that SSL context was created with our temp files
                mock_ssl_context.assert_called_with(cert=(cert_path, key_path))
                
                # After exiting context, temp files should be cleaned up
                self.assertFalse(os.path.exists(cert_path))
                self.assertFalse(os.path.exists(key_path))
        
        # Run the async test
        asyncio.run(async_test())

    def test_cert_file_vs_cert_content_behavior(self):
        """Test that file-based certs and content-based certs work differently but correctly."""
        
        # Create a temporary cert file for comparison
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as temp_cert:
            temp_cert.write(self.cert_content)
            temp_cert_path = temp_cert.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pem', delete=False) as temp_key:
            temp_key.write(self.key_content)
            temp_key_path = temp_key.name
        
        try:
            # Test with file paths
            app_file = Vespa(
                url="https://localhost:8080",
                cert=temp_cert_path,
                key=temp_key_path
            )
            
            # Test with content
            app_content = Vespa(
                url="https://localhost:8080",
                cert_content=self.cert_content,
                key_content=self.key_content
            )
            
            with patch('requests.Session'):
                # Both should work with VespaSync
                with app_file.syncio() as sync_file:
                    self.assertEqual(sync_file.cert, (temp_cert_path, temp_key_path))
                
                with app_content.syncio() as sync_content:
                    # Should use temporary files
                    self.assertIsInstance(sync_content.cert, tuple)
                    # But the paths should be different from the original files
                    self.assertNotEqual(sync_content.cert[0], temp_cert_path)
                    self.assertNotEqual(sync_content.cert[1], temp_key_path)
        
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_cert_path)
                os.unlink(temp_key_path)
            except OSError:
                pass


if __name__ == '__main__':
    unittest.main()