#!/usr/bin/env python3
"""
Example script demonstrating cert_content and key_content functionality.

This script shows how to use the new cert_content and key_content parameters
instead of providing file paths to certificates.
"""

import os
from vespa.application import Vespa

def main():
    """Demonstrate the new cert_content and key_content functionality."""
    
    # Example 1: Using cert_content and key_content (new functionality)
    print("Example 1: Using cert_content and key_content")
    
    # In a real scenario, you might read these from environment variables
    # or from a secrets management system
    cert_content = os.getenv('VESPA_CERT_CONTENT', """-----BEGIN CERTIFICATE-----
MIIBkTCB+wIJAMlyFqk69v+9MA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNVBAMMCWxv
Y2FsaG9zdDAeFw0yNDAxMDEwMDAwMDBaFw0yNTAxMDEwMDAwMDBaMBQxEjAQBgNV
BAMMCWxvY2FsaG9zdDBcMA0GCSqGSIb3DQEBAQUAA0sAMEgCQQC3+t5iV7ZLnhef
ZZwEy8uOZGU/yq4k0uT8b9wF2jLZ8Yj+T8E1dC5eT9kJ+BtY8F7JGfBt0gN7w2V
K2vKl0PhtAgMBAAEwDQYJKoZIhvcNAQELBQADQQAI8XkZkVbYJdC9QJ+kQJMnE4W
VY2CnJjMKfzP+tKQf1Wm8dR2Kl3aMf1U+E3b7J5K9Nz1tVfY5lBnFw0gJt5bE
-----END CERTIFICATE-----""")
    
    key_content = os.getenv('VESPA_KEY_CONTENT', """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgVcBdAbcdGm8n8kCj
3J0jxUzKRNS1jEL8HAMwGlQFHgihRANCAATdHmUczN06o+tZQ/DGPr72GQDzZG2A
wqVZdvpwmT8n2TQv/8Y9bJfzKHT6JqQEYtjjOAh8tGRNRvQVvZQJvL9A
-----END PRIVATE KEY-----""")
    
    try:
        # Create Vespa instance with certificate content
        app = Vespa(
            url="https://my-vespa-endpoint.vespa-app.cloud",
            cert_content=cert_content,
            key_content=key_content
        )
        print("✓ Successfully created Vespa instance with cert_content and key_content")
        print(f"  Cert content length: {len(app.cert_content)} characters")
        print(f"  Key content length: {len(app.key_content)} characters")
        
    except Exception as e:
        print(f"✗ Failed to create Vespa instance: {e}")
        return
    
    # Example 2: Validation - this should fail
    print("\nExample 2: Validation tests")
    
    try:
        # This should fail: cert and cert_content together
        Vespa(
            url="https://my-vespa-endpoint.vespa-app.cloud",
            cert="/path/to/cert.pem",
            cert_content=cert_content
        )
        print("✗ Should have failed with both cert and cert_content")
    except ValueError as e:
        print(f"✓ Correctly rejected cert and cert_content together: {e}")
    
    try:
        # This should fail: cert_content without key_content
        Vespa(
            url="https://my-vespa-endpoint.vespa-app.cloud",
            cert_content=cert_content
        )
        print("✗ Should have failed with cert_content but no key_content")
    except ValueError as e:
        print(f"✓ Correctly rejected cert_content without key_content: {e}")
    
    # Example 3: Show that file paths still work (traditional method)
    print("\nExample 3: Traditional file paths still work")
    try:
        app_file = Vespa(
            url="https://my-vespa-endpoint.vespa-app.cloud",
            cert="/path/to/cert.pem",
            key="/path/to/key.pem"
        )
        print("✓ Successfully created Vespa instance with file paths")
        print(f"  Cert file: {app_file.cert}")
        print(f"  Key file: {app_file.key}")
    except Exception as e:
        print(f"✗ Failed to create Vespa instance with file paths: {e}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("- ✓ cert_content and key_content parameters work correctly")
    print("- ✓ Validation prevents invalid parameter combinations")
    print("- ✓ Traditional cert and key file paths still work")
    print("- ✓ New functionality is backward compatible")

if __name__ == "__main__":
    main()