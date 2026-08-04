# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Unit tests for vespa.validation.

The expectations mirror the Vespa Cloud controller's
``com.yahoo.vespa.hosted.controller.api.identifiers.IdentifierTest``.
"""

import re
import unittest

from vespa.validation import (
    MAX_NAME_LENGTH,
    MAX_NAME_LENGTH_LEGACY,
    to_vespa_identifier,
    validate_application_name,
    validate_application_package_name,
    validate_cloud_names,
    validate_instance_name,
    validate_tenant_name,
)

# The pattern as written in the controller, used to assert that our (linear time)
# formulation accepts and rejects exactly the same names.
CONTROLLER_PATTERN = re.compile(r"^(?=.{1,40}$)[a-z](-?[a-z0-9]+)*$")

VALID_NAMES = [
    "a",
    "myapp",
    "my-app",
    "search2",
    "my-search-app",
    "a1-2b-3c",
    "msbe",
    "a" * MAX_NAME_LENGTH,
]

INVALID_NAMES = [
    "",
    "2fast",  # does not start with a letter
    "-app",  # leading dash
    "app-",  # trailing dash
    "my--app",  # double-dash
    "My_App",  # uppercase and underscore
    "MixedCaseApplication",
    "underscore_application",
    "application.with.dots",
    "`",
    "app name",  # space
    "app\n",  # trailing newline
    "a" * (MAX_NAME_LENGTH + 1),  # too long
]


class TestNamePattern(unittest.TestCase):
    def test_equivalent_to_controller_pattern(self):
        for name in VALID_NAMES + INVALID_NAMES:
            with self.subTest(name=name):
                # Java's Matcher.matches() requires the whole input to be consumed,
                # which fullmatch replicates.
                expected = CONTROLLER_PATTERN.fullmatch(name) is not None
                actual = True
                try:
                    validate_tenant_name(name)
                except ValueError:
                    actual = False
                self.assertEqual(expected, actual)


class TestValidateTenantName(unittest.TestCase):
    def test_valid(self):
        for name in VALID_NAMES:
            with self.subTest(name=name):
                self.assertEqual(validate_tenant_name(name), name)

    def test_invalid(self):
        for name in INVALID_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_tenant_name(name)

    def test_reserved(self):
        for name in ["api", "default"]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as ctx:
                    validate_tenant_name(name)
                self.assertIn("reserved", str(ctx.exception))

    def test_not_a_string(self):
        with self.assertRaises(TypeError):
            validate_tenant_name(None)


class TestValidateApplicationName(unittest.TestCase):
    def test_valid(self):
        for name in VALID_NAMES:
            with self.subTest(name=name):
                self.assertEqual(validate_application_name(name), name)

    def test_invalid(self):
        for name in INVALID_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_application_name(name)

    def test_reserved(self):
        for name in ["api", "default"]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_application_name(name)

    def test_cannot_be_longer_than_40_characters(self):
        with self.assertRaises(ValueError):
            validate_application_name("application-name-longer-than-40-characters")

    def test_legacy_endpoints_limits_length_to_20_characters(self):
        name = "a" * MAX_NAME_LENGTH_LEGACY
        self.assertEqual(validate_application_name(name, legacy_endpoints=True), name)
        with self.assertRaises(ValueError):
            validate_application_name(
                "longer-than-20-characters", legacy_endpoints=True
            )
        # Without legacy endpoints, the same name is fine.
        self.assertEqual(
            validate_application_name("longer-than-20-characters"),
            "longer-than-20-characters",
        )

    def test_error_message_states_the_limit(self):
        with self.assertRaises(ValueError) as ctx:
            validate_application_name(
                "longer-than-20-characters", legacy_endpoints=True
            )
        self.assertIn("no more than 20 characters", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            validate_application_name("a" * 41)
        self.assertIn("no more than 40 characters", str(ctx.exception))


class TestValidateInstanceName(unittest.TestCase):
    def test_valid(self):
        for name in VALID_NAMES:
            with self.subTest(name=name):
                self.assertEqual(validate_instance_name(name), name)

    def test_invalid(self):
        for name in INVALID_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_instance_name(name)

    def test_default_is_allowed(self):
        # InstanceId extends SerializedIdentifier, not NonDefaultIdentifier, and
        # 'default' is the default instance name in Vespa Cloud.
        self.assertEqual(validate_instance_name("default"), "default")

    def test_api_is_reserved(self):
        with self.assertRaises(ValueError):
            validate_instance_name("api")

    def test_tester_suffix_is_reserved(self):
        with self.assertRaises(ValueError) as ctx:
            validate_instance_name("myinstance-t")
        self.assertIn("tester", str(ctx.exception))
        # A name merely containing '-t' is fine.
        self.assertEqual(validate_instance_name("my-test"), "my-test")


class TestValidateApplicationPackageName(unittest.TestCase):
    def test_valid(self):
        for name in VALID_NAMES + ["my_app", "my_app-2", "a_b_c"]:
            with self.subTest(name=name):
                self.assertEqual(validate_application_package_name(name), name)

    def test_invalid(self):
        for name in [
            "",
            "2fast",  # does not start with a letter
            "-app",  # leading dash
            "_app",  # leading underscore
            "My_App",  # uppercase
            "app.with.dots",
            "app name",  # space
            "app\n",  # trailing newline
            "a" * (MAX_NAME_LENGTH + 1),  # too long
        ]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_application_package_name(name)

    def test_cloud_rules_do_not_apply(self):
        # An application package name is not a Vespa Cloud name, so the reserved names
        # and the no-double-dash and no-trailing-dash rules do not apply.
        for name in ["api", "default", "my--app", "my-app-"]:
            with self.subTest(name=name):
                self.assertEqual(validate_application_package_name(name), name)

    def test_derived_names_are_valid(self):
        # Whatever is accepted must yield a valid schema name and cluster ids.
        vespa_identifier = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
        ncname = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.\-]*")
        for name in VALID_NAMES + ["my_app", "my--app", "my-app-", "api"]:
            with self.subTest(name=name):
                self.assertTrue(vespa_identifier.fullmatch(to_vespa_identifier(name)))
                for cluster_id in (f"{name}_container", f"{name}_content"):
                    self.assertTrue(ncname.fullmatch(cluster_id))


class TestValidateCloudNames(unittest.TestCase):
    def test_all_valid(self):
        validate_cloud_names(
            tenant="my-tenant", application="my-app", instance="default"
        )

    def test_none_is_not_validated(self):
        validate_cloud_names()
        validate_cloud_names(tenant="my-tenant")

    def test_reports_which_name_is_invalid(self):
        with self.assertRaises(ValueError) as ctx:
            validate_cloud_names(tenant="my-tenant", application="my--app")
        self.assertIn("application", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            validate_cloud_names(tenant="My-Tenant")
        self.assertIn("tenant", str(ctx.exception))

    def test_legacy_endpoints_applies_to_application_only(self):
        long_name = "longer-than-20-characters"
        with self.assertRaises(ValueError):
            validate_cloud_names(application=long_name, legacy_endpoints=True)
        # Tenant and instance names are never restricted to 20 characters.
        validate_cloud_names(
            tenant=long_name, instance=long_name, legacy_endpoints=True
        )


class TestToVespaIdentifier(unittest.TestCase):
    def test_dashes_become_underscores(self):
        self.assertEqual(to_vespa_identifier("my-app"), "my_app")
        self.assertEqual(to_vespa_identifier("myapp"), "myapp")
        self.assertEqual(to_vespa_identifier("my-search-app"), "my_search_app")


if __name__ == "__main__":
    unittest.main()
