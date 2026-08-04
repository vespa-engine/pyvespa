# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Validation of Vespa Cloud names for tenant, application and instance, and of the
name of an application package.

The Vespa Cloud rules implemented here mirror the ones enforced by the controller in
``com.yahoo.vespa.hosted.controller.api.identifiers`` (``Identifier``, ``TenantId``,
``ApplicationId`` and ``InstanceId``):

* Must start with a lowercase letter.
* May only contain lowercase letters, digits and dashes.
* No double-dashes, and may not end with a dash.
* At most 40 characters (20 if the tenant still supports legacy endpoints).
* ``api`` is reserved for all three, and ``default`` is reserved for tenant and
  application names (but is the default instance name, hence allowed for instances).
* Instance names ending in ``-t`` are reserved for tester instances.

The controller expresses the character rules as ``^(?=.{1,40}$)[a-z](-?[a-z0-9]+)*$``.
:data:`NAME_PATTERN` is an equivalent, but unambiguous, formulation of the same
language, with the length check done separately to avoid the exponential backtracking
that the nested quantifiers of the original would allow.

These rules apply to names that are given to Vespa Cloud. The name of an application
package is a separate, more permissive thing — see
:func:`validate_application_package_name`.
"""

import re
from typing import Iterable, Optional

# Note: The legacy limit of 20 characters is due to level 7 routing endpoint hostnames
# being 'cluster--application--tenant', which can be at most 63 characters. It is only
# relevant for tenants and applications listed in the endpoint-config feature flag.
MAX_NAME_LENGTH: int = 40
MAX_NAME_LENGTH_LEGACY: int = 20

#: Equivalent to the controller's ``[a-z](-?[a-z0-9]+)*``, without the length check.
NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(-[a-z0-9]+)*")

#: Application package names, which are not subject to the Vespa Cloud rules. See
#: :func:`validate_application_package_name`.
PACKAGE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")

#: Reserved for the Vespa Cloud API, and therefore not a valid name of anything.
RESERVED_API = "api"
#: Reserved name for tenants and applications. Allowed (and default) for instances.
RESERVED_DEFAULT = "default"
#: Suffix reserved for tester instances.
TESTER_INSTANCE_SUFFIX = "-t"

_CLOUD_RULES = (
    "must start with a lowercase letter, may only contain lowercase letters, digits "
    "and dashes, may not contain double-dashes, may not end with a dash, and may "
    "contain no more than {max_length} characters"
)

_PACKAGE_RULES = (
    "must start with a lowercase letter, may only contain lowercase letters, digits, "
    "underscores and dashes, and may contain no more than {max_length} characters"
)


def _validate_name(
    name: str,
    kind: str,
    pattern: "re.Pattern" = NAME_PATTERN,
    rules: str = _CLOUD_RULES,
    max_length: int = MAX_NAME_LENGTH,
    reserved: Iterable[str] = (RESERVED_API,),
) -> str:
    """Validate ``name`` as a name of the given kind.

    Args:
        name (str): The name to validate.
        kind (str): What the name names, e.g. ``"tenant"``. Used in error messages.
        pattern (re.Pattern, optional): Pattern the name must match in full.
        rules (str, optional): Description of the rules, for error messages.
        max_length (int, optional): Maximum length of the name. Defaults to 40.
        reserved (Iterable[str], optional): Names that are not allowed for this kind.

    Returns:
        str: The validated name, unchanged.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is not a valid name of the given kind.
    """
    if not isinstance(name, str):
        raise TypeError(
            "Invalid {} name: expected a string, got {}.".format(
                kind, type(name).__name__
            )
        )
    if name in reserved:
        raise ValueError(
            "Invalid {kind} name '{name}': '{name}' is reserved.".format(
                kind=kind, name=name
            )
        )
    if len(name) > max_length or not pattern.fullmatch(name):
        raise ValueError(
            "Invalid {kind} name '{name}': {kind} names {rules}.".format(
                kind=kind, name=name, rules=rules.format(max_length=max_length)
            )
        )
    return name


def validate_tenant_name(name: str) -> str:
    """Validate a Vespa Cloud tenant name.

    Args:
        name (str): The tenant name to validate.

    Returns:
        str: The validated tenant name, unchanged.

    Raises:
        ValueError: If the name is not a valid tenant name.

    Example:
        ```python
        from vespa.validation import validate_tenant_name

        validate_tenant_name("my-tenant")
        'my-tenant'
        ```
    """
    return _validate_name(name, "tenant", reserved=(RESERVED_API, RESERVED_DEFAULT))


def validate_application_name(name: str, legacy_endpoints: bool = False) -> str:
    """Validate a Vespa Cloud application name.

    Args:
        name (str): The application name to validate.
        legacy_endpoints (bool, optional): Set to True if the tenant still supports
            legacy (level 7 routing) endpoints, in which case the name may contain no
            more than 20 characters. Defaults to False.

    Returns:
        str: The validated application name, unchanged.

    Raises:
        ValueError: If the name is not a valid application name.

    Example:
        ```python
        from vespa.validation import validate_application_name

        validate_application_name("my-app")
        'my-app'
        ```
    """
    return _validate_name(
        name,
        "application",
        max_length=MAX_NAME_LENGTH_LEGACY if legacy_endpoints else MAX_NAME_LENGTH,
        reserved=(RESERVED_API, RESERVED_DEFAULT),
    )


def validate_instance_name(name: str) -> str:
    """Validate a Vespa Cloud application instance name.

    Note that ``default`` is a valid instance name, and that instance names ending in
    ``-t`` are reserved for tester instances.

    Args:
        name (str): The instance name to validate.

    Returns:
        str: The validated instance name, unchanged.

    Raises:
        ValueError: If the name is not a valid instance name.

    Example:
        ```python
        from vespa.validation import validate_instance_name

        validate_instance_name("default")
        'default'
        ```
    """
    _validate_name(name, "instance")
    if name.endswith(TESTER_INSTANCE_SUFFIX):
        raise ValueError(
            "Invalid instance name '{}': instance names ending in '{}' are reserved "
            "for tester instances.".format(name, TESTER_INSTANCE_SUFFIX)
        )
    return name


def validate_application_package_name(name: str) -> str:
    """Validate the name of an :class:`vespa.package.ApplicationPackage`.

    An application package name is not a Vespa Cloud application name — the Vespa Cloud
    tenant, application and instance names are given to :class:`vespa.deployment.VespaCloud`
    separately, and validated there — so the Vespa Cloud rules deliberately do not apply
    here, and self-hosted applications are free to use underscores. The name must,
    however, be usable for what pyvespa derives from it:

    * The default schema name, which is a Vespa identifier (``[a-zA-Z_][a-zA-Z0-9_]*``)
      once dashes are replaced by underscores, hence the leading letter.
    * The ``<name>_container`` and ``<name>_content`` cluster ids, which are ``xsd:NCName``
      in services.xml, and which end up verbatim in Vespa Cloud endpoint names (where
      only underscores are translated, to dashes), hence lowercase only and the length
      limit.
    * The Docker container name used by :class:`vespa.deployment.VespaDocker`.

    Args:
        name (str): The application package name to validate.

    Returns:
        str: The validated name, unchanged.

    Raises:
        ValueError: If the name is not a valid application package name.

    Example:
        ```python
        from vespa.validation import validate_application_package_name

        validate_application_package_name("my_app")
        'my_app'
        ```
    """
    return _validate_name(
        name,
        "application package",
        pattern=PACKAGE_NAME_PATTERN,
        rules=_PACKAGE_RULES,
        reserved=(),
    )


def to_vespa_identifier(name: str) -> str:
    """Convert a name to a Vespa identifier.

    Vespa identifiers, such as schema and document type names, may contain underscores
    but not dashes, so dashes are replaced by underscores. This is the inverse of the
    controller's ``Identifier.toDns()``.

    Args:
        name (str): The name to convert.

    Returns:
        str: The name with dashes replaced by underscores.

    Example:
        ```python
        from vespa.validation import to_vespa_identifier

        to_vespa_identifier("my-app")
        'my_app'
        ```
    """
    return name.replace("-", "_")


def validate_cloud_names(
    tenant: Optional[str] = None,
    application: Optional[str] = None,
    instance: Optional[str] = None,
    legacy_endpoints: bool = False,
) -> None:
    """Validate any combination of Vespa Cloud tenant, application and instance names.

    Arguments that are None are not validated.

    Args:
        tenant (str, optional): Tenant name. Defaults to None.
        application (str, optional): Application name. Defaults to None.
        instance (str, optional): Instance name. Defaults to None.
        legacy_endpoints (bool, optional): Whether the tenant supports legacy endpoints,
            which limits the application name to 20 characters. Defaults to False.

    Raises:
        ValueError: If any of the given names is invalid.
    """
    if tenant is not None:
        validate_tenant_name(tenant)
    if application is not None:
        validate_application_name(application, legacy_endpoints=legacy_endpoints)
    if instance is not None:
        validate_instance_name(instance)
