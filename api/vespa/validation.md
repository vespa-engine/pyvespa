## `vespa.validation`

Validation of Vespa Cloud names for tenant, application and instance, and of the name of an application package.

The Vespa Cloud rules implemented here mirror the ones enforced by the controller in `com.yahoo.vespa.hosted.controller.api.identifiers` (`Identifier`, `TenantId`, `ApplicationId` and `InstanceId`):

- Must start with a lowercase letter.
- May only contain lowercase letters, digits and dashes.
- No double-dashes, and may not end with a dash.
- At most 40 characters (20 if the tenant still supports legacy endpoints).
- `api` is reserved for all three, and `default` is reserved for tenant and application names (but is the default instance name, hence allowed for instances).
- Instance names ending in `-t` are reserved for tester instances.

The controller expresses the character rules as `^(?=.{1,40}$)[a-z](-?[a-z0-9]+)*$`. :data:`NAME_PATTERN` is an equivalent, but unambiguous, formulation of the same language, with the length check done separately to avoid the exponential backtracking that the nested quantifiers of the original would allow.

These rules apply to names that are given to Vespa Cloud. The name of an application package is a separate, more permissive thing — see :func:`validate_application_package_name`.

### `validate_tenant_name(name)`

Validate a Vespa Cloud tenant name.

Parameters:

| Name   | Type  | Description                  | Default    |
| ------ | ----- | ---------------------------- | ---------- |
| `name` | `str` | The tenant name to validate. | *required* |

Returns:

| Name  | Type  | Description                           |
| ----- | ----- | ------------------------------------- |
| `str` | `str` | The validated tenant name, unchanged. |

Raises:

| Type         | Description                             |
| ------------ | --------------------------------------- |
| `ValueError` | If the name is not a valid tenant name. |

Example

```python
from vespa.validation import validate_tenant_name

validate_tenant_name("my-tenant")
'my-tenant'
```

### `validate_application_name(name, legacy_endpoints=False)`

Validate a Vespa Cloud application name.

Parameters:

| Name               | Type   | Description                                                                                                                                                    | Default    |
| ------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `name`             | `str`  | The application name to validate.                                                                                                                              | *required* |
| `legacy_endpoints` | `bool` | Set to True if the tenant still supports legacy (level 7 routing) endpoints, in which case the name may contain no more than 20 characters. Defaults to False. | `False`    |

Returns:

| Name  | Type  | Description                                |
| ----- | ----- | ------------------------------------------ |
| `str` | `str` | The validated application name, unchanged. |

Raises:

| Type         | Description                                  |
| ------------ | -------------------------------------------- |
| `ValueError` | If the name is not a valid application name. |

Example

```python
from vespa.validation import validate_application_name

validate_application_name("my-app")
'my-app'
```

### `validate_instance_name(name)`

Validate a Vespa Cloud application instance name.

Note that `default` is a valid instance name, and that instance names ending in `-t` are reserved for tester instances.

Parameters:

| Name   | Type  | Description                    | Default    |
| ------ | ----- | ------------------------------ | ---------- |
| `name` | `str` | The instance name to validate. | *required* |

Returns:

| Name  | Type  | Description                             |
| ----- | ----- | --------------------------------------- |
| `str` | `str` | The validated instance name, unchanged. |

Raises:

| Type         | Description                               |
| ------------ | ----------------------------------------- |
| `ValueError` | If the name is not a valid instance name. |

Example

```python
from vespa.validation import validate_instance_name

validate_instance_name("default")
'default'
```

### `validate_application_package_name(name)`

Validate the name of an :class:`vespa.package.ApplicationPackage`.

An application package name is not a Vespa Cloud application name — the Vespa Cloud tenant, application and instance names are given to :class:`vespa.deployment.VespaCloud` separately, and validated there — so the Vespa Cloud rules deliberately do not apply here, and self-hosted applications are free to use underscores. The name must, however, be usable for what pyvespa derives from it:

- The default schema name, which is a Vespa identifier (`[a-zA-Z_][a-zA-Z0-9_]*`) once dashes are replaced by underscores, hence the leading letter.
- The `<name>_container` and `<name>_content` cluster ids, which are `xsd:NCName` in services.xml, and which end up verbatim in Vespa Cloud endpoint names (where only underscores are translated, to dashes), hence lowercase only and the length limit.
- The Docker container name used by :class:`vespa.deployment.VespaDocker`.

Parameters:

| Name   | Type  | Description                               | Default    |
| ------ | ----- | ----------------------------------------- | ---------- |
| `name` | `str` | The application package name to validate. | *required* |

Returns:

| Name  | Type  | Description                    |
| ----- | ----- | ------------------------------ |
| `str` | `str` | The validated name, unchanged. |

Raises:

| Type         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `ValueError` | If the name is not a valid application package name. |

Example

```python
from vespa.validation import validate_application_package_name

validate_application_package_name("my_app")
'my_app'
```

### `to_vespa_identifier(name)`

Convert a name to a Vespa identifier.

Vespa identifiers, such as schema and document type names, may contain underscores but not dashes, so dashes are replaced by underscores. This is the inverse of the controller's `Identifier.toDns()`.

Parameters:

| Name   | Type  | Description          | Default    |
| ------ | ----- | -------------------- | ---------- |
| `name` | `str` | The name to convert. | *required* |

Returns:

| Name  | Type  | Description                                   |
| ----- | ----- | --------------------------------------------- |
| `str` | `str` | The name with dashes replaced by underscores. |

Example

```python
from vespa.validation import to_vespa_identifier

to_vespa_identifier("my-app")
'my_app'
```

### `validate_cloud_names(tenant=None, application=None, instance=None, legacy_endpoints=False)`

Validate any combination of Vespa Cloud tenant, application and instance names.

Arguments that are None are not validated.

Parameters:

| Name               | Type   | Description                                                                                                          | Default |
| ------------------ | ------ | -------------------------------------------------------------------------------------------------------------------- | ------- |
| `tenant`           | `str`  | Tenant name. Defaults to None.                                                                                       | `None`  |
| `application`      | `str`  | Application name. Defaults to None.                                                                                  | `None`  |
| `instance`         | `str`  | Instance name. Defaults to None.                                                                                     | `None`  |
| `legacy_endpoints` | `bool` | Whether the tenant supports legacy endpoints, which limits the application name to 20 characters. Defaults to False. | `False` |

Raises:

| Type         | Description                           |
| ------------ | ------------------------------------- |
| `ValueError` | If any of the given names is invalid. |
