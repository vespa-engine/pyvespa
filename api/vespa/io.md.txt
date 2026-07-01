## `vespa.io`

### `VespaResponse(json, status_code, url, operation_type)`

Bases: `object`

Class to represent a Vespa HTTP API response.

#### `get_status_code()`

Return status code of the response.

#### `is_successfull()`

[Deprecated] Use is_successful() instead

#### `is_successful()`

True if status code is 200.

#### `get_json()`

Return json of the response.

### `VespaQueryResponse(json, status_code, url, request_body=None)`

Bases: `VespaResponse`

#### `get_json()`

For debugging when the response does not have hits.

Returns:

| Type   | Description                    |
| ------ | ------------------------------ |
| `Dict` | JSON object with full response |
