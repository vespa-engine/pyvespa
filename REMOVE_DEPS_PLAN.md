# Plan: Remove `requests`, `requests_toolbelt`, and `httpx` from runtime dependencies

Branch: `thomasht86/remove-old-dependencies`
Owner: @thomasht86
Status: Proposed
Last updated: 2026-05-13

## Goal

Drop `requests`, `requests_toolbelt`, and `httpx` from the **runtime** dependency
group in `pyproject.toml`. The migration to `httpr` (Rust-based HTTP client) is
already partially complete; this plan finishes it and pushes any remaining
test-only usage into dev/test extras.

End state:

```toml
dependencies = [
    "httpr>=0.4.0",
    "docker",
    "jinja2",
    "cryptography>=46.0.7",
    "aiohttp",
    "tenacity>=8.4.1",
    "typing_extensions",
    "python-dateutil",
    "fastcore>=1.7.8",
    "lxml",
    "certifi",
]
```

`requests` stays only in the `unittest`/`dev`/`feed` extras (where appropriate).
`requests_toolbelt` and `httpx` disappear entirely from runtime; either replaced
or moved to extras if a test still needs them.

## Current state — what blocks removal

### `vespa/` runtime imports (must be removed/replaced)

| File | What it imports | Use |
|------|-----------------|-----|
| `vespa/application.py:14-16` | `Session`, `Response`, `ConnectionError`, `HTTPError`, `JSONDecodeError` from `requests` | Type hints, exception handling, hybrid `raise_for_status` |
| `vespa/application.py:38` | `import httpx` | Type hints on `VespaAsync` ctor: `httpx.Timeout`, `httpx.AsyncClient`; deprecation shim |
| `vespa/application.py:2050,2052,2128,2141-2147` | `httpx.Timeout`, `httpx.AsyncClient` | Backwards-compat: accept the old type and emit `DeprecationWarning` |
| `vespa/models.py:8,276,279,287` | `requests.head`, `requests.get`, `requests.RequestException` | Validating user-supplied model/tokenizer URLs |
| `vespa/deployment.py:29` | `MultipartEncoder` from `requests_toolbelt` | Building multipart bodies for Vespa Cloud deploys (3 call sites: lines 2367, 2456, and `Union[..., MultipartEncoder, ...]` type hints at 1683/1702/1721/1728) |

### `tests/` imports (allowed to keep — move to extras)

| File | Import | Notes |
|------|--------|-------|
| `tests/unit/test_application.py:10,20` | `requests.HTTPError`, `requests.Response`, `httpx` | Mocks for hybrid `raise_for_status`; comments indicate active migration away from `httpx.AsyncClient` |
| `tests/integration/test_integration_*.py` | `requests`, `requests.HTTPError` | Integration tests hit Vespa endpoints directly |

`requests-mock` is already declared in the `unittest` extra but isn't used anywhere
in the code — flag for removal during cleanup.

### Scripts / non-runtime (no action required from this plan)

`feed_to_vespa.py`, `results/inference/onnxbench.py`,
`vespacli/utils/*`, `.github/scripts/linkcheck.py` — all sit outside the installed
package and don't affect runtime deps.

## Strategy

Migration is roughly **60–70% done** for `httpx` (sync path is fully on `httpr`;
async path uses `httpr.AsyncClient` but keeps an `httpx`-typed surface for
back-compat). The remaining work splits cleanly into four small PRs that can
ship and be reverted independently.

PRs are ordered so each leaves `master` in a green, releasable state. PR 1 and
PR 2 are independent and can be done in parallel; PR 3 depends on PRs 1+2; PR 4
is the final cleanup.

---

## PR 1 — Replace `requests_toolbelt.MultipartEncoder` in deployment

**Branch:** `thomasht86/remove-deps-1-toolbelt`
**Scope:** ~50–80 LOC change, deployment.py only

### Changes

- Implement a small internal multipart builder (probably in
  `vespa/_multipart.py` or inline in deployment.py — single ~40-line function)
  that produces the same `(bytes, content_type)` pair `MultipartEncoder` does.
  We already serialize to bytes via `to_string()` immediately on lines 1736,
  2376, 2403 — we never stream — so the full streaming API of
  `requests_toolbelt` isn't needed.
- Alternative: use Python stdlib `email.mime.multipart` / `email.encoders`, or
  build the boundary manually (~30 lines). No external dep needed.
- Replace the 3 instantiations at `deployment.py:2367`, `:2456`, and (via
  helper) `:1728`.
- Update `Union[..., MultipartEncoder, ...]` type hints at
  `deployment.py:1683, 1702, 1721` to use the new type (or `bytes`).

### Tests

- Existing integration tests (`tests/integration/test_integration_vespa_cloud*.py`)
  exercise the deploy path. Run them locally against a Vespa Cloud dev
  instance.
- Add a unit test that builds a multipart body and asserts the boundary,
  `Content-Type`, and that `application/zip` parts are preserved byte-for-byte.

### Deliverable

`requests_toolbelt` is no longer imported anywhere in `vespa/`. Do **not** yet
remove it from `pyproject.toml` — that happens in PR 4 alongside the others to
keep one pin-removal commit.

### Risk

Low. Multipart format is a stable spec; the Vespa Cloud deploy endpoint accepts
standard multipart. The main risk is content-hash mismatch (`api_key` auth path
at `deployment.py:2374`) if byte output differs — round-trip test against the
existing `MultipartEncoder.to_string()` output to verify identical bytes for
the same input + boundary.

---

## PR 2 — Replace `requests` in `vespa/models.py`

**Branch:** `thomasht86/remove-deps-2-models`
**Scope:** ~10 LOC, models.py only

### Changes

- `vespa/models.py:_validate_single_url` uses `requests.head` then `requests.get(stream=True)`
  to validate user-supplied URLs.
- Replace with `httpr.head(...)` / `httpr.get(...)` (sync `httpr` API), or use
  `urllib.request` from stdlib since this is a one-shot validation with no
  pooling concerns.
- Recommendation: **stdlib `urllib.request`** — one fewer thing depending on
  the httpr surface for a non-hot-path check.
- Update the `except requests.RequestException` clause to catch the new
  exception type.

### Tests

- Existing unit tests in `tests/unit/test_models.py` (if any) — verify they
  pass.
- Add a unit test using `pytest-httpserver` or a local socket to cover the
  HEAD-fails-then-GET-succeeds branch.

### Deliverable

`vespa/models.py` no longer imports `requests`.

### Risk

Very low. Pure refactor of a validation helper.

---

## PR 3 — Drop `httpx` and `requests` from `vespa/application.py`

**Branch:** `thomasht86/remove-deps-3-application`
**Scope:** ~80–120 LOC, application.py + test_application.py

This is the biggest PR. It has two halves that should be split if review gets
heavy — they're called out as PR 3a / 3b below in case we need to break it up.

### PR 3a — Remove `httpx` type-hint shim

- `application.py:2050` — change `timeout: Union[httpx.Timeout, int, float]` to
  `timeout: Union[int, float]`. Drop the `isinstance(timeout, httpx.Timeout)`
  branch and the associated `DeprecationWarning` (lines 2141–2147). This is a
  breaking change for any caller still passing `httpx.Timeout`; the warning has
  been live long enough — confirm by checking when the deprecation was
  introduced (`git blame application.py | grep -i httpx.Timeout`).
- `application.py:2052` — change `client: Optional[Union[httpx.AsyncClient,
  httpr.AsyncClient]]` to `Optional[httpr.AsyncClient]`. Same call out — any
  caller passing `httpx.AsyncClient` will hit a `TypeError` at runtime. This is
  the explicit point of the dep removal, so it's intentional, but it should be
  called out in the changelog/release notes.
- Update docstrings throughout `VespaAsync` (lines 348–387, 885, 1038, 1051,
  1094, 1106) that reference `httpx.AsyncClient` to say `httpr.AsyncClient`.
- `application.py:325-326, 371-373, 387` — same treatment on the sync
  `get_async_session` factory.
- Remove `import httpx` at line 38.
- Update `tests/unit/test_application.py` — comments at lines 863, 1071, 1089,
  1106 already say "Changed from httpx_client"; the test still imports
  `httpx` at line 20. Drop the import and any test that explicitly constructs
  an `httpx.Timeout` / `httpx.AsyncClient`.

### PR 3b — Remove `requests` from sync path

`requests` is used in `application.py` for three things:

1. **Exception types** in `raise_for_status` and `_is_connection_error`:
   `ConnectionError`, `HTTPError`, `JSONDecodeError`.
   - `ConnectionError` is shadowed by Python's builtin `ConnectionError` — the
     `requests.exceptions.ConnectionError` import was used to disambiguate.
     Switch to handling the builtin + `httpr.RequestError` / `httpr.ConnectError`
     which the code already checks for (line 88-89).
   - `HTTPError` is raised manually at line 227 wrapping httpr responses. Define
     a local `class HTTPError(Exception)` in `vespa/exceptions.py` or use
     `httpr.HTTPStatusError` if it exists. Prefer a local exception so callers
     who currently catch `requests.HTTPError` get a deprecation path via an
     alias.
   - `JSONDecodeError` — switch to `json.JSONDecodeError` (stdlib).

2. **The `Session` and `Response` type hints** at lines 14-15. These are dead
   weight on a code path that returns `httpr.Response`. Either drop the type
   annotation entirely or use a `Protocol` describing the duck-typed surface
   `raise_for_status` needs (`status_code`, `.json()`, `.text`,
   `raise_for_status` attr probe).

3. **Docstrings** at lines 426, 1425 mention `requests.Session` — update to
   reflect `httpr.Client`.

The `hasattr(response, "raise_for_status")` branch at line 212 can stay as
defensive code, but the comment "requests/httpx Response" should be updated to
"compatibility with response objects that have a built-in raise_for_status".

### Tests

- `tests/unit/test_application.py:10` imports `from requests.models import
  HTTPError, Response`. Update tests to construct mock `httpr.Response`-like
  objects or use the new local `HTTPError`.

### Deliverable

After this PR, `grep -rn "import httpx\|from httpx\|import requests\|from
requests" vespa/` returns no hits.

### Risk

Medium. This is a small **public API break**: anyone catching
`requests.HTTPError` from `raise_for_status` won't catch our new exception
type. Two mitigations:

- Re-export the new exception under the old name with a deprecation shim, OR
- Bump the minor version and call it out in CHANGELOG.

Recommend the deprecation shim — re-export `HTTPError` from `vespa.exceptions`
and document the upcoming change. Defer the actual `requests` removal from
`pyproject.toml` to PR 4 so the shim still works during the deprecation window
if we choose that path.

---

## PR 4 — Remove from `pyproject.toml`

**Branch:** `thomasht86/remove-deps-4-pyproject`
**Scope:** `pyproject.toml` + final integration test run

### Changes

- Delete the three lines from `[project.dependencies]`:

  ```toml
  # Temporarily keeping for comparison testing - will remove after migration
  "requests",
  "requests_toolbelt",
  "httpx[http2]",
  ```

- Decide per-library where (if anywhere) they should live in extras:

  | Library | Decision | Where |
  |---------|----------|-------|
  | `requests` | Keep in `unittest` extra (integration tests use it) | `unittest` extra |
  | `requests_toolbelt` | Drop entirely — nothing imports it after PR 1 | (removed) |
  | `httpx[http2]` | Drop entirely — nothing imports it after PR 3 | (removed) |
  | `requests-mock` | Drop from `unittest` — already unused | (removed) |

- The `feed` extra at line 84 already pins `requests<=2.31.0` for the feeding
  scripts — leave untouched. (Question: is the `<=2.31.0` pin still needed? If
  it was for compatibility with another lib that's since updated, also worth
  cleaning up — out of scope for this plan.)

### Tests

- `uv sync --extra dev && uv run pytest tests/unit/ -v`
- `uv sync --extra dev && uv run pytest tests/integration/ -v` against Docker
- `uv pip compile pyproject.toml` and inspect the lock — confirm `requests`,
  `requests_toolbelt`, and `httpx` are gone from the runtime resolution and
  appear only when dev/unittest extras are installed.
- `uv build` and inspect the wheel — `requests-*.whl` and `httpx-*.whl` should
  not be transitive runtime deps.

### Deliverable

Runtime dep tree shrinks by 3 direct + their transitives (`httpcore`, `h2`,
`hyperframe`, `hpack`, `idna`, `sniffio`, `anyio` indirectly, `charset-normalizer`,
`urllib3` direct from requests). The PR description should include a `pipdeptree`
diff to make the win visible.

### Risk

Low at this point — all the real work happened in PRs 1–3. This PR is the dep
declaration cleanup + final sanity check.

---

## Cross-cutting concerns

### Public API impact

The biggest risk in the whole effort is downstream users who:

- Pass `httpx.Timeout` or `httpx.AsyncClient` to `VespaAsync` (PR 3a).
- Catch `requests.HTTPError` from `raise_for_status` (PR 3b).
- Catch `requests.exceptions.ConnectionError` directly from our calls.

Mitigations: deprecation warnings have been in place for `httpx.Timeout` for
some time; check `git log -S "httpx.Timeout is deprecated"` to confirm the
window. For `HTTPError`, re-export from `vespa.exceptions` as a compatibility
alias for one minor version, then drop in the version after.

### CHANGELOG / release notes

PR 3 and PR 4 both warrant entries. Suggest:

> **Breaking:** `pyvespa` no longer depends on `requests`, `requests_toolbelt`,
> or `httpx` at runtime. Callers that previously passed `httpx.AsyncClient` or
> `httpx.Timeout` to `VespaAsync` must switch to `httpr.AsyncClient` and plain
> `float` timeouts. The deprecation warning for `httpx.Timeout` has been
> removed. `requests.HTTPError` raised by `raise_for_status` is replaced by
> `vespa.exceptions.HTTPError` (aliased for one release for backwards
> compatibility).

### Verification checklist (post-PR-4)

- [ ] `grep -rn "requests\|httpx\|toolbelt" vespa/` returns no matches
- [ ] `uv sync` resolves without `requests`/`httpx`/`requests-toolbelt`
- [ ] Unit + integration tests pass
- [ ] Notebook examples in `docs/` still execute
- [ ] `mktestdocs` passes

### Out of scope (suggested follow-ups)

- Audit the `feed` extra's `requests<=2.31.0` pin — confirm it's still needed.
- Migrate `feed_to_vespa.py` and `.github/scripts/linkcheck.py` off `requests`/`httpx`
  if/when convenient — these don't affect the installed package.
- Drop the unused `requests-mock` dev dep.
