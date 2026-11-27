# Stability

From 1.0.0 keepout follows [semantic versioning](https://semver.org/): a
release that breaks something below is a new major version (2.0.0); a minor
version (1.1.0) adds without breaking; a patch (1.0.1) fixes without adding.
The [changelog](changelog.md) lists every change by version.

## What is covered

- **The Python API**: every name importable from `keepout` and its packages
  that does not start with an underscore and is documented in the
  [API reference](reference/index.md): its parameters, their defaults,
  return types, and the exceptions it raises. A module or name whose own
  name starts with `_` is internal, as is anything under `keepout.cli` and
  `keepout.dashboard` other than what the reference documents.
- **The command line**: the commands, their options and arguments, and the
  [exit statuses](cli.md).
- **The command line's JSON output**: the fields and their meaning, as
  [JSON output](cli-json.md) documents them. New fields may be added in a
  minor version; a field is never removed or redefined within 1.x.
- **The store's schema**: a store made by any 1.x keepout opens in a later
  1.x, after `keepout store upgrade` when the release notes say a
  migration was added ([Upgrading](store.md#upgrading)). The schema changes
  only through such migrations.

Not covered: numerical results to the last digit (a fix to a method may
change a Pc in its later digits, and the changelog says so), the wording of
messages and of the dashboard's pages, the layout of generated reports, the
`tools/` scripts, and the test suite.

## Deprecation

A covered name, option or field is not removed in a 1.x release. When one
is to go:

1. It keeps working for at least one minor release, and using it warns: a
   `DeprecationWarning` from Python, a warning on standard error from the
   command line. The warning names what to use instead.
2. The changelog lists it under **Deprecated** in the release that starts
   the warning.
3. It is removed in the next major release (2.0.0), listed under
   **Removed**.

## Supported Python versions

keepout 1.x supports the Python versions its `requires-python` names
(3.11 and later at 1.0.0) and is tested on each in CI. Dropping a Python
version that has reached its end of life is not a breaking change; it is
done in a minor release and listed in the changelog.
