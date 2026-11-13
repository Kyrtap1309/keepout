# Space-Track data

[Space-Track.org](https://www.space-track.org) is where the 18th Space
Defense Squadron (18 SDS) of USSPACECOM publishes its space situational
awareness data. keepout reads two of its classes:

- **`gp`**, the full GP catalog: the element sets of every tracked object
  in orbit, about 32 000 (31 930 on 2026-10-15). CelesTrak's public groups
  have about 20 000 and lack most rocket bodies and debris (see
  [The public catalog](screening.md#the-public-catalog)). `screen` and the
  synthetic operator take it like CelesTrak's catalog.
- **`cdm_public`**, public summaries of the conjunction data messages
  (CDMs) of approaches to come (see [Public CDMs](#public-cdms)).

CelesTrak's catalog needs no account and is keepout's default; Space-Track
needs one. The `keepout` commands on this page need the `cli` extra (see
[Install](index.md#install)); the Python examples need only the library.

## Account and credentials

An account is free (sign up at <https://www.space-track.org>), and the
user agreement asks each user to have their own. keepout reads the
credentials only from two environment variables, and only when a request
must be sent:

| Variable | Value |
|---|---|
| `SPACETRACK_IDENTITY` | the account's user name (the e-mail address it was opened with) |
| `SPACETRACK_PASSWORD` | its password |

One way to keep them out of the shell history is a file only you can read,
loaded into the environment of the command that needs it. The snippet
below creates the file, then runs one command in a subshell that exports
each line of it as is. It does not use `source`, which would read the file
as shell code (see below):

```bash
mkdir -p ~/.config/keepout
install -m 600 /dev/null ~/.config/keepout/spacetrack.env  # readable by you only
"$EDITOR" ~/.config/keepout/spacetrack.env  # SPACETRACK_IDENTITY=... and SPACETRACK_PASSWORD=...
(while IFS= read -r line || [ -n "$line" ]; do export "$line"; done \
    < ~/.config/keepout/spacetrack.env
 keepout fetch catalog --source spacetrack)
```

The file holds the two lines `SPACETRACK_IDENTITY=...` and
`SPACETRACK_PASSWORD=...` and nothing else. Write the values as they are,
without quotes. The loop exports each line exactly as written, so a password
with `$`, `` ` ``, `#`, quotes or spaces arrives unchanged. Do not load the
file with `.` or `source`. The shell would read it as code: it could change
or cut such a password, so that the login is refused, and even run a
`$(...)` in it.

In the repository, `uv` can load them from a `.env` file instead (git
ignores it; `.env.example` is the template):

```bash
cp .env.example .env && chmod 600 .env && "$EDITOR" .env
uv run --env-file .env keepout fetch catalog --source spacetrack
```

There the values must be in single quotes (`SPACETRACK_PASSWORD='...'`).
Unquoted or in double quotes, `uv` expands `$NAME` in them, which changes a
password holding a `$`. A password with a `'` cannot be written that way,
so use the file and loop above.

keepout never writes the credentials anywhere and never shows them. They
are hidden in `repr`, and any text from the server quoted in an error has
both replaced by `<hidden>`. A missing variable is an `AuthError` naming
it, and a refused login is an `AuthError` too.

## Fetching

```bash
keepout fetch catalog --source spacetrack   # the full catalog
keepout screen --fleet fleet.toml --source spacetrack
keepout fetch cdm                            # the public CDM summaries
```

`screen --source spacetrack` uses the cached catalog while it is current
and downloads it otherwise. In Python, one client logs in once for both
classes and logs out when closed:

```python
from keepout.sources.spacetrack import SpaceTrackClient

with SpaceTrackClient() as client:  # credentials from the environment
    catalog = client.fetch_catalog()  # a keepout.sources.catalog.Catalog
    upcoming = client.fetch_public_cdms()
print(len(catalog.omms), "element sets")
for row in upcoming.cdms[:5]:  # by TCA
    print(
        row.tca.isot,
        row.object1.catalog_number,
        row.object2.catalog_number,
        f"{row.range_at_tca:.0f} m",
        row.pc,
    )
```

`keepout.sources.spacetrack.fetch_catalog()` and `fetch_public_cdms()` do
the same with a client of their own. The catalog is a `Catalog` with one
part, `gp`, named after the query. It holds the newest element set of each
object not decayed with an epoch in the last 10 days, the query
Space-Track recommends.

## Limits

Space-Track suspends accounts that query too often
([API use guidelines](https://www.space-track.org/documentation#/api)).
keepout's client therefore enforces the published limits itself, and never
retries:

| Limit | Space-Track's rule | What the client does |
|---|---|---|
| Requests | fewer than 30 a minute and 300 an hour | waits before a request that would make more than 20 in a minute or 200 in an hour (logins and logouts count), a third below the limits to leave room for the website and other use of the account |
| `gp` | the full catalog once an hour | a query within an hour of the last is not sent |
| `cdm_public` | all public CDMs once every 8 hours | a query within 8 hours of the last is not sent |
| Rate-limit answer | | HTTP 500 "violated your query rate limit" (or 429) raises `RateLimited` and stops the client: it sends nothing more, not even the logout |
| Expired session | | after HTTP 401, one new login and the same query once more |

Every answer is asked for gzipped and cached gzipped, each with a
`.meta.json`, in keepout's cache directory, `<cache>` below. It is
`--cache-dir` if given, else `$KEEPOUT_CACHE_DIR`, else the user cache
directory (e.g. `~/.cache/keepout` on Linux):

- `<cache>/spacetrack/gp/catalog.json.gz`, about 5 MB instead of 36 MB of
  JSON;
- `<cache>/spacetrack/cdm_public/upcoming.json.gz`.

The time of each class's last query is recorded in a ledger,
`<cache>/spacetrack/requests.json`, before the query is sent. A query that
fails therefore still counts.

Within a class's interval, keepout returns the cached copy without
contacting Space-Track. If there is no cached copy, it raises
`RateLimited`, also offline. The error's `retry_after` gives the seconds to
wait, and the command line prints when to try again. `max_age` (the seconds
a cached copy is used) can be longer than the interval, not shorter.

The ledger is per cache directory. Processes sharing one share the limits, but two sending at
the same instant could both send.

Space-Track asks that a scheduled hourly `gp` query runs at a random minute,
not at the top or bottom of the hour; pick one for a cron job.

## Public CDMs

`cdm_public` lists a summary of each CDM 18 SDS issued for approaches still
to come. keepout reads each row as a `PublicCdm` with these fields:
`cdm_id`, `created`, `emergency_reportable`, `tca`, `range_at_tca` [m],
`pc` (Space-Track's Pc) and the two objects. Each object is a
`PublicCdmObject`: catalog number, name, object type, radar cross-section
class and exclusion volume (the last as given).

- **Summaries, not CDMs.** There are no states, covariances or hard-body
  radii, so keepout cannot compute a Pc from a row and `keepout assess`
  cannot read them. Full CDMs (class `cdm`) are for the objects' operators
  under an SSA sharing agreement.
- **Emergency-reportable events.** The class lists the approaches 18 SDS
  rates emergency reportable, not every conjunction. In keepout's live
  check of 2026-10-15 every row was flagged `EMERGENCY_REPORTABLE = Y`.
- **`MIN_RNG` is in metres.** Space-Track does not document the unit; the
  live check settled it against keepout's own screening of the same pairs.
- **Not redistributable.** Unlike GP data, CDMs (public summaries included)
  may not be redistributed without Space-Track's approval (an orbital data
  request). keepout commits none. Its tests use invented rows, and the live
  check stays on the maintainer's disk.

## Data and citation

GP element sets are basic space situational awareness data, which the
Space-Track.org user agreement allows to be redistributed with citation.
Cite data you download as: *USSPACECOM/18 SDS via Space-Track.org,
retrieved YYYY-MM-DD*. keepout commits one such catalog for its tests and
the [screening benchmark](screening.md#validation-and-performance):
`tests/screening/spacetrack-data/spacetrack-gp.csv.gz`, the 31 930 element
sets retrieved on 2026-10-15, with a `NOTICE`; it is not covered by keepout's
MIT licence.

Maintainers make that snapshot with `tools/spacetrack_snapshot.py`:

- `download RAW` sends four requests once (login, `gp`, `cdm_public`,
  logout) and saves each answer unchanged.
- `build RAW` writes the committed files offline from the saved answers.
- `verify` checks them.
- `compare RAW` prints, locally, how the public CDMs compare with keepout's
  screening of the same pairs.

Run `download` at a random minute, and only when the frozen catalog must
change.
