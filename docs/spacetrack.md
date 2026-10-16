# Space-Track data

[Space-Track.org](https://www.space-track.org) is where the 18th Space
Defense Squadron (18 SDS) of USSPACECOM publishes its space situational
awareness data. keepout reads two of its classes:

- **`gp`**, the full GP catalog: the element sets of every tracked object
  in orbit, about 32 000 (31 930 on 2026-10-15), against the 20 000 of
  CelesTrak's public groups, which lack most rocket bodies and debris (see
  [The public catalog](screening.md#the-public-catalog)). `screen` and the
  synthetic operator take it like CelesTrak's catalog.
- **`cdm_public`**, public summaries of the conjunction data messages
  (CDMs) of approaches to come (see [Public CDMs](#public-cdms)).

CelesTrak's catalog needs no account and is keepout's default; Space-Track
needs one.

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
loaded into the environment of the command that needs it:

```bash
mkdir -p ~/.config/keepout
install -m 600 /dev/null ~/.config/keepout/spacetrack.env  # readable by you only
"$EDITOR" ~/.config/keepout/spacetrack.env  # SPACETRACK_IDENTITY=... and SPACETRACK_PASSWORD=...
(while IFS= read -r line || [ -n "$line" ]; do export "$line"; done \
    < ~/.config/keepout/spacetrack.env
 keepout fetch catalog --source spacetrack)
```

The file holds the two lines `SPACETRACK_IDENTITY=...` and
`SPACETRACK_PASSWORD=...` and nothing else, the values written as they are,
without quotes. The loop exports each line exactly as written, so a password
with `$`, `` ` ``, `#`, quotes or spaces arrives unchanged. Do not load the
file with `.` or `source`: the shell would read it as code, change or cut
such a password (the login is then refused) and even run a `$(...)` in it.

keepout never writes the credentials anywhere and never shows them: they
are hidden in `repr`, and any text from the server quoted in an error has
both replaced by `<hidden>`. A missing variable is an `AuthError` naming
it; a refused login is an `AuthError` too.

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
    catalog = client.fetch_catalog()  # a keepout.sources.celestrak.Catalog
    upcoming = client.fetch_public_cdms()
print(len(catalog.omms), "element sets")
for row in upcoming.cdms[:5]:  # by TCA
    print(
        row.tca.isot,
        row.object1.catalog_number,
        row.object2.catalog_number,
        f"{row.min_range:.0f} m",
        row.probability,
    )
```

`keepout.sources.spacetrack.fetch_catalog()` and `fetch_public_cdms()` do
the same with a client of their own. The catalog is a `Catalog` with one
group, `space-track`: the element sets of the objects not decayed with an
epoch in the last 10 days (the query Space-Track recommends), newest per
object.

## Limits

Space-Track suspends accounts that query too often
([API use guidelines](https://www.space-track.org/documentation#/api)), so
keepout's client enforces the published limits itself, and never retries:

| Limit | Space-Track's rule | What the client does |
|---|---|---|
| Requests | fewer than 30 a minute and 300 an hour | waits before a request that would make more than 20 in a minute or 200 in an hour (logins and logouts count), a third below the limits to leave room for the website and other use of the account |
| `gp` | the full catalog once an hour | a query within an hour of the last is not sent |
| `cdm_public` | all public CDMs once every 8 hours | a query within 8 hours of the last is not sent |
| Rate-limit answer | | HTTP 500 "violated your query rate limit" (or 429) raises `RateLimited` and stops the client: it sends nothing more, not even the logout |
| Expired session | | after HTTP 401, one new login and the same query once more |

Every answer is cached (`<cache>/spacetrack/gp/catalog.json`,
`<cache>/spacetrack/cdm_public/upcoming.json`, each with a `.meta.json`), and
the time of each class's last query is recorded in a ledger,
`<cache>/spacetrack/requests.json`, before the query is sent, so a query
that fails still counts. Within a class's interval the cached copy is
returned without contacting Space-Track; without one, `RateLimited` is
raised (also offline) with `retry_after`, the seconds to wait, and the
command line prints when to try again. `max_age` (seconds a cached copy is
used) can be longer than the interval, not shorter. The ledger is per cache
directory (`--cache-dir`, `$KEEPOUT_CACHE_DIR` or the user cache): processes
sharing one share the limits, but two sending at the same instant could both
send.

Space-Track asks that a scheduled hourly `gp` query runs at a random minute,
not at the top or bottom of the hour; pick one for a cron job.

## Public CDMs

`cdm_public` lists, for approaches still to come, a summary of each CDM 18
SDS issued. keepout reads each row as a `PublicCdm`: `cdm_id`, `created`,
`emergency_reportable`, `tca`, `min_range` [m], `probability` (Space-Track's
Pc) and the two objects (`PublicCdmObject`: catalog number, name, object
type, radar cross-section class and exclusion volume, the last as given).

- **Summaries, not CDMs.** There are no states, covariances or hard-body
  radii, so keepout cannot compute a Pc from a row and `keepout assess`
  cannot read them. Full CDMs (class `cdm`) are for the objects' operators
  under an SSA sharing agreement.
- **Emergency-reportable events.** The class lists the approaches 18 SDS
  rates emergency reportable, not every conjunction: in keepout's live
  check of 2026-10-15 every row was flagged `EMERGENCY_REPORTABLE = Y`.
- **`MIN_RNG` is in metres.** Space-Track does not document the unit; the
  live check settled it against keepout's own screening of the same pairs.
- **Not redistributable.** Unlike GP data, CDMs (public summaries included)
  may not be redistributed without Space-Track's approval (an orbital data
  request). keepout commits none: its tests use invented rows, and the live
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
`download RAW` sends four requests once (login, `gp`, `cdm_public`, logout)
and saves each answer unchanged, `build RAW` writes the committed files
offline from the saved answers, `verify` checks them, and `compare RAW`
prints, locally, how the public CDMs compare with keepout's screening of the
same pairs. Run `download` at a random minute, and only when the frozen
catalog must change.
