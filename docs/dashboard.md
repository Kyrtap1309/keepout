# The dashboard

The dashboard shows a [conjunction store](store.md) in a web browser: the
events to come with their risk, each event's Pc trend and geometry, and
its manoeuvre plan. It also takes new CDM files and queues the work a
[worker](store.md#background-jobs) does. It is a Streamlit app run by
`keepout dashboard`, on the machine that holds the store, or with its
worker and a demo store by `docker compose up` ([Deployment](deployment.md)).

It needs the `dashboard` extra ([Install](index.md#install)), which adds
[Streamlit](https://streamlit.io/) and
[Plotly](https://plotly.com/python/) to the `store` extra.
`import keepout` loads neither, and neither does `keepout --help`. Without
the extra, `keepout dashboard` prints the line to install it and exits
with status 1.

!!! warning "For this machine only"
    The dashboard listens on 127.0.0.1 and has no authentication: anyone
    who reaches its port, other users of a shared machine included, can read
    every event, store CDMs and queue or cancel jobs. Do not expose it or
    forward its port. From another machine, use an SSH tunnel:
    `ssh -L 8501:127.0.0.1:8501 HOST`, then open http://127.0.0.1:8501.

## Running it

The dashboard reads a store that exists already. It computes no Pc or
manoeuvre plan itself: it queues them as jobs for a
[worker](cli.md#worker) on the same store. Run the worker in a second
terminal:

```bash
keepout store add cdms/*.kvn --store keepout.db   # a store, e.g. from the CLI walk-through
keepout worker --store keepout.db                 # in a second terminal; Ctrl+C stops it
keepout dashboard --store keepout.db              # http://127.0.0.1:8501, Ctrl+C stops it
```

Without a worker, the dashboard still shows everything the store holds,
but **Compute**, **Plan manoeuvre** and uploads queue jobs that wait in
the store until a worker runs.

The options (`keepout dashboard --help` lists them all):

- **`--store FILE`** (or `$KEEPOUT_STORE`): the store. A missing store,
  or one made by an older keepout, is an error before Streamlit starts
  (`keepout store upgrade` migrates it, see
  [Upgrading](store.md#upgrading)).
- **`--settings FILE`**: the [operator settings](cli.md#settings) every
  page computes and assesses with, as `keepout store add --settings` does.
  A bad file is an error before Streamlit starts. Give the dashboard the
  settings the store was filled with: a Pc cached with another method,
  covariance policy or hard-body radius shows as not computed (see
  [Jobs](#jobs)).
- **`--host ADDRESS`**: the IP address to listen on, 127.0.0.1 (this
  machine only) by default. Any address that is not a loopback address,
  such as `0.0.0.0` (every interface), prints a warning as Streamlit
  starts: the dashboard has no authentication. A container needs
  `0.0.0.0` to be reached from its host; [Deployment](deployment.md)
  publishes it on the host's 127.0.0.1 only. A value that is not an IP
  address, or a scoped IPv6 address such as `fe80::1%eth0`, is a usage
  error.
- **`--port N`**: the port, 8501 by default.
- **`--as-of TIME`**: see [Data time](#data-time-as-of).

`keepout dashboard` runs headless (no browser opens) until Ctrl+C or
SIGTERM, which it passes on to Streamlit; Streamlit is killed if it has not
stopped 10 s later, or on a second Ctrl+C.

## The demo

`examples/dashboard/` holds 37 CDMs in 12 events for trying the dashboard
without any download: the [synthetic operator](synthetic.md)'s example
fleet, issued as updates every 12 hours from 2026-10-06 00:00 to
2026-10-07 12:00 UTC.
[Its README](https://github.com/Kyrtap1309/keepout/tree/main/examples/dashboard)
says how they are made.

```bash
keepout store add examples/dashboard/cdms/*.kvn \
    --settings examples/dashboard/settings.toml --store demo.db
keepout worker --store demo.db        # in a second terminal
keepout dashboard --store demo.db --settings examples/dashboard/settings.toml \
    --as-of 2026-10-07T12:00
```

Seen at the last issue, 7 events are still ahead; the other 5 are listed
with the past ones. Every covariance is assumed (badge
`assumed (GP age model)`), and every run screened the same element sets,
so each event's Pc trend is flat: only the time to TCA, and so the
decision, changes from one CDM to the next.

`docker compose up` serves the same events with their Pc and a manoeuvre
plan precomputed, and an update CDM to upload that turns one event red
([Deployment](deployment.md#the-demo)).

NASA CARA's Metop-B test CDM can be added too:
`uv run python tools/dashboard_seed.py --cara` copies it from keepout's
cache. Add `examples/dashboard/cdms/*.cdm` to `keepout store add` and set
`hbr_m = 10.0` under `[pc]` in the settings, as the README explains. Its
TCA is in 2020, so it is listed with the past events.

## Data time (`--as-of`)

Every page shows, at its top, the time it sees the data at:
`Data as of 2026-10-07T12:00:00 UTC (fixed)`. By default it is the
clock, read again on every page. `--as-of TIME` (or `KEEPOUT_AS_OF`; empty
means the clock; a bad value is a usage error naming whichever gave it)
fixes it, so that frozen
data such as the demo's look as they did then, and the page says
`(fixed)`. The time decides which events are past, how far each TCA is
(`in 30 h`, `2.5 d ago`) and nothing else: the risk of an event comes from
its CDMs, whatever the time.

## The pages

The dashboard opens on the Events list. The Event page and the planner
show the event named in the address, e.g.
`http://127.0.0.1:8501/event?event=3befe6f1305d`: each row of the Events
list links to both, and they are not in the navigation.

### Events list

The events whose latest TCA is ahead, the soonest first. **Show past
events** adds those whose TCA has gone by. Each row has the latest Pc
with its [provenance](#provenance-badges) and the risk, as
[`keepout assess`](cli.md#assess) decides it, e.g. `yellow · monitor`. An
outlined chip says why there is none: `not computed` (the store lacks a
Pc for the settings), `failed` (the latest Pc could not be computed) or
`issued after TCA`.

The page also takes CDM files: see [Uploading CDMs](#uploading-cdms).

### Event page

One event, its latest Pc, its risk and the reasons for it, then:

- **Pc trend**: each CDM's Pc against the hours to TCA, with the
  thresholds and the decision point;
- **CDMs**: every CDM with its miss distance, Pc and risk;
- **Encounter plane** of the latest CDM: the miss vector, the combined
  covariance's ellipses and the hard-body circle (see [2-D Pc](pc-2d.md));
- **Geometry at TCA**: the secondary's motion in the primary's RTN frame,
  with the 3-sigma ellipsoid and the hard-body sphere.

With a maximum Pc (no covariance) the plots leave the covariance out and
say so. The trend and the risk wait until every CDM of the event has its
Pc; the Pc the store has are shown meanwhile.

### Manoeuvre planner

The plan of the event's latest CDM, as
[`keepout maneuver --event ID`](cli.md#stored-events) shows it: whether a
burn can be recommended (or that none is needed, when the baseline
already meets the target), the smallest burn each way against the baseline,
the heatmap of the Pc over burn time and Δv with its cells as a table,
and what the re-screen of each burn found, with a table of the new or
closer approaches: each with its risk level in words and its assumed Pc.
Without a stored catalog for the event
(CDMs added with `store add`, not `screen --store`) there is no
re-screen, and the page says so.

**What if** plans the event for another target Pc or combined hard-body
radius without changing the settings file. Its form starts from the
settings shown, so changing one value keeps the other. Such a plan is
cached under its own settings, as any other, and the page says which
settings it shows; **Back to the operator's settings** returns to them.

## Provenance badges

Every Pc on the pages has a badge saying what it rests on. Hovering over
the badge says more.

| Badge | The Pc rests on |
|---|---|
| `CDM covariance` | the covariances the CDM gives, from orbit determination |
| `assumed (GP age model)` | covariances keepout's [synthetic operator](synthetic.md) assumed from the age of the GP data ([the model](assumptions.md)); its CDMs say so in their comments |
| `assumed covariance` | covariances the CDM marks as assumed (`COVARIANCE_METHOD = DEFAULT`), from a model it does not name |
| `max-Pc (no covariance)` | no covariance: a [maximum Pc](pc-3d.md), an upper bound |

## Uploading CDMs

**Upload CDMs** on the Events list takes CDM files, KVN or XML, plain or
gzip-compressed: at most 50 files of 1 MB each, and a gzip file may not
decompress to more than 1 MB. All are read before any is stored: a file
over the limits, not UTF-8 text or not a CDM, or a `MESSAGE_ID` stored
with other content stops the whole upload, and the error names the file.
Otherwise every CDM joins its event, as with
[`keepout store add`](cli.md#store), and a refresh job is queued for
each event that lacks a Pc. The page says what it stored and queued, and
follows the jobs.

The worker then computes the Pc. Each upload is kept with its file name:
its CDMs' origin is `upload:<file name>`.

## Jobs

Every computation the dashboard asks for is a
[background job](store.md#background-jobs) in the store, run by
`keepout worker`:

- **Compute**, on an event whose Pc are not all computed for the
  settings, queues a refresh job (`refresh` in `keepout jobs list`): the
  worker computes the Pc its CDMs lack and caches them, as
  `keepout store rebuild` would.
- **Plan manoeuvre**, on the planner, queues a `maneuver` job, as
  `keepout maneuver --event ID --queue` does. **What if** queues one for
  its own settings.
- **Cancel** cancels a queued or running job, as `keepout jobs cancel`
  does. If it cannot (the job has ended meanwhile), a toast says why.

Queueing is idempotent, as from the command line: nothing is queued for a
result already cached, and the same job queued or running is shown
instead of a second one. While a job shown is queued or running, its
status is read again every 2 s; the page shows the result once it has
succeeded, or the error if it failed. `keepout jobs list` lists the same
jobs with their kind.

## Errors

A keepout error, such as a store that cannot be read, shows as its
message on the page, never as a traceback. The Event page and the
planner of an id that is not in the store say `No such event: ID`, with
a link back to the Events list. An unexpected exception (a bug) shows
only its type in the browser; its message and traceback go to the
terminal. Text from the data, such as a `MESSAGE_ID`, an object's
designator or an uploaded file's name, is shown as it is, never read as
Markdown.

## Speed

The targets are an Events list in under 1 s and an Event page in under
2 s on a store of 1 000 events and 20 000 CDMs. On 2026-11-14
(`tools/dashboard_benchmark.py run`, Intel Core i5-14600K) their reads took
0.04 s and 0.05 s. CI fails on a slowdown of more than 20 %.
