# The dashboard

The dashboard shows a [conjunction store](store.md) in a web browser: the
events to come with their risk, each event's Pc trend and geometry, and
its manoeuvre plan. It also takes new CDM files and queues the work a
[worker](store.md#background-jobs) does. It is a Streamlit app run by
`keepout dashboard`, on the machine that holds the store.

```bash
pip install "keepout[cli,dashboard] @ git+https://github.com/Kyrtap1309/keepout"
```

The `dashboard` extra installs [Streamlit](https://streamlit.io/) and
[Plotly](https://plotly.com/python/), with the `store` extra it needs.
`import keepout` loads neither, and neither does `keepout --help`. Without
the extra, `keepout dashboard` prints the line to install it and exits
with status 1. In a clone, `uv sync --extra cli --extra dashboard`.

!!! warning "For this machine only"
    The dashboard listens on 127.0.0.1, this machine's own address, and
    has no authentication: whoever reaches its port can read every event,
    store CDMs, queue jobs and cancel them. On a shared machine that is
    every user of the machine. Do not expose it. Do not
    forward its port, put it behind a public proxy or run Streamlit
    yourself on another address. To use it from another machine, reach
    it through an SSH tunnel to the machine it runs on, e.g.
    `ssh -L 8501:127.0.0.1:8501 HOST`, then open http://127.0.0.1:8501.

## Running it

The dashboard reads a store that exists already. It never computes a Pc
or a manoeuvre plan in its own process: it queues them as jobs, and a
[worker](cli.md#worker) on the same store computes them. So run the two
side by side:

```bash
keepout store add cdms/*.kvn --store keepout.db   # a store, e.g. from the CLI walk-through
keepout worker --store keepout.db &               # computes what the dashboard queues
keepout dashboard --store keepout.db              # http://127.0.0.1:8501, Ctrl+C stops it
```

Without a worker, the dashboard still shows everything the store holds.
What it queues waits in the store until a worker runs.

```bash
keepout dashboard --help
```

```text
 Usage: keepout dashboard [OPTIONS]

 Run the dashboard on a store (needs the dashboard extra).

 Serves the dashboard on http://127.0.0.1:PORT, this machine only, until
 stopped with Ctrl+C: the events, each event's risk and the manoeuvre
 planner. It reads the store and queues jobs for keepout worker; it
 computes no Pc or plan itself. The store must exist and be at this
 keepout's schema (keepout store upgrade). Exits with Streamlit's status.

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ *  --store           FILE             The conjunction store, a SQLite file.  │
│                                       [env var: KEEPOUT_STORE]               │
│                                       [required]                             │
│    --settings        FILE             Operator settings, TOML: [pc] method,  │
│                                       policy, hbr_m; [risk] green_below,     │
│                                       red_from; [maneuver] target_pc         │
│                                       (default: the library's). Where the    │
│                                       command has --method, --hbr or         │
│                                       --target-pc, that option wins over the │
│                                       file.                                  │
│    --port            N [1<=x<=65535]  The port to serve on. [default: 8501]  │
│    --as-of           TIME             See the data at this time, ISO 8601    │
│                                       UTC: the events whose latest TCA is    │
│                                       before it are past (default: the       │
│                                       clock, read on each page).             │
│    --help                             Show this message and exit.            │
╰──────────────────────────────────────────────────────────────────────────────╯

 Examples:
 keepout dashboard --store keepout.db
 keepout dashboard --store keepout.db --as-of 2026-10-07T12:00 --port 8502
```

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
- **`--port N`**: the port, 8501 by default. The address is always
  127.0.0.1.
- **`--as-of TIME`**: see [The time the data are seen at](#the-time-the-data-are-seen-at).

`keepout dashboard` runs Streamlit headless (it opens no browser) until
Ctrl+C, and exits with Streamlit's status. It hands the store, the
settings and the time to the app as the environment variables
`KEEPOUT_STORE`, `KEEPOUT_SETTINGS` and `KEEPOUT_AS_OF`; only its options
set them, not variables left in the shell. In the browser, an unexpected
exception shows only its type (`--client.showErrorDetails type`); its
message and traceback stay on the terminal (see [Errors](#errors)).

## The demo

`examples/dashboard/` holds a store's worth of CDMs for trying the
dashboard without any download: the [synthetic operator](synthetic.md)'s
example fleet screened against the frozen public catalog of 2026-10-06,
as an operator receives updates. It ran every 12 hours over the first two
days of the week, from 2026-10-06 00:00 to 2026-10-07 12:00 UTC, and each
run issued a CDM for every approach still ahead: 37 CDMs in 12 events, up
to 4 per event. Its `settings.toml` is keepout's defaults.

```bash
keepout store add examples/dashboard/cdms/*.kvn \
    --settings examples/dashboard/settings.toml --store demo.db
keepout worker --store demo.db &
keepout dashboard --store demo.db --settings examples/dashboard/settings.toml \
    --as-of 2026-10-07T12:00
```

Seen at the last issue, 2026-10-07 12:00 UTC, 7 events are still ahead;
the other 5 are listed with the past ones. Every covariance is assumed
(keepout's model of GP data errors, badge `assumed (GP age model)`). Every
run screened the same frozen element sets, so an event's CDMs give the same
Pc: its trend is flat, and only the time to TCA, and so the decision,
changes from one CDM to the next.

`tools/dashboard_seed.py` writes these files, and CI checks them with
`--check`. NASA CARA's Metop-B test CDM is not redistributed:
`uv run python tools/dashboard_seed.py --cara` adds it to the folder from
keepout's local cache (`keepout fetch cara` downloads it), and git ignores
it. The command above takes only the `.kvn` files: add
`examples/dashboard/cdms/*.cdm` to `keepout store add` to store it too. Its
TCA is in 2020, so it is listed with the past events. It has no `HBR`
keyword (CARA gives 10 m), so set `hbr_m = 10.0` under `[pc]` in the
settings (a copy of `examples/dashboard/settings.toml`). Without it,
`store add` skips its Pc, the event shows `not computed`, and the
`refresh` job **Compute** queues for it fails for want of a hard-body
radius.

## The time the data are seen at

Every page shows, at its top, the time it sees the data at:
`Data as of 2026-10-07T12:00:00 UTC (fixed)`. By default it is the
clock, read again on every page. `--as-of TIME` fixes it, so that frozen
data such as the demo's look as they did then, and the page says
`(fixed)`. The time decides which events are past, how far each TCA is
(`in 30 h`, `2.5 d ago`) and nothing else: the risk of an event comes from
its CDMs, whatever the time.

## The pages

The sidebar has three pages. The event and the planner show the event of
the address's `event` parameter, e.g.
`http://127.0.0.1:8501/event?event=3befe6f1305d`; without one, they ask
to pick an event from the list. Reach them through the links: each row of
the list links to its event's page and to its planner, and the event page
links to its planner. The planner chosen in the sidebar has no event.

### Events

The stored events whose latest TCA is ahead, the soonest first, 25 to a
page with Previous and Next. **Show past events** adds those whose TCA
has gone by. Each row has the event's id, a link to its page, the two
objects, the TCA with how far it is, the latest Pc with its
[provenance](#provenance-badges), the risk, the number of CDMs, the
kind of job running for it, if any, and a link to its manoeuvre planner.

The risk is a chip in the level's colour that names the level and the
action, e.g. `yellow · monitor`, from the event's Pc with the settings'
thresholds, as [`keepout assess`](cli.md#assess) decides it. An outlined
chip says why there is none: `not computed` (a Pc the store lacks for the
settings), `failed` (the latest Pc could not be computed) or `issued after
TCA`.

The page also takes CDM files: see [Uploading CDMs](#uploading-cdms).

### Event

One event: its objects and TCA, the latest Pc with its provenance, the
risk with the action and the reasons for it, a link to its manoeuvre
planner, then

- **Pc trend**: the Pc of each CDM against the hours to TCA, coloured by
  level, with the thresholds and the decision point, and its data table;
- **CDMs**: every CDM, oldest first, with its creation time, its time to
  TCA, its miss distance, its Pc and provenance, and its risk;
- **Encounter plane** of the latest CDM: the miss vector, the combined
  covariance's 1, 2 and 3-sigma ellipses and the hard-body circle, with a
  table of the miss, the radius and the ellipse's axes;
- **Geometry at TCA**: the secondary's motion relative to the primary in
  the primary's RTN frame around TCA (a straight line, as the 2-D Pc
  assumes), with the combined covariance's 3-sigma ellipsoid around the
  secondary and the hard-body sphere around the primary, and a table of
  the position and velocity at TCA.

With a maximum Pc (no covariance) the plots leave the covariance out and
say so. The trend and the risk wait until every CDM of the event has its
Pc; the Pc the store has are shown meanwhile.

### Manoeuvre planner

The plan of the event's latest CDM, as
[`keepout maneuver --event ID`](cli.md#stored-events) shows it: whether a
burn can be recommended, the smallest burn each way against the baseline,
the heatmap of the Pc over burn time and Δv with its cells as a table,
and what the re-screen of each burn found, each new or closer approach
with its risk and its assumed Pc. Without a stored catalog for the event
(CDMs added with `store add`, not `screen --store`) there is no
re-screen, and the page says so.

**What if** plans the event for another target Pc or combined hard-body
radius without changing the settings file. Such a plan is cached under its
own settings, as any other, and the page says which settings it shows;
**Back to the operator's settings** returns to them.

## Provenance badges

Every Pc on the pages has a badge saying what it rests on. Hovering over
the badge says more.

| Badge | The Pc rests on |
|---|---|
| `CDM covariance` | the covariances the CDM gives, from orbit determination |
| `assumed (GP age model)` | covariances keepout's [synthetic operator](synthetic.md) assumed from the age of the GP data ([the model](assumptions.md)); its CDMs say so in their comments |
| `assumed covariance` | covariances the CDM marks as assumed (`COVARIANCE_METHOD = DEFAULT`), from a model it does not name |
| `max-Pc (no covariance)` | no covariance: a [maximum Pc](pc-3d.md), an upper bound |

Pc from ESA's Kelvins data do not appear: the store keeps CDMs, and
Kelvins rows are not CDMs (see the [Kelvins report](risk-kelvins.md)).

## Uploading CDMs

**Upload CDMs** on the events page takes CDM files, KVN or XML, plain or
gzip-compressed: at most 50 files of 1 MB each, and a gzip file may not
decompress to more than 1 MB. All are read before any is stored: a file
over the limits, not UTF-8 text or not a CDM, or a `MESSAGE_ID` stored
with other content stops the whole upload, and the error names the file.
Otherwise every CDM joins its event, as with
[`keepout store add`](cli.md#store), and a `refresh` job is queued for
each event that lacks a Pc. The page says what it stored and queued, and
follows the jobs.

The worker then computes the Pc. Each upload is kept with its file name:
its CDMs' origin is `upload:<file name>`.

## Jobs

Every computation the dashboard asks for is a
[background job](store.md#background-jobs) in the store, run by
`keepout worker`:

- **Compute**, on an event whose Pc are not all computed for the
  settings, queues a `refresh` job: the worker computes the Pc its CDMs
  lack and caches them, as `keepout store rebuild` would.
- **Plan manoeuvre**, on the planner, queues a `maneuver` job, as
  `keepout maneuver --event ID --queue` does. **What if** queues one for
  its own settings.
- **Cancel** cancels a queued or running job, as `keepout jobs cancel`
  does.

Queueing is idempotent, as from the command line: nothing is queued for a
result already cached, and the same job queued or running is shown
instead of a second one. While a job shown is queued or running, its
status is read again every 2 s; the page shows the result once it has
succeeded, or the error if it failed. `keepout jobs list` lists the same
jobs with their kind.

## Errors

A keepout error, such as a store that cannot be read, shows as its
message on the page, never as a traceback. The event page and the planner
of an id that is not in the store say `No such event: ID`, with a link
back to the list. An unexpected exception (a bug) shows only its type in
the browser: `keepout dashboard` runs Streamlit with
`--client.showErrorDetails type`, so its message and traceback go to the
terminal only. Text from the data, such as a `MESSAGE_ID`, an object's
designator or an uploaded file's name, is shown as it is, never read as
Markdown.

## Speed

The targets are an events list in under 1 s and an event page in under
2 s on a store of 1 000 events and 20 000 CDMs, a year of a 20-satellite
fleet with margin. The list reads one page of events in a fixed number of
queries and parses only each event's latest CDM; the event page reads the
event in one transaction. The manoeuvre grid is a job, so no page waits for
it.

`tools/dashboard_benchmark.py run` builds such a store (about 90 s) and
times the reads each page makes, without Streamlit's rendering. Measured
on 2026-11-14 (median of 5 warm reads per run, over 5 runs, Intel Core
i5-14600K): the events list's first page in 0.039 to 0.041 s and the event
page's data in 0.048 to 0.053 s. The first read in a fresh process takes
0.32 s for the list and 0.19 s for the event page. CI times both against
the merge base, as for screening, and fails on a slowdown of more than
20 %.
