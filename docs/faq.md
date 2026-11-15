# FAQ and troubleshooting

## Getting started

**Where do I get a CDM to try keepout on?**
:   The repository has two hand-written examples: `tests/cdm/data/tracss_v2.kvn`
    (CDM 2.0, an `HBR` for one object only) and `tests/cdm/data/b1_example.kvn`
    (CDM 1.0, no `HBR`). Both need a combined hard-body radius, as in
    [Probability of collision](index.md#probability-of-collision).

    For a week of CDMs on real satellites, run the
    [synthetic operator](cli.md#a-week-of-a-synthetic-operator). Its CDMs
    carry assumed covariances.

**Can keepout tell me whether my satellite is at risk?**
:   Only from CDMs with covariances from orbit determination, such as those
    18 SDS or TraCSS send to the satellite's operator. From public data
    alone keepout finds close approaches, not risk
    ([Limits of GP data](screening.md#limits-of-gp-data)).

**Is it "maneuver" or "manoeuvre"?**
:   Commands, options, the command line's output and API names use the US
    "maneuver", as the CCSDS keywords do. The prose of these pages uses the
    British "manoeuvre".

## Errors

**`CDM ... gives no HBR for OBJECT1 ...`**
:   The CDM gives no hard-body radius (`HBR`) for one or both objects, and
    keepout never assumes one. Pass `combined_hard_body_radius=` (in metres)
    in Python, `--hbr METRES` on the command line, or `hbr_m` in the
    [settings file](cli.md#settings).

**`CovarianceError`**
:   A covariance is not symmetric or not positive semi-definite. The
    default policy, `"repair"`, fixes small defects and records them in
    `Covariance.repairs`. This error means either `policy="raise"` was
    chosen or the projected covariance has no positive variance at all
    ([Covariance validation and repair](frames-and-units.md#covariance-validation-and-repair)).

**`BrokenProcessPool` or a script that runs twice with `workers=`**
:   With `workers` above 1, keepout starts processes that import the main
    module again. Put the call under `if __name__ == "__main__":`
    ([Workers](screening.md#workers)). A script read from standard input
    (`python -`) cannot use workers.

**Screening finds nothing, and `skipped` lists everything as `stale`**
:   The window ends more than 30 days after the element sets' epoch. With
    a downloaded catalog, start the window from now (`Time.now()`, or omit
    `--start`). With a frozen catalog, start on its retrieval day
    ([Element age](screening.md#element-age)).

**`SourceUnavailable` from CelesTrak**
:   CelesTrak refuses repeated downloads of unchanged data (HTTP 403) and
    blocks clients that ignore its policy. keepout returns its cached copy
    when it has one; without one, wait and try again later
    ([CelesTrak's public catalog](screening.md#celestraks-public-catalog)).

**`AuthError` or `RateLimited` from Space-Track**
:   `AuthError`: `SPACETRACK_IDENTITY` or `SPACETRACK_PASSWORD` is missing
    or wrong ([Account and credentials](spacetrack.md#account-and-credentials)).
    `RateLimited`: the query would break Space-Track's limits; its
    `retry_after` says how long to wait ([Limits](spacetrack.md#limits)).

**`... was made by an older keepout: run keepout store upgrade`**
:   The store's schema is older than this keepout's. Back it up, then run
    `keepout store upgrade --store FILE` ([Upgrading](store.md#upgrading)).

**`--store needs the store extra`**, **`keepout validate needs the validation extra`**
:   Install the extra the message names, `keepout[cli,store]` or
    `keepout[cli,validation]`; the message gives the `pip` command. In a
    clone: `uv sync --extra cli --extra store` (or `--extra validation`).
    See [Install](index.md#install).

## Performance

**`compute_pc` takes a minute on some conjunctions**
:   In `auto` mode a slow encounter uses Hall's 3-D method (seconds to a
    minute) and may fall back to Monte Carlo. For a batch, choose a method.
    Foster takes milliseconds: `compute_pc(conjunction, "foster")` for one
    conjunction, or for every CDM of an event
    `event.observations(compute=computation("foster"))` with
    `from keepout.pc import computation`; `keepout assess --method foster`
    on the command line ([Choosing a method](pc-3d.md#choosing-a-method)).

**Manoeuvre planning uses every CPU core and is still slow**
:   The numerical libraries start a thread per core by default, which slows
    the planner down. The `keepout` command limits them to one thread; in
    Python, set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` before
    starting it ([Time](maneuver.md#time)).

## Using keepout's results

**How do I cite keepout?**
:   See [Citing keepout](index.md#citing-keepout). Cite the data you used
    as well: GP data as *USSPACECOM/18 SDS via Space-Track.org, retrieved
    YYYY-MM-DD*, and Kelvins and CARA data as given in
    [Data and licences](index.md#data-and-licences).
