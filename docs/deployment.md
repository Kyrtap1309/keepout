# Deployment with Docker

`docker compose up` runs the [dashboard](dashboard.md) and a
[worker](store.md#background-jobs) on a demo store, from a clone of the
repository, with Docker and Docker Compose v2 as the only requirements:

```bash
git clone https://github.com/Kyrtap1309/keepout && cd keepout
docker compose up --wait     # builds the image the first time (a few minutes)
```

Then open http://127.0.0.1:8501. `docker compose down` stops it and keeps
the data; `docker compose down -v` also deletes them, and the next `up`
starts from the demo again.

!!! warning "For this machine only"
    The dashboard has no authentication: anyone who reaches its port can
    read every event, store CDMs and queue or cancel jobs. `compose.yaml`
    publishes the port on the host's 127.0.0.1 only. Do not change that to
    `0.0.0.0` or put the port behind a public proxy. From another machine,
    use an SSH tunnel: `ssh -L 8501:127.0.0.1:8501 HOST`, then open
    http://127.0.0.1:8501.

## What runs

`compose.yaml` runs three services from one image, `keepout:local`, built
from `Dockerfile`:

- **`init`** gives the `data` volume the demo store if it has no store
  (`/data/keepout.db`), then exits. A volume that has a store, from a
  previous `up` or your own, is left as it is, once init has checked that
  this keepout can open it ([Updating](#updating)).
- **`dashboard`** is `keepout dashboard` on that store. Inside the container
  it listens on every address (`--host 0.0.0.0`, which prints a warning in
  its log); the host publishes it on 127.0.0.1:8501 only. Its healthcheck
  asks Streamlit's `/_stcore/health`, and `up --wait` waits for it.
- **`worker`** is `keepout worker` on the same store: it computes the Pc of
  uploaded CDMs and the manoeuvre plans queued from the dashboard.
  `docker compose stop` sends it SIGTERM; it puts its running job back in
  the queue and exits. Its healthcheck asks for a heartbeat file under a
  minute old: compose sets `KEEPOUT_HEARTBEAT_FILE=/tmp/keepout-worker.alive`
  (`keepout worker --heartbeat-file`), which the worker touches each time it
  looks at the queue (every 2 s when idle) or renews its job's lease (every
  20 s while a job computes). A worker run without it touches no file.

Both run under an init process (`init: true`), which reaps the child
processes Streamlit and the jobs run in. Both run as the user `keepout`
(uid 10001), never root, and restart unless stopped.

`KEEPOUT_PORT` picks another host port when 8501 is taken:
`KEEPOUT_PORT=8601 docker compose up --wait`, then http://127.0.0.1:8601.

## The demo

The image holds a store built from `examples/dashboard/` when it is built
(`tools/dashboard_seed.py --store`): the 12 events and 37 CDMs of the
[demo](dashboard.md#the-demo), their Pc, and the manoeuvre plan of event
`3befe6f1305d`, so the planner shows its heatmap at once.

`compose.yaml` fixes the time the data are seen at to 2026-10-07 18:00 UTC
(`KEEPOUT_AS_OF`, the dashboard's `--as-of`): 6 events are ahead. To see the event's Pc rise, upload
`examples/dashboard/demo/*.kvn` on the Events page (**Upload CDMs**). It is
a fifth CDM of event `3befe6f1305d`, issued at 18:00 with a smaller miss
and covariance. The worker computes its Pc within seconds, and the event
turns red with a rising trend:

![The demo: the events list, the planner, the upload, and the event turning red](img/dashboard/dashboard.gif)

The planner names the catalog its re-screen used as
`stored copy of tests/screening/data/celestrak-public.csv.gz`: the store
keeps a copy of the catalog, labelled by its path in the repository.

## The Iridium-Cosmos case study

A second profile serves the [case study](case-study-iridium-cosmos.md) as an
operator would have seen it two days before the 2009 collision:

```bash
docker compose build
docker compose --profile iridium up --wait iridium-dashboard iridium-worker
```

then <http://127.0.0.1:8502> (`KEEPOUT_IRIDIUM_PORT` picks another port).
`docker compose --profile iridium up` alone starts the demo on 8501 too.

- **Its store**, built into the image by
  `tools/dashboard_seed.py --iridium-store`, holds the synthetic CDMs of
  Iridium 33 and Cosmos 2251 created by the decision point, and nothing
  later. The dashboard's time is fixed there, so the event is two days
  ahead.
- **Its manoeuvre plan** (to green, `examples/dashboard/iridium-settings.toml`)
  is precomputed, re-screened against the 2009 catalog as known then.
- **Its own volume**, `iridium-data`, and **its own worker**, which computes
  the planner's what-if plans against that same catalog.

## Your own data

- **Your own time.** `KEEPOUT_AS_OF= docker compose up --wait` (an empty
  value) makes the dashboard read the clock; put `KEEPOUT_AS_OF=` in a
  `.env` file next to `compose.yaml` to keep it.
- **CDM files.** Upload them on the Events page, or add them from the
  command line:

    ```bash
    docker compose cp ./cdms worker:/tmp/cdms
    docker compose exec worker sh -c \
        'keepout store add /tmp/cdms/*.kvn --store /data/keepout.db --settings /opt/keepout/settings.toml'
    ```

- **Your settings.** The dashboard reads `/opt/keepout/settings.toml`, a
  copy of `examples/dashboard/settings.toml`, and queues each job with
  them: the worker reads no settings file. Mount your own over it in a
  `compose.override.yaml` next to `compose.yaml`, which Compose reads too,
  in the worker as well if you run `keepout store add` there:

    ```yaml
    services:
      dashboard:
        volumes:
          - ./settings.toml:/opt/keepout/settings.toml:ro
      worker:
        volumes:
          - ./settings.toml:/opt/keepout/settings.toml:ro
    ```

    A Pc cached with other settings shows as not computed; **Compute**
    queues it for the worker.

- **Without the demo.** Every demo event is before 2026-10-14; delete them
  with `docker compose exec worker keepout store purge --store
  /data/keepout.db --before 2026-10-14 --yes`.

## Backups

Stop the services, so that SQLite's write-ahead log is merged into the
file, then copy the whole volume:

```bash
docker compose stop
docker compose cp worker:/data/. ./keepout-backup/
docker compose start
```

`keepout-backup/` then holds `keepout.db` and its catalog copies
(`keepout.db.catalogs/`), which the re-screens need.

## Updating

After `git pull`, rebuild and restart: `docker compose up --build --wait`.
The volume keeps its store. If the new keepout changed the store's schema,
`init` fails, `up` says so, and its log (`docker compose logs init`) gives
the command. Stop the services, so that nothing holds the store, migrate it
in a container of its own, then start again
([Upgrading](store.md#upgrading)):

```bash
docker compose stop dashboard worker
docker compose run --rm --no-deps worker keepout store upgrade --store /data/keepout.db
docker compose up --wait
```

The base images (`python:3.13-slim` and `uv`) are pinned by digest in
`Dockerfile`, so a build gives the same image until they are changed. To
take their security updates, look up the current digests
(`docker buildx imagetools inspect python:3.13-slim`), replace them in
`Dockerfile` and rebuild.
