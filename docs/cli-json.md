# JSON output of the command line

Every `keepout` command that takes `--json` prints one JSON document on
standard output instead of tables, for scripts and other tools. It is
strict JSON, with no `NaN`. Errors still go to standard error with the
same exit status as without `--json` (see
[Exit status](cli.md#exit-status)). Every time is UTC, in ISO 8601. This
page gives the keys each command prints; [Command line](cli.md) explains
the commands themselves.

## `screen` and `assess`

Both commands describe each event in the same shape:

| Key | Type | Meaning |
|---|---|---|
| `event_id` | string | the event's identifier |
| `tca` | string | TCA of the latest CDM |
| `object1`, `object2` | object | `object_id` and `name` of each object |
| `miss_distance_m` | number | miss distance of the latest CDM [m] |
| `cdms` | array of strings | the `MESSAGE_ID`s of the event's CDMs |
| `pc` | number or `null` | the latest Pc |
| `level` | string or `null` | `green`, `yellow` or `red` |
| `trend` | string | `rising`, `falling`, `steady` or `undetermined` |
| `action` | string | `ignore`, `monitor`, `prepare_maneuver` or `maneuver` |
| `reasons` | array of strings | why the action was chosen |
| `notes` | array of strings | the trend's notes |
| `observations` | array of objects | one per CDM (below) |
| `assumed_covariance` | boolean | true when a CDM of the event has an object with `COVARIANCE_METHOD = DEFAULT` |

Each observation has these keys:

| Key | Type | Meaning |
|---|---|---|
| `message_id` | string | the CDM's `MESSAGE_ID` |
| `time_to_tca_s` | number | time from the CDM's creation to TCA [s] |
| `pc` | number or `null` | keepout's Pc |
| `method` | string or `null` | the method that gave it |
| `error` | string or `null` | why no Pc was computed |
| `reported_pc` | number or `null` | the originator's `COLLISION_PROBABILITY` |
| `diagnostics` | array of strings | the Pc's diagnostics |
| `assumed_covariance` | boolean | whether the CDM has an assumed covariance |

With `--store`, `event_id` is the store's id of the event. For `assess`
and `store add`, `cdms` and `observations` cover every CDM the store holds
of it; for `screen`, they are the approach's own CDM.

At the top level, `screen` gives:

| Key | Type | Meaning |
|---|---|---|
| `fleet` | array of numbers | the fleet's catalog numbers |
| `catalog` | string | where the catalog came from |
| `catalog_objects` | number | the element sets screened |
| `start`, `end` | string | the window |
| `events` | array of objects | the events, each with `relative_speed_m_s` and `cdm_file` added |
| `assumed_covariance_events` | number | the events with an assumed covariance (every event, for `screen`) |
| `skipped` | array of objects | the element sets left out: `catalog_number`, `role`, `reason`, `detail` |
| `cdm_dir` | string or `null` | the `--cdm-dir` |
| `store` | object or `null` | what `--store` did (below); `null` without it |

At the top level, `assess` gives:

| Key | Type | Meaning |
|---|---|---|
| `cdm_count` | number | the CDMs read |
| `events` | array of objects | the events |
| `assumed_covariance_events` | number | the events with an assumed covariance |
| `maneuver` | boolean | true when the command exits with status 3 |
| `store` | object or `null` | what `--store` did (below); `null` without it |

`keepout store add` gives the same keys as `assess`, with `store` always
set. A `store` has these keys:

| Key | Type | Meaning |
|---|---|---|
| `path` | string | the store's file |
| `added` | array of strings | the `MESSAGE_ID`s stored |
| `already_stored` | array of strings | the `MESSAGE_ID`s given that were stored before, with the same content |
| `events` | array of strings | the ids of the events the CDMs given belong to |
| `new_events` | array of strings | the ids of those events that the command created |

## `store upgrade`, `store rebuild` and `store purge`

Each gives `path`, the store's file, and:

| Command | Key | Type | Meaning |
|---|---|---|---|
| `store upgrade` | `from` | string or `null` | the schema revision before; `null` for none |
| `store upgrade` | `to` | string | the revision after |
| `store rebuild` | `computed` | number | the Pc observations computed |
| `store rebuild` | `skipped` | array of strings | the `MESSAGE_ID`s of the CDMs skipped for lack of `HBR` |
| `store purge` | `before` | string | the time given |
| `store purge` | `deleted` | number | the events deleted |

## `maneuver`

| Key | Type | Meaning |
|---|---|---|
| `event` | string | the event's identifier |
| `primary`, `secondary` | string | the objects' `OBJECT_DESIGNATOR`s |
| `target_pc` | number | the target Pc |
| `method` | string | the event's own Pc method |
| `assumed_covariance` | boolean | whether the latest CDM has an assumed covariance |
| `baseline` | object | the event without a burn: `pc`, `range_at_tca_m`, `method`, `meets_target`, `notes` (empty: the baseline's own are in the plan's `notes`) |
| `notes` | array of strings | the plan's notes |
| `options` | array of objects | one per burn (below); empty with no burn on the grid |
| `refusal` | string or `null` | why no burn can be recommended, the text's last line, when the command exits with status 4; `null` otherwise |

Each option has the five keys of `baseline` and these; its `notes` are
what its own Pc says that the plan's do not, such as Monte Carlo standing
in where Hall's method did not converge:

| Key | Type | Meaning |
|---|---|---|
| `direction` | string | `prograde` or `retrograde` |
| `burn_lead_s` | number | time from the burn to TCA [s] |
| `delta_v_m_s` | number | signed Δv [m/s] |
| `rescreen` | object or `null` | the re-screen (below); `null` without a catalog |

A `rescreen` has these keys:

| Key | Type | Meaning |
|---|---|---|
| `conflict` | boolean | a new or closer approach is red |
| `secondary_screened` | boolean | whether the event's secondary was screened |
| `event_range_m` | number or `null` | the event's range after the burn [m] |
| `catalog_objects` | number | the element sets screened |
| `skipped_objects` | number | the element sets left out |
| `window_start`, `window_end` | string | the window; the start is the burn |
| `new_approaches` | array of objects | approaches the burn brings |
| `closer_approaches` | array of objects | approaches the burn brings closer |
| `event_returns` | array of objects | the event's own pair back at or above `target_pc` |

`event_range_m` is `null` when the secondary has no approach within 10
minutes of the CDM's TCA, or was not screened. Read `secondary_screened`
to tell the two apart. An event return is a new or closer approach of the
event's own secondary, more than 10 minutes from its TCA, at or above
`target_pc`. Any event return makes the burn count like a conflict for
status 4.

Each approach has `secondary`, `tca`, `range_at_tca_m`,
`relative_speed_m_s`, `pc`, `level` and `assumed_covariance` (always true).

## `maneuver --queue`, `jobs list` and `jobs cancel`

`maneuver --queue` gives `event`, `job` (its id, `null` when the plan is
cached), `status` (`null` likewise) and `cached`. `jobs cancel` gives
`job`, `event` and `status`. `jobs list` gives `path` and `jobs`, the
newest first, each with these keys:

| Key | Type | Meaning |
|---|---|---|
| `job` | number | the job's id |
| `event` | string | the event's id |
| `cdm` | string | the `MESSAGE_ID` of the CDM it plans from |
| `status` | string | `queued`, `running`, `succeeded`, `failed` or `cancelled` |
| `attempt` | number | how many times a worker took it (a stop does not count) |
| `worker` | string or `null` | the worker running it, or the one that ended it. `null` while it is queued, if it was cancelled while queued, or if it failed after two workers stopped without an answer |
| `created`, `started`, `finished` | string or `null` | its times (`created` is never `null`) |
| `error` | string or `null` | why it failed |

`keepout worker` has no `--json`: it prints how many jobs it ran.
