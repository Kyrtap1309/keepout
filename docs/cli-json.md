# JSON output of the command line

Every `keepout` command but `worker` takes `--json`. It then prints one
JSON document on standard output instead of tables, for scripts and other
tools. It is
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

For instance, `keepout assess --json tests/cdm/data/tracss_v2.kvn --hbr 20`,
from a clone of the repository, prints this (exit status 3):

```json
{
  "cdm_count": 1,
  "events": [
    {
      "event_id": "5ede3a14c345",
      "tca": "2025-06-02T08:15:30.123456",
      "object1": {
        "object_id": "12345",
        "name": "SATELLITE A"
      },
      "object2": {
        "object_id": "30337",
        "name": "FENGYUN 1C DEB"
      },
      "miss_distance_m": 412.5,
      "cdms": [
        "TRACSS-TEST-0001"
      ],
      "pc": 0.045044354503398956,
      "level": "red",
      "trend": "undetermined",
      "action": "maneuver",
      "reasons": [
        "latest Pc 4.50e-02 is red, 0.84 d before TCA",
        "trend undetermined",
        "within the decision point (2.00 d before TCA)",
        "diagnostic: states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach"
      ],
      "notes": [
        "fewer than 2 observations with a Pc before TCA: direction undetermined"
      ],
      "observations": [
        {
          "message_id": "TRACSS-TEST-0001",
          "time_to_tca_s": 72930.123456,
          "pc": 0.045044354503398956,
          "method": "hall3d",
          "error": null,
          "reported_pc": 1.2e-05,
          "diagnostics": [
            "states are 300 m from closest approach along the relative velocity; 2-D Pc uses the straight-line closest approach"
          ],
          "assumed_covariance": false
        }
      ],
      "assumed_covariance": false
    }
  ],
  "assumed_covariance_events": 0,
  "maneuver": true,
  "store": null
}
```

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

## `validate`

| Key | Type | Meaning |
|---|---|---|
| `cara_commit` | string | the commit of CARA's files compared with |
| `monte_carlo` | boolean | whether `--monte-carlo` was given |
| `comparisons` | number | the comparisons made: one per case and method |
| `failed` | number | the comparisons that failed (status 1 if any) |
| `summary` | array of objects | per suite and method: `suite`, `method`, `passed`, `total` and `worst_relative_error` (`null` when no case is judged by a relative bound) |
| `outcomes` | array of objects | one per comparison (below) |

Each outcome has these keys:

| Key | Type | Meaning |
|---|---|---|
| `suite`, `case_id` | string | the case |
| `method` | string | keepout's method |
| `reference` | number | CARA's Pc |
| `value` | number | keepout's Pc |
| `error` | number | the error: relative if `relative`, else absolute |
| `relative` | boolean | whether the case is judged by a relative bound |
| `passed` | boolean | whether the value meets the case's criterion |
| `ci` | array of 2 numbers or `null` | a Monte Carlo Pc's confidence interval |

## `fetch`

`fetch cara` gives `source` (`"cara"`), `commit`, the pinned commit of
CARA's repository, `files`, the number of files, and `directory`, where
they are in the cache.

`fetch kelvins` gives `source` (`"kelvins"`), `doi`, the data set's
Zenodo DOI, and `path`, the cached raw table.

`fetch catalog` gives:

| Key | Type | Meaning |
|---|---|---|
| `source` | string | `celestrak` or `spacetrack` |
| `objects` | number | the objects, one element set each |
| `duplicates` | number | the older element sets of the same objects left out |
| `parts` | array of objects | one per download (a CelesTrak group, Space-Track's `gp` query): `name`, `objects` and the retrieval keys below |

`fetch cdm` gives `source` (`"spacetrack"`), `retrieval`, with the
retrieval keys below, and `cdms`, the summaries in TCA order, each with
these keys:

| Key | Type | Meaning |
|---|---|---|
| `cdm_id` | number | the CDM's number at Space-Track |
| `created` | string | when the CDM was made |
| `tca` | string | TCA |
| `emergency_reportable` | boolean | whether the approach meets 18 SDS's emergency reporting criteria |
| `range_at_tca_m` | number | range at TCA [m] |
| `pc` | number or `null` | the Pc Space-Track reports |
| `object1`, `object2` | object | `catalog_number` (number), and `name`, `object_type`, `rcs_size` and `exclusion_volume` (strings or `null`) |

A retrieval has `url`, where the copy came from, `path`, the cached copy,
`retrieved`, when the source sent it, and `checked`, when the source last
confirmed it current.

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
