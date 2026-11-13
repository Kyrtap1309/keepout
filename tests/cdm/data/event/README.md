# Sample event

Three CDMs of one made-up encounter, as a screening service would send
them over four days before TCA. They are keepout's own, built with
`keepout.cdm.make_cdm`; none was issued by a screening provider. Each says
so in its first `COMMENT`.

Every CDM has the objects, TCA and states of `../tracss_v2.kvn` (written in
GCRF) and an `HBR` of 10 m per object. Only the creation date and the
covariances change: each CDM's covariances are those of `tracss_v2.kvn`
multiplied by a factor that shrinks as TCA nears, so the Pc rises.

| Created (UTC) | Covariance factor |
|---|---|
| 2025-05-29 08:00 | 1000 |
| 2025-05-30 20:00 | 25 |
| 2025-06-01 12:00 | 1 |

`tests/cdm/test_sample_event.py` builds them and checks that these files
match; run it as a script to write them again:

```bash
uv run python tests/cdm/test_sample_event.py
```

The first example of `docs/risk.md` reads them.
