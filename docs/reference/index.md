# API reference

These pages are generated from keepout's docstrings. Each starts with a
line on what the module is for and a link to the guide that explains it.

- **Start here:** `compute_pc`, `screen`, `assess`, `ConjunctionEvent` and
  `Conjunction` are importable from `keepout` itself. The pages under
  *Core types* and *Pc and risk* cover them.
- **Units and frames:** every quantity is in SI units (metres, seconds,
  m/s) and every state in GCRF, unless a name or a docstring says
  otherwise. See [Frames and units](../frames-and-units.md).
- **Terms:** Pc, TCA, HBR, CDM and the other abbreviations are explained
  in the [Glossary](../concepts.md#glossary).
- **Errors:** keepout raises subclasses of
  [`KeepoutError`](errors.md#keepout.errors.KeepoutError).

## Where to find what

| Area | Package or module | Page |
|---|---|---|
| Conjunctions, objects, states | `keepout.conjunction`, `keepout.objects`, `keepout.state`, `keepout.orbits` | [Conjunction](conjunction.md), [Objects and state vectors](objects.md) |
| Covariances, frames, units | `keepout.covariance`, `keepout.frames`, `keepout.units` | [Covariance](covariance.md), [Frames](frames.md), [Units](units.md) |
| Encounter plane, orbit elements | `keepout.encounter`, `keepout.equinoctial` | [Encounter plane](encounter.md), [Equinoctial elements](equinoctial.md) |
| Propagation | `keepout.propagation` | [Propagation](propagation.md) |
| Probability of collision | `keepout.pc` | [Probability of collision](pc.md) |
| Events and risk | `keepout.events`, `keepout.risk` | [Conjunction events](events.md), [Risk](risk.md) |
| Manoeuvres and settings | `keepout.maneuver`, `keepout.settings` | [Manoeuvres](maneuver.md), [Operator settings](settings.md) |
| Screening | `keepout.screening` | [Screening](screening.md) |
| Synthetic operator | `keepout.synthetic` | [Synthetic operator](synthetic.md) |
| Message formats | `keepout.cdm`, `keepout.omm`, `keepout.ccsds` | [CDM](cdm.md), [Element sets (OMM)](omm.md), [CCSDS formats](ccsds.md) |
| Data sources | `keepout.sources` | [GP catalog](catalog.md), [CelesTrak data](celestrak.md), [Space-Track data](spacetrack.md), [CARA reference data](cara.md), [Kelvins data](kelvins.md), [Data cache](cache.md) |
| Conjunction store | `keepout.store` | [Conjunction store](store.md) |
| Validation | `keepout.validation` | [Validation tooling](validation.md) |
| Errors | `keepout.errors` | [Errors](errors.md) |

`keepout.cli` is the `keepout` command. It is documented in
[Command line](../cli.md), not here.

## What `keepout` exports

`import keepout` gives these names; the rest is in the packages above.

- Pc: [`compute_pc`][keepout.pc.compute_pc],
  [`PcResult`][keepout.pc.PcResult], [`Method`][keepout.pc.Method] and
  [`MethodChoice`][keepout.pc.MethodChoice].
- Conjunctions and their parts:
  [`Conjunction`][keepout.conjunction.Conjunction],
  [`ObjectState`][keepout.objects.ObjectState],
  [`StateVector`][keepout.state.StateVector],
  [`RefFrame`][keepout.state.RefFrame],
  [`Covariance`][keepout.covariance.Covariance] and
  [`make_covariance`][keepout.covariance.make_covariance].
- Events and risk: [`ConjunctionEvent`][keepout.events.ConjunctionEvent],
  [`PcObservation`][keepout.risk.PcObservation] and
  [`assess`][keepout.risk.assess].
- Screening: [`screen`][keepout.screening.screen],
  [`ScreeningResult`][keepout.screening.ScreeningResult],
  [`ScreeningVolume`][keepout.screening.ScreeningVolume] and
  [`CloseApproach`][keepout.screening.CloseApproach].
- Errors: [`KeepoutError`][keepout.errors.KeepoutError] and every
  exception of [Errors](errors.md).
- [`__version__`][keepout.__version__], keepout's version.

## The `keepout` package

::: keepout
    options:
      members:
        - __version__
      show_root_heading: false

## The command line package

::: keepout.cli
    options:
      members: false
      heading_level: 3
