# Propagation

The propagators: SGP4 for GP element sets, Cowell integration for manoeuvre planning, and SGP4 plus a burn for the re-screen.

::: keepout.propagation

## The propagation modules

::: keepout.propagation.base
    options:
      heading_level: 3
      members:
        - object_indices

::: keepout.propagation.sgp4_prop
    options:
      heading_level: 3
      members:
        - Sgp4Sampler

::: keepout.propagation.forces
    options:
      heading_level: 3
      members: false

::: keepout.propagation.cowell
    options:
      heading_level: 3
      members:
        - SPAN_SLACK

::: keepout.propagation.maneuvered
    options:
      heading_level: 3
      members: false
