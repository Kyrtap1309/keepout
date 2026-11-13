# Screening

Finding the close approaches of primaries against a catalog. See [Screening](../screening.md).

::: keepout.screening

## The screening modules

::: keepout.screening.pipeline
    options:
      heading_level: 3
      members: false

::: keepout.screening.result
    options:
      heading_level: 3
      members: false

::: keepout.screening.volume
    options:
      heading_level: 3
      members:
        - NASA_GP_DATA_RADIUS
        - SFS_DEEP_SPACE_INCLINATION
        - SFS_DEEP_SPACE_PERIOD
        - SFS_ECCENTRICITY
        - SFS_LEO_PERIGEES
        - SOCRATES_RADIUS
        - per_primary

::: keepout.screening.filters
    options:
      heading_level: 3
      members: false

::: keepout.screening.tca
    options:
      heading_level: 3
      members:
        - DEFAULT_STEP
        - DERIVATIVE_STEP
        - GRAVITY_MARGIN
        - MIN_RADIUS
        - PERTURBATION
        - SEARCH_REACH
        - SLOW_STEP
        - SPEED_BOUND_MARGIN
        - TCA_TOLERANCE
        - speed_bounds

::: keepout.screening.search_result
    options:
      heading_level: 3
      members: false
