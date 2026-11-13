# Synthetic operator

The synthetic operator: a fleet file, a covariance model of GP errors and a run that issues and assesses CDMs. See [Synthetic operator](../synthetic.md).

::: keepout.synthetic

## The synthetic operator's modules

::: keepout.synthetic.covariance_model
    options:
      heading_level: 3
      members:
        - FLOHRER_SIGMAS
        - GEO_LOWER_ALTITUDE
        - GEO_UPPER_ALTITUDE
        - LEO_UPPER_ALTITUDE
        - LEVIT_MARSHALL_RATE
        - PUBLISHED_PARAMETERS
        - SAFETY_FACTOR

::: keepout.synthetic.fleet
    options:
      heading_level: 3
      members:
        - FLEET_KEYS
        - SATELLITE_KEYS

::: keepout.synthetic.operator
    options:
      heading_level: 3
      members:
        - DEFAULT_MODEL
        - HEADER_COMMENTS
