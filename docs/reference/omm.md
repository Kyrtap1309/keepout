# Element sets (OMM)

Reading GP element sets in every format they come in: CCSDS OMM (KVN, XML), CelesTrak CSV and JSON, and TLE. See [Element sets](../screening.md#element-sets).

::: keepout.omm
    options:
      members:
        - load
        - loads
        - load_data
        - Omm
        - deduplicate
        - Deduplicated

## The OMM modules

::: keepout.omm.model
    options:
      heading_level: 3
      members:
        - MAX_CATALOG_NUMBER
        - MAX_ELEMENT_SET_NO
        - SGP4_EPOCH_DIGITS
        - InvalidElement

::: keepout.omm.kvn
    options:
      heading_level: 3

::: keepout.omm.xml
    options:
      heading_level: 3

::: keepout.omm.celestrak
    options:
      heading_level: 3

::: keepout.omm.tle
    options:
      heading_level: 3

::: keepout.omm.keywords
    options:
      heading_level: 3
