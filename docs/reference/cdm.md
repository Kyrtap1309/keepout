# CDM

Reading, writing and building Conjunction Data Messages. See [Reading a CDM](../index.md#reading-a-cdm) and [Frames and units](../frames-and-units.md).

::: keepout.cdm
    options:
      members:
        - load
        - loads
        - read_text
        - format_of
        - CdmFormat
        - Cdm
        - CdmObject
        - Section
        - Field
        - ObjectMetadata
        - ObjectType
        - CovarianceMethod
        - Maneuverable
        - make_cdm
        - make_message_id
        - ORIGINATOR

## The CDM modules

::: keepout.cdm.model
    options:
      heading_level: 3
      members:
        - KEYWORD_PATTERN
        - USER_DEFINED_PREFIX
        - DuplicateKeyword
        - covariance_dimension
        - format_field
        - parse_field

::: keepout.cdm.kvn
    options:
      heading_level: 3
      members:
        - dump
        - dumps
        - load
        - loads

::: keepout.cdm.xml
    options:
      heading_level: 3
      members:
        - NDM_NAMESPACE
        - dump
        - dumps
        - load
        - loads

::: keepout.cdm.builder
    options:
      heading_level: 3
      members:
        - MESSAGE_ID_PRECISION
        - SCREEN_VOLUME_COMMENT
        - TCA_PRECISION
        - RtnSemiAxes

::: keepout.cdm.keywords
    options:
      heading_level: 3
      members:
        - ADDITIONAL_PARAMETERS
        - COVARIANCE_AXES
        - HEADER
        - KEYWORDS
        - METADATA_GROUP
        - PHYSICAL_PARAMETERS
        - POSITION_KEYWORDS
        - RELATIVE
        - REQUIRED
        - SEGMENT
        - STATE_KEYWORDS
        - VELOCITY_KEYWORDS
        - VERSION_1_0
        - VERSION_2_0
        - VERSION_KEYWORD
        - SectionKind
        - covariance_entries
        - covariance_matrix
        - covariance_names
