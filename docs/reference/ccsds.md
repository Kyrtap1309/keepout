# CCSDS formats

The machinery shared by the CDM and OMM readers and writers: keywords, KVN, XML and epochs. Most users need only [CDM](cdm.md) and [Element sets](omm.md).

::: keepout.ccsds
    options:
      members: false

## Epochs and files

::: keepout.ccsds.epochs
    options:
      heading_level: 3
      members:
        - parse_epoch
        - format_epoch

::: keepout.ccsds.files
    options:
      heading_level: 3
      members:
        - read_file

## Internals

The rest of the CCSDS machinery: message families, keyword values and the KVN and XML readers and writers. The CDM and OMM modules use them.

::: keepout.ccsds.message
    options:
      heading_level: 3

::: keepout.ccsds.keywords
    options:
      heading_level: 3
      members: false

::: keepout.ccsds.kvn
    options:
      heading_level: 3
      members: false

::: keepout.ccsds.xml
    options:
      heading_level: 3
      members: false
