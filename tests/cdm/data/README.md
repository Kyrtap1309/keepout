# Test data

The CDMs here are keepout's own, written by hand; none was downloaded or
issued by a screening provider. Each says so in its first `COMMENT`.

`b1_example.kvn` and `b1_example.xml` are one message in KVN and in XML,
modelled on the CDM example in the annex of CCSDS 508.0-B-1, *Conjunction
Data Message* (Blue Book, June 2013): a version 1.0 CDM between two objects
in EME2000, with a full 6x6 covariance per object. The XML is the KVN
message in the layout of the NDM/XML 2.0.0 CDM schema.

`tracss_v2.kvn` is a version 2.0 CDM modelled on the TraCSS CDM
specification v2.1: `CLASSIFICATION` and `CONJUNCTION_ID` in the header,
states in ITRF, a hard-body radius (`HBR`) for one object only, and
TraCSS's extra and `USER_DEFINED_` keywords. Its objects, states and
covariances are made up.

# Schemas

`tests/cdm/xsd/ndmxml-2.0.0-cdm-1.0.xsd` and
`tests/cdm/xsd/ndmxml-2.0.0-common-2.0.xsd` are the CDM 1.0 and common
modules of the NDM/XML 2.0.0 schema (version 2.0, Rev 0, 07/18/2020),
copied unchanged from the SANA registry
(https://sanaregistry.org/r/ndmxml_unqualified/). `tests/cdm/xsd/cdm.xsd`
is keepout's: an entry point declaring the `<cdm>` root element as
`ndmxml-2.0.0-master-2.0.xsd` does, importing only those two modules.
CCSDS publishes no schema for CDM 2.0, so only version 1.0 messages are
validated.
