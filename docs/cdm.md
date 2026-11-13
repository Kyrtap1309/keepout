# Working with CDMs

A Conjunction Data Message (CDM) warns that two objects will pass close.
It gives the time of closest approach (TCA), each object's state and
covariance at TCA and, often, the originator's probability of collision
(Pc). `keepout.cdm` reads CDMs, gives typed access to their fields,
converts them between KVN and XML, and builds CDMs of keepout's own.

The examples read sample CDMs committed to the repository; run them from
a clone (see [Install](index.md#install)) or use your own files in their
place:

- `tests/cdm/data/tracss_v2.kvn`: a CDM 2.0 in KVN, modelled on TraCSS's.
- `tests/cdm/data/b1_example.kvn` and `b1_example.xml`: one CDM 1.0 in
  KVN and in XML, modelled on the example of CCSDS 508.0-B-1.
- `tests/cdm/data/event/`: three CDMs of one made-up encounter.

## Loading

`keepout.cdm.load` reads a file and `keepout.cdm.loads` a string. Both
tell KVN from XML by content (XML starts with `<`) and read CDM 1.0 and
2.0. `load` also reads gzip-compressed files, recognised by content
whatever the file name. `keepout.cdm.kvn` and `keepout.cdm.xml` have their
own `load` and `loads` that read one format only.

```python
import keepout.cdm

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")  # KVN or XML, gzipped or not
print(cdm.version, cdm.originator, cdm.message_id)
print(cdm.creation_date.isot, cdm.tca.isot)
print(cdm.miss_distance, cdm.collision_probability)  # [m], the originator's Pc
print(cdm.conjunction_id)  # CDM 2.0: shared by every CDM of one event

xml = keepout.cdm.load("tests/cdm/data/b1_example.xml")  # CDM 1.0 in XML
print(xml.version, xml.message_id, xml.conjunction_id)
```

```text
2.0 TRACSS TRACSS-TEST-0001
2025-06-01T12:00:00.000 2025-06-02T08:15:30.123456
412.5 1.2e-05
dd8c054b-6bea-48fb-a245-6cb23331b156
1.0 201113719185 None
```

A `Cdm` keeps every field as read, in order, with its comments. Times are
astropy `Time` objects in UTC.

## Reading fields

`cdm.object1` and `cdm.object2` are typed views of the two object
sections. Values are in SI units whatever the file's units: a KVN state in
km is read in metres. Keywords keepout does not type (those beyond CCSDS
508.0-B-1, apart from `HBR`) are kept as text.

```python
import keepout.cdm

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")
for item in (cdm.object1, cdm.object2):
    print(item.object_id, item.name, item.ref_frame, item.hard_body_radius)
    print(item.position, item.velocity)  # in ref_frame [m], [m/s]
    print(item.covariance_rtn.diagonal())  # [m², m²/s²], as written
print(cdm.object1.user_defined)  # USER_DEFINED_* keywords, as text
print(cdm.header.get("CLASSIFICATION"))  # any keyword, by name
print(cdm.relative.get("COLLISION_PROBABILITY_METHOD"))
print(cdm.object1.section.extras)  # keywords beyond CCSDS 508.0-B-1, as text
```

```text
12345 SATELLITE A ITRF 5.0
[6778137.       0.       0.] [   0. 5900. 4400.]
[1.e+02 1.e+04 1.e+02 1.e-02 1.e-02 1.e-02]
30337 FENGYUN 1C DEB ITRF None
[6.778137e+06 3.000000e+02 1.000000e+02] [    0. -5900.  4400.]
[4.e+02 4.e+04 4.e+02 4.e-02 4.e-02 4.e-02]
{'USER_DEFINED_RUN_ID': '8e815679-b659-48df-957e-9c5b4921af52', 'USER_DEFINED_MEETS_ALERTABLE_CRITERIA': 'NO'}
UNCLASSIFIED
FOSTER-1992
{'DENSITY_FORECAST_UNCERTAINTY': '2.046460670000000E-01', 'USER_DEFINED_RUN_ID': '8e815679-b659-48df-957e-9c5b4921af52', 'USER_DEFINED_MEETS_ALERTABLE_CRITERIA': 'NO'}
```

`covariance_rtn` is the covariance as written, 6x6 to 9x9 and not
checked. `cdm.header`, `cdm.relative` and `object1.section` are the raw
sections; `get` returns a field's value by keyword and `comments` the
section's comments.

## States and covariances

`cdm.object_states()` gives both objects in GCRF, SI units, ready for the
Pc methods. It converts the states from the CDM's `REF_FRAME` (GCRF,
EME2000, ITRF or TEME) and checks each covariance, repairing a matrix that
is not quite a valid covariance and recording what it changed.
`cdm.conjunction()` adds the combined hard-body radius.

```python
import keepout.cdm

cdm = keepout.cdm.load("tests/cdm/data/tracss_v2.kvn")
primary, secondary = cdm.object_states()  # both in GCRF, SI units
print(primary.object_id, primary.state.frame, primary.state.r.round(1))
print(primary.covariance.frame, primary.covariance.repairs)
print(cdm.object2.has_covariance, cdm.assumed_covariance)
conjunction = cdm.conjunction(combined_hard_body_radius=20.0)  # [m]
print(f"{conjunction.plane.miss_distance:.1f} m")
```

```text
12345 GCRF [6559034.  1709371.   -16274.8]
GCRF ()
True False
100.0 m
```

- `has_covariance` is false when the CDM gives an all-zero covariance, as
  operators do when they have none. The object's covariance is then
  `None`, and only the maximum-Pc methods accept it.
- `assumed_covariance` is true when an object's `COVARIANCE_METHOD` is
  `DEFAULT`: the covariance is assumed, not from orbit determination.
- `policy="raise"` raises `CovarianceError` instead of repairing.

[Frames, units and covariances](frames-and-units.md) explains the frames,
the RTN frame and the covariance checks and repairs.
[2-D probability of collision](pc-2d.md) computes the Pc of the
conjunction.

## Converting and writing

`keepout.cdm.kvn` and `keepout.cdm.xml` write a CDM: `dumps` returns the
text and `dump` writes a file. Loading one format and writing the other
converts it.

```python
import keepout.cdm

cdm = keepout.cdm.load("tests/cdm/data/b1_example.kvn")
text = keepout.cdm.xml.dumps(cdm)  # KVN to XML
print(text.splitlines()[1])
again = keepout.cdm.loads(text)
print(again.message_id, again.tca.isot, again.object1.covariance_rtn.shape)
keepout.cdm.kvn.dump(again, "b1_again.kvn")  # and back to KVN, in a file
print(keepout.cdm.load("b1_again.kvn") == cdm)
```

```text
<cdm id="CCSDS_CDM_VERS" version="1.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd">
201113719185 2010-03-13T22:37:52.618 (6, 6)
True
```

- KVN is written with the CCSDS units (km for the state).
- XML is not namespace-qualified. Version 1.0 messages follow the CCSDS
  508.0-B-1 schema; version 2.0 messages put the 2.0 keywords in
  `physicalParameters` and `userDefinedParameters`. Comments move to the
  start of their element, where the schema allows them.
- Writing raises `CdmWriteError` for a value KVN or XML cannot hold: a
  line break in a KVN value, a character XML 1.0 does not allow.

## Building a CDM

`keepout.cdm.make_cdm` writes a CDM of a `Conjunction` you built from your
own states, for instance from your orbit determination. `ObjectMetadata`
gives what the CDM says about each object besides its state and
covariance: name, international designator, object type, manoeuvrability
and covariance method.

```python
import numpy as np
from astropy.time import Time

import keepout.cdm
from keepout import Conjunction, ObjectState, StateVector, make_covariance
from keepout.cdm import Maneuverable, ObjectMetadata, ObjectType, make_cdm

tca = Time("2025-06-02T08:15:30.123456", scale="utc")


def object_state(object_id, r, v, sigmas_rtn, radius):
    """One object at TCA: GCRF state [m, m/s], RTN 1-sigma [m, m/s], HBR [m]."""
    state = StateVector(tca, np.array(r), np.array(v), "GCRF")
    rtn = make_covariance(np.diag(np.square(sigmas_rtn)), "RTN", tca)
    return ObjectState(object_id, state, rtn.in_gcrf(state), radius)


primary = object_state(
    "12345",
    [6778137.0, 0.0, 0.0],
    [0.0, 5900.0, 4400.0],
    [10.0, 100.0, 10.0, 0.1, 0.1, 0.1],
    5.0,
)
secondary = object_state(
    "30337",
    [6778137.0, 300.0, 100.0],
    [0.0, -5900.0, 4400.0],
    [20.0, 200.0, 20.0, 0.2, 0.2, 0.2],
    15.0,
)
cdm = make_cdm(
    Conjunction(primary, secondary, combined_hard_body_radius=20.0),
    ObjectMetadata(
        "SATELLITE A",
        "1997-030E",
        object_type=ObjectType.PAYLOAD,
        maneuverable=Maneuverable.YES,
    ),
    ObjectMetadata("FENGYUN 1C DEB", "1999-025AA", object_type=ObjectType.DEBRIS),
    created=Time("2025-06-01T12:00:00", scale="utc"),
    originator="MY-OPS",
)
print(cdm.version, cdm.message_id)  # 2.0: both objects have an HBR
print(f"{cdm.miss_distance:.1f} m", cdm.object1.hard_body_radius)
keepout.cdm.kvn.dump(cdm, f"{cdm.message_id}.kvn")
```

```text
2.0 12345_30337_20250602T081530.123Z_20250601T120000.000Z
316.2 m 5.0
```

- The states are written in GCRF and each covariance in its object's RTN
  frame; an unknown covariance (`None`) is written as zeros.
- The message is version 2.0 when both objects have a hard-body radius
  (`HBR` is a CDM 2.0 keyword), else 1.0. The two radii must add up to the
  conjunction's combined radius: a radius for one object only is a
  `ValueError`.
- `created` is the `CREATION_DATE`; keepout reads no clock, so the same
  inputs give the same message. The default `MESSAGE_ID` is built from
  both ids, the TCA and `created`, and is also a portable file name.
- `screening_period=` and `screening_volume=` add the screening keywords,
  `message_for=` and `comments=` the header's.

The [synthetic operator](synthetic.md#the-cdms) issues its CDMs this way.

## Many CDMs and events

A screening service sends a new CDM of an encounter every few hours. The
CDMs of one encounter make an *event*. `group_events` sorts CDMs about any
objects into events; `ConjunctionEvent.from_cdms` makes one event of CDMs
you know belong together, and refuses CDMs that do not.

```python
from pathlib import Path

import keepout.cdm
from keepout.events import group_events

paths = [
    *Path("tests/cdm/data/event").glob("*.kvn"),  # three CDMs of one encounter
    Path("tests/cdm/data/b1_example.xml"),  # the same objects, TCA in 2010
]
cdms = [keepout.cdm.load(path) for path in paths]
for event in group_events(cdms):  # by first TCA
    print(event.event_id, event.primary_id, event.secondary_id, event.latest.tca.isot)
    for cdm in event.cdms:  # by CREATION_DATE
        print("   ", cdm.creation_date.isot, cdm.message_id)
```

```text
23c7fe5cf923 12345 30337 2010-03-13T22:37:52.618
    2010-03-12T22:31:12.000 201113719185
4f4ae3491c9e 12345 30337 2025-06-02T08:15:30.123456
    2025-05-29T08:00:00.000 12345_30337_20250602T081530.123Z_20250529T080000.000Z
    2025-05-30T20:00:00.000 12345_30337_20250602T081530.123Z_20250530T200000.000Z
    2025-06-01T12:00:00.000 12345_30337_20250602T081530.123Z_20250601T120000.000Z
```

An event's id is derived from the `MESSAGE_ID` of its first CDM, so it
stays the same as later CDMs arrive. The grouping rule (same objects, TCAs
within 10 minutes, one `CONJUNCTION_ID`) is in
[Risk assessment](risk.md#events-and-observations), which assesses an
event's Pc over time. On the command line, `keepout assess` groups the
CDM files it is given in the same way.

## Errors

Every error is a `keepout.errors.KeepoutError`; the ones below are also
importable from `keepout`.

- `CdmParseError`: the file is not a valid CDM, not UTF-8 text or broken
  gzip data. Its message names the line and keyword (KVN) or the element
  path (XML); they are also in its `line`, `keyword` and `path`.
- `CdmWriteError`: the CDM cannot be written in the requested format.
- `EventError`: CDMs do not make one event (other objects, a repeated
  `MESSAGE_ID`, TCAs too far apart, different `CONJUNCTION_ID` values).
- `HardBodyRadiusError`: no combined hard-body radius was given and an
  object has no `HBR`.
- `UnsupportedFrame`: a `REF_FRAME` other than GCRF, EME2000, ITRF or
  TEME, when converting the states.

```python
from pathlib import Path

import keepout.cdm
from keepout import CdmParseError

text = Path("tests/cdm/data/tracss_v2.kvn").read_text()
broken = text.replace("MISS_DISTANCE                  = 412.5", "MISS_DISTANCE = far")
try:
    keepout.cdm.loads(broken)
except CdmParseError as error:
    print(error)
    print(error.line, error.keyword)

try:
    keepout.cdm.loads(text.replace("OBJECT_DESIGNATOR              = 30337\n", ""))
except CdmParseError as error:
    print(error)

xml = Path("tests/cdm/data/b1_example.xml").read_text()
try:
    keepout.cdm.loads(xml.replace(">715<", ">far<"))
except CdmParseError as error:
    print(error)
    print(error.path)
```

```text
invalid number 'far' (line 10, keyword MISS_DISTANCE)
10 MISS_DISTANCE
OBJECT2 section is missing required keyword(s): OBJECT_DESIGNATOR
invalid number 'far' (at cdm/body/relativeMetadataData/MISS_DISTANCE, keyword MISS_DISTANCE)
cdm/body/relativeMetadataData/MISS_DISTANCE
```

KVN line numbers count from the file's first line, XML ones from its
first non-blank line. A CDM 1.0 has no `HBR`, so its conjunction needs
the radius:

```python
import keepout.cdm
from keepout import HardBodyRadiusError

cdm = keepout.cdm.load("tests/cdm/data/b1_example.kvn")  # CDM 1.0: no HBR
try:
    cdm.conjunction()
except HardBodyRadiusError as error:
    print(error)
```

```text
no hard-body radius for the primary (12345) and the secondary (30337): pass combined_hard_body_radius= (in metres) or give both objects one
```

## Supported CDM versions

- **CCSDS 508.0-B-1** (June 2013), including Technical Corrigendum 2, which
  allows namespace-qualified `ndm:` XML. Every keyword is typed.
- **CDM 2.0 as issued by TraCSS**, based on the 508.0-P-1.1 draft. It is
  read and written. Keywords beyond 508.0-B-1 are kept as text, except
  `HBR`, which is typed.
- JSON and CSV variants are not supported.
- Space-Track's public CDM summaries (`cdm_public`) are not CDMs. They carry
  no states or covariances (see [Space-Track data](spacetrack.md#public-cdms)).
- `keepout.cdm.make_cdm` writes CDMs of keepout's own. It writes version 2.0
  when both objects have a hard-body radius, else 1.0.

The standards are CCSDS 508.0-B-1, *Conjunction Data Message*, Blue Book,
2013 (Cor. 2, 2021), and the Office of Space Commerce's *TraCSS CDM
Specification 001*, v2.1.
