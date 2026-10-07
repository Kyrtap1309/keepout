# Test data

Data: USSPACECOM/18 SDS via Space-Track.org, retrieved from CelesTrak
(celestrak.org) on 2026-10-06.

`celestrak_sample.csv` holds the header and eight rows, unchanged, of
CelesTrak's `active` group in CSV
(`https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv`) as
downloaded on 2026-10-06: CALSPHERE 1 (900), ISS (ZARYA) (25544), GOES 18
(51850, geostationary), THEMIS E (30798, eccentricity 0.84), EXPRESS-MD2
(38745, non-zero `MEAN_MOTION_DDOT`), SARAMAGO (100000), STARLINK-38128
(100001, negative `BSTAR`) and CREW DRAGON 13 (100882, docked to the ISS:
same elements under its own name and catalog number). The last three have
six-digit catalog numbers, which have no TLE form.

`celestrak_sample.json` holds the same eight element sets in the layout of
CelesTrak's JSON format (one object per element set, numbers as JSON
numbers). It was converted from the CSV rows, not downloaded: each
number is the CSV value written as a JSON number.

This data is basic space situational awareness data that Space-Track's user
agreement allows to redistribute with citation. It is not covered by
keepout's MIT licence.

`ccsds_omm_g7.kvn`, `ccsds_omm_g8.kvn`, `ccsds_omm_g9.kvn` and
`ccsds_omm_g10.xml` are figures G-7 to G-10 of CCSDS 502.0-B-3, *Orbit Data
Messages* (Blue Book, April 2023, https://public.ccsds.org/Pubs/502x0b3e1.pdf):
the standard's OMM examples for GOES 9 in KVN (without and with a covariance,
and with units and a user-defined parameter) and in XML. They are copied
unchanged except for trailing blanks and, in the XML, the schema location,
which the document breaks over two lines. Figure G-6, the TLE they are based
on, is in `tests/omm/omm_builders.py`.
