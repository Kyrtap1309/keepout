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
