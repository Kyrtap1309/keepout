# Test data

`kelvins_excerpt.csv` holds every row of six events (`event_id` 30, 78, 97,
505, 907 and 5490) of the raw table `raw_data/raw_data_2015-2019.gz` of ESA's
Collision Avoidance Challenge data set, unchanged, with its header:

> ESA, *Collision Avoidance Challenge dataset*, Zenodo, 2021,
> doi:[10.5281/zenodo.4463683](https://doi.org/10.5281/zenodo.4463683).
> © ESA, licensed under
> [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

The events cover ESA's -30 floor of `risk`, a CDM issued after TCA, empty
velocity sigmas and solar indices, and rising, falling and red Pc.
`tests/sources/test_kelvins_data.py` (marked `validation`) checks that the
rows match the published table.
