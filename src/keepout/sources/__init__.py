"""External data sources.

Each module downloads or reads one source, within its usage policy, and
keeps its downloads in the cache (:mod:`keepout.sources.cache`):

- :mod:`keepout.sources.catalog`: the GP catalog, whichever source it came
  from;
- :mod:`keepout.sources.celestrak` and :mod:`keepout.sources.socrates`:
  CelesTrak's public catalog and its SOCRATES reports;
- :mod:`keepout.sources.spacetrack`: Space-Track.org's full catalog and
  public CDM summaries;
- :mod:`keepout.sources.cara` and :mod:`keepout.sources.cara_legacy`: NASA
  CARA's validation files;
- :mod:`keepout.sources.kelvins`: ESA's Kelvins collision avoidance data.
"""

__all__: list[str] = []
