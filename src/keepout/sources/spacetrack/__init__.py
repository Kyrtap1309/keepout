"""Space-Track.org: the full GP catalog, within the site's published limits.

Space-Track.org (https://www.space-track.org) is where the 18th Space
Defense Squadron publishes its general perturbations (GP) element sets, for
every tracked object in orbit: about 32 000, against the 20 000 of
CelesTrak's public groups (:mod:`keepout.sources.celestrak`). It needs a
free account. :func:`fetch_catalog` downloads the catalog with the query
Space-Track recommends for it (:data:`CATALOG`: element sets of objects not
decayed, with an epoch in the last 10 days, as OMM JSON) and gives the same
:class:`~keepout.sources.catalog.Catalog` as CelesTrak's client.
:func:`fetch_public_cdms` downloads the public summaries of the conjunction
data messages of approaches still to come (``cdm_public``, :data:`PUBLIC_CDMS`)
as :class:`PublicCdm` rows (:mod:`~keepout.sources.spacetrack.public_cdm`).
:class:`SpaceTrackClient` is the session underneath, for other queries.

**Credentials** come only from the environment: ``SPACETRACK_IDENTITY``
(the account's user name) and ``SPACETRACK_PASSWORD``
(:meth:`Credentials.from_env`), read only when a request must be sent.
Each user needs an account of their own (Space-Track user agreement).
Neither value appears in a ``repr``, an error message or a cached file:
server text quoted in an error has both replaced by ``<hidden>``.

**Limits.** Space-Track suspends accounts that query too often
(https://www.space-track.org/documentation#/api, "API Use Guidelines"), so
the client enforces the published limits itself:

- Fewer than 30 requests a minute and 300 an hour: :class:`Throttle` waits
  before a request that would make more than :data:`PER_MINUTE` (20) in a
  minute or :data:`PER_HOUR` (200) in an hour, a third below the limits to
  leave room for other use of the same account (the website, a browser).
  Logins and logouts count. The count is the client's own; the class
  limits below keep separate clients and processes far below it.
- Each request class has its own limit (:data:`CLASS_INTERVALS`): ``gp``
  once an hour, ``cdm_public`` (all public CDM summaries) once every 8
  hours. Every answer is cached on disk, and the time of each class's last
  query is recorded there too (``<cache>/spacetrack/requests.json``),
  before the query is sent, so a failed query counts. A copy younger than
  ``max_age`` is returned without contacting Space-Track; a query of a
  class asked for less than its interval ago raises
  :class:`~keepout.errors.RateLimited` without contacting it either. The
  record is per cache directory: processes sharing one share the limit.
  It is read again after a login, so a query another process recorded
  meanwhile is kept and still refuses its class; two processes recording
  at the same instant may both send.
- An answer saying the rate limit was violated (HTTP 500 "violated your
  query rate limit", or 429) raises :class:`~keepout.errors.RateLimited`
  and stops the client: it sends nothing more, not even the logout.
  Nothing is retried, except one fresh login after HTTP 401 (the session,
  two hours, expired) followed by the same query once more.
- The session is closed (``/ajaxauth/logout``) when the client is closed.

Space-Track asks that the hourly ``gp`` query is sent at a random minute,
not at the top or bottom of the hour; a scheduled job should pick one.

Answers are asked for gzipped (``Accept-Encoding: gzip``) and cached
gzipped under ``<cache>/spacetrack/<class>/`` (the catalog in
``gp/catalog.json.gz``: about 5 MB instead of 36 MB of JSON), each with a
``.meta.json`` file as CelesTrak's
(:func:`~keepout.sources.cache.read_copy`).

GP element sets are basic space situational awareness data of USSPACECOM
(18th Space Defense Squadron), which Space-Track.org's user agreement
allows to redistribute with citation; cite them as such. Other classes
(``cdm_public`` among them) may not be redistributed without approval.
"""

from keepout.sources.spacetrack._client import (
    PER_HOUR,
    PER_MINUTE,
    SpaceTrackClient,
    Throttle,
    fetch_catalog,
    fetch_public_cdms,
    read_catalog,
)
from keepout.sources.spacetrack._credentials import (
    HIDDEN,
    IDENTITY_VARIABLE,
    PASSWORD_VARIABLE,
    Credentials,
)
from keepout.sources.spacetrack._query import (
    BASE_URL,
    CATALOG,
    CLASS_INTERVALS,
    LOGIN_URL,
    LOGOUT_URL,
    PUBLIC_CDMS,
    QUERY_URL,
    Query,
    RequestClass,
)
from keepout.sources.spacetrack._transport import (
    Request,
    Response,
    Transport,
    UrllibTransport,
)
from keepout.sources.spacetrack.public_cdm import (
    PUBLIC_CDM_COLUMNS,
    PublicCdm,
    PublicCdmObject,
    PublicCdms,
    read_public_cdms,
)

__all__ = [
    "BASE_URL",
    "CATALOG",
    "CLASS_INTERVALS",
    "HIDDEN",
    "IDENTITY_VARIABLE",
    "LOGIN_URL",
    "LOGOUT_URL",
    "PASSWORD_VARIABLE",
    "PER_HOUR",
    "PER_MINUTE",
    "PUBLIC_CDMS",
    "PUBLIC_CDM_COLUMNS",
    "QUERY_URL",
    "Credentials",
    "PublicCdm",
    "PublicCdmObject",
    "PublicCdms",
    "Query",
    "Request",
    "RequestClass",
    "Response",
    "SpaceTrackClient",
    "Throttle",
    "Transport",
    "UrllibTransport",
    "fetch_catalog",
    "fetch_public_cdms",
    "read_catalog",
    "read_public_cdms",
]
