"""Exception hierarchy for keepout."""

__all__ = [
    "AuthError",
    "CaraFormatError",
    "CcsdsParseError",
    "CdmParseError",
    "CdmWriteError",
    "ChecksumMismatch",
    "CovarianceError",
    "EventError",
    "FleetError",
    "GeometryError",
    "HardBodyRadiusError",
    "HttpStatusError",
    "KeepoutError",
    "KelvinsFormatError",
    "ManeuverError",
    "OmmParseError",
    "OmmWriteError",
    "PcConvergenceError",
    "PropagationError",
    "RateLimited",
    "SourceFormatError",
    "SourceUnavailable",
    "UnsupportedElements",
    "UnsupportedFrame",
]


class KeepoutError(Exception):
    """Base class for all keepout errors."""


class SourceUnavailable(KeepoutError):
    """A remote data source could not be reached or returned an error."""


class HttpStatusError(SourceUnavailable):
    """A server answered a download with an HTTP status other than 200 OK.

    Parameters
    ----------
    message
        What happened.
    url
        The URL asked for.
    status
        The HTTP status code, e.g. 403 or 503.
    """

    def __init__(self, message: str, *, url: str, status: int) -> None:
        super().__init__(message)
        self.url = url
        self.status = status


class AuthError(SourceUnavailable):
    """A data source that needs an account refused the login, or no
    credentials were given (Space-Track)."""


class RateLimited(SourceUnavailable):
    """A request would exceed, or exceeded, a data source's published limits.

    Parameters
    ----------
    message
        What happened.
    retry_after
        Seconds until keepout allows the request again; ``None`` when the
        server refused it, as it does not say when (Space-Track).
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ChecksumMismatch(KeepoutError):
    """Downloaded data does not match its expected SHA-256 digest."""


class SourceFormatError(KeepoutError):
    """A reference data file from an external source could not be parsed."""


class CaraFormatError(SourceFormatError):
    """A NASA CARA data file could not be parsed."""


class KelvinsFormatError(SourceFormatError):
    """ESA's Kelvins collision avoidance data could not be parsed."""


class CcsdsParseError(KeepoutError):
    """A CCSDS message (CDM, OMM) or a related element-set format could not
    be parsed.

    The message names where the problem is, e.g.
    ``bad number (record 3, line 4, keyword BSTAR)``.

    Parameters
    ----------
    message
        What is wrong.
    record
        1-based position of the object in a file of several (a CSV data row,
        a JSON array element, a message of several).
    line
        1-based line number in a text format (KVN, CSV, TLE, JSON).
    keyword
        The keyword (CSV column, JSON key) involved.
    path
        Element path in an XML message, e.g. ``cdm/body/segment[1]/data/stateVector/X``.
    """

    def __init__(
        self,
        message: str,
        *,
        record: int | None = None,
        line: int | None = None,
        keyword: str | None = None,
        path: str | None = None,
    ) -> None:
        context: list[str] = []
        if record is not None:
            context.append(f"record {record}")
        if line is not None:
            context.append(f"line {line}")
        if path is not None:
            context.append(f"at {path}")
        if keyword is not None:
            context.append(f"keyword {keyword}")
        super().__init__(f"{message} ({', '.join(context)})" if context else message)
        self.record = record
        self.line = line
        self.keyword = keyword
        self.path = path


class CdmParseError(CcsdsParseError):
    """A Conjunction Data Message could not be parsed (see :class:`CcsdsParseError`)."""


class OmmParseError(CcsdsParseError):
    """Orbit Mean-Elements Message data (CCSDS OMM, CelesTrak CSV or JSON, TLE)
    could not be parsed (see :class:`CcsdsParseError`)."""


class CdmWriteError(KeepoutError):
    """A Conjunction Data Message cannot be written in the requested format."""


class OmmWriteError(KeepoutError):
    """An Orbit Mean-Elements Message cannot be written in the requested format."""


class UnsupportedElements(KeepoutError, ValueError):
    """An element set SGP4 cannot be initialised with.

    E.g. a catalog number above :data:`~keepout.propagation.MAX_SATREC_NUMBER`
    or an epoch in a leap second. It is also a :class:`ValueError`, as
    raised for any other invalid argument.
    """


class CovarianceError(KeepoutError):
    """A covariance matrix is malformed, asymmetric or not positive semi-definite."""


class PcConvergenceError(KeepoutError):
    """A probability-of-collision method did not converge; no value is returned."""


class FleetError(KeepoutError):
    """A synthetic operator's fleet file is invalid, or a fleet satellite has
    no element set in the catalog."""


class EventError(KeepoutError, ValueError):
    """CDMs do not make one conjunction event: a ``MESSAGE_ID`` repeats, they
    are about different objects, their TCAs spread too far, or they give
    different ``CONJUNCTION_ID`` values.

    It is also a :class:`ValueError`, as raised for any other invalid
    argument of :class:`~keepout.events.ConjunctionEvent`.
    """


class HardBodyRadiusError(KeepoutError, ValueError):
    """No combined hard-body radius: an object has none (no ``HBR`` in its
    CDM) and none was given.

    It is also a :class:`ValueError`, as raised for any other invalid
    argument.
    """


class GeometryError(KeepoutError):
    """A reference frame or encounter geometry cannot be computed."""


class UnsupportedFrame(GeometryError, ValueError):
    """A state vector or covariance is given in a frame keepout does not support.

    It is also a :class:`ValueError`, as raised for any other invalid argument
    of :class:`~keepout.state.StateVector` or :class:`~keepout.covariance.Covariance`.
    """


class PropagationError(KeepoutError):
    """A numerical propagation failed: the integrator gave up, or an orbit
    fell below the Earth's surface."""


class ManeuverError(KeepoutError):
    """An avoidance manoeuvre cannot be planned or checked for this event."""
