"""Exception hierarchy for keepout."""

__all__ = [
    "CaraFormatError",
    "CdmParseError",
    "CdmWriteError",
    "ChecksumMismatch",
    "CovarianceError",
    "GeometryError",
    "KeepoutError",
    "KelvinsFormatError",
    "OmmParseError",
    "PcConvergenceError",
    "SourceFormatError",
    "SourceUnavailable",
    "UnsupportedFrame",
]


class KeepoutError(Exception):
    """Base class for all keepout errors."""


class SourceUnavailable(KeepoutError):
    """A remote data source could not be reached or returned an error."""


class ChecksumMismatch(KeepoutError):
    """Downloaded data does not match its expected SHA-256 digest."""


class SourceFormatError(KeepoutError):
    """A reference data file from an external source could not be parsed."""


class CaraFormatError(SourceFormatError):
    """A NASA CARA data file could not be parsed."""


class KelvinsFormatError(SourceFormatError):
    """ESA's Kelvins collision avoidance data could not be parsed."""


class CdmParseError(KeepoutError):
    """A Conjunction Data Message could not be parsed.

    Parameters
    ----------
    message
        What is wrong.
    line
        1-based line number in a KVN message.
    keyword
        The CDM keyword involved.
    path
        Element path in an XML message, e.g. ``cdm/body/segment[1]/data/stateVector/X``.
    """

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        keyword: str | None = None,
        path: str | None = None,
    ) -> None:
        context: list[str] = []
        if line is not None:
            context.append(f"line {line}")
        if path is not None:
            context.append(f"at {path}")
        if keyword is not None:
            context.append(f"keyword {keyword}")
        super().__init__(f"{message} ({', '.join(context)})" if context else message)
        self.line = line
        self.keyword = keyword
        self.path = path


class OmmParseError(KeepoutError):
    """An Orbit Mean-Elements Message could not be parsed.

    Parameters
    ----------
    message
        What is wrong.
    record
        1-based position of the object in a file of several (a CSV data row,
        a JSON array element, a segment of an XML message).
    line
        1-based line number in a text format (KVN, CSV, TLE, JSON).
    keyword
        The OMM keyword (CSV column, JSON key) involved.
    path
        Element path in an XML message.
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


class CdmWriteError(KeepoutError):
    """A Conjunction Data Message cannot be written in the requested format."""


class CovarianceError(KeepoutError):
    """A covariance matrix is malformed, asymmetric or not positive semi-definite."""


class PcConvergenceError(KeepoutError):
    """A probability-of-collision method did not converge; no value is returned."""


class GeometryError(KeepoutError):
    """A reference frame or encounter geometry cannot be computed."""


class UnsupportedFrame(GeometryError, ValueError):
    """A state vector or covariance is given in a frame keepout does not support.

    It is also a :class:`ValueError`, as raised for any other invalid argument
    of :class:`~keepout.state.StateVector` or :class:`~keepout.covariance.Covariance`.
    """
