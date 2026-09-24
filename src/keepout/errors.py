"""Exception hierarchy for keepout."""


class KeepoutError(Exception):
    """Base class for all keepout errors."""


class SourceUnavailable(KeepoutError):
    """A remote data source could not be reached or returned an error."""


class ChecksumMismatch(KeepoutError):
    """Downloaded data does not match its expected SHA-256 digest."""


class CaraFormatError(KeepoutError):
    """A NASA CARA data file could not be parsed."""


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


class CovarianceError(KeepoutError):
    """A covariance matrix is malformed, asymmetric or not positive semi-definite."""


class GeometryError(KeepoutError):
    """A reference frame or encounter geometry cannot be computed."""
