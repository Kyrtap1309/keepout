"""Exception hierarchy for keepout."""


class KeepoutError(Exception):
    """Base class for all keepout errors."""


class SourceUnavailable(KeepoutError):
    """A remote data source could not be reached or returned an error."""


class ChecksumMismatch(KeepoutError):
    """Downloaded data does not match its expected SHA-256 digest."""


class CaraFormatError(KeepoutError):
    """A NASA CARA data file could not be parsed."""
