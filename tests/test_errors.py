import inspect

import pytest

from keepout import errors
from keepout.errors import (
    AuthError,
    CcsdsParseError,
    CdmParseError,
    EventError,
    GeometryError,
    HardBodyRadiusError,
    HttpStatusError,
    KeepoutError,
    OmmParseError,
    RateLimited,
    SourceUnavailable,
    UnsupportedElements,
    UnsupportedFrame,
)

ERRORS = [
    value
    for _, value in inspect.getmembers(errors, inspect.isclass)
    if issubclass(value, BaseException)
    and value is not KeepoutError
    and value.__module__ == errors.__name__
]
"""Every exception class defined in ``keepout.errors`` but the base."""


def test_the_error_module_exports_every_error_it_defines() -> None:
    assert {error.__name__ for error in ERRORS} == set(errors.__all__) - {
        "KeepoutError"
    }


@pytest.mark.parametrize("error", ERRORS, ids=lambda error: error.__name__)
def test_every_error_is_a_keepout_error(error: type[BaseException]) -> None:
    assert issubclass(error, KeepoutError)


def test_cdm_parse_error_reports_line_and_keyword() -> None:
    error = CdmParseError("bad number", line=12, keyword="X")
    assert str(error) == "bad number (line 12, keyword X)"
    assert error.line == 12
    assert error.keyword == "X"
    assert error.path is None


def test_cdm_parse_error_reports_xml_path() -> None:
    error = CdmParseError("bad unit", path="cdm/body/segment[1]/data/stateVector/X")
    assert str(error) == "bad unit (at cdm/body/segment[1]/data/stateVector/X)"


@pytest.mark.parametrize("error", [CdmParseError, OmmParseError])
def test_a_parse_error_without_context_is_its_message(
    error: type[CcsdsParseError],
) -> None:
    assert str(error("empty message")) == "empty message"


def test_omm_parse_error_reports_record_line_path_and_keyword() -> None:
    error = OmmParseError(
        "bad number", record=3, line=4, path="omm/body/segment[3]", keyword="BSTAR"
    )
    assert str(error) == (
        "bad number (record 3, line 4, at omm/body/segment[3], keyword BSTAR)"
    )
    assert (error.record, error.line, error.path, error.keyword) == (
        3,
        4,
        "omm/body/segment[3]",
        "BSTAR",
    )


@pytest.mark.parametrize("error", [CdmParseError, OmmParseError])
def test_message_parse_errors_share_the_ccsds_base(
    error: type[CcsdsParseError],
) -> None:
    assert issubclass(error, CcsdsParseError)
    assert not issubclass(error, ValueError)
    raised = error("bad unit", record=2, keyword="X")
    assert str(raised) == "bad unit (record 2, keyword X)"
    assert (raised.record, raised.line, raised.path) == (2, None, None)


def test_unsupported_elements_are_a_keepout_and_value_error() -> None:
    # A ValueError like any other invalid argument, and a keepout error that
    # screening catches to leave the element set out.
    assert issubclass(UnsupportedElements, KeepoutError)
    assert issubclass(UnsupportedElements, ValueError)


@pytest.mark.parametrize("error", [EventError, HardBodyRadiusError])
def test_cdm_data_errors_are_keepout_and_value_errors(
    error: type[KeepoutError],
) -> None:
    # A ValueError like the other invalid arguments of an event, and a keepout
    # error the command line reports as a data error, not a bug.
    assert issubclass(error, KeepoutError)
    assert issubclass(error, ValueError)


def test_an_unsupported_frame_is_a_geometry_and_value_error() -> None:
    # A ValueError like StateVector's other argument errors, and the keepout
    # error of a message (e.g. a CDM) in a frame keepout cannot handle.
    assert issubclass(UnsupportedFrame, GeometryError)
    assert issubclass(UnsupportedFrame, ValueError)


def test_an_http_status_error_is_a_source_error_with_url_and_status() -> None:
    error = HttpStatusError("cannot download u: HTTP 503", url="u", status=503)
    assert isinstance(error, SourceUnavailable)
    assert str(error) == "cannot download u: HTTP 503"
    assert (error.url, error.status) == ("u", 503)


def test_account_and_limit_errors_are_source_errors() -> None:
    # A caller that handles an unavailable source handles these too.
    assert issubclass(AuthError, SourceUnavailable)
    error = RateLimited("try again in 40 min", retry_after=2400.0)
    assert isinstance(error, SourceUnavailable)
    assert str(error) == "try again in 40 min"
    assert error.retry_after == 2400.0
    assert RateLimited("refused").retry_after is None
