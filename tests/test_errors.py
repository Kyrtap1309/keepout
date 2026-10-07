import inspect

import pytest

from keepout import errors
from keepout.errors import (
    CdmParseError,
    GeometryError,
    KeepoutError,
    OmmParseError,
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


def test_the_error_module_defines_errors() -> None:
    assert len(ERRORS) >= 8


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


def test_cdm_parse_error_without_context() -> None:
    assert str(CdmParseError("empty message")) == "empty message"


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
    assert str(OmmParseError("empty message")) == "empty message"


def test_an_unsupported_frame_is_a_geometry_and_value_error() -> None:
    # A ValueError like StateVector's other argument errors, and the keepout
    # error of a message (e.g. a CDM) in a frame keepout cannot handle.
    assert issubclass(UnsupportedFrame, GeometryError)
    assert issubclass(UnsupportedFrame, ValueError)
