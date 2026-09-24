from keepout.errors import CdmParseError, CovarianceError, GeometryError, KeepoutError


def test_new_errors_are_keepout_errors() -> None:
    for error in (CdmParseError, CovarianceError, GeometryError):
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
