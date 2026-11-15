import gzip
from pathlib import Path
from xml.etree import ElementTree

import pytest

from keepout.ccsds import files, kvn, xml
from keepout.ccsds.message import COMMENT, Family, KeywordValues, Located
from keepout.errors import CdmParseError, CdmWriteError, OmmParseError, OmmWriteError

# Families of made-up widths, not keepout's CDM and OMM ones.
FAMILY_A = Family(
    "CDM", CdmParseError, CdmWriteError, keyword_width=8, unknown_is_text=False
)
FAMILY_B = Family(
    "OMM", OmmParseError, OmmWriteError, keyword_width=4, unknown_is_text=True
)
FAMILIES = [FAMILY_A, FAMILY_B]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_repeated_keyword_is_the_familys_parse_error(family: Family) -> None:
    found = KeywordValues(family, record=2)
    found.add("A", Located("1", line=3))
    found.add(COMMENT, Located("x", line=4))
    found.add(COMMENT, Located("y", line=5))
    assert found.values == {"A": Located("1", line=3)}
    with pytest.raises(family.parse_error, match="duplicate keyword") as info:
        found.add("A", Located("2", line=6, path="m/A"))
    error = info.value
    assert (error.record, error.line, error.path, error.keyword) == (2, 6, "m/A", "A")


@pytest.mark.parametrize("family", FAMILIES)
def test_xml_values_name_user_defined_parameters(family: Family) -> None:
    element = ElementTree.fromstring(
        '<data><g><X units="km"> 1 </X></g>'
        '<USER_DEFINED parameter="RUN">a</USER_DEFINED><EMPTY/></data>'
    )
    assert list(xml.values(element, "data", family, groups={"g"})) == [
        ("X", Located("1", "km", path="data/g/X")),
        ("USER_DEFINED_RUN", Located("a", path="data/USER_DEFINED")),
        ("EMPTY", Located("", path="data/EMPTY")),
    ]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("attribute", ["", ' parameter=""'])
def test_a_user_defined_element_needs_a_parameter(
    family: Family, attribute: str
) -> None:
    element = ElementTree.fromstring(f"<data><USER_DEFINED{attribute}/></data>")
    with pytest.raises(
        family.parse_error, match="USER_DEFINED has no parameter attribute"
    ) as info:
        list(xml.values(element, "data", family, record=1))
    assert (info.value.record, info.value.path) == (1, "data/USER_DEFINED")


@pytest.mark.parametrize("family", FAMILIES)
def test_xml_errors_are_the_familys(family: Family) -> None:
    with pytest.raises(family.parse_error, match=f"not allowed in {family.name} XML"):
        xml.parse("<!DOCTYPE x><x/>", family)
    with pytest.raises(family.parse_error, match="expected exactly one <a>"):
        xml.single(ElementTree.fromstring("<x/>"), "a", "x", family)
    with pytest.raises(family.write_error, match=r"A: XML 1\.0 cannot hold"):
        xml.add_leaf(ElementTree.Element("x"), "A", "\x00", None, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_kvn_lines_are_the_familys(family: Family) -> None:
    width = family.keyword_width
    assert kvn.format_line("A", "1", "m", family) == f"{'A':<{width}} = 1 [m]"
    with pytest.raises(family.write_error, match="KVN cannot hold a line break"):
        kvn.format_line("A", "1\n2", None, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_broken_file_is_the_familys_parse_error(
    family: Family, tmp_path: Path
) -> None:
    path = tmp_path / "message"
    path.write_bytes(b"\xff")
    with pytest.raises(family.parse_error, match="not UTF-8 text"):
        files.read_file(path, family)


@pytest.mark.parametrize("family", FAMILIES)
def test_decode_file_caps_the_size(family: Family) -> None:
    text = "A = 1\n"
    size = len(text)
    assert files.decode_file(text.encode(), "f", family, max_bytes=size) == text
    with pytest.raises(family.parse_error, match=f"^f: larger than {size - 1} bytes$"):
        files.decode_file(text.encode(), "f", family, max_bytes=size - 1)


@pytest.mark.parametrize("family", FAMILIES)
def test_decode_file_caps_gzip_data_once_decompressed(family: Family) -> None:
    text = "A = 1\n" * 1000
    packed = gzip.compress(text.encode())
    size = len(text)
    assert len(packed) < size
    assert files.decode_file(packed, "f", family, max_bytes=size) == text
    # Several members, as gzip.decompress reads them.
    twice = packed + packed
    assert files.decode_file(twice, "f", family, max_bytes=2 * size) == text * 2
    with pytest.raises(family.parse_error, match=r"than \d+ bytes once"):
        files.decode_file(twice, "f", family, max_bytes=2 * size - 1)
    # A bomb is not inflated past the limit.
    bomb = gzip.compress(b" " * 50_000_000)
    with pytest.raises(family.parse_error, match="once decompressed"):
        files.decode_file(bomb, "f", family, max_bytes=len(bomb) + 1)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"\x1f\x8bnot gzip", "invalid gzip data"),
        (gzip.compress(b"A = 1\n")[:-4], "invalid gzip data: truncated"),
        (gzip.compress(b"A = 1\n") + b"junk", "invalid gzip data: trailing bytes"),
    ],
)
def test_decode_file_with_a_cap_refuses_broken_gzip(data: bytes, match: str) -> None:
    with pytest.raises(FAMILY_A.parse_error, match=match):
        files.decode_file(data, "f", FAMILY_A, max_bytes=1000)
    # As without a cap.
    with pytest.raises(FAMILY_A.parse_error, match="invalid gzip data"):
        files.decode_file(data, "f", FAMILY_A)
