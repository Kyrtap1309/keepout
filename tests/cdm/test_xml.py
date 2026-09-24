from pathlib import Path
from xml.etree import ElementTree

import pytest
from cdm_builders import DATA, assert_equivalent, cdms
from hypothesis import given, settings

from keepout.cdm import Cdm, kvn, xml
from keepout.errors import CdmParseError

B1_KVN = DATA / "b1_example.kvn"
B1_XML = DATA / "b1_example.xml"
TRACSS = DATA / "tracss_v2.kvn"


def qualify(text: str) -> str:
    """Rewrite a message with every element in the ndm: namespace (TC2)."""
    root = ElementTree.fromstring(text)
    for element in root.iter():
        element.tag = f"{{{xml.NDM_NAMESPACE}}}{element.tag}"
    ElementTree.register_namespace("ndm", xml.NDM_NAMESPACE)
    return ElementTree.tostring(root, encoding="unicode")


def b1_xml() -> str:
    return B1_XML.read_text(encoding="utf-8")


def test_xml_fixture_matches_kvn_fixture() -> None:
    assert_equivalent(xml.load(B1_XML), kvn.load(B1_KVN))


def test_namespace_qualified_xml_is_accepted() -> None:
    assert_equivalent(xml.loads(qualify(b1_xml())), kvn.load(B1_KVN))


@pytest.mark.parametrize("path", [B1_KVN, TRACSS])
def test_fixture_round_trip(path: Path) -> None:
    cdm = kvn.load(path)
    assert_equivalent(xml.loads(xml.dumps(cdm)), cdm)


def test_version_1_uses_b1_layout() -> None:
    text = xml.dumps(kvn.load(B1_KVN))
    assert 'version="1.0"' in text
    assert (
        'xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/'
        'ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd"'
    ) in text
    assert "<additionalParameters>" in text
    assert "<relativeStateVector>" in text
    assert '<X units="km">' in text
    assert "<CCSDS_CDM_VERS" not in text


def test_version_2_uses_cdm_2_groups() -> None:
    text = xml.dumps(kvn.load(TRACSS))
    assert 'version="2.0"' in text
    assert "<physicalParameters>" in text
    assert "<userDefinedParameters>" in text
    assert "<additionalParameters>" not in text
    assert "<CLASSIFICATION>UNCLASSIFIED</CLASSIFICATION>" in text


@settings(max_examples=50, deadline=None)
@given(cdms())
def test_round_trip_property(cdm: Cdm) -> None:
    assert_equivalent(xml.loads(xml.dumps(cdm)), cdm)


@settings(max_examples=25, deadline=None)
@given(cdms())
def test_kvn_to_xml_property(cdm: Cdm) -> None:
    assert_equivalent(xml.loads(xml.dumps(kvn.loads(kvn.dumps(cdm)))), cdm)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<cdm>", "malformed XML"),
        ('<oem version="1.0"/>', "root element must be <cdm>, got <oem>"),
        ("<cdm/>", "<cdm> has no version attribute"),
        ('<cdm version="1.0"><header/></cdm>', "expected exactly one <body>, found 0"),
        (
            '<?xml version="1.0"?>'
            '<!DOCTYPE cdm [<!ENTITY a "aaaa">]><cdm version="1.0"/>',
            "DOCTYPE declarations are not allowed",
        ),
    ],
)
def test_structural_errors(text: str, message: str) -> None:
    with pytest.raises(CdmParseError, match=message):
        xml.loads(text)


def test_one_segment_is_rejected() -> None:
    root = ElementTree.fromstring(b1_xml())
    body = root.find("body")
    assert body is not None
    body.remove(body.findall("segment")[1])
    with pytest.raises(CdmParseError, match="expected 2 <segment> elements, found 1"):
        xml.loads(ElementTree.tostring(root, encoding="unicode"))


def test_bad_unit_reports_the_element_path() -> None:
    text = b1_xml().replace(
        '<X units="km">2570.097065</X>', '<X units="m">2570.097065</X>'
    )
    with pytest.raises(CdmParseError, match=r"does not match expected \[km\]") as info:
        xml.loads(text)
    assert info.value.path == "cdm/body/segment[1]/data/stateVector/X"
    assert info.value.keyword == "X"


def test_duplicate_keyword() -> None:
    text = b1_xml().replace(
        '<MISS_DISTANCE units="m">715</MISS_DISTANCE>',
        '<MISS_DISTANCE units="m">715</MISS_DISTANCE>'
        '<MISS_DISTANCE units="m">1</MISS_DISTANCE>',
    )
    with pytest.raises(CdmParseError, match="duplicate keyword") as info:
        xml.loads(text)
    assert info.value.keyword == "MISS_DISTANCE"


def test_missing_required_keyword() -> None:
    text = b1_xml().replace('<X_DOT units="km/s">-2.888612500</X_DOT>', "")
    with pytest.raises(
        CdmParseError, match=r"OBJECT2 section is missing required keyword\(s\): X_DOT"
    ):
        xml.loads(text)


def test_file_round_trip(tmp_path: Path) -> None:
    cdm = xml.load(B1_XML)
    target = tmp_path / "out.xml"
    xml.dump(cdm, target)
    assert_equivalent(xml.load(target), cdm)
