import dataclasses
from pathlib import Path
from xml.etree import ElementTree

import pytest
import xmlschema
from cdm_builders import B1_KVN, B1_XML, TRACSS, assert_equivalent, cdms, minimal_cdm
from hypothesis import given, settings

from keepout.cdm import Cdm, Field, Section, kvn, xml
from keepout.errors import CdmParseError

# NDM/XML 2.0.0 CDM module (CCSDS 508.0-B-1). CCSDS publishes no schema for
# CDM 2.0, so only version 1.0 output is validated.
SCHEMA = xmlschema.XMLSchema(Path(__file__).parent / "xsd" / "cdm.xsd")


def qualify(text: str) -> str:
    """Rewrite a message with every element in the ndm: namespace (TC2)."""
    root = ElementTree.fromstring(text)
    for element in root.iter():
        element.tag = f"{{{xml.NDM_NAMESPACE}}}{element.tag}"
    return ElementTree.tostring(root, encoding="unicode")


def with_segment2(cdm: Cdm, *extra: Field) -> Cdm:
    segment = Section("segment", (*cdm.segments[1].fields, *extra))
    return dataclasses.replace(cdm, segments=(cdm.segments[0], segment))


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


def test_version_1_output_is_valid_against_the_ccsds_schema() -> None:
    SCHEMA.validate(xml.dumps(kvn.load(B1_KVN)))


def test_version_1_uses_b1_layout() -> None:
    root = ElementTree.fromstring(xml.dumps(kvn.load(B1_KVN)))
    assert root.get("version") == "1.0"
    xsi = "http://www.w3.org/2001/XMLSchema-instance"
    location = root.get(f"{{{xsi}}}noNamespaceSchemaLocation")
    assert location == (
        "https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd"
    )
    segment = root.find("body/segment")
    assert segment is not None
    assert segment.find("data/additionalParameters") is not None
    x = segment.find("data/stateVector/X")
    assert x is not None
    assert x.get("units") == "km"
    assert root.find("body/relativeMetadataData/relativeStateVector") is not None
    assert root.find(".//CCSDS_CDM_VERS") is None


def test_version_2_uses_cdm_2_groups() -> None:
    root = ElementTree.fromstring(xml.dumps(kvn.load(TRACSS)))
    assert root.get("version") == "2.0"
    assert root.find("body/segment/data/physicalParameters") is not None
    assert root.find(".//additionalParameters") is None
    assert root.findtext("header/CLASSIFICATION") == "UNCLASSIFIED"


def test_user_defined_parameters_use_the_parameter_attribute() -> None:
    cdm = kvn.load(TRACSS)
    root = ElementTree.fromstring(xml.dumps(cdm))
    group = root.find("body/segment/data/userDefinedParameters")
    assert group is not None
    assert [(e.tag, e.get("parameter")) for e in group] == [
        ("USER_DEFINED", "RUN_ID"),
        ("USER_DEFINED", "MEETS_ALERTABLE_CRITERIA"),
    ]
    assert xml.loads(xml.dumps(cdm)).object1.user_defined == cdm.object1.user_defined


def test_several_user_defined_parameters_are_read() -> None:
    text = xml.dumps(kvn.load(TRACSS))
    assert text.count("<USER_DEFINED ") == 2
    assert xml.loads(text).object1.user_defined == {
        "USER_DEFINED_RUN_ID": "8e815679-b659-48df-957e-9c5b4921af52",
        "USER_DEFINED_MEETS_ALERTABLE_CRITERIA": "NO",
    }


def test_user_defined_without_parameter_is_rejected() -> None:
    text = xml.dumps(kvn.load(TRACSS)).replace(' parameter="RUN_ID"', "", 1)
    with pytest.raises(CdmParseError, match="USER_DEFINED has no parameter") as info:
        xml.loads(text)
    assert info.value.path is not None
    assert info.value.path.endswith("userDefinedParameters/USER_DEFINED")


def test_empty_group_is_structure_not_a_keyword() -> None:
    text = b1_xml().replace("<data>", "<data><userDefinedParameters/>", 1)
    assert_equivalent(xml.loads(text), xml.load(B1_XML))


def test_mid_block_comments_move_to_the_start_of_their_element() -> None:
    # KVN allows a comment anywhere; NDM/XML only at the start of an element.
    base = minimal_cdm()
    relative = Section(
        "relative", (*base.relative.fields, Field("COMMENT", "late note"))
    )
    cdm = dataclasses.replace(base, relative=relative)
    text = xml.dumps(cdm)
    SCHEMA.validate(text)
    block = ElementTree.fromstring(text).find("body/relativeMetadataData")
    assert block is not None
    assert (block[0].tag, block[0].text) == ("COMMENT", "late note")
    assert xml.loads(text).relative.comments == ("late note",)


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
