import dataclasses
from pathlib import Path
from xml.etree import ElementTree

import pytest
import xmlschema
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st
from omm_builders import CCSDS_XML, GOES_9, ISS, SAMPLE_CSV, csv_text, records

from keepout.errors import OmmParseError, OmmWriteError
from keepout.omm import Omm, xml
from keepout.omm.celestrak import load_csv, loads_csv

# The OMM 3.0 module of NDM/XML 4.0.0 (CCSDS 502.0-B-3).
SCHEMA = xmlschema.XMLSchema(Path(__file__).parent / "xsd" / "omm.xsd")

CREATED = Time("2026-10-06T12:00:00", scale="utc", precision=0)

(ISS_OMM,) = loads_csv(csv_text([ISS]))
ISS_XML = xml.dumps([ISS_OMM], creation_date=CREATED)


def iss_xml(old: str, new: str) -> str:
    """:data:`ISS_XML` with ``old`` replaced by ``new`` (``old`` must occur)."""
    assert old in ISS_XML, old
    return ISS_XML.replace(old, new, 1)


def assert_valid(text: str) -> None:
    """Every ``<omm>`` of an ``<ndm>`` is valid against the CCSDS schema."""
    root = ElementTree.fromstring(text)
    assert root.tag == "ndm"
    for message in root:
        SCHEMA.validate(ElementTree.tostring(message, encoding="unicode"))


def test_the_ccsds_example_is_valid_and_read() -> None:
    SCHEMA.validate(str(CCSDS_XML))
    (goes,) = loads_csv(csv_text([{**GOES_9, "OBJECT_NAME": "GOES-9"}]))
    # It omits EPHEMERIS_TYPE and CLASSIFICATION_TYPE: the defaults apply.
    assert xml.load(CCSDS_XML) == (goes,)


def test_written_omms_are_valid_against_the_ccsds_schema() -> None:
    omms = load_csv(SAMPLE_CSV)
    text = xml.dumps(omms, creation_date=CREATED)
    assert_valid(text)
    root = ElementTree.fromstring(text)
    location = root.get(
        "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation"
    )
    assert location == xml.SCHEMA_LOCATION
    assert [message.get("version") for message in root] == ["3.0"] * len(omms)
    assert xml.loads(text) == omms


def test_writer_layout_and_units() -> None:
    root = ElementTree.fromstring(ISS_XML)
    assert root.findtext("omm/header/CREATION_DATE") == "2026-10-06T12:00:00"
    assert root.findtext("omm/header/ORIGINATOR") == "UNKNOWN"
    segment = root.find("omm/body/segment")
    assert segment is not None
    assert [child.tag for child in segment.iter()][:4] == [
        "segment",
        "metadata",
        "OBJECT_NAME",
        "OBJECT_ID",
    ]
    assert segment.findtext("metadata/MEAN_ELEMENT_THEORY") == "SGP4"
    motion = segment.find("data/meanElements/MEAN_MOTION")
    assert motion is not None
    assert (motion.text, motion.get("units")) == ("15.48747543", "rev/day")
    bstar = segment.find("data/tleParameters/BSTAR")
    assert bstar is not None
    assert bstar.get("units") == "1/ER"


def test_a_single_omm_and_a_qualified_ndm_are_read() -> None:
    root = ElementTree.fromstring(ISS_XML)
    (message,) = root
    assert xml.loads(ElementTree.tostring(message, encoding="unicode")) == (ISS_OMM,)
    for element in root.iter():
        element.tag = f"{{urn:ccsds:schema:ndmxml}}{element.tag}"
    qualified = ElementTree.tostring(root, encoding="unicode")
    assert xml.loads(qualified) == (ISS_OMM,)


def test_ndm_comments_and_message_id_are_read_past() -> None:
    text = iss_xml(
        "<omm ", "<MESSAGE_ID>42</MESSAGE_ID><COMMENT>catalog</COMMENT><omm "
    )
    assert xml.loads(text) == (ISS_OMM,)


def test_version_2_is_read() -> None:
    assert xml.loads(iss_xml('version="3.0"', 'version="2.0"')) == (ISS_OMM,)


def test_other_elements_are_read_past() -> None:
    text = iss_xml(
        "</tleParameters>",
        "</tleParameters><userDefinedParameters>"
        '<USER_DEFINED parameter="A">1</USER_DEFINED>'
        '<USER_DEFINED parameter="B">2</USER_DEFINED></userDefinedParameters>',
    ).replace("<EPOCH>", "<COMMENT>x</COMMENT><EPOCH>")
    assert xml.loads(text) == (ISS_OMM,)


def test_an_empty_ndm_holds_no_omms() -> None:
    assert xml.loads("<ndm/>") == ()
    assert xml.loads(xml.dumps([])) == ()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '<!DOCTYPE omm [<!ENTITY a "a">]><omm/>',
            "DOCTYPE declarations are not allowed",
        ),
        ("<ndm>\n<omm>", r"malformed XML: .* \(line 2\)"),
        ("<cdm/>", "root element must be <omm> or <ndm>, got <cdm>"),
        ("<ndm><cdm/></ndm>", r"<ndm> holds a <cdm>; only OMMs are read \(at ndm\)"),
        (
            iss_xml(' version="3.0"', ""),
            r"<omm> has no version attribute \(record 1, at ndm/omm\[1\]\)",
        ),
        (
            iss_xml('version="3.0"', 'version="1.0"'),
            (
                r"unsupported OMM version '1.0', expected 2.0 or 3.0 "
                r"\(record 1, at ndm/omm\[1\], keyword CCSDS_OMM_VERS\)"
            ),
        ),
        (
            iss_xml("<segment>", "<segment/><segment>"),
            (
                r"expected exactly one <segment>, found 2 "
                r"\(record 1, at ndm/omm\[1\]/body\)"
            ),
        ),
        (
            iss_xml("<metadata>", "<metadata><NORAD_CAT_ID>1</NORAD_CAT_ID>"),
            (
                r"duplicate keyword \(record 1, at ndm/omm\[1\]/body/segment/data/"
                r"tleParameters/NORAD_CAT_ID, keyword NORAD_CAT_ID\)"
            ),
        ),
        (
            ISS_XML.replace(
                "</ndm>",
                ElementTree.tostring(
                    ElementTree.fromstring(
                        iss_xml("<REV_AT_EPOCH>58893</REV_AT_EPOCH>", "")
                    )[0],
                    encoding="unicode",
                )
                + "</ndm>",
            ),
            (
                r"missing keyword \(record 2, at ndm/omm\[2\]/body/segment, "
                r"keyword REV_AT_EPOCH\)"
            ),
        ),
        (
            iss_xml(
                "</tleParameters>",
                "</tleParameters><userDefinedParameters>"
                "<USER_DEFINED>1</USER_DEFINED></userDefinedParameters>",
            ),
            (
                r"USER_DEFINED has no parameter attribute \(record 1, at ndm/omm"
                r"\[1\]/body/segment/data/userDefinedParameters/USER_DEFINED\)"
            ),
        ),
        (
            iss_xml(
                "</tleParameters>",
                "</tleParameters><userDefinedParameters>"
                '<USER_DEFINED parameter="A">1</USER_DEFINED>'
                '<USER_DEFINED parameter="A">2</USER_DEFINED>'
                "</userDefinedParameters>",
            ),
            (
                r"duplicate keyword \(record 1, at .*/USER_DEFINED, "
                r"keyword USER_DEFINED_A\)"
            ),
        ),
        (
            iss_xml('units="deg"', 'units="rad"'),
            (
                r"unit \[rad\] does not match expected \[deg\] \(record 1, "
                r"at ndm/omm\[1\]/body/segment/data/meanElements/INCLINATION, "
                r"keyword INCLINATION\)"
            ),
        ),
        (
            iss_xml(">0.00068585<", ">-0.1<"),
            (
                r"eccentricity must be in \[0, 1\).* \(record 1, at ndm/omm\[1\]/body/"
                r"segment/data/meanElements/ECCENTRICITY, keyword ECCENTRICITY\)"
            ),
        ),
    ],
)
def test_invalid_xml_is_refused_with_its_place(text: str, message: str) -> None:
    with pytest.raises(OmmParseError, match=message):
        xml.loads(text)


def test_a_character_xml_cannot_hold_is_refused() -> None:
    broken = dataclasses.replace(ISS_OMM, object_name="A\x00B")
    with pytest.raises(OmmWriteError, match=r"OBJECT_NAME: XML 1\.0 cannot hold"):
        xml.dumps([broken])


def test_a_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "omm.xml"
    path.write_text(ISS_XML, encoding="utf-8")
    assert xml.load(path) == (ISS_OMM,)


@given(st.lists(records(), max_size=4))
def test_written_elements_are_valid_and_read_back_equal(
    values: list[dict[str, str]],
) -> None:
    omms: tuple[Omm, ...] = loads_csv(csv_text(values))
    text = xml.dumps(omms, creation_date=CREATED)
    assert_valid(text)
    assert xml.loads(text) == omms


@pytest.mark.parametrize("precision", [3, 9])
def test_written_epochs_do_not_depend_on_the_time_precision(precision: int) -> None:
    omm = dataclasses.replace(ISS_OMM, epoch=Time(ISS_OMM.epoch, precision=precision))
    text = xml.dumps([omm], creation_date=CREATED)
    assert text == ISS_XML
    assert xml.loads(text) == (omm,)


def test_an_epoch_finer_than_microseconds_is_written_and_read_exactly() -> None:
    epoch = "2026-10-06T00:20:47.860512345"
    omm = dataclasses.replace(ISS_OMM, epoch=Time(epoch, scale="utc", precision=3))
    text = xml.dumps([omm], creation_date=CREATED)
    assert_valid(text)
    assert f"<EPOCH>{epoch}</EPOCH>" in text
    assert xml.loads(text) == (omm,)
