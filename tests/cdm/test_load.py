import gzip
from pathlib import Path

import pytest
from builders import B1_KVN
from cdm_builders import B1_XML, TRACSS, assert_equivalent

import keepout.cdm
from keepout.errors import CdmParseError


def test_load_detects_the_format() -> None:
    assert_equivalent(keepout.cdm.load(B1_XML), keepout.cdm.load(B1_KVN))


def test_loads_tolerates_bom_and_leading_whitespace() -> None:
    text = B1_XML.read_text(encoding="utf-8")
    cdm = keepout.cdm.loads("﻿\n  " + text)
    assert cdm.message_id == "201113719185"
    kvn_text = TRACSS.read_text(encoding="utf-8")
    assert keepout.cdm.loads("\n" + kvn_text).version == "2.0"


def test_a_gzip_file_is_read_whatever_its_name(tmp_path: Path) -> None:
    path = tmp_path / "message.kvn"
    path.write_bytes(gzip.compress(B1_KVN.read_bytes()))
    assert_equivalent(keepout.cdm.load(path), keepout.cdm.load(B1_KVN))


def test_a_file_that_is_not_utf8_is_a_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.kvn"
    path.write_bytes(b"CCSDS_CDM_VERS = 1.0\n\xff\n")
    with pytest.raises(CdmParseError, match=r"broken\.kvn: not UTF-8 text"):
        keepout.cdm.load(path)
