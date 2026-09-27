from cdm_builders import B1_KVN, B1_XML, TRACSS, assert_equivalent

import keepout.cdm


def test_load_detects_the_format() -> None:
    assert_equivalent(keepout.cdm.load(B1_XML), keepout.cdm.load(B1_KVN))


def test_loads_tolerates_bom_and_leading_whitespace() -> None:
    text = B1_XML.read_text(encoding="utf-8")
    cdm = keepout.cdm.loads("﻿\n  " + text)
    assert cdm.message_id == "201113719185"
    kvn_text = TRACSS.read_text(encoding="utf-8")
    assert keepout.cdm.loads("\n" + kvn_text).version == "2.0"
