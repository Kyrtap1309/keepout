from cdm_builders import DATA, assert_equivalent

import keepout.cdm


def test_load_detects_the_format() -> None:
    from_kvn = keepout.cdm.load(DATA / "b1_example.kvn")
    from_xml = keepout.cdm.load(DATA / "b1_example.xml")
    assert_equivalent(from_xml, from_kvn)


def test_loads_tolerates_bom_and_leading_whitespace() -> None:
    text = (DATA / "b1_example.xml").read_text(encoding="utf-8")
    cdm = keepout.cdm.loads("﻿\n  " + text)
    assert cdm.message_id == "201113719185"
    kvn_text = (DATA / "tracss_v2.kvn").read_text(encoding="utf-8")
    assert keepout.cdm.loads("\n" + kvn_text).version == "2.0"
