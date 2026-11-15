from typing import cast

import pytest

from keepout.sources.spacetrack import CATALOG, CLASS_INTERVALS, Query, RequestClass
from keepout.units import HOUR

CDM_QUERY = Query("cdm_public", "TCA/>now/orderby/TCA asc/format/json", "upcoming")


def test_the_catalog_query_is_space_tracks_recommended_one() -> None:
    assert CATALOG.url == (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        "decay_date/null-val/epoch/%3Enow-10/orderby/norad_cat_id/format/json"
    )
    assert CATALOG.interval == HOUR


def test_the_class_limits_are_the_published_ones() -> None:
    assert dict(CLASS_INTERVALS) == {"gp": HOUR, "cdm_public": 8 * HOUR}
    assert CDM_QUERY.interval == 8 * HOUR


def test_a_query_url_is_its_class_and_quoted_path() -> None:
    assert CDM_QUERY.url.endswith(
        "/class/cdm_public/TCA/%3Enow/orderby/TCA%20asc/format/json"
    )
    assert Query("gp", "NORAD_CAT_ID/25544,43873/format/json", "a").url.endswith(
        "/NORAD_CAT_ID/25544,43873/format/json"
    )


@pytest.mark.parametrize(
    ("request_class", "path", "name", "message"),
    [
        ("gp_history", "format/json", "x", "no Space-Track limit known"),
        ("cdm", "format/json", "x", "no Space-Track limit known"),
        ("gp", "format/json", "../x", "not a cache file name"),
        ("gp", "format/json", "Catalog", "not a cache file name"),
        ("gp", "format/json", "", "not a cache file name"),
        ("gp", "", "x", "not a Space-Track query path"),
        ("gp", "/format/json", "x", "not a Space-Track query path"),
        ("gp", "format/json/", "x", "not a Space-Track query path"),
        ("gp", "format/json?x=1", "x", "not a Space-Track query path"),
        ("gp", "format/json#x", "x", "not a Space-Track query path"),
        ("gp", "a/../b", "x", "not a Space-Track query path"),
    ],
)
def test_a_query_is_checked(
    request_class: str, path: str, name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Query(cast("RequestClass", request_class), path, name)
