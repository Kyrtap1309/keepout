"""The store groups CDMs into events as :func:`keepout.events.group_events`
does, whatever their order and however they are split into ingests."""

import itertools
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from store_builders import cdm

from keepout.cdm import Cdm
from keepout.events import DEFAULT_TCA_TOLERANCE, group_events
from keepout.store import CdmInput, open_store

TOLERANCE = DEFAULT_TCA_TOLERANCE


@st.composite
def events_of_pairs(draw: st.DrawFn) -> list[Cdm]:
    """The CDMs of realistic events: the TCAs of one event spread over no more
    than the tolerance (less a second, for the millisecond rounding of the
    TCAs), the events of one pair more than twice the tolerance apart; an
    event's CDMs give its CONJUNCTION_ID or none."""
    cdms: list[Cdm] = []
    for secondary in draw(
        st.lists(
            st.sampled_from(["30337", "40000"]), min_size=1, max_size=2, unique=True
        )
    ):
        start = 0.0
        for number in range(draw(st.integers(1, 3))):
            start += draw(st.floats(3.1 * TOLERANCE, 6 * TOLERANCE))
            conjunction_id = draw(st.sampled_from([None, f"{secondary}-{number}"]))
            offsets = draw(
                st.lists(st.floats(0.0, TOLERANCE - 1.0), min_size=1, max_size=4)
            )
            for offset in offsets:
                given_id = draw(st.sampled_from([None, conjunction_id]))
                cdms.append(
                    cdm(
                        f"M{len(cdms)}",
                        tca=round(start + offset, 3),
                        secondary=secondary,
                        conjunction_id=given_id,
                        created=-draw(st.floats(0.0, 7 * 86_400.0)),
                    )
                )
    return cdms


@st.composite
def permutations_and_batches(draw: st.DrawFn, cdms: list[Cdm]) -> list[list[Cdm]]:
    """``cdms`` in any order, split into ingests of any size."""
    shuffled = draw(st.permutations(cdms))
    cuts = sorted(draw(st.sets(st.integers(1, max(1, len(cdms) - 1)))))
    bounds = [0, *(c for c in cuts if c < len(cdms)), len(cdms)]
    return [list(shuffled[a:b]) for a, b in itertools.pairwise(bounds)]


def _partition(events: list[list[str]]) -> set[frozenset[str]]:
    return {frozenset(event) for event in events}


@pytest.mark.slow
@settings(max_examples=200)
@given(st.data())
def test_grouping_equals_group_events_in_any_order_and_batching(
    data: st.DataObject,
) -> None:
    cdms = data.draw(events_of_pairs())
    batches = data.draw(permutations_and_batches(cdms))
    expected = _partition(
        [[c.message_id for c in event.cdms] for event in group_events(cdms)]
    )
    with (
        tempfile.TemporaryDirectory() as directory,
        open_store(Path(directory) / "keepout.db") as store,
    ):
        for batch in batches:
            store.ingest([CdmInput.of(c, origin="test") for c in batch])
        stored = [[c.message_id for c in event.cdms] for event in store.events()]
    assert _partition(stored) == expected
