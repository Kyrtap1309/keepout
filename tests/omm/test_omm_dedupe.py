import dataclasses

import astropy.units as u
from hypothesis import given
from hypothesis import strategies as st
from omm_builders import ISS

from keepout.omm import Omm, deduplicate
from keepout.omm._io import parse_record

ISS_OMM = parse_record(ISS)


def element_set(catalog_number: int, day: float, revolution: int = 0) -> Omm:
    """The ISS elements as another object, ``day`` days later."""
    return dataclasses.replace(
        ISS_OMM,
        catalog_number=catalog_number,
        epoch=ISS_OMM.epoch + day * u.day,
        rev_at_epoch=ISS_OMM.rev_at_epoch + revolution,
    )


def test_the_newest_element_set_of_each_object_is_kept() -> None:
    older, newer, other = element_set(1, 0.0), element_set(1, 0.5), element_set(2, 0.0)
    result = deduplicate([older, other, newer])
    assert result.kept == (newer, other)
    assert result.dropped == (older,)


def test_an_exact_copy_is_not_reported() -> None:
    first, newer = element_set(1, 0.0), element_set(1, 1.0)
    copy = dataclasses.replace(first)
    assert copy is not first
    result = deduplicate([first, newer, copy, dataclasses.replace(newer)])
    assert result.kept == (newer,)
    assert result.kept[0] is newer
    assert result.dropped == (first,)
    assert result.dropped[0] is first


def test_of_two_different_sets_with_the_same_epoch_the_first_is_kept() -> None:
    first, second = element_set(7, 0.0), element_set(7, 0.0, revolution=1)
    assert deduplicate([first, second]) == ((first,), (second,))
    assert deduplicate([second, first]) == ((second,), (first,))


def test_nothing_in_nothing_out() -> None:
    assert deduplicate([]) == ((), ())
    assert deduplicate(iter([ISS_OMM])) == ((ISS_OMM,), ())


@given(
    st.lists(
        st.tuples(st.integers(1, 4), st.integers(0, 3), st.integers(0, 1)).map(
            lambda key: element_set(key[0], float(key[1]), key[2])
        ),
        max_size=12,
    )
)
def test_one_newest_set_per_object_and_every_set_accounted_for(
    omms: list[Omm],
) -> None:
    kept, dropped = deduplicate(omms)
    numbers = [omm.catalog_number for omm in omms]
    assert [omm.catalog_number for omm in kept] == list(dict.fromkeys(numbers))
    for omm in kept:
        same = [other for other in omms if other.catalog_number == omm.catalog_number]
        assert all(omm.epoch >= other.epoch for other in same)
    listed = [*kept, *dropped]
    assert all(any(omm == other for other in listed) for omm in omms)
    assert all(
        not any(a == b for b in listed[index + 1 :]) for index, a in enumerate(listed)
    )
    assert [omms.index(omm) for omm in dropped] == sorted(
        omms.index(omm) for omm in dropped
    )
