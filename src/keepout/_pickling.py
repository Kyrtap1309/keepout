"""Private pickling support for frozen dataclasses holding read-only mappings."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = ["reduce_with_mappings"]

_Instance = TypeVar("_Instance", bound="DataclassInstance")


def reduce_with_mappings(
    instance: _Instance,
) -> tuple[type[_Instance], tuple[object, ...]]:
    """The ``__reduce__`` of a dataclass instance: its class and its init
    fields, with each :class:`~types.MappingProxyType` as a plain
    :class:`dict` (a mapping proxy does not pickle).

    The instance is rebuilt by calling the class with the fields
    positionally, in declaration order, so ``__post_init__`` makes the
    mappings read-only again; the init fields must not be keyword-only.
    Adding a field needs no change to the class's ``__reduce__``.
    """
    values = (
        getattr(instance, item.name)
        for item in dataclasses.fields(instance)
        if item.init
    )
    return type(instance), tuple(
        dict(value) if isinstance(value, MappingProxyType) else value
        for value in values
    )
