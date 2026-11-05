"""Hypothesis strategies of CDMs, for the CDM tests and the store's
round-trip test."""

import string

from astropy.time import Time
from hypothesis import strategies as st

from keepout.ccsds.epochs import parse_epoch
from keepout.cdm.keywords import REQUIRED, SEGMENT, covariance_names
from keepout.cdm.model import Cdm, Field, Section

words = (
    # Mixed case, punctuation (including KVN's "=", "[" and "]" and XML's
    # "<", "&" and quotes) and non-ASCII letters. No line breaks, tabs or
    # control characters (KVN is line-based; XML 1.0 forbids most controls),
    # and no edge whitespace, which neither format keeps. A trailing "[...]"
    # is excluded too: KVN reads it as the field's unit, so it cannot be told
    # apart from text.
    st.text(
        alphabet=string.ascii_letters
        + string.digits
        + " -._/:,;()+#'\"&<>=[]!?*%@"
        + "\u00c4\u00e9\u00df\u00f8\u00c5\u00f1\u00b0\u00b5",
        max_size=16,
    )
    .map(str.strip)
    .filter(lambda text: bool(text) and not text.endswith("]"))
)
reals = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_subnormal=False
).filter(lambda x: x == 0.0 or abs(x) > 1e-200)


@st.composite
def epochs(draw: st.DrawFn) -> Time:
    """UTC epochs 1990-2025 with 0 to 6 fractional digits."""
    date = (
        f"{draw(st.integers(1990, 2025)):04d}-{draw(st.integers(1, 12)):02d}-"
        f"{draw(st.integers(1, 28)):02d}"
    )
    clock = (
        f"{draw(st.integers(0, 23)):02d}:{draw(st.integers(0, 59)):02d}:"
        f"{draw(st.integers(0, 59)):02d}"
    )
    digits = draw(st.text(alphabet=string.digits, max_size=6))
    return parse_epoch(f"{date}T{clock}" + (f".{digits}" if digits else ""))


def comments() -> st.SearchStrategy[list[Field]]:
    return st.lists(words.map(lambda text: Field("COMMENT", text)), max_size=2)


def extras(prefix: str) -> st.SearchStrategy[list[Field]]:
    units = st.none() | st.sampled_from(["m", "kg", "n/a"])
    return st.lists(st.tuples(words, units), max_size=2).map(
        lambda items: [
            Field(f"{prefix}_{index}", value, unit)
            for index, (value, unit) in enumerate(items)
        ]
    )


_REQUIRED_TEXT = tuple(
    name
    for name in REQUIRED["segment"]
    if SEGMENT[name].kind == "str" and name not in ("OBJECT", "REF_FRAME")
)


@st.composite
def segments(draw: st.DrawFn, index: int) -> Section:
    # Comments open the section, where both KVN and NDM/XML put them.
    fields = [*draw(comments()), Field("OBJECT", f"OBJECT{index}")]
    fields += [Field(name, draw(words)) for name in _REQUIRED_TEXT]
    fields.append(
        Field("REF_FRAME", draw(st.sampled_from(["EME2000", "GCRF", "ITRF"])))
    )
    if draw(st.booleans()):
        fields.append(Field("TIME_LASTOB_START", draw(epochs())))
    if draw(st.booleans()):
        fields.append(Field("OBS_USED", draw(st.integers(0, 10**6))))
    if draw(st.booleans()):
        fields.append(Field("HBR", draw(reals)))
    fields += [
        Field(name, draw(reals)) for name in ("X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT")
    ]
    fields += [
        Field(name, draw(reals)) for name in covariance_names(draw(st.integers(6, 9)))
    ]
    fields += draw(extras("USER_DEFINED_TEST"))
    fields += draw(extras("X_KEEPOUT_SEG"))
    return Section("segment", tuple(fields))


@st.composite
def cdms(draw: st.DrawFn) -> Cdm:
    """Valid CDMs (any covariance values) exercising comments and extras."""
    header = Section(
        "header",
        (
            Field("CCSDS_CDM_VERS", draw(st.sampled_from(["1.0", "2.0"]))),
            *draw(comments()),
            Field("CREATION_DATE", draw(epochs())),
            Field("ORIGINATOR", draw(words)),
            Field("MESSAGE_ID", draw(words)),
            *draw(extras("X_KEEPOUT_HDR")),
        ),
    )
    relative_fields = [
        *draw(comments()),
        Field("TCA", draw(epochs())),
        Field("MISS_DISTANCE", draw(reals)),
    ]
    if draw(st.booleans()):
        relative_fields += [
            Field(f"RELATIVE_POSITION_{axis}", draw(reals)) for axis in "RTN"
        ]
    if draw(st.booleans()):
        relative_fields.append(Field("COLLISION_PROBABILITY", draw(reals)))
    relative_fields += draw(extras("X_KEEPOUT_REL"))
    return Cdm(
        header,
        Section("relative", tuple(relative_fields)),
        (draw(segments(1)), draw(segments(2))),
    )
