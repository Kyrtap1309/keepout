from pathlib import Path

import openpyxl
import pytest

from keepout.errors import CaraFormatError
from keepout.sources import cara

Cell = str | int | float | None
"""A worksheet cell written by :func:`workbook_with`."""


def workbook_with(path: Path, *rows: tuple[Cell, ...]) -> Path:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    return path


def test_alfano_urls_are_percent_encoded() -> None:
    url = cara.ALFANO_2009_TCA[0].url
    assert url.endswith("/Alfano_2009_Test_Cases/Case1%20data%20at%20TCA.xls")


def test_manifest_covers_the_2d_validation_data() -> None:
    assert len(cara.ALFANO_2009_TCA) == 12
    assert len(cara.PC_METHOD_CDMS) == 53
    assert len({file.path for file in cara.MANIFEST}) == len(cara.MANIFEST)


def test_load_csv_returns_rows_by_header(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("Case,HBR\n1,15\n2,4\n", encoding="utf-8")
    assert cara.load_csv(path) == [
        {"Case": "1", "HBR": "15"},
        {"Case": "2", "HBR": "4"},
    ]


def test_load_csv_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(CaraFormatError, match="no header row"):
        cara.load_csv(path)


def test_load_xlsx_returns_rows_by_header(tmp_path: Path) -> None:
    path = workbook_with(
        tmp_path / "table.xlsx",
        ("Conjunction_ID", "HBR_m", "Pc2D"),
        ("A", 10, 1e-4),
        (None, None, None),
    )
    assert cara.load_xlsx(path) == [{"Conjunction_ID": "A", "HBR_m": 10, "Pc2D": 1e-4}]


def test_load_xlsx_ignores_empty_cells_after_the_header(tmp_path: Path) -> None:
    # CARA's Alfano09_CARA_Supplementary_Data.xlsx has two empty columns.
    path = workbook_with(
        tmp_path / "t.xlsx", ("Case", "Time", None, None), (1, 21600, None, None), (2,)
    )
    assert cara.load_xlsx(path) == [
        {"Case": 1, "Time": 21600},
        {"Case": 2, "Time": None},
    ]


def test_load_xlsx_rejects_values_past_the_header(tmp_path: Path) -> None:
    path = workbook_with(tmp_path / "t.xlsx", ("Case", "Time"), (1, 2, 3))
    with pytest.raises(CaraFormatError, match="row 2 has values past the header"):
        cara.load_xlsx(path)


@pytest.mark.parametrize("header", [("A", None, "B"), ("A", "B", "A")])
def test_load_xlsx_rejects_empty_or_repeated_header_names(
    tmp_path: Path, header: tuple[str | None, ...]
) -> None:
    path = workbook_with(tmp_path / "table.xlsx", header, (1, 2, 3))
    with pytest.raises(CaraFormatError, match="empty or repeated names"):
        cara.load_xlsx(path)


def test_load_xlsx_rejects_a_non_workbook(tmp_path: Path) -> None:
    path = tmp_path / "bad.xlsx"
    path.write_bytes(b"not a zip file")
    with pytest.raises(CaraFormatError, match=r"bad\.xlsx: not a readable \.xlsx"):
        cara.load_xlsx(path)


def test_load_xls_rejects_a_non_workbook(tmp_path: Path) -> None:
    path = tmp_path / "bad.xls"
    path.write_bytes(b"not an xls file" * 100)
    with pytest.raises(CaraFormatError, match=r"bad\.xls: not a readable \.xls"):
        cara.load_xls_rows(path)
