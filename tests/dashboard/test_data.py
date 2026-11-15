"""The pages' data without Streamlit: the events list's page
(``keepout.dashboard.listing``) and the event page's data path
(``keepout.dashboard.event_data``), on the seed store."""

from pathlib import Path

import dashboard_seed as seed
import plotly.graph_objects as go

from keepout.dashboard import event_data
from keepout.dashboard.listing import PAGE_SIZE, read_page
from keepout.settings import load_settings


def test_read_page_moves_back_to_the_last_page(seed_db: Path) -> None:
    settings = load_settings(seed.SETTINGS)
    first = read_page(seed_db, settings, as_of=seed.seed_as_of(), include_past=True)
    assert first.offset == 0
    assert first.limit == PAGE_SIZE
    beyond = read_page(
        seed_db,
        settings,
        as_of=seed.seed_as_of(),
        include_past=True,
        offset=10 * PAGE_SIZE,
    )
    assert beyond.offset == (first.total - 1) // PAGE_SIZE * PAGE_SIZE
    assert beyond.items


def test_read_event_of_no_event(seed_db: Path) -> None:
    settings = load_settings(seed.SETTINGS)
    assert event_data.read_event(seed_db, "no-such-event", settings) is None


def test_event_data_path(seed_db: Path) -> None:
    settings = load_settings(seed.SETTINGS)
    page = read_page(seed_db, settings, as_of=seed.seed_as_of())
    view = event_data.read_event(seed_db, page.items[0].event_id, settings)
    assert view is not None
    assert not view.missing
    trend = event_data.trend_of(view, settings)
    assert trend is not None
    assert isinstance(event_data.trend_plot(trend), go.Figure)
    conjunction = event_data.latest_conjunction(view, settings)
    max_pc = event_data.is_max_pc(view, conjunction)
    assert isinstance(event_data.encounter_plot(conjunction, max_pc=max_pc), go.Figure)
    figure = event_data.geometry_of(conjunction)
    assert isinstance(event_data.geometry_plot(figure, max_pc=max_pc), go.Figure)
