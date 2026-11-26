"""The pages' data without Streamlit: the events list's page
(``keepout.dashboard.listing``) and the event page's data path
(``keepout.dashboard.event_data``), on the seed store."""

from pathlib import Path

import dashboard_seed as seed
import plotly.graph_objects as go

from keepout.dashboard import event_data
from keepout.dashboard.listing import PAGE_SIZE, read_page
from keepout.settings import load_settings


def _names(figure: go.Figure) -> list[str]:
    """The names of a figure's traces."""
    return [str(trace.get("name", "")) for trace in figure.to_dict()["data"]]


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
    assert not view.detail.missing
    trend = event_data.trend_of(view)
    assert trend is not None
    # Every CDM of an upcoming event is a point of its trend.
    assert trend.labels == tuple(cdm.message_id for cdm in view.detail.event.cdms)
    assert _names(event_data.trend_plot(trend))
    conjunction = event_data.latest_conjunction(view, settings)
    assert conjunction.tca == view.detail.event.latest.tca
    # The seed's covariances are from the GP age model: not a max-Pc.
    assert not event_data.is_max_pc(view, conjunction)
    encounter = _names(event_data.encounter_plot(conjunction, max_pc=False))
    assert {"1 sigma", "hard-body radius"} <= set(encounter)
    figure = event_data.geometry_of(conjunction)
    geometry = _names(event_data.geometry_plot(figure, max_pc=False))
    assert any("ellipsoid" in name for name in geometry)
