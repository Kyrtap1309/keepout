"""keepout's dashboard (the ``dashboard`` extra: Streamlit and Plotly).

``keepout dashboard`` runs Streamlit on :mod:`keepout.dashboard.app`, which
reads its store, settings and time from the environment
(:mod:`~keepout.dashboard.context`) and shows three pages: the events, an
event and the manoeuvre planner. :mod:`~keepout.dashboard.widgets` holds
what the pages share, and :mod:`~keepout.dashboard.plots` draws the figures
of :mod:`keepout.figures` with Plotly. The dashboard reads the store and
queues jobs; it never computes a Pc or a plan in its own process.

This package imports nothing: ``keepout dashboard`` finds the app's file
without loading Streamlit.
"""

__all__: list[str] = []
