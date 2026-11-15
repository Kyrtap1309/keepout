# Only the parts of Plotly the dashboard draws with are typed: the figure
# and the traces of keepout.dashboard.plots, make_subplots, and the names
# Streamlit's st.plotly_chart annotates its figure with (graph_objs,
# basedatatypes). Importing anything else is an error, so a new use is typed
# here before it is relied on.
