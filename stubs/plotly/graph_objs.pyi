# Plotly's older name of plotly.graph_objects, which Streamlit's
# st.plotly_chart annotates its figure with (go.Figure | go.Data).
from plotly.graph_objects import Figure as Figure

# A list of traces, Plotly's old figure data. Typed so that Streamlit's
# figure type is not Any: keepout passes Figures only.
class Data: ...
