
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Drug Avoidance System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

html_path = (
    Path(__file__).parent
    / "drug_avoidance_system_eda_presentation.html"
)

if not html_path.exists():
    st.error(
        "Presentation HTML file not found. "
        "Place it in the same folder as app.py."
    )
else:
    st.iframe(html_path, height=900)
