"""Streamlit UI for Xuanzhi.

Run from the repo root:

    streamlit run streamlit_app.py

The app is organised as one ``main.py`` with a sidebar that switches
between six views (Overview, Ingest, Knowledge Graph, Paper Explorer,
Cross-Literature, Figure Source Lookup). Every view degrades gracefully
when its prerequisite data is missing, so the app is demo-able at any
stage of the pipeline.
"""
