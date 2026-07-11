import streamlit as st
import os

st.set_page_config(page_title="AIC 2026", layout="wide")

st.title("🎬 AIC 2026 - Multimodal Retrieval System")
st.markdown("### Team 2: Unified Search Application")

st.info("System is initializing...")

es_url = os.environ.get("ELASTICSEARCH_URL", "Not Set")
st.success(f"Backend configured! Expecting Elasticsearch at: `{es_url}`")

st.write("Ready for Team 2 to build the NLP Pipeline!")
