import streamlit as st
import pandas as pd

st.set_page_config(page_title="BRVM App", layout="wide")
st.title("🏠 Home")

# Charger une seule fois et stocker dans la session
if "data" not in st.session_state:
    st.session_state["data"] = {
        "base": pd.read_csv("Base_complète.csv", encoding="utf-8-sig"),
        "indices": pd.read_csv("60_Cours_indices.csv", encoding="utf-8-sig"),
        "ratios": pd.read_csv("ratios.csv", encoding="utf-8-sig"),
    }

st.success("✅ Données chargées et disponibles pour toutes les pages")

# petit aperçu
st.dataframe(st.session_state["data"]["base"].head())

