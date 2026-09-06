import streamlit as st

st.set_page_config(
    page_title="Agent scientifique - Cité du Vin",
    page_icon=None,
    layout="centered"
)

st.title("Agent scientifique - Cité du Vin")

question = st.text_input(
    "Posez votre question scientifique :"
)

if st.button("Analyser"):
    if question:
        st.write("Question reçue :", question)
    else:
        st.warning("Veuillez saisir une question.")