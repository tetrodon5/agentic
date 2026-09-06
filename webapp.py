import streamlit as st
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent


# -----------------------------
# Configuration
# -----------------------------

load_dotenv()

st.set_page_config(
    page_title="Agent scientifique - Cité du Vin",
    layout="wide"
)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# -----------------------------
# Agents
# -----------------------------

orchestrator = OrchestratorAgent(client)
researcher = ResearcherAgent(client)
critic = CriticAgent(client)
communicator = CommunicatorAgent(client)


# -----------------------------
# Interface
# -----------------------------

st.title("Agent scientifique - Cité du Vin")

st.caption(
    "Démonstration d’un écosystème agentique pour la recherche "
    "et la médiation scientifique."
)

st.write(
    "Une question est analysée par plusieurs agents spécialisés "
    "avant de produire une réponse scientifique vulgarisée."
)

question = st.text_area(
    "Posez votre question scientifique",
    placeholder="Ex. Pourquoi certains vins vieillissent-ils mieux que d'autres ?",
    height=90
)

launch = st.button("Lancer l'analyse scientifique")


# -----------------------------
# Exécution
# -----------------------------

if launch:

    if not question.strip():
        st.warning("Veuillez saisir une question.")
        st.stop()

    start_time = time.time()

    with st.status("Analyse agentique en cours...", expanded=True) as status:

        st.write("Orchestrateur — planification")
        orchestrator_result = orchestrator.run(question)

        st.write("Chercheur — recherche scientifique")
        research_result = researcher.run(
            question,
            orchestrator_result["text"]
        )

        st.write("Critique scientifique — vérification scientifique")
        critic_result = critic.run(
            question,
            research_result["text"]
        )

        st.write("Communication — synthèse")
        communicator_result = communicator.run(
            question,
            research_result["text"],
            critic_result["text"]
        )

        status.update(
            label="Analyse terminée",
            state="complete",
            expanded=False
        )

    elapsed_time = time.time() - start_time

    total_tokens = (
        orchestrator_result["total_tokens"]
        + research_result["total_tokens"]
        + critic_result["total_tokens"]
        + communicator_result["total_tokens"]
    )


    # -----------------------------
    # Réponse finale
    # -----------------------------

    st.divider()

    st.header("Réponse scientifique")

    st.markdown(communicator_result["text"])


    # -----------------------------
    # Processus agentique
    # -----------------------------

    st.divider()

    st.subheader("Processus agentique")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown("**Orchestrateur**")
    c1.caption("Planifie")

    c2.markdown("**Chercheur**")
    c2.caption("Analyse")

    c3.markdown("**Critique scientifique**")
    c3.caption("Challenge")

    c4.markdown("**Communication**")
    c4.caption("Vulgarise")


    # -----------------------------
    # Observabilité
    # -----------------------------

    st.divider()

    st.subheader("Observabilité")

    m1, m2, m3 = st.columns(3)

    m1.metric("Agents", "4")
    m2.metric("Tokens", f"{total_tokens:,}".replace(",", " "))
    m3.metric("Temps", f"{elapsed_time:.1f} s")


    # -----------------------------
    # Détails
    # -----------------------------

    st.divider()

    with st.expander("Voir le plan de l'Orchestrator"):
        st.markdown(orchestrator_result["text"])

    with st.expander("Voir l'analyse du Researcher"):
        st.markdown(research_result["text"])

    with st.expander("Voir la critique scientifique"):
        st.markdown(critic_result["text"])