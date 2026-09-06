import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent

load_dotenv()

st.set_page_config(
    page_title="Agent scientifique - Cité du Vin",
    layout="wide"
)

st.title("Agent scientifique - Cité du Vin")
st.caption("Démonstration d’un écosystème agentique multi-agents")

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

orchestrator = OrchestratorAgent(client)
researcher = ResearcherAgent(client)
critic = CriticAgent(client)
communicator = CommunicatorAgent(client)

question = st.text_input(
    "Posez votre question scientifique :",
    placeholder="Ex. Pourquoi certains vins vieillissent-ils mieux que d'autres ?"
)

if st.button("Lancer l'analyse"):

    if not question:
        st.warning("Veuillez saisir une question.")
        st.stop()

    st.divider()

    with st.status("Analyse agentique en cours...", expanded=True) as status:

        st.write("Orchestrateur : analyse de la question")
        orchestrator_result = orchestrator.run(question)
        st.write("✓ Plan scientifique établi")

        st.write("Researcher : recherche et analyse scientifique")
        research_result = researcher.run(
            question,
            orchestrator_result["text"]
        )
        st.write("✓ Recherche terminée")

        st.write("Critic : vérification et challenge scientifique")
        critic_result = critic.run(
            question,
            research_result["text"]
        )
        st.write("✓ Vérification terminée")

        st.write("Communicator : synthèse et vulgarisation")
        communicator_result = communicator.run(
            question,
            research_result["text"],
            critic_result["text"]
        )
        st.write("✓ Réponse finale produite")

        status.update(
            label="Analyse terminée",
            state="complete",
            expanded=False
        )

    total_tokens = (
        orchestrator_result["total_tokens"]
        + research_result["total_tokens"]
        + critic_result["total_tokens"]
        + communicator_result["total_tokens"]
    )

    st.divider()

    st.subheader("Réponse scientifique")

    st.markdown(
        communicator_result["text"]
    )

    st.divider()

    st.subheader("Processus agentique")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Orchestrator",
        f'{orchestrator_result["total_tokens"]} tokens'
    )

    col2.metric(
        "Researcher",
        f'{research_result["total_tokens"]} tokens'
    )

    col3.metric(
        "Critic",
        f'{critic_result["total_tokens"]} tokens'
    )

    col4.metric(
        "Communicator",
        f'{communicator_result["total_tokens"]} tokens'
    )

    st.metric(
        "Consommation totale",
        f"{total_tokens} tokens"
    )

    st.divider()

    st.subheader("Voir le raisonnement des agents")

    with st.expander("Orchestrator - Plan scientifique"):
        st.markdown(orchestrator_result["text"])

    with st.expander("Researcher - Analyse scientifique"):
        st.markdown(research_result["text"])

    with st.expander("Critic - Vérification scientifique"):
        st.markdown(critic_result["text"])

    with st.expander("Communicator - Synthèse finale"):
        st.markdown(communicator_result["text"])