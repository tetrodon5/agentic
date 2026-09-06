import os
import time
import random
import json
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from domain_guard import (
    is_wine_domain,
    get_out_of_domain_message
)

# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Agent scientifique - Agent du Vin",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# CLE OPENAI
# ==================================================

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Clé OPENAI_API_KEY introuvable.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# ==================================================
# IMPORTS PROJET
# ==================================================

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent

from rag_search import search

from memory import (
    load_memory,
    save_memory_entry,
    search_memory,
    validate_memory_entry,
    reject_memory_entry,
    unvalidate_memory_entry
)


# ==================================================
# CLIENT OPENAI
# ==================================================

client = OpenAI(
    api_key=api_key
)


# ==================================================
# TARIFICATION
# ==================================================

INPUT_PRICE_PER_MILLION_USD = 0.20
OUTPUT_PRICE_PER_MILLION_USD = 1.20

USD_TO_EUR = 0.8606


def calculate_cost_eur(
    input_tokens,
    output_tokens
):

    input_cost_usd = (
        input_tokens
        / 1_000_000
        * INPUT_PRICE_PER_MILLION_USD
    )

    output_cost_usd = (
        output_tokens
        / 1_000_000
        * OUTPUT_PRICE_PER_MILLION_USD
    )

    return (
        input_cost_usd
        + output_cost_usd
    ) * USD_TO_EUR


def format_eur(
    value,
    decimals=5
):

    return (
        f"{value:.{decimals}f} €"
        .replace(".", ",")
    )


# ==================================================
# STATUTS MEMOIRE
# ==================================================

def memory_status_label(status):

    labels = {
        "generated_unvalidated":
            "En attente de validation",

        "generated_validated":
            "Validée",

        "generated_rejected":
            "Rejetée"
    }

    return labels.get(
        status,
        status
    )


# ==================================================
# VALIDATION MEMOIRE
# ==================================================

def validate_memory_callback(
    memory_id
):

    result = validate_memory_entry(
        memory_id
    )

    if result["success"]:

        st.session_state.memory_message = (
            f"Mémoire #{memory_id} validée."
        )

        st.session_state.memory_message_type = (
            "success"
        )

        payload = st.session_state.get(
            "analysis_payload"
        )

        if payload:

            entry = payload.get(
                "memory_entry"
            )

            if (
                entry
                and entry["id"] == memory_id
            ):

                entry["status"] = (
                    "generated_validated"
                )


def reject_memory_callback(
    memory_id
):

    result = reject_memory_entry(
        memory_id
    )

    if result["success"]:

        st.session_state.memory_message = (
            f"Mémoire #{memory_id} rejetée."
        )

        st.session_state.memory_message_type = (
            "warning"
        )

        payload = st.session_state.get(
            "analysis_payload"
        )

        if payload:

            entry = payload.get(
                "memory_entry"
            )

            if (
                entry
                and entry["id"] == memory_id
            ):

                entry["status"] = (
                    "generated_rejected"
                )


# ==================================================
# FEEDBACK UTILISATEUR
# ==================================================

def useful_feedback_callback():

    st.session_state.user_feedback = (
        "Utile"
    )

    st.session_state.feedback_message = (
        "Merci pour votre retour."
    )

    st.session_state.feedback_type = (
        "success"
    )


def review_feedback_callback():

    st.session_state.user_feedback = (
        "À revoir"
    )

    payload = st.session_state.get(
        "analysis_payload"
    )

    if payload:

        entry = payload.get(
            "memory_entry"
        )

        if entry:

            memory_id = entry["id"]

            unvalidate_memory_entry(
                memory_id
            )

            entry["status"] = (
                "generated_unvalidated"
            )

    st.session_state.feedback_message = (
        "Merci. Cette réponse est signalée pour révision."
    )

    st.session_state.feedback_type = (
        "warning"
    )


# ==================================================
# PROVENANCE
# ==================================================

def calculate_provenance(
    rag_results,
    memory_results
):

    if rag_results:

        rag_strength = (
            sum(
                r["score"]
                for r in rag_results
            )
            / len(rag_results)
        )

    else:

        rag_strength = 0


    if memory_results:

        memory_strength = max(
            r["score"]
            for r in memory_results
        )

    else:

        memory_strength = 0


    if (
        rag_strength >= 0.60
        and rag_strength
        >= memory_strength * 0.85
    ):

        label = (
            "Principalement documentaire"
        )

        detail = (
            "Les passages documentaires constituent "
            "le socle principal de la réponse."
        )

    elif (
        rag_strength > 0
        and memory_strength > 0
    ):

        label = (
            "Sources mixtes"
        )

        detail = (
            "La réponse combine documents retrouvés "
            "et analyses précédemment générées."
        )

    elif memory_strength > 0:

        label = (
            "Mémoire secondaire"
        )

        detail = (
            "Le corpus apporte peu de contexte ; "
            "la mémoire contribue davantage."
        )

    else:

        label = (
            "Connaissances générales"
        )

        detail = (
            "Aucun contexte documentaire ou mémoire "
            "suffisamment pertinent n'a été retrouvé."
        )


    return (
        label,
        detail,
        rag_strength,
        memory_strength
    )


# ==================================================
# FRONTIERES DE CONNAISSANCE
# ==================================================

def build_knowledge_boundaries(
    rag_results,
    memory_results
):

    knows = []
    limits = []


    if rag_results:

        avg_score = (
            sum(
                r["score"]
                for r in rag_results
            )
            / len(rag_results)
        )

        knows.append(
            f"{len(rag_results)} passage(s) documentaire(s) "
            "ont été retrouvés dans le corpus."
        )

        if avg_score >= 0.70:

            knows.append(
                "La correspondance avec le corpus "
                "documentaire est relativement forte."
            )

        elif avg_score >= 0.55:

            limits.append(
                "La correspondance documentaire "
                "avec la question reste partielle."
            )

        else:

            limits.append(
                "Le corpus actuel apporte peu "
                "d'éléments directement liés à la question."
            )

    else:

        limits.append(
            "Aucun passage documentaire "
            "n'a été retrouvé."
        )


    validated = [
        item
        for item in memory_results
        if item["status"]
        == "generated_validated"
    ]


    unvalidated = [
        item
        for item in memory_results
        if item["status"]
        == "generated_unvalidated"
    ]


    if validated:

        knows.append(
            f"{len(validated)} analyse(s) "
            "précédemment validée(s) ont été retrouvées."
        )


    if unvalidated:

        limits.append(
            f"{len(unvalidated)} mémoire(s) utilisée(s) "
            "ne sont pas encore validées humainement."
        )


    limits.append(
        "Une affirmation absente des sources "
        "ne peut pas être considérée comme confirmée "
        "uniquement parce qu'elle paraît plausible."
    )


    return (
        knows,
        limits
    )


# ==================================================
# MEDIATION
# ==================================================

def build_agent_question(
    question,
    audience,
    response_level
):

    audience_instructions = {

        "Visiteur":
            (
                "Répondre pour un visiteur curieux non spécialiste. "
                "Employer un vocabulaire accessible, concret et pédagogique."
            ),

        "Œnologue":
            (
                "Répondre pour un œnologue ou professionnel du vin. "
                "Le niveau scientifique et technique peut être plus élevé."
            )
    }


    level_instructions = {

        "Court":
            (
                "La réponse finale doit être concise : "
                "environ 2 à 4 paragraphes courts."
            ),

        "Standard":
            (
                "La réponse finale doit être structurée et pédagogique, "
                "sans devenir trop longue."
            ),

        "Approfondi":
            (
                "La réponse finale peut être détaillée, "
                "avec mécanismes scientifiques, nuances et limites."
            )
    }


    return (
        f"{question}\n\n"
        f"PUBLIC CIBLE : {audience}\n"
        f"{audience_instructions[audience]}\n\n"
        f"NIVEAU DE REPONSE : {response_level}\n"
        f"{level_instructions[response_level]}"
    )


# ==================================================
# STATISTIQUES GLOBALES
# ==================================================

def get_memory_statistics():

    memory = load_memory()

    total = len(memory)

    validated = sum(
        1
        for item in memory
        if item.get("status")
        == "generated_validated"
    )

    pending = sum(
        1
        for item in memory
        if item.get(
            "status",
            "generated_unvalidated"
        )
        == "generated_unvalidated"
    )

    rejected = sum(
        1
        for item in memory
        if item.get("status")
        == "generated_rejected"
    )


    total_tokens = sum(
        item.get(
            "total_tokens",
            0
        ) or 0
        for item in memory
    )


    total_cost = sum(
        item.get(
            "total_cost",
            0
        ) or 0
        for item in memory
    )


    return {
        "memory":
            memory,

        "total":
            total,

        "validated":
            validated,

        "pending":
            pending,

        "rejected":
            rejected,

        "tokens":
            total_tokens,

        "cost":
            total_cost
    }


# ==================================================
# DESIGN
# ==================================================

st.markdown(
    """
<style>

:root {
    --wine: #701C3A;
    --wine-dark: #4E1328;
    --wine-light: #F5EBEF;
    --cream: #FCFAF7;
    --ink: #242A35;
    --muted: #706C69;
    --border: #DDD3CD;
    --doc: #F4F1EC;
    --memory: #F4EDF2;
    --soft-green: #EFF5F0;
    --soft-orange: #FAF2E9;
}

.stApp {
    background: var(--cream);
}

.block-container {
    max-width: 1380px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    color: var(--ink);
}

h1 {
    font-size: 3rem !important;
    line-height: 1.05 !important;
    letter-spacing: -1px;
}

.hero-kicker {
    color: #8A8582;
    font-size: 0.80rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.hero-subtitle {
    font-size: 1.4rem;
    font-weight: 650;
    color: var(--ink);
    margin-top: 0.8rem;
    margin-bottom: 1rem;
}

.hero-copy {
    color: #53504E;
    line-height: 1.7;
    max-width: 760px;
}

.hero-tags {
    margin-top: 1.4rem;
    color: var(--wine);
    font-size: 0.8rem;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    font-weight: 650;
}

.demo-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.35rem 1.5rem;
    min-height: 220px;
    box-shadow: 0 8px 28px rgba(70,43,53,0.06);
}

.demo-card-label {
    color: #8C8987;
    font-size: 0.75rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}

.demo-card-title {
    color: var(--wine);
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
}

.section-kicker {
    color: var(--wine);
    font-size: 0.75rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.source-card {
    background: var(--doc);
    border-left: 5px solid #8C7B68;
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
}

.memory-card {
    background: var(--memory);
    border-left: 5px solid var(--wine);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    margin-bottom: 0.8rem;
}

.knowledge-card {
    background: var(--soft-green);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    min-height: 90px;
    border: 1px solid #DDE8DF;
}

.limit-card {
    background: var(--soft-orange);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    min-height: 90px;
    border: 1px solid #EDDFD1;
}

.provenance-map {
    background: white;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    margin-top: 0.7rem;
    margin-bottom: 1.2rem;
}

.map-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
}

.map-node {
    background: #FFFFFF;
    border: 1px solid #D9CFCA;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    min-width: 125px;
    text-align: center;
    font-size: 0.88rem;
}

.map-node-main {
    background: #F3E7EC;
    border-color: #D4B4C1;
    color: #4E1328;
    font-weight: 700;
}

.map-arrow {
    color: #8D7981;
    font-size: 1.35rem;
}

.card-title {
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.25rem;
}

.card-meta {
    color: #625D5A;
    font-size: 0.93rem;
    line-height: 1.6;
}

.provenance-badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    background: #F1E4E9;
    color: var(--wine-dark);
    font-weight: 650;
    margin-bottom: 0.4rem;
}

.flow-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.85rem 0.7rem;
    text-align: center;
    min-height: 86px;
}

.flow-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--wine);
}

.flow-result {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.35rem;
}

.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--wine);
    background: white;
    color: var(--wine);
    font-weight: 600;
    min-height: 44px;
}

.stButton > button:hover {
    background: var(--wine-light);
    border-color: var(--wine-dark);
    color: var(--wine-dark);
}

div[data-testid="stTextArea"] textarea {
    border-radius: 10px;
    background: #F3F5F7;
    border: 1px solid #E4E6E8;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 4px 18px rgba(60,40,45,0.04);
}

hr {
    border-color: #DDD4CE;
}

</style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# AGENTS
# ==================================================

orchestrator = OrchestratorAgent(
    client
)

researcher = ResearcherAgent(
    client
)

critic = CriticAgent(
    client
)

communicator = CommunicatorAgent(
    client
)


# ==================================================
# QUESTIONS
# ==================================================

question_bank = [

    "Pourquoi certains vins vieillissent-ils mieux que d'autres ?",

    "Quel rôle joue le terroir dans le goût du vin ?",

    "Pourquoi la couleur d'un vin rouge évolue-t-elle avec le temps ?",

    "Comment le changement climatique modifie-t-il les vins de Bordeaux ?",

    "Pourquoi certains vins sentent-ils les fruits rouges alors qu'ils n'en contiennent pas ?",

    "Comment les levures transforment-elles le jus de raisin en vin ?",

    "Pourquoi certains vins sont-ils plus acides que d'autres ?",

    "Quel est le rôle des tanins dans le vin ?",

    "Pourquoi sert-on certains vins plus frais que d'autres ?",

    "Comment le bois d'une barrique influence-t-il le vin ?"
]


if "suggestions" not in st.session_state:

    st.session_state.suggestions = (
        random.sample(
            question_bank,
            3
        )
    )


if "question" not in st.session_state:

    st.session_state.question = ""


# ==================================================
# MESSAGES
# ==================================================

if "memory_message" in st.session_state:

    if (
        st.session_state.memory_message_type
        == "success"
    ):

        st.success(
            st.session_state.memory_message
        )

    else:

        st.warning(
            st.session_state.memory_message
        )

    del st.session_state[
        "memory_message"
    ]

    del st.session_state[
        "memory_message_type"
    ]


if "feedback_message" in st.session_state:

    if (
        st.session_state.feedback_type
        == "success"
    ):

        st.success(
            st.session_state.feedback_message
        )

    else:

        st.warning(
            st.session_state.feedback_message
        )

    del st.session_state[
        "feedback_message"
    ]

    del st.session_state[
        "feedback_type"
    ]


# ==================================================
# HERO
# ==================================================

hero_left, hero_right = st.columns(
    [2.5, 1]
)


with hero_left:

    st.markdown(
        '<div class="hero-kicker">'
        'FCCV | DÉMONSTRATION IA'
        '</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Agent scientifique"
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'Recherche, contrôle scientifique et médiation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-copy">'
        "Explorez les connaissances scientifiques "
        "et culturelles du vin à travers une expérience "
        "de médiation adaptée à votre niveau."
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-tags">'
        'Patrimoine · Science · Culture · Partage'
        '</div>',
        unsafe_allow_html=True
    )


with hero_right:

    st.markdown(
        '<div class="demo-card">'
        '<div class="demo-card-label">'
        'Expérience scientifique'
        '</div>'
        '<div class="demo-card-title">'
        'Explorez le monde du vin'
        '</div>'
        '<div style="line-height:1.9;color:#3E3A3A;">'
        'Posez votre question<br>'
        'Choisissez votre niveau<br>'
        'Découvrez une réponse adaptée<br>'
        'Approfondissez si vous le souhaitez'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ==================================================
# EXPERIENCE
# ==================================================

st.markdown(
    '<div class="section-kicker">'
    'Expérience'
    '</div>',
    unsafe_allow_html=True
)

st.header(
    "Configurer l'expérience"
)


mode_col, audience_col, level_col = (
    st.columns(3)
)


with mode_col:

    display_mode = st.radio(
        "Mode d'affichage",
        [
            "Visiteur",
            "Expert"
        ],
        horizontal=True
    )


with audience_col:

    audience = st.radio(
        "Public cible",
        [
            "Visiteur",
            "Œnologue"
        ],
        horizontal=True
    )


with level_col:

    response_level = st.radio(
        "Niveau de réponse",
        [
            "Court",
            "Standard",
            "Approfondi"
        ],
        index=1,
        horizontal=True
    )


st.divider()


# ==================================================
# QUESTION
# ==================================================

st.markdown(
    '<div class="section-kicker">'
    'Explorer'
    '</div>',
    unsafe_allow_html=True
)

st.header(
    "Que souhaitez-vous explorer ?"
)

st.caption(
    "Choisissez une question ou posez librement la vôtre."
)


q1, q2, q3 = st.columns(3)


for column, suggestion in zip(
    [q1, q2, q3],
    st.session_state.suggestions
):

    with column:

        if st.button(
            suggestion,
            width="stretch"
        ):

            st.session_state.question = (
                suggestion
            )


question = st.text_area(
    "Votre question scientifique",
    value=st.session_state.question,
    placeholder="Posez librement votre question...",
    height=100
)


launch = st.button(
    "Lancer l'analyse scientifique"
)


# ==================================================
# EXECUTION
# ==================================================

if launch:

    if not question.strip():

        st.warning(
            "Veuillez saisir ou sélectionner une question."
        )

        st.stop()


    if not is_wine_domain(question):

        st.warning(
            get_out_of_domain_message()
        )

        st.stop()


    st.session_state.user_feedback = None

    start_time = time.time()


    agent_question = build_agent_question(
        question,
        audience,
        response_level
    )


    # ==================================================
    # MODE VISITEUR
    # ==================================================

    if display_mode == "Visiteur":

        status_container = st.status(
            "Analyse scientifique en cours...",
            expanded=False
        )


    # ==================================================
    # MODE EXPERT
    # ==================================================

    else:

        st.markdown(
            '<div class="section-kicker">'
            'Live Agent Flow'
            '</div>',
            unsafe_allow_html=True
        )

        st.subheader(
            "L'écosystème travaille"
        )

        status_container = st.status(
            "Analyse agentique en cours...",
            expanded=True
        )


    with status_container as status:


        # ==============================================
        # RAG
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**RAG documentaire**"
            )


        rag_results = search(
            question,
            top_k=2,
            minimum_score=0.55
        )


        if display_mode == "Expert":

            st.caption(
                f"{len(rag_results)} passage(s) "
                "pertinent(s) retrouvé(s)"
            )


        rag_context = ""

        for result in rag_results:

            rag_context += (
                f"\nSource : {result['source']} "
                f"| Chunk : {result['chunk_id']} "
                f"| Score : {result['score']:.3f}\n"
                f"{result['text']}\n"
            )


        # ==============================================
        # MEMOIRE
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**Mémoire sémantique**"
            )


        memory_results = search_memory(
            question,
            top_k=2,
            minimum_score=0.70
        )


        if display_mode == "Expert":

            st.caption(
                f"{len(memory_results)} mémoire(s) "
                "proche(s) retrouvée(s)"
            )


        memory_context = ""

        for result in memory_results:

            memory_context += (
                f"\nMémoire ID : {result['id']} "
                f"| Score : {result['score']:.3f} "
                f"| Statut : {result['status']}\n"
                f"Question précédente : {result['question']}\n"
                f"Réponse précédente : {result['answer']}\n"
            )


        # ==============================================
        # ORCHESTRATEUR
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**Orchestrateur**"
            )


        orchestrator_result = (
            orchestrator.run(
                agent_question
            )
        )


        if display_mode == "Expert":

            st.caption(
                "Plan scientifique établi"
            )


        # ==============================================
        # CHERCHEUR
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**Chercheur**"
            )


        research_result = (
            researcher.run(
                agent_question,
                orchestrator_result["text"],
                rag_context,
                memory_context
            )
        )


        if display_mode == "Expert":

            st.caption(
                "Analyse scientifique terminée"
            )


        # ==============================================
        # CRITIQUE
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**Critique scientifique**"
            )


        critic_result = (
            critic.run(
                agent_question,
                research_result["text"],
                rag_context=rag_context
            )
        )


        if display_mode == "Expert":

            st.caption(
                "Vérification et nuances terminées"
            )


        # ==============================================
        # MEDIATEUR
        # ==============================================

        if display_mode == "Expert":

            st.markdown(
                "**Médiateur scientifique**"
            )


        communicator_result = (
            communicator.run(
                agent_question,
                research_result["text"],
                critic_result["text"]
            )
        )


        if display_mode == "Expert":

            st.caption(
                f"Réponse adaptée au public : "
                f"{audience}"
            )


        if display_mode == "Expert":

            status.update(
                label="Analyse terminée",
                state="complete",
                expanded=False
            )

        else:

            status.update(
                label="Réponse prête",
                state="complete",
                expanded=False
            )


    elapsed_time = (
        time.time()
        - start_time
    )


    # ==================================================
    # PROVENANCE
    # ==================================================

    (
        provenance_label,
        provenance_detail,
        rag_strength,
        memory_strength

    ) = calculate_provenance(
        rag_results,
        memory_results
    )


    # ==================================================
    # FRONTIERES
    # ==================================================

    knows, limits = (
        build_knowledge_boundaries(
            rag_results,
            memory_results
        )
    )


    # ==================================================
    # CONFIANCE
    # ==================================================

    confidence_score = (
        rag_strength * 0.8
        + memory_strength * 0.2
    )

    confidence_score = min(
        confidence_score,
        0.95
    )

    confidence_percent = round(
        confidence_score * 100
    )


    # ==================================================
    # COUTS
    # ==================================================

    agents_data = [

        {
            "Agent":
                "Orchestrateur",
            "Fonction":
                "Planification",
            "Entrée":
                orchestrator_result["input_tokens"],
            "Sortie":
                orchestrator_result["output_tokens"]
        },

        {
            "Agent":
                "Chercheur",
            "Fonction":
                "Analyse scientifique",
            "Entrée":
                research_result["input_tokens"],
            "Sortie":
                research_result["output_tokens"]
        },

        {
            "Agent":
                "Critique scientifique",
            "Fonction":
                "Contrôle",
            "Entrée":
                critic_result["input_tokens"],
            "Sortie":
                critic_result["output_tokens"]
        },

        {
            "Agent":
                "Médiateur scientifique",
            "Fonction":
                "Médiation",
            "Entrée":
                communicator_result["input_tokens"],
            "Sortie":
                communicator_result["output_tokens"]
        }
    ]


    df = pd.DataFrame(
        agents_data
    )


    df["Tokens"] = (
        df["Entrée"]
        + df["Sortie"]
    )


    df["Coût (€)"] = df.apply(
        lambda row:
            calculate_cost_eur(
                row["Entrée"],
                row["Sortie"]
            ),
        axis=1
    )


    total_tokens = int(
        df["Tokens"].sum()
    )


    total_cost = float(
        df["Coût (€)"].sum()
    )


    df["Pondération (%)"] = (
        df["Tokens"]
        / total_tokens
        * 100
    ).round(1)


    df["Coût estimé (€)"] = (
        df["Coût (€)"]
        .map(
            lambda x:
                format_eur(
                    x,
                    6
                )
        )
    )


    # ==================================================
    # MEMORISATION
    # ==================================================

    sources_used = [

        {
            "source":
                result["source"],

            "chunk_id":
                result["chunk_id"],

            "score":
                result["score"]
        }

        for result in rag_results
    ]


    memory_entry = save_memory_entry(
        question=question,
        answer=communicator_result["text"],
        critique=critic_result["text"],
        sources=sources_used,
        total_tokens=total_tokens,
        total_cost=total_cost
    )


    # ==================================================
    # SESSION
    # ==================================================

    st.session_state.analysis_payload = {

        "question":
            question,

        "audience":
            audience,

        "response_level":
            response_level,

        "rag_results":
            rag_results,

        "memory_results":
            memory_results,

        "orchestrator_result":
            orchestrator_result,

        "research_result":
            research_result,

        "critic_result":
            critic_result,

        "communicator_result":
            communicator_result,

        "elapsed_time":
            elapsed_time,

        "confidence_percent":
            confidence_percent,

        "provenance_label":
            provenance_label,

        "provenance_detail":
            provenance_detail,

        "rag_strength":
            rag_strength,

        "memory_strength":
            memory_strength,

        "knows":
            knows,

        "limits":
            limits,

        "df":
            df,

        "total_tokens":
            total_tokens,

        "total_cost":
            total_cost,

        "memory_entry":
            memory_entry
    }

def speak_text_controls(
    text: str,
    play_label: str = "Lire la réponse",
    stop_label: str = "Arrêter"
):

    text_json = json.dumps(
        text,
        ensure_ascii=False
    )

    html = f"""
    <div style="
        display:flex;
        gap:10px;
        align-items:center;
        padding-top:4px;
    ">

        <button
            id="playButton"
            style="
                background:white;
                color:#701C3A;
                border:1px solid #701C3A;
                border-radius:8px;
                padding:10px 18px;
                font-weight:600;
                cursor:pointer;
                font-size:14px;
            "
        >
            ▶ {play_label}
        </button>

        <button
            id="stopButton"
            style="
                background:white;
                color:#666666;
                border:1px solid #B8B0AC;
                border-radius:8px;
                padding:10px 18px;
                font-weight:600;
                cursor:pointer;
                font-size:14px;
            "
        >
            ■ {stop_label}
        </button>

    </div>

    <script>

        const textToRead = {text_json};

        document
            .getElementById("playButton")
            .addEventListener(
                "click",
                function() {{

                    window.speechSynthesis.cancel();

                    const speech =
                        new SpeechSynthesisUtterance(
                            textToRead
                        );

                    speech.lang = "fr-FR";
                    speech.rate = 1.0;
                    speech.pitch = 1.0;
                    speech.volume = 1.0;

                    const voices =
                        window.speechSynthesis.getVoices();

                    const frenchVoice =
                        voices.find(
                            voice =>
                                voice.lang
                                &&
                                voice.lang
                                    .toLowerCase()
                                    .startsWith("fr")
                        );

                    if (frenchVoice) {{
                        speech.voice = frenchVoice;
                    }}

                    window.speechSynthesis.speak(
                        speech
                    );
                }}
            );


 document
            .getElementById("stopButton")
            .addEventListener(
                "click",
                function() {{

                    window.speechSynthesis.cancel();

                }}
            );

    </script>
    """

    st.iframe(
        html,
        height=65,
        width="stretch"
    )


# ==================================================
# RESULTATS
# ==================================================
if "analysis_payload" in st.session_state:

    payload = (
        st.session_state.analysis_payload
    )


    rag_results = (
        payload["rag_results"]
    )

    memory_results = (
        payload["memory_results"]
    )

    orchestrator_result = (
        payload["orchestrator_result"]
    )

    research_result = (
        payload["research_result"]
    )

    critic_result = (
        payload["critic_result"]
    )

    communicator_result = (
        payload["communicator_result"]
    )

    memory_entry = (
        payload["memory_entry"]
    )

    df = payload["df"]

    total_tokens = (
        payload["total_tokens"]
    )

    total_cost = (
        payload["total_cost"]
    )

    elapsed_time = (
        payload["elapsed_time"]
    )


# ==================================================
    # REPONSE — TOUJOURS VISIBLE
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">'
        'Résultat'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Réponse scientifique"
    )

    st.markdown(
        communicator_result["text"]
    )

    speak_text_controls(
        communicator_result["text"]
    )


    # ==================================================
    # MODE EXPERT UNIQUEMENT
    # ==================================================

    if display_mode == "Expert":


        # ==============================================
        # PROVENANCE
        # ==============================================

        st.markdown(
            '<div class="provenance-badge">'
            f'{payload["provenance_label"]}'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption(
            payload["provenance_detail"]
        )


        # ==============================================
        # SYNTHÈSE
        # ==============================================

        st.subheader(
            "Synthèse du traitement"
        )


        f1, f2, f3, f4, f5 = (
            st.columns(5)
        )


        flow_data = [

            (
                f1,
                "RAG",
                f"{len(rag_results)} passages"
            ),

            (
                f2,
                "Mémoire",
                f"{len(memory_results)} analyses"
            ),

            (
                f3,
                "Orchestrateur",
                "Plan établi"
            ),

            (
                f4,
                "Critique",
                "Contrôle effectué"
            ),

            (
                f5,
                "Médiateur",
                payload["audience"]
            )
        ]


        for (
            column,
            title,
            result
        ) in flow_data:

            with column:

                st.markdown(
                    '<div class="flow-card">'
                    f'<div class="flow-title">'
                    f'{title}'
                    '</div>'
                    f'<div class="flow-result">'
                    f'{result}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


        # ==============================================
        # REPERES
        # ==============================================

        st.subheader(
            "Repères"
        )


        r1, r2, r3, r4 = (
            st.columns(4)
        )


        r1.metric(
            "Sources documentaires",
            len(rag_results)
        )

        r2.metric(
            "Mémoires réutilisées",
            len(memory_results)
        )

        r3.metric(
            "Agents mobilisés",
            4
        )

        r4.metric(
            "Confiance documentaire",
            f'{payload["confidence_percent"]}%'
        )


        # ==============================================
        # INCERTITUDE
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Maîtrise de l’incertitude'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Ce que l'agent sait — "
            "et ce qu'il ne peut pas confirmer"
        )


        knows_col, limits_col = (
            st.columns(2)
        )


        with knows_col:

            st.markdown(
                '<div class="knowledge-card">'
                '<div class="card-title">'
                'Ce que l’agent peut étayer'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            for item in payload["knows"]:

                st.markdown(
                    f"- {item}"
                )


        with limits_col:

            st.markdown(
                '<div class="limit-card">'
                '<div class="card-title">'
                'Ce qui reste à confirmer'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            for item in payload["limits"]:

                st.markdown(
                    f"- {item}"
                )


        # ==============================================
        # SOURCES
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Sources'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Contexte documentaire"
        )


        if not rag_results:

            st.caption(
                "Aucun passage documentaire retrouvé."
            )


        for result in rag_results:

            score_percent = (
                result["score"]
                * 100
            )

            source_html = (
                '<div class="source-card">'
                '<div class="card-title">'
                'Source documentaire'
                '</div>'
                '<div class="card-meta">'
                f'{result["source"]}<br>'
                f'Pertinence : '
                f'{score_percent:.1f} %'
                '</div>'
                '</div>'
            )

            st.markdown(
                source_html,
                unsafe_allow_html=True
            )


            with st.expander(
                f"Voir le passage — chunk "
                f"{result['chunk_id']}"
            ):

                st.markdown(
                    result["text"]
                )


        # ==============================================
        # KNOWLEDGE MAP
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Knowledge Map'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Carte de connaissance et de provenance"
        )

        st.caption(
            "Vue simplifiée du chemin emprunté "
            "par l'information."
        )


        st.markdown(
            '<div class="provenance-map">'
            '<div class="map-row">'
            '<div class="map-node map-node-main">'
            'Question'
            '</div>'
            '<div class="map-arrow">→</div>'
            f'<div class="map-node">'
            f'RAG<br>{len(rag_results)} passages'
            f'</div>'
            '<div class="map-arrow">+</div>'
            f'<div class="map-node">'
            f'Mémoire<br>{len(memory_results)} analyses'
            f'</div>'
            '<div class="map-arrow">→</div>'
            '<div class="map-node">'
            'Orchestrateur'
            '</div>'
            '<div class="map-arrow">→</div>'
            '<div class="map-node">'
            'Chercheur'
            '</div>'
            '<div class="map-arrow">→</div>'
            '<div class="map-node">'
            'Critique'
            '</div>'
            '<div class="map-arrow">→</div>'
            '<div class="map-node">'
            'Médiateur'
            '</div>'
            '<div class="map-arrow">→</div>'
            '<div class="map-node map-node-main">'
            'Réponse'
            '</div>'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


        # ==============================================
        # CRITIQUE
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Contrôle'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Points de vigilance scientifique"
        )

        st.markdown(
            critic_result["text"]
        )


        # ==============================================
        # MEMOIRE UTILISEE
        # ==============================================

        st.subheader(
            "Mémoire utilisée pour cette analyse"
        )


        if not memory_results:

            st.caption(
                "Aucune mémoire pertinente retrouvée."
            )


        for result in memory_results:

            score_percent = (
                result["score"]
                * 100
            )

            status_label = (
                memory_status_label(
                    result["status"]
                )
            )


            memory_html = (
                '<div class="memory-card">'
                '<div class="card-title">'
                f'Analyse #{result["id"]}'
                '</div>'
                '<div class="card-meta">'
                f'Pertinence : '
                f'{score_percent:.1f} %<br>'
                f'Statut : {status_label}'
                '</div>'
                '</div>'
            )

            st.markdown(
                memory_html,
                unsafe_allow_html=True
            )


        # ==============================================
        # AGENTS
        # ==============================================

        st.subheader(
            "Travail des agents"
        )


        with st.expander(
            "Orchestrateur — plan scientifique"
        ):

            st.markdown(
                orchestrator_result["text"]
            )


        with st.expander(
            "Chercheur — analyse scientifique"
        ):

            st.markdown(
                research_result["text"]
            )


        with st.expander(
            "Critique scientifique — vérification"
        ):

            st.markdown(
                critic_result["text"]
            )


        with st.expander(
            "Médiateur scientifique — synthèse"
        ):

            st.markdown(
                communicator_result["text"]
            )


        # ==============================================
        # OBSERVABILITE
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Observabilité'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Performance de l'analyse"
        )


        m1, m2, m3, m4 = (
            st.columns(4)
        )


        m1.metric(
            "Agents",
            "4"
        )

        m2.metric(
            "Tokens",
            f"{total_tokens:,}"
            .replace(",", " ")
        )

        m3.metric(
            "Temps",
            f"{elapsed_time:.1f} s"
        )

        m4.metric(
            "Coût estimé",
            format_eur(
                total_cost
            )
        )


        display_df = df[
            [
                "Agent",
                "Fonction",
                "Entrée",
                "Sortie",
                "Tokens",
                "Pondération (%)",
                "Coût estimé (€)"
            ]
        ]


        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )


        left, right = (
            st.columns(2)
        )


        with left:

            st.caption(
                "Pondération du traitement"
            )

            chart_df = (
                df[
                    [
                        "Agent",
                        "Pondération (%)"
                    ]
                ]
                .set_index("Agent")
            )

            st.bar_chart(
                chart_df,
                height=220,
                width="stretch"
            )


        with right:

            st.caption(
                "Coût estimé par agent (€)"
            )

            cost_chart = (
                df[
                    [
                        "Agent",
                        "Coût (€)"
                    ]
                ]
                .set_index("Agent")
            )

            st.bar_chart(
                cost_chart,
                height=220,
                width="stretch"
            )


        # ==============================================
        # DASHBOARD GLOBAL
        # ==============================================

        st.divider()

        stats = get_memory_statistics()


        st.markdown(
            '<div class="section-kicker">'
            'Écosystème'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Tableau de bord global"
        )


        g1, g2, g3, g4 = (
            st.columns(4)
        )


        g1.metric(
            "Analyses mémorisées",
            stats["total"]
        )

        g2.metric(
            "Validées",
            stats["validated"]
        )

        g3.metric(
            "En attente",
            stats["pending"]
        )

        g4.metric(
            "Rejetées",
            stats["rejected"]
        )


        g5, g6 = (
            st.columns(2)
        )


        g5.metric(
            "Tokens cumulés",
            f'{stats["tokens"]:,}'
            .replace(",", " ")
        )

        g6.metric(
            "Coût cumulé enregistré",
            format_eur(
                stats["cost"]
            )
        )


        if stats["total"] > 0:

            governance_df = pd.DataFrame(
                {
                    "Statut": [
                        "Validées",
                        "En attente",
                        "Rejetées"
                    ],

                    "Nombre": [
                        stats["validated"],
                        stats["pending"],
                        stats["rejected"]
                    ]
                }
            ).set_index(
                "Statut"
            )

            st.caption(
                "Répartition de la mémoire"
            )

            st.bar_chart(
                governance_df,
                height=220,
                width="stretch"
            )


        # ==============================================
        # HISTORIQUE + VALIDATION
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Historique'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Analyses mémorisées"
        )


        memory_history = (
            stats["memory"]
        )


        if not memory_history:

            st.caption(
                "Aucune analyse mémorisée."
            )

        else:

            status_filter = st.selectbox(
                "Filtrer les analyses",
                [
                    "Toutes",
                    "En attente de validation",
                    "Validées",
                    "Rejetées"
                ]
            )


            def memory_matches_filter(
                entry,
                selected_filter
            ):

                status_value = entry.get(
                    "status",
                    "generated_unvalidated"
                )

                if selected_filter == "Toutes":
                    return True

                if (
                    selected_filter
                    == "En attente de validation"
                ):

                    return (
                        status_value
                        == "generated_unvalidated"
                    )

                if selected_filter == "Validées":

                    return (
                        status_value
                        == "generated_validated"
                    )

                if selected_filter == "Rejetées":

                    return (
                        status_value
                        == "generated_rejected"
                    )

                return True


            filtered_history = [
                item
                for item in reversed(
                    memory_history
                )
                if memory_matches_filter(
                    item,
                    status_filter
                )
            ]


            if not filtered_history:

                st.caption(
                    "Aucune analyse pour ce filtre."
                )


            for item in filtered_history:

                history_id = (
                    item.get("id")
                )

                history_status = (
                    item.get(
                        "status",
                        "generated_unvalidated"
                    )
                )

                history_status_label = (
                    memory_status_label(
                        history_status
                    )
                )

                history_question = (
                    item.get(
                        "question",
                        ""
                    )
                )


                with st.expander(
                    f"#{history_id} — "
                    f"{history_status_label} — "
                    f"{history_question}"
                ):


                    st.markdown(
                        "**Question**"
                    )

                    st.write(
                        history_question
                    )


                    st.markdown(
                        "**Réponse mémorisée**"
                    )

                    st.markdown(
                        item.get(
                            "answer",
                            ""
                        )
                    )


                    history_critique = (
                        item.get(
                            "critique",
                            ""
                        )
                    )


                    if history_critique:

                        with st.expander(
                            "Voir la critique scientifique"
                        ):

                            st.markdown(
                                history_critique
                            )


                    h1, h2, h3 = (
                        st.columns(3)
                    )


                    history_tokens = (
                        item.get(
                            "total_tokens",
                            0
                        ) or 0
                    )

                    history_cost = (
                        item.get(
                            "total_cost",
                            0
                        ) or 0
                    )


                    h1.metric(
                        "Statut",
                        history_status_label
                    )

                    h2.metric(
                        "Tokens",
                        f"{history_tokens:,}"
                        .replace(",", " ")
                    )

                    h3.metric(
                        "Coût",
                        format_eur(
                            history_cost
                        )
                    )


                    if (
                        history_status
                        == "generated_unvalidated"
                    ):

                        st.markdown(
                            "**Décision de validation**"
                        )


                        approve_col, reject_col, spacer = (
                            st.columns(
                                [1, 1, 4]
                            )
                        )


                        with approve_col:

                            if st.button(
                                "Valider",
                                key=
                                    f"history_validate_"
                                    f"{history_id}"
                            ):

                                result = (
                                    validate_memory_entry(
                                        history_id
                                    )
                                )

                                if result["success"]:

                                    if (
                                        memory_entry["id"]
                                        == history_id
                                    ):

                                        memory_entry["status"] = (
                                            "generated_validated"
                                        )

                                    st.rerun()


                        with reject_col:

                            if st.button(
                                "Rejeter",
                                key=
                                    f"history_reject_"
                                    f"{history_id}"
                            ):

                                result = (
                                    reject_memory_entry(
                                        history_id
                                    )
                                )

                                if result["success"]:

                                    if (
                                        memory_entry["id"]
                                        == history_id
                                    ):

                                        memory_entry["status"] = (
                                            "generated_rejected"
                                        )

                                    st.rerun()


                    elif (
                        history_status
                        == "generated_validated"
                    ):

                        st.success(
                            "Cette analyse est validée."
                        )


                    elif (
                        history_status
                        == "generated_rejected"
                    ):

                        st.warning(
                            "Cette analyse a été rejetée."
                        )


        # ==============================================
        # PROJECTION
        # ==============================================

        st.subheader(
            "Projection de coût"
        )


        p1, p2, p3 = (
            st.columns(3)
        )


        p1.metric(
            "1 analyse",
            format_eur(
                total_cost
            )
        )

        p2.metric(
            "100 analyses",
            format_eur(
                total_cost * 100,
                2
            )
        )

        p3.metric(
            "1 000 analyses",
            format_eur(
                total_cost * 1000,
                2
            )
        )


        # ==============================================
        # CAPITALISATION
        # ==============================================

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Capitalisation'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Mémoire de l'écosystème"
        )


        memory_id = (
            memory_entry["id"]
        )


        if memory_entry["duplicate"]:

            duplicate_percent = (
                memory_entry[
                    "duplicate_score"
                ]
                * 100
            )

            st.info(
                f"Cette analyse est déjà présente "
                f"en mémoire. "
                f"Similarité : "
                f"{duplicate_percent:.1f} %. "
                f"Mémoire existante : #{memory_id}. "
                f"Aucun doublon créé."
            )

        else:

            st.success(
                f"Nouvelle analyse mémorisée — "
                f"#{memory_id}."
            )


        current_status = (
            memory_entry["status"]
        )


        st.caption(
            f"Statut : "
            f"{memory_status_label(current_status)}"
        )


        if (
            current_status
            == "generated_unvalidated"
        ):

            st.markdown(
                "**Validation humaine**"
            )


            validate_col, reject_col, spacer = (
                st.columns(
                    [1, 1, 4]
                )
            )


            with validate_col:

                st.button(
                    "Valider",
                    key=
                        f"validate_{memory_id}",
                    on_click=
                        validate_memory_callback,
                    args=(
                        memory_id,
                    )
                )


            with reject_col:

                st.button(
                    "Rejeter",
                    key=
                        f"reject_{memory_id}",
                    on_click=
                        reject_memory_callback,
                    args=(
                        memory_id,
                    )
                )


        elif (
            current_status
            == "generated_validated"
        ):

            st.success(
                "Cette mémoire a été validée."
            )


        elif (
            current_status
            == "generated_rejected"
        ):

            st.warning(
                "Cette mémoire a été rejetée."
            )


    # ==================================================
    # FEEDBACK — VISITEUR ET EXPERT
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">'
        'Votre avis'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "Cette réponse vous a-t-elle été utile ?"
    )


    feedback_col1, feedback_col2, spacer = (
        st.columns(
            [1, 1, 4]
        )
    )


    with feedback_col1:

        st.button(
            "Utile",
            key="feedback_useful",
            on_click=
                useful_feedback_callback
        )


    with feedback_col2:

        st.button(
            "À revoir",
            key="feedback_review",
            on_click=
                review_feedback_callback
        )


    if st.session_state.get(
        "user_feedback"
    ):

        st.caption(
            f"Votre avis : "
            f"{st.session_state.user_feedback}"
        )
