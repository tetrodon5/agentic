import os
import time
import random

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Agent scientifique - Cité du Vin",
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
    save_memory_entry,
    search_memory,
    validate_memory_entry,
    reject_memory_entry
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
# MEMOIRE
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

        if "analysis_payload" in st.session_state:

            entry = (
                st.session_state
                .analysis_payload
                .get("memory_entry")
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

        if "analysis_payload" in st.session_state:

            entry = (
                st.session_state
                .analysis_payload
                .get("memory_entry")
            )

            if (
                entry
                and entry["id"] == memory_id
            ):

                entry["status"] = (
                    "generated_rejected"
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
        and rag_strength >= memory_strength * 0.85
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
            "Peu de contexte documentaire pertinent ; "
            "la mémoire générée contribue davantage."
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
# CONSIGNE DE MEDIATION
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

orchestrator = OrchestratorAgent(client)
researcher = ResearcherAgent(client)
critic = CriticAgent(client)
communicator = CommunicatorAgent(client)


# ==================================================
# SESSION
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
# MESSAGE MEMOIRE
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

    del st.session_state["memory_message"]
    del st.session_state["memory_message_type"]


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
        "Une démonstration d'écosystème agentique appliqué "
        "aux connaissances scientifiques et culturelles du vin. "
        "Le système combine recherche documentaire, mémoire sémantique, "
        "contrôle scientifique et médiation adaptée au public."
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
        '<div class="demo-card-label">Démonstrateur</div>'
        '<div class="demo-card-title">'
        'Écosystème multi-agents + RAG'
        '</div>'
        '<div style="line-height:1.9;color:#3E3A3A;">'
        'RAG documentaire<br>'
        'Mémoire sémantique<br>'
        'Orchestration<br>'
        'Vérification scientifique<br>'
        'Médiation adaptative'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


st.divider()


# ==================================================
# MODE
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


mode_col, audience_col, level_col = st.columns(3)


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
    "Choisissez une question ou interrogez directement l'agent scientifique."
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


    start_time = time.time()

    agent_question = build_agent_question(
        question,
        audience,
        response_level
    )


    # ==================================================
    # LIVE FLOW
    # ==================================================

    st.markdown(
        '<div class="section-kicker">'
        'Live Agent Flow'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader(
        "L'écosystème travaille"
    )


    with st.status(
        "Analyse agentique en cours...",
        expanded=True
    ) as status:


        # ==============================================
        # RAG
        # ==============================================

        st.markdown(
            "**RAG documentaire**"
        )

        rag_results = search(
            question,
            top_k=2
        )

        st.caption(
            f"{len(rag_results)} passage(s) pertinent(s) retrouvé(s)"
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

        st.markdown(
            "**Mémoire sémantique**"
        )

        memory_results = search_memory(
            question,
            top_k=2,
            minimum_score=0.70
        )

        st.caption(
            f"{len(memory_results)} mémoire(s) proche(s) retrouvée(s)"
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

        st.markdown(
            "**Orchestrateur**"
        )

        orchestrator_result = (
            orchestrator.run(
                agent_question
            )
        )

        st.caption(
            "Plan scientifique établi"
        )


        # ==============================================
        # RESEARCHER
        # ==============================================

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

        st.caption(
            "Analyse scientifique terminée"
        )


        # ==============================================
        # CRITIC
        # ==============================================

        st.markdown(
            "**Critique scientifique**"
        )

        critic_result = (
            critic.run(
                agent_question,
                research_result["text"]
            )
        )

        st.caption(
            "Vérification et nuances terminées"
        )


        # ==============================================
        # COMMUNICATOR
        # ==============================================

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

        st.caption(
            f"Réponse adaptée au public : {audience}"
        )


        status.update(
            label="Analyse terminée",
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
    # SCORE DOCUMENTAIRE
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

        for result
        in rag_results
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

        "df":
            df,

        "total_tokens":
            total_tokens,

        "total_cost":
            total_cost,

        "memory_entry":
            memory_entry
    }


# ==================================================
# RESULTATS
# ==================================================

if "analysis_payload" in st.session_state:

    payload = (
        st.session_state.analysis_payload
    )


    rag_results = payload["rag_results"]
    memory_results = payload["memory_results"]

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
    # REPONSE
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


    # provenance

    st.markdown(
        '<div class="provenance-badge">'
        f'{payload["provenance_label"]}'
        '</div>',
        unsafe_allow_html=True
    )

    st.caption(
        payload["provenance_detail"]
    )


    st.markdown(
        communicator_result["text"]
    )


    # ==================================================
    # FLOW SYNTHETIQUE
    # ==================================================

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
                f'<div class="flow-title">{title}</div>'
                f'<div class="flow-result">{result}</div>'
                '</div>',
                unsafe_allow_html=True
            )


    # ==================================================
    # INDICATEURS
    # ==================================================

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


    # ==================================================
    # SOURCES
    # ==================================================

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


    for result in rag_results:

        score_percent = (
            result["score"] * 100
        )

        source_html = (
            '<div class="source-card">'
            '<div class="card-title">'
            'Source documentaire'
            '</div>'
            '<div class="card-meta">'
            f'{result["source"]}<br>'
            f'Pertinence : {score_percent:.1f} %'
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


    # ==================================================
    # MODE EXPERT
    # ==================================================

    if display_mode == "Expert":

        st.divider()

        st.markdown(
            '<div class="section-kicker">'
            'Mode Expert'
            '</div>',
            unsafe_allow_html=True
        )

        st.header(
            "Analyse interne de l'écosystème"
        )


        # ==============================================
        # MEMOIRE
        # ==============================================

        st.subheader(
            "Mémoire générée"
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
                'Mémoire générée'
                '</div>'
                '<div class="card-meta">'
                f'Analyse #{result["id"]}<br>'
                f'Pertinence : {score_percent:.1f} %<br>'
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

        st.subheader(
            "Observabilité & coûts"
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


    # ==================================================
    # CAPITALISATION
    # ==================================================

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
            f"Cette analyse est déjà présente en mémoire. "
            f"Similarité : {duplicate_percent:.1f} %. "
            f"Mémoire existante : #{memory_id}. "
            f"Aucun doublon créé."
        )

    else:

        st.success(
            f"Nouvelle analyse mémorisée — #{memory_id}."
        )


    current_status = (
        memory_entry["status"]
    )


    st.caption(
        f"Statut : "
        f"{memory_status_label(current_status)}"
    )


    # ==================================================
    # VALIDATION
    # ==================================================

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