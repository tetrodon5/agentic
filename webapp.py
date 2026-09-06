import streamlit as st
import os
import time
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent

from rag_search import search
from memory import save_memory_entry, search_memory


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

st.set_page_config(
    page_title="Agent scientifique - Cité du Vin",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


# ==================================================
# TARIFICATION GPT-5.6 LUNA
# ==================================================

INPUT_PRICE_PER_MILLION = 0.20
OUTPUT_PRICE_PER_MILLION = 1.20


def calculate_cost(input_tokens, output_tokens):
    input_cost = (
        input_tokens / 1_000_000
    ) * INPUT_PRICE_PER_MILLION

    output_cost = (
        output_tokens / 1_000_000
    ) * OUTPUT_PRICE_PER_MILLION

    return input_cost + output_cost


# ==================================================
# DESIGN
# ==================================================

st.markdown(
    """
    <style>

    :root {
        --wine: #701C3A;
        --wine-dark: #4E1328;
        --cream-light: #FCFAF7;
        --ink: #242A35;
        --border: #D9CEC7;
    }

    .stApp {
        background: var(--cream-light);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
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
        font-size: 0.82rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        font-size: 1.45rem;
        line-height: 1.3;
        color: var(--ink);
        font-weight: 600;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
    }

    .hero-copy {
        color: #4F4D4C;
        font-size: 1rem;
        line-height: 1.7;
        max-width: 760px;
    }

    .hero-tags {
        margin-top: 1.4rem;
        color: var(--wine);
        font-size: 0.82rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        font-weight: 600;
    }

    .demo-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        min-height: 210px;
        box-shadow: 0 6px 22px rgba(70, 43, 53, 0.05);
    }

    .demo-card-title {
        color: var(--wine);
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .demo-card-label {
        color: #8C8987;
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .section-kicker {
        color: var(--wine);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--wine);
        background: white;
        color: var(--wine);
        font-weight: 600;
        min-height: 46px;
    }

    .stButton > button:hover {
        background: #F5EBEF;
        border-color: var(--wine-dark);
        color: var(--wine-dark);
    }

    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
        background: #F3F5F7;
        border: 1px solid #E4E6E8;
        min-height: 100px;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 18px rgba(60, 40, 45, 0.04);
    }

    hr {
        border-color: #DDD4CE;
        margin-top: 1.8rem;
        margin-bottom: 1.8rem;
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
# QUESTIONS SUGGÉRÉES
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
    st.session_state.suggestions = random.sample(question_bank, 3)

if "question" not in st.session_state:
    st.session_state.question = ""


# ==================================================
# HERO
# ==================================================

hero_left, hero_right = st.columns([2.5, 1])

with hero_left:

    st.markdown(
        '<div class="hero-kicker">FCCV | DÉMONSTRATION IA</div>',
        unsafe_allow_html=True
    )

    st.title("Agent scientifique")

    st.markdown(
        '<div class="hero-subtitle">'
        'Recherche, contrôle scientifique et médiation'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-copy">'
        "Une démonstration d'écosystème agentique appliqué aux "
        "connaissances scientifiques et culturelles du vin. "
        "Le système combine recherche documentaire, mémoire générée, "
        "contrôle scientifique et médiation."
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
        """
        <div class="demo-card">
            <div class="demo-card-label">Démonstrateur</div>
            <div class="demo-card-title">Écosystème multi-agents + RAG</div>
            <div style="line-height:1.9;color:#3E3A3A;">
                RAG documentaire<br>
                Mémoire sémantique<br>
                Orchestration<br>
                Vérification<br>
                Médiation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()


# ==================================================
# QUESTIONS
# ==================================================

st.markdown(
    '<div class="section-kicker">Explorer</div>',
    unsafe_allow_html=True
)

st.header("Que souhaitez-vous explorer ?")

st.caption(
    "Choisissez une question ou interrogez directement l'agent scientifique."
)

q1, q2, q3 = st.columns(3)

with q1:
    if st.button(
        st.session_state.suggestions[0],
        width="stretch"
    ):
        st.session_state.question = st.session_state.suggestions[0]

with q2:
    if st.button(
        st.session_state.suggestions[1],
        width="stretch"
    ):
        st.session_state.question = st.session_state.suggestions[1]

with q3:
    if st.button(
        st.session_state.suggestions[2],
        width="stretch"
    ):
        st.session_state.question = st.session_state.suggestions[2]


question = st.text_area(
    "Votre question scientifique",
    value=st.session_state.question,
    placeholder="Posez librement votre question...",
    height=100
)

launch = st.button(
    "Lancer l'analyse scientifique",
    type="primary"
)


# ==================================================
# EXECUTION
# ==================================================

if launch:

    if not question.strip():
        st.warning("Veuillez saisir ou sélectionner une question.")
        st.stop()

    start_time = time.time()


    # ==================================================
    # RAG DOCUMENTAIRE
    # ==================================================

    rag_results = search(
        question,
        top_k=2
    )

    rag_context = ""

    for result in rag_results:
        rag_context += (
            f"\nSource : {result['source']} "
            f"| Chunk : {result['chunk_id']} "
            f"| Score : {result['score']:.3f}\n"
            f"{result['text']}\n"
        )


    # ==================================================
    # MÉMOIRE SÉMANTIQUE
    # ==================================================

    memory_results = search_memory(
        question,
        top_k=2,
        minimum_score=0.70
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


    # ==================================================
    # AGENTS
    # ==================================================

    with st.status(
        "Analyse agentique en cours...",
        expanded=True
    ) as status:

        st.write("RAG — recherche des passages documentaires")
        st.write(f"{len(rag_results)} passages retrouvés")

        st.write("Mémoire — recherche des analyses précédentes")
        st.write(f"{len(memory_results)} mémoire(s) proche(s) retrouvée(s)")

        st.write("Orchestrateur — analyse et planification")
        orchestrator_result = orchestrator.run(question)

        st.write("Chercheur — investigation scientifique")
        research_result = researcher.run(
            question,
            orchestrator_result["text"],
            rag_context,
            memory_context
        )

        st.write("Critique scientifique — vérification et challenge")
        critic_result = critic.run(
            question,
            research_result["text"]
        )

        st.write("Médiateur scientifique — synthèse et vulgarisation")
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


    # ==================================================
    # COUTS
    # ==================================================

    agents_data = [
        {
            "Agent": "Orchestrateur",
            "Fonction": "Planification",
            "Entrée": orchestrator_result["input_tokens"],
            "Sortie": orchestrator_result["output_tokens"]
        },
        {
            "Agent": "Chercheur",
            "Fonction": "Analyse scientifique",
            "Entrée": research_result["input_tokens"],
            "Sortie": research_result["output_tokens"]
        },
        {
            "Agent": "Critique scientifique",
            "Fonction": "Contrôle",
            "Entrée": critic_result["input_tokens"],
            "Sortie": critic_result["output_tokens"]
        },
        {
            "Agent": "Médiateur scientifique",
            "Fonction": "Vulgarisation",
            "Entrée": communicator_result["input_tokens"],
            "Sortie": communicator_result["output_tokens"]
        }
    ]

    df = pd.DataFrame(agents_data)

    df["Tokens"] = df["Entrée"] + df["Sortie"]

    df["Coût ($)"] = df.apply(
        lambda row: calculate_cost(
            row["Entrée"],
            row["Sortie"]
        ),
        axis=1
    )

    total_tokens = int(df["Tokens"].sum())
    total_cost = float(df["Coût ($)"].sum())

    df["Pondération (%)"] = (
        df["Tokens"] / total_tokens * 100
    ).round(1)

    df["Coût estimé ($)"] = df["Coût ($)"].map(
        lambda x: f"{x:.6f}"
    )


    # ==================================================
    # REPONSE
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Résultat</div>',
        unsafe_allow_html=True
    )

    st.header("Réponse scientifique")

    st.markdown(communicator_result["text"])


    # ==================================================
    # SOURCES RAG
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Grounding</div>',
        unsafe_allow_html=True
    )

    st.header("Sources documentaires utilisées")

    for result in rag_results:

        score_percent = result["score"] * 100

        with st.expander(
            f"{result['source']} — pertinence {score_percent:.1f}%"
        ):
            st.caption(
                f"Chunk {result['chunk_id']}"
            )
            st.markdown(result["text"])


    # ==================================================
    # MEMOIRE UTILISEE
    # ==================================================

    st.subheader("Mémoire réutilisée")

    if not memory_results:

        st.caption(
            "Aucune analyse antérieure suffisamment proche n'a été utilisée."
        )

    else:

        for result in memory_results:

            score_percent = result["score"] * 100

            with st.expander(
                f"Mémoire #{result['id']} — pertinence {score_percent:.1f}%"
            ):
                st.write(
                    f"Statut : {result['status']}"
                )
                st.write(
                    f"Question précédente : {result['question']}"
                )

                st.warning(
                    "Cette mémoire est générée par l'IA et n'est pas "
                    "considérée comme une source documentaire validée."
                )


    # ==================================================
    # OBSERVABILITE
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Pilotage du système</div>',
        unsafe_allow_html=True
    )

    st.header("Observabilité & coûts")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Agents",
        "4"
    )

    m2.metric(
        "Tokens",
        f"{total_tokens:,}".replace(",", " ")
    )

    m3.metric(
        "Temps",
        f"{elapsed_time:.1f} s"
    )

    m4.metric(
        "Coût estimé",
        f"${total_cost:.5f}"
    )


    # ==================================================
    # TABLEAU
    # ==================================================

    display_df = df[
        [
            "Agent",
            "Fonction",
            "Entrée",
            "Sortie",
            "Tokens",
            "Pondération (%)",
            "Coût estimé ($)"
        ]
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        height=180
    )


    # ==================================================
    # GRAPHIQUES
    # ==================================================

    left, right = st.columns(2)

    with left:

        st.caption("Pondération du traitement")

        chart_df = (
            df[["Agent", "Pondération (%)"]]
            .set_index("Agent")
        )

        st.bar_chart(
            chart_df,
            height=220,
            width="stretch"
        )

    with right:

        st.caption("Coût estimé par agent")

        cost_chart = (
            df[["Agent", "Coût ($)"]]
            .set_index("Agent")
        )

        st.bar_chart(
            cost_chart,
            height=220,
            width="stretch"
        )


    # ==================================================
    # PROJECTION
    # ==================================================

    st.subheader("Projection de coût")

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "1 analyse",
        f"${total_cost:.5f}"
    )

    p2.metric(
        "100 analyses",
        f"${total_cost * 100:.2f}"
    )

    p3.metric(
        "1 000 analyses",
        f"${total_cost * 1000:.2f}"
    )


    # ==================================================
    # DETAILS
    # ==================================================

    st.divider()

    st.markdown(
        '<div class="section-kicker">Transparence</div>',
        unsafe_allow_html=True
    )

    st.header("Explorer le travail des agents")

    with st.expander("Orchestrateur — plan scientifique"):
        st.markdown(orchestrator_result["text"])

    with st.expander("Chercheur — analyse scientifique"):
        st.markdown(research_result["text"])

    with st.expander("Critique scientifique — vérification"):
        st.markdown(critic_result["text"])

    with st.expander("Médiateur scientifique — synthèse"):
        st.markdown(communicator_result["text"])


    # ==================================================
    # SAUVEGARDE MEMOIRE
    # ==================================================

    sources_used = [
        {
            "source": result["source"],
            "chunk_id": result["chunk_id"],
            "score": result["score"]
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

    st.caption(
        f"Nouvelle analyse mémorisée — ID {memory_entry['id']} "
        f"| statut : {memory_entry['status']}"
    )