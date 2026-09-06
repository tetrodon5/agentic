from dotenv import load_dotenv
from openai import OpenAI
import os

from agents.orchestrator import OrchestratorAgent
from agents.researcher import ResearcherAgent
from agents.critic import CriticAgent
from agents.communicator import CommunicatorAgent

from rag_search import search
from memory import save_memory_entry, search_memory


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Agents
# --------------------------------------------------

orchestrator = OrchestratorAgent(client)
researcher = ResearcherAgent(client)
critic = CriticAgent(client)
communicator = CommunicatorAgent(client)


# --------------------------------------------------
# Question utilisateur
# --------------------------------------------------

question = input(
    "\nPosez votre question scientifique : "
)


# --------------------------------------------------
# 1. Recherche RAG documentaire
# --------------------------------------------------

rag_results = search(
    question,
    top_k=2,
    minimum_score=0.55
)

rag_context = ""

for result in rag_results:
    rag_context += (
        f"\nSource : {result['source']} "
        f"| Chunk : {result['chunk_id']} "
        f"| Score : {result['score']:.3f}\n"
        f"{result['text']}\n"
    )


print("\n=== RAG DOCUMENTAIRE ===")

for result in rag_results:
    print(
        f"{result['source']} "
        f"| Chunk {result['chunk_id']} "
        f"| Score {result['score']:.3f}"
    )


# --------------------------------------------------
# 2. Recherche dans la mémoire générée
# --------------------------------------------------

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


print("\n=== MÉMOIRE GÉNÉRÉE ===")

if not memory_results:
    print("Aucune mémoire suffisamment proche.")

for result in memory_results:
    print(
        f"ID {result['id']} "
        f"| Score {result['score']:.3f} "
        f"| Statut {result['status']}"
    )


# --------------------------------------------------
# 3. Orchestrateur
# --------------------------------------------------

orchestrator_result = orchestrator.run(
    question
)

print("\n=== ORCHESTRATEUR ===")
print(orchestrator_result["text"])


# --------------------------------------------------
# 4. Chercheur avec RAG + mémoire
# --------------------------------------------------

research_result = researcher.run(
    question,
    orchestrator_result["text"],
    rag_context,
    memory_context
)

print("\n=== CHERCHEUR ===")
print(research_result["text"])


# --------------------------------------------------
# 5. Critique scientifique
# --------------------------------------------------

critic_result = critic.run(
    question,
    research_result["text"],
    rag_context=rag_context
)

print("\n=== CRITIQUE SCIENTIFIQUE ===")
print(critic_result["text"])


# --------------------------------------------------
# 6. Médiateur scientifique
# --------------------------------------------------

communicator_result = communicator.run(
    question,
    research_result["text"],
    critic_result["text"]
)

print("\n=== RÉPONSE FINALE ===")
print(communicator_result["text"])


# --------------------------------------------------
# 7. Consommation
# --------------------------------------------------

total_tokens = (
    orchestrator_result["total_tokens"]
    + research_result["total_tokens"]
    + critic_result["total_tokens"]
    + communicator_result["total_tokens"]
)

print("\n=== CONSOMMATION ===")

print(
    "Orchestrateur :",
    orchestrator_result["total_tokens"],
    "tokens"
)

print(
    "Chercheur :",
    research_result["total_tokens"],
    "tokens"
)

print(
    "Critique :",
    critic_result["total_tokens"],
    "tokens"
)

print(
    "Médiateur :",
    communicator_result["total_tokens"],
    "tokens"
)

print(
    "Total :",
    total_tokens,
    "tokens"
)


# --------------------------------------------------
# 8. Sources documentaires utilisées
# --------------------------------------------------

sources_used = [
    {
        "source": result["source"],
        "chunk_id": result["chunk_id"],
        "score": result["score"]
    }
    for result in rag_results
]


# --------------------------------------------------
# 9. Sauvegarde de la nouvelle réponse
# --------------------------------------------------

memory_entry = save_memory_entry(
    question=question,
    answer=communicator_result["text"],
    critique=critic_result["text"],
    sources=sources_used,
    total_tokens=total_tokens
)


print("\n=== MÉMOIRE ===")

print(
    f"Réponse enregistrée dans la mémoire "
    f"(ID {memory_entry['id']})"
)
