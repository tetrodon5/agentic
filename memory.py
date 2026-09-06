from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import os
import json
import math


MEMORY_FILE = Path("data/memory.json")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def load_memory():
    if not MEMORY_FILE.exists():
        return []

    return json.loads(
        MEMORY_FILE.read_text(encoding="utf-8")
    )


def save_memory_entry(
    question,
    answer,
    critique,
    sources=None,
    total_tokens=None,
    total_cost=None
):
    memory = load_memory()

    content_for_embedding = (
        f"Question : {question}\n\n"
        f"Réponse : {answer}"
    )

    embedding = create_embedding(
        content_for_embedding
    )

    entry = {
        "id": len(memory) + 1,
        "created_at": datetime.now().isoformat(),
        "source_type": "generated_memory",
        "status": "generated_unvalidated",
        "question": question,
        "answer": answer,
        "critique": critique,
        "sources": sources or [],
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "embedding": embedding
    }

    memory.append(entry)

    MEMORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    MEMORY_FILE.write_text(
        json.dumps(
            memory,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    return entry


def search_memory(
    question,
    top_k=2,
    minimum_score=0.75
):
    memory = load_memory()

    if not memory:
        return []

    question_embedding = create_embedding(
        question
    )

    results = []

    for entry in memory:

        score = cosine_similarity(
            question_embedding,
            entry["embedding"]
        )

        if score >= minimum_score:
            results.append({
                "id": entry["id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "critique": entry["critique"],
                "status": entry["status"],
                "source_type": entry["source_type"],
                "score": score
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    question = input(
        "\nQuestion à rechercher dans la mémoire : "
    )

    results = search_memory(
        question,
        top_k=2,
        minimum_score=0.70
    )

    print("\n=== MÉMOIRE RETROUVÉE ===")

    if not results:
        print("Aucune mémoire suffisamment proche.")

    for result in results:

        print(
            f"\nID : {result['id']} "
            f"| Score : {result['score']:.3f} "
            f"| Statut : {result['status']}"
        )

        print(
            f"Question mémorisée : "
            f"{result['question']}"
        )