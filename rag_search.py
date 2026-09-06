from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import math
from pathlib import Path


INDEX_FILE = Path("data/index.json")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    norm_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    norm_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def load_index():
    return json.loads(
        INDEX_FILE.read_text(encoding="utf-8")
    )


def search(question, top_k=2):

    index = load_index()

    question_embedding = create_embedding(question)

    results = []

    for item in index:

        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "text": item["text"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


if __name__ == "__main__":

    question = input(
        "\nQuestion à rechercher dans le RAG : "
    )

    results = search(question)

    print("\n=== PASSAGES RETROUVÉS ===")

    for result in results:

        print(
            f"\nSource : {result['source']} "
            f"| Chunk : {result['chunk_id']} "
            f"| Score : {result['score']:.3f}"
        )

        print(result["text"])