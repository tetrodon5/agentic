import json
import numpy as np
from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

INDEX_FILE = Path("data/index.json")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


_index_data = None
_embeddings_matrix = None


def load_index():

    global _index_data
    global _embeddings_matrix

    if _index_data is None:

        with open(
            INDEX_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            _index_data = json.load(f)

        _embeddings_matrix = np.array(
            [
                item["embedding"]
                for item in _index_data
            ],
            dtype=np.float32
        )

        norms = np.linalg.norm(
            _embeddings_matrix,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1

        _embeddings_matrix = (
            _embeddings_matrix
            / norms
        )

    return (
        _index_data,
        _embeddings_matrix
    )


def create_query_embedding(
    question
):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    vector = np.array(
        response.data[0].embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        vector
    )

    if norm != 0:
        vector = vector / norm

    return vector


def search(
    question,
    top_k=2
):

    index_data, matrix = (
        load_index()
    )

    query_vector = (
        create_query_embedding(
            question
        )
    )

    scores = (
        matrix
        @ query_vector
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]


    results = []

    for idx in top_indices:

        item = index_data[
            int(idx)
        ]

        results.append(
            {
                "source":
                    item["source"],

                "chunk_id":
                    item["chunk_id"],

                "text":
                    item["text"],

                "score":
                    float(
                        scores[idx]
                    )
            }
        )

    return results