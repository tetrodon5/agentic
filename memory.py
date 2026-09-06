from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from datetime import datetime
import os
import json
import math


# ==================================================
# CONFIGURATION
# ==================================================

MEMORY_FILE = Path("data/memory.json")

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ==================================================
# EMBEDDINGS
# ==================================================

def create_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ==================================================
# SIMILARITE COSINUS
# ==================================================

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(
        sum(
            a * a
            for a in vector_a
        )
    )

    norm_b = math.sqrt(
        sum(
            b * b
            for b in vector_b
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (
        norm_a * norm_b
    )


# ==================================================
# CHARGEMENT MEMOIRE
# ==================================================

def load_memory():

    if not MEMORY_FILE.exists():
        return []

    try:

        return json.loads(
            MEMORY_FILE.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError:

        return []


# ==================================================
# ECRITURE MEMOIRE
# ==================================================

def write_memory(memory):

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


# ==================================================
# RECHERCHE MEMOIRE
# ==================================================

def search_memory(
    question,
    top_k=2,
    minimum_score=0.75,
    include_unvalidated=True,
    include_rejected=False
):

    memory = load_memory()

    if not memory:
        return []

    question_embedding = create_embedding(
        question
    )

    results = []

    for entry in memory:

        status = entry.get(
            "status",
            "generated_unvalidated"
        )

        # ------------------------------------------
        # FILTRAGE DES STATUTS
        # ------------------------------------------

        if (
            status == "generated_unvalidated"
            and not include_unvalidated
        ):
            continue

        if (
            status == "generated_rejected"
            and not include_rejected
        ):
            continue

        embedding = entry.get(
            "embedding"
        )

        if not embedding:
            continue

        score = cosine_similarity(
            question_embedding,
            embedding
        )

        if score >= minimum_score:

            results.append(
                {
                    "id": entry["id"],
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "critique": entry.get(
                        "critique",
                        ""
                    ),
                    "status": status,
                    "source_type": entry.get(
                        "source_type",
                        "generated_memory"
                    ),
                    "score": score
                }
            )


    # ==================================================
    # CLASSEMENT
    #
    # Une mémoire validée passe avant une mémoire
    # non validée, puis classement par similarité.
    # ==================================================

    def ranking_key(result):

        validation_priority = 0

        if (
            result["status"]
            == "generated_validated"
        ):
            validation_priority = 2

        elif (
            result["status"]
            == "generated_unvalidated"
        ):
            validation_priority = 1

        return (
            validation_priority,
            result["score"]
        )


    results.sort(
        key=ranking_key,
        reverse=True
    )

    return results[:top_k]


# ==================================================
# DETECTION DE DOUBLON
# ==================================================

def find_duplicate(
    question,
    threshold=0.92
):

    memory = load_memory()

    if not memory:
        return None

    question_embedding = create_embedding(
        question
    )

    best_match = None
    best_score = 0

    for entry in memory:

        status = entry.get(
            "status",
            "generated_unvalidated"
        )

        # Une mémoire rejetée ne bloque pas
        # la création d'une nouvelle analyse.

        if (
            status
            == "generated_rejected"
        ):
            continue

        embedding = entry.get(
            "embedding"
        )

        if not embedding:
            continue

        score = cosine_similarity(
            question_embedding,
            embedding
        )

        if score > best_score:

            best_score = score
            best_match = entry


    if (
        best_match is not None
        and best_score >= threshold
    ):

        return {
            "entry": best_match,
            "score": best_score
        }

    return None


# ==================================================
# SAUVEGARDE D'UNE NOUVELLE MEMOIRE
# ==================================================

def save_memory_entry(
    question,
    answer,
    critique,
    sources=None,
    total_tokens=None,
    total_cost=None,
    duplicate_threshold=0.92
):

    memory = load_memory()


    # ==================================================
    # VERIFICATION DOUBLON
    # ==================================================

    duplicate = find_duplicate(
        question,
        threshold=duplicate_threshold
    )

    if duplicate:

        existing_entry = duplicate[
            "entry"
        ]

        return {
            "id": existing_entry["id"],
            "created": False,
            "duplicate": True,
            "duplicate_score":
                duplicate["score"],
            "status":
                existing_entry.get(
                    "status",
                    "generated_unvalidated"
                )
        }


    # ==================================================
    # EMBEDDING
    # ==================================================

    content_for_embedding = (
        f"Question : {question}\n\n"
        f"Réponse : {answer}"
    )

    embedding = create_embedding(
        content_for_embedding
    )


    # ==================================================
    # GENERATION ID
    # ==================================================

    existing_ids = [
        entry.get(
            "id",
            0
        )
        for entry in memory
    ]

    if existing_ids:

        new_id = (
            max(existing_ids)
            + 1
        )

    else:

        new_id = 1


    # ==================================================
    # NOUVELLE ENTREE
    # ==================================================

    now = datetime.now().isoformat()

    entry = {
        "id": new_id,
        "created_at": now,
        "updated_at": now,
        "source_type":
            "generated_memory",
        "status":
            "generated_unvalidated",
        "question": question,
        "answer": answer,
        "critique": critique,
        "sources": sources or [],
        "total_tokens":
            total_tokens,
        "total_cost":
            total_cost,
        "embedding":
            embedding
    }


    memory.append(
        entry
    )

    write_memory(
        memory
    )


    return {
        "id": entry["id"],
        "created": True,
        "duplicate": False,
        "duplicate_score": None,
        "status": entry["status"]
    }


# ==================================================
# MODIFICATION DU STATUT
# ==================================================

def set_memory_status(
    memory_id,
    status
):

    allowed_statuses = [
        "generated_unvalidated",
        "generated_validated",
        "generated_rejected"
    ]

    if status not in allowed_statuses:

        raise ValueError(
            f"Statut invalide : {status}"
        )


    memory = load_memory()


    for entry in memory:

        if entry.get("id") == memory_id:

            entry["status"] = status

            entry["updated_at"] = (
                datetime.now().isoformat()
            )

            write_memory(
                memory
            )

            return {
                "success": True,
                "id": memory_id,
                "status": status
            }


    return {
        "success": False,
        "id": memory_id,
        "status": None
    }


# ==================================================
# VALIDATION
# ==================================================

def validate_memory_entry(
    memory_id
):

    return set_memory_status(
        memory_id,
        "generated_validated"
    )


# ==================================================
# REJET
# ==================================================

def reject_memory_entry(
    memory_id
):

    return set_memory_status(
        memory_id,
        "generated_rejected"
    )


# ==================================================
# REMISE EN ATTENTE
# ==================================================

def unvalidate_memory_entry(
    memory_id
):

    return set_memory_status(
        memory_id,
        "generated_unvalidated"
    )