from pathlib import Path
import json
import math


MEMORY_FILE = Path("data/memory.json")
DUPLICATE_THRESHOLD = 0.92


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
        MEMORY_FILE.read_text(
            encoding="utf-8"
        )
    )


def save_memory(memory):
    MEMORY_FILE.write_text(
        json.dumps(
            memory,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def cleanup_memory():
    memory = load_memory()

    if not memory:
        print("Mémoire vide.")
        return

    cleaned = []
    removed = []

    for entry in memory:

        duplicate_found = False

        for kept_entry in cleaned:

            score = cosine_similarity(
                entry["embedding"],
                kept_entry["embedding"]
            )

            if score >= DUPLICATE_THRESHOLD:

                removed.append({
                    "removed_id": entry["id"],
                    "kept_id": kept_entry["id"],
                    "score": score
                })

                duplicate_found = True
                break

        if not duplicate_found:
            cleaned.append(entry)

    # Renumérotation propre
    for index, entry in enumerate(cleaned, start=1):
        entry["id"] = index

    save_memory(cleaned)

    print("\n=== NETTOYAGE MÉMOIRE ===")
    print(f"Entrées avant : {len(memory)}")
    print(f"Entrées après : {len(cleaned)}")
    print(f"Doublons supprimés : {len(removed)}")

    if removed:
        print("\nDoublons détectés :")

        for item in removed:
            print(
                f"ID supprimé {item['removed_id']} "
                f"→ conservé {item['kept_id']} "
                f"| similarité {item['score']:.3f}"
            )


if __name__ == "__main__":
    cleanup_memory()