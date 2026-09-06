from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
import json


DOCUMENTS_DIR = Path("data/documents")
INDEX_FILE = Path("data/index.json")


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size=500):
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"

            current_chunk += paragraph

        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


documents = load_documents()

index = []

print(f"Documents trouvés : {len(documents)}")

for document in documents:

    print(f"\nTraitement : {document['source']}")

    chunks = chunk_text(document["text"])

    for chunk_id, chunk in enumerate(chunks, start=1):

        print(f"Embedding chunk {chunk_id}/{len(chunks)}")

        embedding = create_embedding(chunk)

        index.append({
            "source": document["source"],
            "chunk_id": chunk_id,
            "text": chunk,
            "embedding": embedding
        })


INDEX_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

INDEX_FILE.write_text(
    json.dumps(index, ensure_ascii=False),
    encoding="utf-8"
)

print("\nIndex créé avec succès")
print(f"Chunks indexés : {len(index)}")
print(f"Fichier : {INDEX_FILE}")