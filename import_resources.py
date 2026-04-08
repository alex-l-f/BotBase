"""
Resource Importer
=================
Reads .txt and .json files from a directory and produces the artifacts
expected by embedding_service.py:

    <output_dir>/
        database.db                  — SQLite with a 'resources' table
        embedded_texts.pkl           — {text_idx: str}
        text_to_resource_mapping.pkl — {text_idx: resource_id}
        embeddings.bin               — HNSW index (cosine, built from embeddings)

Usage:
    python import_resources.py <input_dir> [--output <output_dir>]
                                           [--model <model_name>]

Input formats
-------------
.json  — Each file is either a single resource object or a list of objects.
         Expected fields: "title", "description".
         Optional: "physical_address", "portal_url", "latitude", "longitude",
                   and any other metadata columns.

.txt   — Each file becomes one resource.  The filename (without extension) is
         used as the title; the file contents become the description.
"""

import argparse
import json
import os
import pickle
import sqlite3
import textwrap

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")


def load_model(model_name: str) -> SentenceTransformer:
    """Load the SentenceTransformer model directly."""
    print(f"Loading model {model_name} ...")
    if torch.cuda.is_available():
        model_kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
        device = "cuda"
    else:
        model_kwargs = {}
        device = "cpu"
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    print(f"  Model loaded on {device}")
    return model


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode texts directly using the model."""
    if hasattr(model, "encode_document"):
        emb = model.encode_document(texts, show_progress_bar=True)
    else:
        emb = model.encode(texts, show_progress_bar=True)
    return emb.astype(np.float32)


def read_resources(input_dir: str) -> list[dict]:
    """Read .txt and .json files from input_dir into resource dicts."""
    resources = []

    for filename in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()

        if ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if "title" not in item or "description" not in item:
                    print(f"  SKIP (missing title/description): {filename}")
                    continue
                resources.append(item)

        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                print(f"  SKIP (empty): {filename}")
                continue
            resources.append({
                "title": os.path.splitext(filename)[0],
                "description": content,
            })

    return resources


def build_database(resources: list[dict], db_path: str) -> list[int]:
    """
    Create a SQLite database with a 'resources' table.
    Returns the list of assigned resource IDs (row order matches input).
    """
    conn = sqlite3.connect(db_path)
    conn.execute(textwrap.dedent("""\
        CREATE TABLE IF NOT EXISTS resources (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            description     TEXT NOT NULL,
            physical_address TEXT DEFAULT '',
            portal_url      TEXT DEFAULT '',
            latitude        REAL,
            longitude       REAL
        )
    """))

    ids = []
    for r in resources:
        cursor = conn.execute(
            "INSERT INTO resources (title, description, physical_address, "
            "portal_url, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?)",
            (
                r["title"],
                r["description"],
                r.get("physical_address", ""),
                r.get("portal_url", ""),
                r.get("latitude"),
                r.get("longitude"),
            ),
        )
        ids.append(cursor.lastrowid)
    conn.commit()
    conn.close()
    return ids


def encode_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Encode texts directly using the model."""
    if hasattr(model, "encode_document"):
        emb = model.encode_document(texts, show_progress_bar=True)
    else:
        emb = model.encode(texts, show_progress_bar=True)
    return emb.astype(np.float32)


def build_hnsw_index(embeddings: np.ndarray, output_path: str):
    """Build and save an HNSW index from the embeddings matrix."""
    num_elements, dim = embeddings.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max(num_elements, 1), M=16, ef_construction=200)
    index.add_items(embeddings, list(range(num_elements)))
    index.set_ef(50)
    index.save_index(output_path)
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Import .txt/.json resources for the embedding service."
    )
    parser.add_argument("input_dir", help="Directory containing .txt and .json files")
    parser.add_argument(
        "--output", "-o",
        default="./processed_resources/imported",
        help="Output directory (default: ./processed_resources/imported)",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory")
        return 1

    # 1. Load embedding model
    model = load_model(args.model)

    # 2. Read resources
    print(f"Reading resources from {args.input_dir} ...")
    resources = read_resources(args.input_dir)
    if not resources:
        print("No valid resources found.")
        return 1
    print(f"  Found {len(resources)} resources")

    # 3. Create output directory and SQLite database
    os.makedirs(args.output, exist_ok=True)
    db_path = os.path.join(args.output, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Building database ...")
    resource_ids = build_database(resources, db_path)

    # 4. Build text chunks — each resource produces one text segment
    #    combining title + description for embedding
    embedded_texts: dict[int, str] = {}
    text_to_resource_id: dict[int, int] = {}
    for text_idx, (res, res_id) in enumerate(zip(resources, resource_ids)):
        chunk = f"{res['title']}: {res['description']}"
        embedded_texts[text_idx] = chunk
        text_to_resource_id[text_idx] = res_id

    # 5. Encode texts
    print(f"Encoding {len(embedded_texts)} text chunks ...")
    texts_ordered = [embedded_texts[i] for i in range(len(embedded_texts))]
    embeddings = encode_texts(model, texts_ordered)
    print(f"  Embedding shape: {embeddings.shape}")

    # 6. Build HNSW index
    index_path = os.path.join(args.output, "embeddings.bin")
    print("Building HNSW index ...")
    build_hnsw_index(embeddings, index_path)

    # 7. Save pickles
    with open(os.path.join(args.output, "embedded_texts.pkl"), "wb") as f:
        pickle.dump(embedded_texts, f)
    with open(os.path.join(args.output, "text_to_resource_mapping.pkl"), "wb") as f:
        pickle.dump(text_to_resource_id, f)

    print(f"\nDone! Output written to {args.output}/")
    print(f"  database.db                  — {len(resources)} resources")
    print(f"  embeddings.bin               — HNSW index ({embeddings.shape[1]}d)")
    print(f"  embedded_texts.pkl           — {len(embedded_texts)} text chunks")
    print(f"  text_to_resource_mapping.pkl — {len(text_to_resource_id)} mappings")
    print(f"\nTo use: add '{args.output}' to PROVIDER_DIRS in embedding_service.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
