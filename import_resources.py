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
                                           [--embedding-url <url>]

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
import requests


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


def encode_texts(texts: list[str], service_url: str) -> np.ndarray:
    """Encode texts via the embedding microservice /encode endpoint."""
    # Batch in chunks of 64 to avoid overloading the service
    all_embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            f"{service_url}/encode",
            json={"texts": batch, "mode": "document"},
            timeout=120,
        )
        resp.raise_for_status()
        all_embeddings.extend(resp.json()["embeddings"])
    return np.array(all_embeddings, dtype=np.float32)


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
        "--embedding-url",
        default="http://localhost:8200",
        help="URL of the embedding microservice (default: http://localhost:8200)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory")
        return 1

    # 1. Read resources
    print(f"Reading resources from {args.input_dir} ...")
    resources = read_resources(args.input_dir)
    if not resources:
        print("No valid resources found.")
        return 1
    print(f"  Found {len(resources)} resources")

    # 2. Create output directory and SQLite database
    os.makedirs(args.output, exist_ok=True)
    db_path = os.path.join(args.output, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    print("Building database ...")
    resource_ids = build_database(resources, db_path)

    # 3. Build text chunks — each resource produces one text segment
    #    combining title + description for embedding
    embedded_texts: dict[int, str] = {}
    text_to_resource_id: dict[int, int] = {}
    for text_idx, (res, res_id) in enumerate(zip(resources, resource_ids)):
        chunk = f"{res['title']}: {res['description']}"
        embedded_texts[text_idx] = chunk
        text_to_resource_id[text_idx] = res_id

    # 4. Encode via embedding service
    print(f"Encoding {len(embedded_texts)} text chunks ...")
    texts_ordered = [embedded_texts[i] for i in range(len(embedded_texts))]
    embeddings = encode_texts(texts_ordered, args.embedding_url)
    print(f"  Embedding shape: {embeddings.shape}")

    # 5. Build HNSW index
    index_path = os.path.join(args.output, "embeddings.bin")
    print("Building HNSW index ...")
    build_hnsw_index(embeddings, index_path)

    # 6. Save pickles
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
