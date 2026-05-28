"""
Transcript Importer
===================
Reads `with summary_*.json` transcript files from data/Transcripts/<topic>/
and produces the artifacts expected by embedding_service.py:

    <output_dir>/
        database.db                  — SQLite with a 'resources' table
        embedded_texts.pkl           — {text_idx: str}
        text_to_resource_mapping.pkl — {text_idx: resource_id}
        embeddings.bin               — HNSW index (cosine)

Key differences vs import_resources.py:
  - One resource per transcript JSON.
  - Each resource produces multiple text chunks: a rich summary chunk plus
    one chunk per transcript section (slide / segment / page). This gives
    section-level retrieval while still returning resource-level results.
  - The DB stores extra columns (source_file, source_type, source_path,
    summary_topic, keywords, takeaways, full_transcript, char_count) so
    examine_resource can return everything the bot needs.
  - portal_url is populated with `/api/file/<provider>/<int_id>` so search
    output already carries a working file URL.

Usage:
    python import_transcripts.py <topic_dir> --topic-slug <slug> \\
                                 [--output <out>] [--source-folder <folder>]

    # Or import every topic in data/Transcripts/ at once:
    python import_transcripts.py --all
"""

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "Transcripts"
PROCESSED_DIR = ROOT / "processed_resources"

# Hard cap per chunk to keep the embedding model in its sweet spot.
# embeddinggemma's context is ~2k tokens; 4k chars is a comfortable
# upper bound that still preserves whole slides / pages.
MAX_CHUNK_CHARS = 4000
# Skip sections shorter than this — usually slide-titles with no body.
MIN_SECTION_CHARS = 60


def slugify(name: str) -> str:
    """Lowercase, replace non-alphanumerics with underscores, collapse runs."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "topic"


def load_model(model_name: str) -> SentenceTransformer:
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
    if hasattr(model, "encode_document"):
        emb = model.encode_document(texts, show_progress_bar=True)
    else:
        emb = model.encode(texts, show_progress_bar=True)
    return emb.astype(np.float32)


def derive_source_file(transcript: dict, fallback: str) -> str:
    """Get the original asset filename (e.g. 'Tactical Breathing.mp4')."""
    src = transcript.get("source_file") or fallback
    return os.path.basename(src)


def read_transcripts(topic_dir: Path) -> list[dict]:
    """Read every `with summary_*.json` in *topic_dir*."""
    files = sorted(topic_dir.glob("with summary_*.json"))
    out = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  SKIP (parse error in {f.name}): {exc}")
            continue
        data["_transcript_json_filename"] = f.name
        out.append(data)
    return out


def build_database(
    resources: list[dict],
    source_folder: str,
    provider_slug: str,
    db_path: str,
) -> list[int]:
    """
    Create a SQLite database with an extended 'resources' table.
    Returns the list of assigned integer resource IDs.

    Column choices keep the existing tools working: search_resources
    reads `description`, `physical_address`, `portal_url` and exposes
    everything else under examine_resource.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(textwrap.dedent("""\
        CREATE TABLE IF NOT EXISTS resources (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            title            TEXT NOT NULL,
            description      TEXT NOT NULL,
            physical_address TEXT DEFAULT '',
            portal_url       TEXT DEFAULT '',
            latitude         REAL,
            longitude        REAL,
            source_file      TEXT,
            source_type      TEXT,
            source_path      TEXT,
            summary_topic    TEXT,
            keywords         TEXT,
            takeaways        TEXT,
            full_transcript  TEXT,
            char_count       INTEGER
        )
    """))

    ids: list[int] = []
    for r in resources:
        summary = r.get("summary") or {}
        source_file = derive_source_file(r, r.get("_transcript_json_filename", ""))
        title = os.path.splitext(source_file)[0]
        summary_text = (summary.get("text") or "").strip()
        topic = summary.get("topic") or ""
        keywords = summary.get("keywords") or []
        takeaways = summary.get("takeaways") or []
        # Description = the AI-generated topic + summary blurb. Search tool
        # truncates this to 100 chars so keep it punchy and informative.
        description = f"{topic}. {summary_text}".strip(". ").strip() or title
        # Filesystem-relative path under data/, used by the file endpoint.
        source_path = f"{source_folder}/{source_file}"
        char_count = (r.get("stats") or {}).get("character_count", 0)

        cursor = conn.execute(
            "INSERT INTO resources "
            "(title, description, physical_address, portal_url, "
            " source_file, source_type, source_path, summary_topic, "
            " keywords, takeaways, full_transcript, char_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                title,
                description,
                source_folder,
                "",  # populated below once the row id is known
                source_file,
                r.get("source_type") or "",
                source_path,
                topic,
                json.dumps(keywords, ensure_ascii=False),
                json.dumps(takeaways, ensure_ascii=False),
                r.get("transcript") or "",
                char_count,
            ),
        )
        rid = cursor.lastrowid
        portal_url = f"/api/file/{provider_slug}/{rid}"
        conn.execute(
            "UPDATE resources SET portal_url = ? WHERE id = ?",
            (portal_url, rid),
        )
        ids.append(rid)
    conn.commit()
    conn.close()
    return ids


def build_chunks(
    resources: list[dict],
    resource_ids: list[int],
) -> tuple[dict[int, str], dict[int, int]]:
    """
    Build (embedded_texts, text_to_resource_id) maps.

    Each resource gets:
      - one summary chunk (always)
      - one chunk per non-trivial transcript section (slide / segment / page)

    All chunks point back to the resource via text_to_resource_id, so the
    embedding service de-duplicates to resource-level results.
    """
    embedded_texts: dict[int, str] = {}
    text_to_resource: dict[int, int] = {}
    text_idx = 0

    for r, rid in zip(resources, resource_ids):
        summary = r.get("summary") or {}
        title = os.path.splitext(derive_source_file(
            r, r.get("_transcript_json_filename", "")))[0]
        topic = summary.get("topic") or ""
        summary_text = (summary.get("text") or "").strip()
        keywords = summary.get("keywords") or []
        takeaways = summary.get("takeaways") or []
        search_terms = summary.get("search_terms") or []

        summary_lines = [f"Title: {title}"]
        if topic:
            summary_lines.append(f"Topic: {topic}")
        if summary_text:
            summary_lines.append(f"Summary: {summary_text}")
        if keywords:
            summary_lines.append(f"Keywords: {', '.join(keywords)}")
        if takeaways:
            summary_lines.append("Key takeaways:")
            summary_lines.extend(f"- {t}" for t in takeaways)
        if search_terms:
            summary_lines.append(f"Related queries: {', '.join(search_terms)}")
        summary_chunk = "\n".join(summary_lines)
        embedded_texts[text_idx] = summary_chunk[:MAX_CHUNK_CHARS]
        text_to_resource[text_idx] = rid
        text_idx += 1

        for section in r.get("sections") or []:
            text = (section.get("text") or "").strip()
            if len(text) < MIN_SECTION_CHARS:
                continue
            label = section.get("label") or section.get("section_type") or "section"
            chunk_lines = [f"{title} — {label}"]
            if topic:
                chunk_lines.append(f"(Topic: {topic})")
            chunk_lines.append(text)
            chunk = "\n".join(chunk_lines)[:MAX_CHUNK_CHARS]
            embedded_texts[text_idx] = chunk
            text_to_resource[text_idx] = rid
            text_idx += 1

    return embedded_texts, text_to_resource


def build_hnsw_index(embeddings: np.ndarray, output_path: str):
    num_elements, dim = embeddings.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max(num_elements, 1), M=16, ef_construction=200)
    index.add_items(embeddings, list(range(num_elements)))
    index.set_ef(50)
    index.save_index(output_path)
    return index


def import_topic(
    topic_dir: Path,
    provider_slug: str,
    source_folder: str,
    output_dir: Path,
    model: SentenceTransformer,
):
    print(f"\n=== Importing topic: {source_folder!r} → {output_dir} ===")

    resources = read_transcripts(topic_dir)
    if not resources:
        print(f"  No transcripts found in {topic_dir}, skipping.")
        return

    print(f"  Found {len(resources)} transcripts")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "database.db"
    resource_ids = build_database(
        resources, source_folder, provider_slug, str(db_path),
    )

    embedded_texts, text_to_resource = build_chunks(resources, resource_ids)
    print(f"  Built {len(embedded_texts)} chunks across "
          f"{len(set(text_to_resource.values()))} resources")

    texts_ordered = [embedded_texts[i] for i in range(len(embedded_texts))]
    embeddings = encode_texts(model, texts_ordered)
    print(f"  Embedding shape: {embeddings.shape}")

    build_hnsw_index(embeddings, str(output_dir / "embeddings.bin"))

    with open(output_dir / "embedded_texts.pkl", "wb") as f:
        pickle.dump(embedded_texts, f)
    with open(output_dir / "text_to_resource_mapping.pkl", "wb") as f:
        pickle.dump(text_to_resource, f)

    print(f"  Done. {len(resources)} resources, {len(embedded_texts)} chunks.")


def discover_topics() -> list[tuple[Path, str, str]]:
    """
    Return [(transcript_dir, provider_slug, source_folder)] for every topic
    folder that has both a Transcripts subfolder and a sibling source folder
    in data/.
    """
    out = []
    if not TRANSCRIPTS_DIR.exists():
        raise SystemExit(f"Transcripts dir not found: {TRANSCRIPTS_DIR}")
    for transcript_dir in sorted(p for p in TRANSCRIPTS_DIR.iterdir() if p.is_dir()):
        source_folder = transcript_dir.name
        source_dir = DATA_DIR / source_folder
        if not source_dir.exists():
            print(f"  WARN: source folder missing for transcripts {source_folder!r}")
        out.append((transcript_dir, slugify(source_folder), source_folder))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "topic_dir", nargs="?",
        help="Path to a Transcripts/<topic> folder (omit when using --all)",
    )
    parser.add_argument(
        "--topic-slug",
        help="Override the auto-derived provider slug (lowercase, underscores)",
    )
    parser.add_argument(
        "--source-folder",
        help="Override the data/<folder> name used for file URLs",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: processed_resources/<slug>)",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Import every topic discovered under data/Transcripts/",
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.all:
        topics = discover_topics()
        for transcript_dir, slug, source_folder in topics:
            output_dir = PROCESSED_DIR / slug
            import_topic(transcript_dir, slug, source_folder, output_dir, model)
        print("\nDone importing all topics.")
        print("Provider slugs:", [slug for _, slug, _ in topics])
        return 0

    if not args.topic_dir:
        parser.error("topic_dir is required unless --all is given")

    topic_dir = Path(args.topic_dir).resolve()
    if not topic_dir.is_dir():
        parser.error(f"{topic_dir} is not a directory")

    source_folder = args.source_folder or topic_dir.name
    slug = args.topic_slug or slugify(source_folder)
    output_dir = Path(args.output) if args.output else PROCESSED_DIR / slug
    import_topic(topic_dir, slug, source_folder, output_dir, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
