"""
E-learning Course Importer
==========================
Indexes the e-learning module's pages (Elearningcourse/<section>/<lesson>/
rag_content.json) into the existing per-topic search providers, so
search_resources surfaces course pages alongside files and the bot can send
them to the user with the open_course_page tool.

For each topic provider this script:
  1. Removes any course pages from a previous run (safe to re-run).
  2. Appends one resource row per lesson to database.db with
     source_type='course_page', a lesson_id column, and portal_url set to
     the SCORM deep link the frontend viewer understands
     (scorm/<package>/scormcontent/index.html#/lessons/<lesson_id>).
  3. Embeds a summary chunk plus one chunk per lesson section and extends
     the provider's HNSW index / pkl artifacts in place.

Lesson IDs are validated against the Rise course JSON inside the SCORM
package so a stale export can't produce dead links.

Restart the embedding service after running — providers load at startup.

Usage:
    python import_elearning.py [--course-dir Elearningcourse]
                               [--model <model_name>] [--dry-run]
"""

import argparse
import base64
import json
import os
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import hnswlib
import numpy as np

ROOT = Path(__file__).resolve().parent
COURSE_META_DIR = ROOT / "Elearningcourse"
PROCESSED_DIR = ROOT / "processed_resources"

DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")

# Same chunking limits as import_transcripts.py.
MAX_CHUNK_CHARS = 4000
MIN_SECTION_CHARS = 60

SOURCE_TYPE = "course_page"

# Section-folder → topic-provider mapping. Rules are matched in order
# against the lowercased folder name. "Improve relations" lands in
# performance because its content (team relations, social support,
# resilience at work) sits closest to the team-resilience material there.
SECTION_TOPIC_RULES: list[tuple[str, str]] = [
    ("skill",                "coping_mental_skills"),
    ("coping with stress",   "coping_stress"),
    ("implement recovery",   "coping_recovery"),
    ("improve performance",  "performance"),
    ("improve relations",    "performance"),
]

# Progress/checkmark markers some lesson titles carry in the authoring tool,
# plus the invisible variation-selector/joiner codepoints that ride with them.
_TITLE_MARKERS = "\u2705\u2714\u2611\ufe0f\u200d \t"  # check emoji + VS16/ZWJ


def provider_for_section(folder_name: str) -> str | None:
    name = folder_name.strip().lower()
    for prefix, provider in SECTION_TOPIC_RULES:
        if name.startswith(prefix):
            return provider
    return None


def clean_title(raw: str) -> str:
    return (raw or "").strip().lstrip(_TITLE_MARKERS).strip()


# ---------------------------------------------------------------------------
# SCORM package discovery / validation
# ---------------------------------------------------------------------------
def find_scorm_package() -> Path:
    """The project-root directory containing an imsmanifest.xml."""
    candidates = [
        p for p in sorted(ROOT.iterdir())
        if p.is_dir() and (p / "imsmanifest.xml").is_file()
    ]
    if not candidates:
        raise SystemExit("No SCORM package (imsmanifest.xml) found in project root.")
    if len(candidates) > 1:
        print(f"WARN: multiple SCORM packages found, using {candidates[0].name!r}")
    return candidates[0]


def rise_lesson_ids(pkg_dir: Path) -> set[str]:
    """Lesson IDs from the Rise course JSON embedded in scormcontent/index.html."""
    index_path = pkg_dir / "scormcontent" / "index.html"
    html = index_path.read_text(encoding="utf-8")
    m = re.search(r'deserialize\("([^"]+)"\)', html)
    if not m:
        raise SystemExit(f"Could not find Rise course JSON in {index_path}")
    course = json.loads(base64.b64decode(m.group(1)))["course"]
    return {
        lesson["id"] for lesson in course.get("lessons", [])
        if lesson.get("type") != "section" and lesson.get("id")
    }


# ---------------------------------------------------------------------------
# Course metadata loading
# ---------------------------------------------------------------------------
def collect_section_text(sections: list[dict]) -> str:
    """Flatten a lesson's nested section tree into learner-facing text."""
    parts: list[str] = []

    def walk(node: dict):
        title = (node.get("title") or "").strip()
        if title:
            title = clean_title(title)
            if not parts or parts[-1] != title:
                parts.append(title)
        for block in node.get("content") or []:
            block = (block or "").strip()
            # Authoring tool often repeats the heading as the first block.
            if block and (not parts or parts[-1] != block):
                parts.append(block)
        for sub in node.get("subsections") or []:
            walk(sub)

    for section in sections or []:
        walk(section)
    return "\n\n".join(parts)


def read_lessons(course_dir: Path) -> dict[str, list[dict]]:
    """
    Walk Elearningcourse/<section>/<lesson>/rag_content.json.
    Returns {provider_slug: [lesson_record, ...]}.
    """
    by_provider: dict[str, list[dict]] = {}
    unmapped: list[str] = []

    for section_dir in sorted(p for p in course_dir.iterdir() if p.is_dir()):
        provider = provider_for_section(section_dir.name)
        if provider is None:
            unmapped.append(section_dir.name)
            continue
        for lesson_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            rc_path = lesson_dir / "rag_content.json"
            if not rc_path.is_file():
                print(f"  WARN: no rag_content.json in {lesson_dir}, skipped")
                continue
            try:
                data = json.loads(rc_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                print(f"  WARN: bad JSON in {rc_path}: {exc}, skipped")
                continue
            data["_meta_dir"] = lesson_dir
            by_provider.setdefault(provider, []).append(data)

    if unmapped:
        rules = ", ".join(repr(p) for p, _ in SECTION_TOPIC_RULES)
        raise SystemExit(
            f"Unmapped section folders: {unmapped}. "
            f"Add a rule to SECTION_TOPIC_RULES (current prefixes: {rules})."
        )
    return by_provider


# ---------------------------------------------------------------------------
# Resource + chunk construction
# ---------------------------------------------------------------------------
def build_resource(lesson: dict, pkg_name: str) -> dict:
    meta = lesson.get("llm_metadata") or {}
    lesson_id = lesson["lesson_id"]
    title = clean_title(lesson.get("lesson_title") or "Untitled page")
    section_header = (lesson.get("section_category_header")
                      or lesson["_meta_dir"].parent.name)
    topic = (meta.get("topic") or "").strip()
    summary = (meta.get("summary") or "").strip()
    description = f"{topic}. {summary}".strip(". ").strip() or title
    full_text = collect_section_text(lesson.get("sections"))

    return {
        "lesson_id": lesson_id,
        "title": title,
        "description": description,
        "physical_address": section_header,
        "portal_url": f"scorm/{pkg_name}/scormcontent/index.html#/lessons/{lesson_id}",
        "source_file": "",
        "source_type": SOURCE_TYPE,
        "source_path": str(lesson["_meta_dir"].relative_to(ROOT)).replace("\\", "/"),
        "summary_topic": topic,
        "keywords": json.dumps(meta.get("keywords") or [], ensure_ascii=False),
        "takeaways": json.dumps(meta.get("takeaways") or [], ensure_ascii=False),
        "full_transcript": full_text,
        "char_count": len(full_text),
        "_meta": meta,
        "_sections": lesson.get("sections") or [],
    }


def build_chunks(resource: dict) -> list[str]:
    """Summary chunk + one chunk per top-level section, like transcripts."""
    meta = resource["_meta"]
    title = resource["title"]
    topic = resource["summary_topic"]

    lines = [f"Title: {title}", f"Course page — {resource['physical_address']}"]
    if topic:
        lines.append(f"Topic: {topic}")
    summary = (meta.get("summary") or "").strip()
    if summary:
        lines.append(f"Summary: {summary}")
    keywords = meta.get("keywords") or []
    if keywords:
        lines.append(f"Keywords: {', '.join(keywords)}")
    takeaways = meta.get("takeaways") or []
    if takeaways:
        lines.append("Key takeaways:")
        lines.extend(f"- {t}" for t in takeaways)
    search_terms = meta.get("search_terms") or []
    if search_terms:
        lines.append(f"Related queries: {', '.join(search_terms)}")
    chunks = ["\n".join(lines)[:MAX_CHUNK_CHARS]]

    for section in resource["_sections"]:
        text = collect_section_text([section])
        if len(text) < MIN_SECTION_CHARS:
            continue
        label = clean_title(section.get("title") or "") or "section"
        chunk_lines = [f"{title} — {label}"]
        if topic:
            chunk_lines.append(f"(Topic: {topic})")
        chunk_lines.append(text)
        chunks.append("\n".join(chunk_lines)[:MAX_CHUNK_CHARS])

    return chunks


# ---------------------------------------------------------------------------
# Provider artifact updates
# ---------------------------------------------------------------------------
def ensure_lesson_id_column(conn: sqlite3.Connection):
    cols = {row[1] for row in conn.execute("PRAGMA table_info(resources)")}
    if "lesson_id" not in cols:
        conn.execute("ALTER TABLE resources ADD COLUMN lesson_id TEXT")


def remove_previous_course_pages(conn: sqlite3.Connection) -> set[int]:
    """Delete rows from an earlier import; returns their resource ids."""
    stale = {
        row[0] for row in conn.execute(
            "SELECT id FROM resources WHERE source_type = ?", (SOURCE_TYPE,)
        )
    }
    if stale:
        conn.execute("DELETE FROM resources WHERE source_type = ?", (SOURCE_TYPE,))
    return stale


def insert_resources(conn: sqlite3.Connection, resources: list[dict]) -> list[int]:
    ids = []
    for r in resources:
        cursor = conn.execute(
            "INSERT INTO resources "
            "(title, description, physical_address, portal_url, "
            " source_file, source_type, source_path, summary_topic, "
            " keywords, takeaways, full_transcript, char_count, lesson_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                r["title"], r["description"], r["physical_address"],
                r["portal_url"], r["source_file"], r["source_type"],
                r["source_path"], r["summary_topic"], r["keywords"],
                r["takeaways"], r["full_transcript"], r["char_count"],
                r["lesson_id"],
            ),
        )
        ids.append(cursor.lastrowid)
    return ids


def update_provider(
    provider_dir: Path,
    resources: list[dict],
    model,
    dry_run: bool = False,
):
    db_path = provider_dir / "database.db"
    texts_path = provider_dir / "embedded_texts.pkl"
    mapping_path = provider_dir / "text_to_resource_mapping.pkl"
    index_path = provider_dir / "embeddings.bin"

    for p in (db_path, texts_path, mapping_path, index_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p} — run import_transcripts.py for this topic first."
            )

    print(f"\n=== {provider_dir.name}: {len(resources)} course pages ===")
    if dry_run:
        for r in resources:
            print(f"  [{r['physical_address']}] {r['title']} -> {r['portal_url']}")
        return

    with open(texts_path, "rb") as f:
        embedded_texts: dict[int, str] = pickle.load(f)
    with open(mapping_path, "rb") as f:
        text_to_resource: dict[int, int] = pickle.load(f)

    conn = sqlite3.connect(db_path)
    try:
        ensure_lesson_id_column(conn)
        stale_ids = remove_previous_course_pages(conn)
        new_ids = insert_resources(conn, resources)
        # Chunks are kept only if they belong to a surviving non-course-page
        # resource. Matching on the live table (rather than the ids we just
        # deleted) also cleans up orphans left by an interrupted earlier run.
        keep_ids = {
            row[0] for row in conn.execute(
                "SELECT id FROM resources WHERE COALESCE(source_type, '') != ?",
                (SOURCE_TYPE,),
            )
        }
        conn.commit()
    finally:
        conn.close()

    drop = {idx for idx, rid in text_to_resource.items() if rid not in keep_ids}
    if stale_ids or drop:
        embedded_texts = {i: t for i, t in embedded_texts.items() if i not in drop}
        text_to_resource = {i: r for i, r in text_to_resource.items() if i not in drop}
        print(f"  Removed {len(stale_ids)} stale course pages ({len(drop)} chunks)")

    dim = model.get_sentence_embedding_dimension()
    old_index = hnswlib.Index(space="cosine", dim=dim)
    old_index.load_index(str(index_path))

    kept_idxs = sorted(embedded_texts.keys())
    kept_vectors = (
        np.asarray(old_index.get_items(kept_idxs), dtype=np.float32)
        if kept_idxs else np.zeros((0, dim), dtype=np.float32)
    )

    # Embed the new chunks.
    next_idx = (max(kept_idxs) + 1) if kept_idxs else 0
    new_texts: list[str] = []
    for resource, rid in zip(resources, new_ids):
        for chunk in build_chunks(resource):
            embedded_texts[next_idx] = chunk
            text_to_resource[next_idx] = rid
            new_texts.append(chunk)
            next_idx += 1

    print(f"  Encoding {len(new_texts)} new chunks ...")
    if hasattr(model, "encode_document"):
        new_vectors = model.encode_document(new_texts, show_progress_bar=True)
    else:
        new_vectors = model.encode(new_texts, show_progress_bar=True)
    new_vectors = new_vectors.astype(np.float32)

    # Rebuild the index: kept vectors under their original ids + new ones.
    all_ids = kept_idxs + list(range(next_idx - len(new_texts), next_idx))
    all_vectors = np.concatenate([kept_vectors, new_vectors], axis=0)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=max(len(all_ids), 1), M=16, ef_construction=200)
    index.add_items(all_vectors, all_ids)
    index.set_ef(50)
    index.save_index(str(index_path))

    with open(texts_path, "wb") as f:
        pickle.dump(embedded_texts, f)
    with open(mapping_path, "wb") as f:
        pickle.dump(text_to_resource, f)

    print(f"  Done: {len(new_ids)} course pages, {len(new_texts)} chunks "
          f"(index now {len(all_ids)} vectors)")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-dir", default=str(COURSE_META_DIR),
        help="Folder holding <section>/<lesson>/rag_content.json metadata",
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show the section→topic mapping and planned rows, write nothing",
    )
    args = parser.parse_args()

    course_dir = Path(args.course_dir)
    if not course_dir.is_dir():
        raise SystemExit(f"Course metadata dir not found: {course_dir}")

    pkg_dir = find_scorm_package()
    valid_ids = rise_lesson_ids(pkg_dir)
    print(f"SCORM package: {pkg_dir.name} ({len(valid_ids)} lessons in course)")

    by_provider = read_lessons(course_dir)

    total, missing = 0, 0
    provider_resources: dict[str, list[dict]] = {}
    for provider, lessons in sorted(by_provider.items()):
        rows = []
        for lesson in lessons:
            lid = lesson.get("lesson_id")
            if lid not in valid_ids:
                print(f"  WARN: lesson_id {lid!r} "
                      f"({lesson.get('lesson_title')!r}) not in SCORM course, skipped")
                missing += 1
                continue
            rows.append(build_resource(lesson, pkg_dir.name))
            total += 1
        provider_resources[provider] = rows

    print(f"Mapped {total} course pages across {len(provider_resources)} topics"
          + (f" ({missing} skipped — not in course)" if missing else ""))

    model = None
    if not args.dry_run:
        # Import lazily so --dry-run works without the ML stack.
        import torch
        from sentence_transformers import SentenceTransformer
        print(f"Loading model {args.model} ...")
        if torch.cuda.is_available():
            model_kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
            device = "cuda"
        else:
            model_kwargs = {}
            device = "cpu"
        model = SentenceTransformer(
            args.model,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        print(f"  Model loaded on {device}")

    for provider, rows in sorted(provider_resources.items()):
        update_provider(PROCESSED_DIR / provider, rows, model, args.dry_run)

    if not args.dry_run:
        print("\nDone. Restart the embedding service to load the updated indexes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
