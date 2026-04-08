"""
Embedding Microservice (Hybrid Search)
=======================================
A FastAPI service that owns the SentenceTransformer model and exposes
encoding + hybrid HNSW/BM25 search over HTTP.

Hybrid search runs dense vector retrieval (HNSW) and sparse keyword
retrieval (BM25) in parallel, then merges results with Reciprocal Rank
Fusion (RRF).  This captures both semantic similarity and exact lexical
matches — the combination significantly outperforms either method alone.

BM25 indices are built at startup from the same embedded_texts corpus
already used for the vector index.  No external search engine required.

Features:
  - Single model instance shared across all requests
  - Dynamic request batching (collects requests over a short window)
  - Hybrid search: HNSW (dense) + BM25 (sparse) with RRF fusion
  - Supports multiple provider databases loaded at startup
  - Language-filtered retrieval (english, french, or all)
  - Returns resource IDs + scores; the client handles DB lookups

Dependencies:
  pip install rank-bm25          # BM25 scoring

Run:
  uvicorn embedding_service:app --host 0.0.0.0 --port 8200 --workers 1
"""

import asyncio
import math
import os
import pickle
import re
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional

import hnswlib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-300m")
PROVIDER_DIRS = [
    "./processed_resources",
]
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "10"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
DEFAULT_RRF_K = int(os.getenv("RRF_K", "60"))

# BM25 candidate over-fetch factor.  We retrieve this many times k from
# each method before fusing so RRF has good coverage of both rankings.
_BM25_OVERFETCH = 4


# ---------------------------------------------------------------------------
# Language helpers
# ---------------------------------------------------------------------------
_LANGUAGE_TAG_MAP: dict[str, str] = {
    "english":  "english",
    "anglais":  "english",
    "french":   "french",
    "français": "french",
    "francais": "french",
}


def normalise_language(raw: Optional[str]) -> str:
    """Return 'english', 'french', or 'all' from any accepted tag value."""
    if raw is None:
        return "all"
    canonical = _LANGUAGE_TAG_MAP.get(raw.strip().lower())
    if canonical:
        return canonical
    if raw.strip().lower() == "all":
        return "all"
    return "all"


# ---------------------------------------------------------------------------
# BM25 tokeniser
# ---------------------------------------------------------------------------
# Simple regex tokeniser: extracts word characters, lowercased.
# Works for Latin-script languages (English, French) without heavy deps.
# BM25's IDF naturally down-weights stopwords so explicit removal is
# unnecessary and would hurt recall on short queries.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase regex tokeniser for BM25 indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in kilometres."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _load_resource_coords(db_path: str) -> dict[str, tuple[float, float]]:
    """
    Load (latitude, longitude) for every resource that has coordinates.
    Returns {str(resource_id): (lat, lon)}.  Silently skips if the DB or
    columns are absent so geo support degrades gracefully.
    """
    coords: dict[str, tuple[float, float]] = {}
    if not os.path.exists(db_path):
        return coords
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, latitude, longitude FROM resources "
            "WHERE latitude IS NOT NULL AND longitude IS NOT NULL "
            "AND latitude != '' AND longitude != ''"
        )
        for rid, lat, lon in cursor.fetchall():
            try:
                coords[str(rid)] = (float(lat), float(lon))
            except (ValueError, TypeError):
                pass
        conn.close()
    except Exception as exc:
        print(f"[geo] Could not load coordinates from {db_path}: {exc}")
    return coords


# ---------------------------------------------------------------------------
# Provider data stores
# ---------------------------------------------------------------------------
@dataclass
class BM25Index:
    """A BM25 index over a set of text chunks."""
    bm25: BM25Okapi
    doc_ids: list[int]  # position in the BM25 arrays → global text ID


@dataclass
class LanguageIndex:
    """An HNSW index for a single language partition."""
    index: hnswlib.Index
    local_to_global: dict[int, int]  # local contiguous ID → global text ID


@dataclass
class ProviderData:
    index: hnswlib.Index                           # combined (all languages)
    text_to_resource_id: dict
    embedded_texts: dict
    # Pre-built reverse mapping: resource_id → list of text indices
    resource_id_to_text_idxs: dict = field(default_factory=dict)
    # Language-specific HNSW indices (may be empty)
    language_indices: dict[str, LanguageIndex] = field(default_factory=dict)
    # BM25 indices
    bm25_all: Optional[BM25Index] = None
    bm25_by_language: dict[str, BM25Index] = field(default_factory=dict)
    # Geo coordinates: resource_id → (lat, lon).  Empty if DB unavailable.
    resource_coords: dict[str, tuple[float, float]] = field(default_factory=dict)


class ServiceState:
    """Holds the model and all loaded provider indexes."""

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.providers: dict[str, ProviderData] = {}

    def load_model(self):
        print(f"Loading model {MODEL_NAME} ...")
        if torch.cuda.is_available():
            model_kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
            device = "cuda"
        else:
            model_kwargs = {}
            device = "cpu"
        self.model = SentenceTransformer(
            MODEL_NAME,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"padding_side": "left"},
        )
        print("Model loaded.")

    # ------------------------------------------------------------------
    # BM25 index builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_bm25(
        embedded_texts: dict[int, str],
        global_ids: list[int],
    ) -> Optional[BM25Index]:
        """Build a BM25Okapi index for the given subset of text IDs."""
        # Filter to IDs that actually exist in embedded_texts
        if isinstance(embedded_texts, dict):
            valid = [(gid, embedded_texts[gid]) for gid in global_ids
                     if gid in embedded_texts]
        else:
            n = len(embedded_texts)
            valid = [(gid, embedded_texts[gid]) for gid in global_ids
                     if 0 <= gid < n]
        if not valid:
            return None
        doc_ids, texts = zip(*valid)
        tokenized = [tokenize(t) for t in texts]
        return BM25Index(bm25=BM25Okapi(tokenized), doc_ids=list(doc_ids))

    # ------------------------------------------------------------------
    # Provider loader
    # ------------------------------------------------------------------
    def load_provider(self, folder: str):
        folder = folder.strip()
        if not folder:
            return
        print(f"Loading provider: {folder}")
        provider_key = os.path.basename(os.path.normpath(folder))

        index_path = os.path.join(folder, "embeddings.bin")
        text_mapping_path = os.path.join(folder, "text_to_resource_mapping.pkl")
        embedded_texts_path = os.path.join(folder, "embedded_texts.pkl")

        for p in [index_path, text_mapping_path, embedded_texts_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing: {p}")

        dim = self.model.get_sentence_embedding_dimension()
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(index_path)

        with open(text_mapping_path, "rb") as f:
            text_to_resource_id = pickle.load(f)
        with open(embedded_texts_path, "rb") as f:
            embedded_texts = pickle.load(f)

        resource_id_to_text_idxs: dict[str, list[int]] = {}
        for idx, res_id in text_to_resource_id.items():
            resource_id_to_text_idxs.setdefault(res_id, []).append(idx)

        # ---------------------------------------------------------------
        # Language-specific HNSW indices
        # ---------------------------------------------------------------
        language_indices: dict[str, LanguageIndex] = {}
        _LANG_FILES = {
            "english": ("embeddings_english.bin", "english_local_to_global.pkl"),
            "french":  ("embeddings_french.bin",  "french_local_to_global.pkl"),
        }
        for lang, (idx_file, map_file) in _LANG_FILES.items():
            idx_path = os.path.join(folder, idx_file)
            map_path = os.path.join(folder, map_file)
            if os.path.exists(idx_path) and os.path.exists(map_path):
                lang_index = hnswlib.Index(space="cosine", dim=dim)
                lang_index.load_index(idx_path)
                with open(map_path, "rb") as f:
                    local_to_global = pickle.load(f)
                language_indices[lang] = LanguageIndex(
                    index=lang_index,
                    local_to_global=local_to_global,
                )
                print(f"  Loaded {lang} HNSW index for '{provider_key}' "
                      f"({lang_index.get_current_count()} segments)")
            else:
                print(f"  No {lang} HNSW index found for '{provider_key}' (skipped)")

        # ---------------------------------------------------------------
        # BM25 indices
        # ---------------------------------------------------------------
        t0 = time.perf_counter()

        # All-languages BM25
        all_ids = (sorted(embedded_texts.keys())
                   if isinstance(embedded_texts, dict)
                   else list(range(len(embedded_texts))))
        bm25_all = self._build_bm25(embedded_texts, all_ids)
        print(f"  Built BM25 (all) for '{provider_key}': "
              f"{len(all_ids)} docs in {time.perf_counter() - t0:.2f}s")

        # Per-language BM25
        bm25_by_language: dict[str, BM25Index] = {}
        for lang, lang_idx_info in language_indices.items():
            t1 = time.perf_counter()
            global_ids = sorted(lang_idx_info.local_to_global.values())
            bm25_idx = self._build_bm25(embedded_texts, global_ids)
            if bm25_idx is not None:
                bm25_by_language[lang] = bm25_idx
                print(f"  Built BM25 ({lang}) for '{provider_key}': "
                      f"{len(bm25_idx.doc_ids)} docs in "
                      f"{time.perf_counter() - t1:.2f}s")

        # Load geo coordinates (optional — no failure if absent)
        db_path = os.path.join(folder, "database.db")
        resource_coords = _load_resource_coords(db_path)
        print(f"  Loaded {len(resource_coords):,} resource coordinates for '{provider_key}'")

        self.providers[provider_key] = ProviderData(
            index=index,
            text_to_resource_id=text_to_resource_id,
            embedded_texts=embedded_texts,
            resource_id_to_text_idxs=resource_id_to_text_idxs,
            language_indices=language_indices,
            bm25_all=bm25_all,
            bm25_by_language=bm25_by_language,
            resource_coords=resource_coords,
        )
        langs = list(language_indices.keys()) or ["none"]
        print(f"Provider '{provider_key}' loaded "
              f"({len(text_to_resource_id)} texts, language indices: {langs})")


state = ServiceState()


# ---------------------------------------------------------------------------
# Dynamic batcher  (unchanged — handles model encoding)
# ---------------------------------------------------------------------------
class InferenceBatcher:
    """
    Collects encode requests arriving within a short time window and
    dispatches them as a single batched call to the model.
    """

    def __init__(self, model: SentenceTransformer, max_batch: int, window_ms: int):
        self.model = model
        self.max_batch = max_batch
        self.window_s = window_ms / 1000.0
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def encode(self, texts: list[str], mode: str = "query") -> np.ndarray:
        """
        Submit texts for encoding; returns when the batch completes.

        mode: "query"    → model.encode_query()
              "document" → model.encode_document() / model.encode()
              "raw"      → model.encode()
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put((texts, mode, future))
        return await future

    def _encode_batch(self, texts: list[str], mode: str) -> np.ndarray:
        """Synchronous encode — always returns float32."""
        if mode == "query":
            emb = self.model.encode_query(texts, show_progress_bar=False)
        elif mode == "document":
            if hasattr(self.model, "encode_document"):
                emb = self.model.encode_document(texts, show_progress_bar=False)
            else:
                emb = self.model.encode(texts, show_progress_bar=False)
        else:
            emb = self.model.encode(texts, show_progress_bar=False)
        return emb.astype(np.float32)

    async def _batch_loop(self):
        """Drain the queue, batch, encode, and resolve futures."""
        while True:
            batch_items: list[tuple[list[str], str, asyncio.Future]] = []
            total_texts = 0

            item = await self._queue.get()
            batch_items.append(item)
            total_texts += len(item[0])

            deadline = asyncio.get_event_loop().time() + self.window_s
            while total_texts < self.max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining,
                    )
                    batch_items.append(item)
                    total_texts += len(item[0])
                except asyncio.TimeoutError:
                    break

            groups: dict[str, list[tuple[list[str], asyncio.Future]]] = {}
            for texts, mode, future in batch_items:
                groups.setdefault(mode, []).append((texts, future))

            for mode, entries in groups.items():
                all_texts = []
                ranges = []
                for texts, future in entries:
                    start = len(all_texts)
                    all_texts.extend(texts)
                    ranges.append((start, start + len(texts), future))

                try:
                    embeddings = await asyncio.to_thread(
                        self._encode_batch, all_texts, mode,
                    )
                    for start, end, future in ranges:
                        if not future.cancelled():
                            future.set_result(embeddings[start:end])
                except Exception as exc:
                    for _, _, future in ranges:
                        if not future.cancelled():
                            future.set_exception(exc)


batcher: Optional[InferenceBatcher] = None


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------
def _hnsw_search(
    search_index: hnswlib.Index,
    query_vectors: np.ndarray,
    text_to_resource_id: dict,
    k: int,
    lang_index_info: Optional[LanguageIndex],
) -> list[tuple[str, float]]:
    """
    Run HNSW knn for each query vector.  Returns a de-duplicated list of
    (resource_id, best_distance) sorted by distance ascending (best first).
    """
    max_k = min(k * 2, search_index.get_current_count())
    resource_best: dict[str, float] = {}  # resource_id → best (lowest) distance

    for vec in query_vectors:
        indices, distances = search_index.knn_query([vec], k=max_k)
        for local_idx, dist in zip(indices[0], distances[0]):
            local_idx = int(local_idx)

            if lang_index_info is not None:
                global_idx = lang_index_info.local_to_global.get(local_idx)
                if global_idx is None:
                    continue
            else:
                global_idx = local_idx

            rid = text_to_resource_id.get(global_idx)
            if rid is None:
                continue
            rid = str(rid)
            dist_f = float(dist)
            # Keep the LOWEST distance (= most similar) per resource
            if rid not in resource_best or dist_f < resource_best[rid]:
                resource_best[rid] = dist_f

    # Sort ascending by distance (most similar first) for correct RRF ranking
    return sorted(resource_best.items(), key=lambda x: x[1])


def _bm25_search(
    bm25_idx: BM25Index,
    query_texts: list[str],
    text_to_resource_id: dict,
    k: int,
) -> list[tuple[str, float]]:
    """
    Run BM25 for each query string.  Returns a de-duplicated list of
    (resource_id, best_score) sorted by score descending (best first).
    """
    resource_best: dict[str, float] = {}

    for text in query_texts:
        tokens = tokenize(text)
        if not tokens:
            continue

        scores = bm25_idx.bm25.get_scores(tokens)

        # argpartition is O(N) vs O(N log N) for a full sort — much faster
        # for large corpora where we only need the top candidates.
        n_candidates = min(k * _BM25_OVERFETCH, len(scores))
        if n_candidates <= 0:
            continue
        top_positions = np.argpartition(scores, -n_candidates)[-n_candidates:]

        for pos in top_positions:
            score = float(scores[pos])
            if score <= 0.0:
                continue
            global_id = bm25_idx.doc_ids[pos]
            rid = text_to_resource_id.get(global_id)
            if rid is None:
                continue
            rid = str(rid)
            if rid not in resource_best or score > resource_best[rid]:
                resource_best[rid] = score

    # Sort descending by BM25 score (most relevant first)
    return sorted(resource_best.items(), key=lambda x: x[1], reverse=True)


def _reciprocal_rank_fusion(
    *rankings: list[tuple[str, float]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[list[float]] = None,
) -> dict[str, float]:
    """
    Merge multiple ranked lists with Reciprocal Rank Fusion.

    Each ranking is a list of (resource_id, _score) already sorted best-first.
    The original scores are ignored — only rank position matters.

    Returns {resource_id: rrf_score} where higher is better.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    rrf_scores: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (rid, _) in enumerate(ranking, start=1):
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + w / (k + rank)
    return rrf_scores


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global batcher
    state.load_model()
    for folder in PROVIDER_DIRS:
        if folder.strip():
            state.load_provider(folder)
    batcher = InferenceBatcher(state.model, MAX_BATCH_SIZE, BATCH_WINDOW_MS)
    batcher.start()
    yield
    await batcher.stop()


app = FastAPI(title="Embedding Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class EncodeRequest(BaseModel):
    texts: list[str]
    mode: str = "query"  # "query", "document", or "raw"


class EncodeResponse(BaseModel):
    embeddings: list[list[float]]


class GeoFilter(BaseModel):
    lat: float = Field(description="Centre latitude in decimal degrees.")
    lon: float = Field(description="Centre longitude in decimal degrees.")
    radius_km: float = Field(default=25.0, description="Hard radius cutoff in kilometres.")
    distance_weight: float = Field(
        default=0.6,
        description=(
            "How much geographic proximity boosts the relevance score.  "
            "A resource at the centre gets a multiplier of (1 + distance_weight); "
            "one at the edge of the radius gets 1.0 (no boost).  "
            "Set to 0 to filter-only without boosting."
        ),
    )
    include_unlocated: bool = Field(
        default=True,
        description=(
            "When True, resources without coordinates are kept with their raw "
            "relevance score (no geo boost).  When False they are excluded."
        ),
    )


class SearchRequest(BaseModel):
    provider: str
    queries: list[str] = Field(default_factory=list)
    passages: list[str] = Field(default_factory=list)
    k: int = 10
    language: Optional[str] = None
    # Backward-compat: accept old field names from existing clients
    positive_queries: list[str] = Field(default_factory=list, exclude=True)
    positive_passages: list[str] = Field(default_factory=list, exclude=True)

    def model_post_init(self, __context):
        # Merge old-style fields into the new ones so callers don't break
        if self.positive_queries and not self.queries:
            self.queries = self.positive_queries
        if self.positive_passages and not self.passages:
            self.passages = self.positive_passages
    # Optional geographic filter / boost
    geo_filter: Optional[GeoFilter] = Field(
        default=None,
        description=(
            "When set, results are filtered to resources within radius_km of the "
            "given coordinates and boosted by proximity.  Resources without "
            "coordinates are kept or dropped based on include_unlocated."
        ),
    )
    # When geo_filter is active we over-fetch internally before applying the
    # radius cut so the final result list still has k items where possible.
    geo_overfetch: int = Field(
        default=5,
        description="Internal over-fetch multiplier applied when geo_filter is set.",
    )

    # RRF fusion tuning
    rrf_k: int = Field(default=DEFAULT_RRF_K, description=(
        "RRF constant.  Lower values increase the influence of top-ranked "
        "results; 60 is the standard default from the original RRF paper."
    ))
    vector_weight: float = Field(default=1.0, description=(
        "Multiplier for the dense (HNSW) ranking's contribution to RRF."
    ))
    bm25_weight: float = Field(default=1.0, description=(
        "Multiplier for the sparse (BM25) ranking's contribution to RRF.  "
        "Set to 0 to disable BM25 and fall back to pure vector search."
    ))


class SearchResult(BaseModel):
    resource_scores: dict[str, float]  # resource_id → RRF score (higher = better)


class HealthResponse(BaseModel):
    status: str
    providers: list[str]
    model: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        providers=list(state.providers.keys()),
        model=MODEL_NAME,
    )


@app.post("/encode", response_model=EncodeResponse)
async def encode(req: EncodeRequest):
    """Encode texts into embeddings (batched automatically)."""
    embeddings = await batcher.encode(req.texts, req.mode)
    return EncodeResponse(embeddings=embeddings.tolist())


@app.post("/search", response_model=SearchResult)
async def search(req: SearchRequest):
    """
    Hybrid search pipeline:
      1. Encode queries into dense vectors  (via the batched model)
      2. Run HNSW (dense) and BM25 (sparse) retrieval in parallel
      3. Fuse results with Reciprocal Rank Fusion
      4. Return top-k resource_id → score

    The optional `language` field restricts results to documents in that
    language.  Accepts: english, anglais, french, français, francais, all,
    or None (= all).
    """
    provider_key = os.path.basename(os.path.normpath(req.provider))
    if provider_key not in state.providers:
        raise HTTPException(
            status_code=404,
            detail=f"Provider '{provider_key}' not loaded. "
                   f"Available: {list(state.providers.keys())}",
        )

    provider = state.providers[provider_key]

    if not req.queries and not req.passages:
        raise HTTPException(status_code=400, detail="At least one query is required.")

    # ---------------------------------------------------------------
    # Resolve language-specific indices
    # ---------------------------------------------------------------
    lang = normalise_language(req.language)
    lang_index_info: Optional[LanguageIndex] = None

    if lang != "all" and lang in provider.language_indices:
        lang_index_info = provider.language_indices[lang]
        search_hnsw = lang_index_info.index
        search_bm25 = provider.bm25_by_language.get(lang, provider.bm25_all)
    elif lang != "all" and lang not in provider.language_indices:
        print(f"[WARN] No '{lang}' index for provider '{provider_key}'; "
              f"falling back to combined index")
        search_hnsw = provider.index
        search_bm25 = provider.bm25_all
    else:
        search_hnsw = provider.index
        search_bm25 = provider.bm25_all

    # ---------------------------------------------------------------
    # Encode queries (dense vectors via the batched model)
    # ---------------------------------------------------------------
    encode_tasks = []
    if req.queries:
        encode_tasks.append(batcher.encode(req.queries, "query"))
    if req.passages:
        encode_tasks.append(batcher.encode(req.passages, "document"))

    encoded = await asyncio.gather(*encode_tasks)
    query_vectors = np.concatenate(encoded, axis=0).astype(np.float32)

    # Collect raw query strings for BM25 (both queries and passages)
    bm25_query_texts = list(req.queries) + list(req.passages)

    k = req.k
    # When geo filtering is active we over-fetch so that after the radius cut
    # we still have roughly k results.
    fetch_k = k * req.geo_overfetch if req.geo_filter is not None else k

    # ---------------------------------------------------------------
    # Run HNSW and BM25 in parallel (both are CPU-bound)
    # ---------------------------------------------------------------
    hnsw_future = asyncio.to_thread(
        _hnsw_search,
        search_hnsw, query_vectors, provider.text_to_resource_id,
        fetch_k, lang_index_info,
    )

    if search_bm25 is not None and req.bm25_weight > 0:
        bm25_future = asyncio.to_thread(
            _bm25_search,
            search_bm25, bm25_query_texts, provider.text_to_resource_id, fetch_k,
        )
        hnsw_ranking, bm25_ranking = await asyncio.gather(
            hnsw_future, bm25_future,
        )
    else:
        hnsw_ranking = await hnsw_future
        bm25_ranking = []

    # ---------------------------------------------------------------
    # Fuse with RRF
    # ---------------------------------------------------------------
    if bm25_ranking:
        rrf_scores = _reciprocal_rank_fusion(
            hnsw_ranking,
            bm25_ranking,
            k=req.rrf_k,
            weights=[req.vector_weight, req.bm25_weight],
        )
    else:
        # No BM25 results — use vector ranking directly (convert distance
        # to a score so higher = better, consistent with the RRF path)
        rrf_scores = {
            rid: 1.0 / (req.rrf_k + rank)
            for rank, (rid, _) in enumerate(hnsw_ranking, start=1)
        }

    # ---------------------------------------------------------------
    # Optional: geo filter + proximity boost
    # ---------------------------------------------------------------
    if req.geo_filter is not None:
        gf = req.geo_filter
        geo_scores: dict[str, float] = {}
        for rid, rrf_score in rrf_scores.items():
            coords = provider.resource_coords.get(rid)
            if coords is None:
                # Resource has no coordinates
                if gf.include_unlocated:
                    geo_scores[rid] = rrf_score  # no boost, no penalty
            else:
                dist_km = haversine_km(gf.lat, gf.lon, coords[0], coords[1])
                if dist_km <= gf.radius_km:
                    # Linear proximity bonus: 1.0 at centre, 0.0 at the edge
                    proximity = max(0.0, 1.0 - dist_km / gf.radius_km)
                    geo_scores[rid] = rrf_score * (1.0 + gf.distance_weight * proximity)
        rrf_scores = geo_scores

    # ---------------------------------------------------------------
    # Sort descending and truncate to k
    # ---------------------------------------------------------------
    sorted_scores = dict(
        sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    )

    return SearchResult(resource_scores=sorted_scores)