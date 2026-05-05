import logging
import math
import re
import time
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from src.core.config import get_settings
from src.models.schemas import RetrievedChunk
from src.utils.hf_paths import resolve_hf_model_path
from src.utils.jurisdiction import detect_chunk_state, infer_state, is_central_reference

logger = logging.getLogger(__name__)

QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "under",
    "is",
    "are",
}


class LegalRetriever:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = chromadb.PersistentClient(path=str(self.settings.vector_db_path))
        try:
            self._collection = self._client.get_collection(name=self.settings.chroma_collection)
        except Exception:  # noqa: BLE001
            collections = self._client.list_collections()
            if not collections:
                raise ValueError(
                    f"No Chroma collections found in {self.settings.vector_db_path}. "
                    "Add data or set CHROMA_COLLECTION correctly."
                )
            self._collection = self._client.get_collection(name=collections[0].name)
            logger.warning(
                "Configured collection '%s' not found. Falling back to '%s'.",
                self.settings.chroma_collection,
                collections[0].name,
            )

        model_source = self.settings.embedding_model_local_path or self.settings.embedding_model_name
        resolved_model_source = resolve_hf_model_path(model_source)
        self._embedding_model = SentenceTransformer(resolved_model_source)

    @staticmethod
    def _prefix_text(text: str, *, is_query: bool) -> str:
        prefix = QUERY_PREFIX if is_query else DOCUMENT_PREFIX
        return f"{prefix}{text.strip()}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _clean_text(text: str) -> str:
        replacements = {
            "\u00e2\u0080\u0094": "-",
            "\u00e2\u0080\u0093": "-",
            "\u00e2\u0080\u0098": "'",
            "\u00e2\u0080\u0099": "'",
            "\u00e2\u0080\u009c": '"',
            "\u00e2\u0080\u009d": '"',
            "\u00e2\u0080\u00a2": "-",
            "\u00e2\u0080\u00af": " ",
            "\u00e2\u0082\u00b9": "Rs.",
            "\u00c2": "",
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        return text

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}

    @staticmethod
    def _keyword_overlap_score(query_text: str, doc_text: str) -> float:
        q = LegalRetriever._tokenize(query_text)
        if not q:
            return 0.0
        d = LegalRetriever._tokenize(doc_text)
        return len(q.intersection(d)) / len(q)

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _expand_query(query_text: str) -> str:
        q = query_text.strip()
        q_lower = q.lower()
        hints = []
        if any(k in q_lower for k in ["section", "act", "statute", "law"]):
            hints.append("indian statute legal section")
        if any(k in q_lower for k in ["contract", "agreement", "clause"]):
            hints.append("contractual obligation liability termination penalty")
        if any(k in q_lower for k in ["compliance", "regulation", "authority"]):
            hints.append("regulatory compliance india authority")
        if not hints:
            return q
        return f"{q} {' '.join(hints)}"

    @staticmethod
    def _jurisdiction_score(
        *,
        target_state: str | None,
        chunk_state: str | None,
        metadata: dict,
        text: str,
    ) -> float:
        if not target_state:
            return 0.0
        if chunk_state == target_state:
            return 0.22
        if chunk_state and chunk_state != target_state:
            return -0.30
        if is_central_reference(metadata, text):
            return 0.08
        return -0.05

    def embed_texts(self, texts: list[str], *, is_query: bool) -> list[list[float]]:
        prefixed = [self._prefix_text(t, is_query=is_query) for t in texts]
        embeddings = self._embedding_model.encode(
            prefixed,
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    @lru_cache(maxsize=1024)
    def _embed_query_cached(self, query: str) -> tuple[float, ...]:
        return tuple(self.embed_texts([query], is_query=True)[0])

    def query(self, query_text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if not query_text.strip():
            return []

        started = time.perf_counter()
        base_k = top_k or self.settings.top_k_retrieval
        fetch_k = max(base_k * 4, base_k + 6)
        expanded_query = self._expand_query(query_text)
        target_state = infer_state(query_text)
        query_embedding = [list(self._embed_query_cached(expanded_query))]

        result = self._collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        if not documents:
            return []

        doc_embeddings = self.embed_texts(documents, is_query=False)
        query_vector = list(self._embed_query_cached(expanded_query))

        candidates = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
            distance = distances[idx] if idx < len(distances) else None
            semantic_score = self._cosine_similarity(query_vector, doc_embeddings[idx])
            keyword_score = self._keyword_overlap_score(query_text, text)
            distance_score = 1.0 / (1.0 + distance) if distance is not None else semantic_score
            chunk_state = detect_chunk_state(metadata, text)
            jurisdiction_score = self._jurisdiction_score(
                target_state=target_state,
                chunk_state=chunk_state,
                metadata=metadata,
                text=text,
            )
            fused_score = (
                (0.58 * semantic_score)
                + (0.23 * keyword_score)
                + (0.12 * distance_score)
                + (0.07 * jurisdiction_score)
            )
            candidates.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                    "embedding": doc_embeddings[idx],
                    "semantic_score": semantic_score,
                    "keyword_score": keyword_score,
                    "jurisdiction_score": jurisdiction_score,
                    "chunk_state": chunk_state,
                    "fused_score": fused_score,
                }
            )

        # Deduplicate repeated chunks by normalized text and keep best scored instance.
        deduped: dict[str, dict] = {}
        for item in candidates:
            key = self._normalize_text(item["text"])
            prev = deduped.get(key)
            if prev is None or item["fused_score"] > prev["fused_score"]:
                deduped[key] = item

        ranked = sorted(deduped.values(), key=lambda x: x["fused_score"], reverse=True)

        # Diversity-aware final selection (MMR-like).
        selected: list[dict] = []
        lambda_relevance = 0.78
        while ranked and len(selected) < base_k:
            best_idx = 0
            best_score = -1e9
            for idx, item in enumerate(ranked):
                max_similarity = 0.0
                if selected:
                    max_similarity = max(
                        self._cosine_similarity(item["embedding"], picked["embedding"]) for picked in selected
                    )
                mmr_score = (lambda_relevance * item["fused_score"]) - ((1 - lambda_relevance) * max_similarity)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(ranked.pop(best_idx))

        chunks: list[RetrievedChunk] = []
        for item in selected:
            enriched_metadata = dict(item["metadata"])
            enriched_metadata["_retrieval"] = {
                "semantic_score": round(item["semantic_score"], 6),
                "keyword_score": round(item["keyword_score"], 6),
                "distance": item["distance"],
                "jurisdiction_score": round(item["jurisdiction_score"], 6),
                "target_state": target_state or "",
                "chunk_state": item["chunk_state"] or "",
                "fused_score": round(item["fused_score"], 6),
            }
            chunks.append(
                RetrievedChunk(
                    text=self._clean_text(item["text"]),
                    metadata=enriched_metadata,
                    score=item["fused_score"],
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Retriever result | query='%s' | requested_k=%s | fetched=%s | deduped=%s | returned=%s | latency_ms=%.2f",
            query_text[:120],
            base_k,
            len(documents),
            len(deduped),
            len(chunks),
            elapsed_ms,
        )
        return chunks


@lru_cache(maxsize=1)
def get_retriever() -> LegalRetriever:
    return LegalRetriever()
