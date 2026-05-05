from collections.abc import Iterator
import logging

from src.models.schemas import CitationItem, RetrievedChunk
from src.services.llm.providers import get_llm_client
from src.services.retriever import get_retriever
from src.utils.jurisdiction import detect_chunk_state, infer_state, is_central_reference

logger = logging.getLogger(__name__)


def _safe_metadata_value(metadata: dict, keys: list[str], default: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


class RAGService:
    def __init__(self) -> None:
        try:
            self.retriever = get_retriever()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retriever unavailable; continuing without Chroma context: %s", exc)
            self.retriever = None
        self.llm = get_llm_client()

    def _retrieve_chunks(self, *, question: str, clause_text: str | None, top_k: int | None) -> list[RetrievedChunk]:
        if self.retriever is None:
            return []

        retrieval_query = question
        if not infer_state(question):
            derived_state = infer_state(clause_text or "")
            if derived_state:
                retrieval_query = f"{question} {derived_state}"

        try:
            return self.retriever.query(retrieval_query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed; answering from uploaded document only: %s", exc)
            return []

    @staticmethod
    def _length_guidance(clause_text: str | None, chunks: list[RetrievedChunk]) -> str:
        combined_text = " ".join([clause_text or ""] + [chunk.text for chunk in chunks])
        source_words = len(combined_text.split())

        if source_words >= 2500:
            practical_target = "300-500 words"
        elif source_words >= 1200:
            practical_target = "200-300 words"
        else:
            practical_target = "100-150 words"

        return (
            "LENGTH CONTROL:\n"
            "- Target summary length: 10-15% of the original document.\n"
            "- For large judgments: 300-500 words.\n"
            "- For medium documents: 200-300 words.\n"
            "- For short documents: 100-150 words.\n"
            f"- Practical target for this request: {practical_target}.\n"
        )

    @staticmethod
    def _build_prompt(*, question: str, clause_text: str | None, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join([f"- {chunk.text}" for chunk in chunks])
        length_guidance = RAGService._length_guidance(clause_text, chunks)
        return (
            "You are an Indian legal analysis assistant for enterprise users (law firms, CA firms, insurance, real estate). "
            "Generate a structured legal summary in plain English.\n\n"
            "Context:\n"
            f"{context}\n\n"
            f"User Clause (if provided):\n{clause_text or 'N/A'}\n\n"
            f"User Query:\n{question}\n\n"
            "Task Requirements:\n"
            f"{length_guidance}\n"
            "Use exactly these sections and order:\n\n"
            "1. Document Metadata\n"
            "- Document Type:\n"
            "- Document Title:\n"
            "- Parties Involved:\n"
            "- Effective Date:\n"
            "- Document Duration / Validity:\n"
            "- Governing Law / Jurisdiction:\n"
            "- Key {document_type} Terms:\n\n"
            "2. Background Context\n"
            "Provide a short explanation of the situation that led to this legal document or dispute. "
            "Focus on events and circumstances relevant to the case.\n\n"
            "3. Legal Issues, Statutes, Arguments, and Takeaways\n"
            "Identify the main legal questions, statutes, or sections involved.\n"
            "Summarize important arguments by parties (for cases) OR key contractual clauses/obligations (for agreements).\n"
            "Include important takeaways as bullets:\n"
            "- legal principles applied\n"
            "- precedent value\n"
            "- compliance implications\n\n"
            "Formatting Rules:\n"
            "STYLE REQUIREMENTS:\n"
            "- Use clear and concise sentences.\n"
            "- Avoid unnecessary legal jargon where possible.\n"
            "- Organize the information using headings and bullet points where helpful.\n"
            "- Keep the output professional, clear, and actionable.\n"
            "- If any metadata is not available in context, write 'Not found in provided context'.\n"
            "- Do not invent citations or facts.\n"
            "- End the response with a section titled 'Follow-up Questions'.\n"
            "- In that section, include 2-4 context-aware questions that help the user continue the conversation.\n"
            "- Follow-up questions must be specific to the current document/context, not generic.\n"
        )

    @staticmethod
    def _build_citations(
        chunks: list[RetrievedChunk],
        *,
        question: str,
        clause_text: str | None,
    ) -> list[CitationItem]:
        max_citations = min(10, len(chunks))
        target_state = infer_state(f"{question}\n{clause_text or ''}")
        citations: list[CitationItem] = []
        seen_keys: set[tuple[str, str]] = set()

        def add_citation(idx: int, chunk: RetrievedChunk, *, strict_filter: bool) -> None:
            nonlocal citations
            if len(citations) >= max_citations:
                return
            metadata = chunk.metadata or {}
            chunk_state = detect_chunk_state(metadata, chunk.text)

            if strict_filter:
                # If target state is present, suppress citations from other states.
                if target_state and chunk_state and chunk_state != target_state:
                    return

                # Prefer target-state and central references. Drop ambiguous references
                # when we already have a specific target state context.
                if target_state and not chunk_state and not is_central_reference(metadata, chunk.text):
                    return

            citation = _safe_metadata_value(
                metadata,
                ["citation", "act_name", "title", "source", "document_name"],
                f"Reference {idx}",
            )
            section = _safe_metadata_value(
                metadata,
                ["section", "section_number", "article", "clause", "provision"],
                "Not found",
            )
            reference_type = _safe_metadata_value(
                metadata,
                ["reference_type", "doc_type", "type", "category"],
                "legal_reference",
            )
            if target_state and chunk_state == target_state:
                reference_type = f"{target_state}_state_act"
            elif is_central_reference(metadata, chunk.text):
                reference_type = "central_act"

            dedupe_key = (citation.strip().lower(), section.strip().lower())
            if dedupe_key in seen_keys:
                return
            seen_keys.add(dedupe_key)

            citations.append(
                CitationItem(
                    citation=citation,
                    section=section,
                    reference_type=reference_type,
                    score=chunk.score,
                )
            )

        # Pass 1: strict state-aware filtering
        for idx, chunk in enumerate(chunks, start=1):
            add_citation(idx, chunk, strict_filter=True)
            if len(citations) >= max_citations:
                return citations

        # Pass 2: relaxed fill to reach up to 10 citations
        for idx, chunk in enumerate(chunks, start=1):
            add_citation(idx, chunk, strict_filter=False)
            if len(citations) >= max_citations:
                break

        return citations

    def answer(
        self,
        *,
        question: str,
        clause_text: str | None = None,
        top_k: int | None = None,
    ) -> tuple[str, list[RetrievedChunk], list[CitationItem]]:
        chunks = self._retrieve_chunks(question=question, clause_text=clause_text, top_k=top_k)
        prompt = self._build_prompt(question=question, clause_text=clause_text, chunks=chunks)
        answer = self.llm.generate(prompt)
        citations = self._build_citations(chunks, question=question, clause_text=clause_text)
        return answer, chunks, citations

    def answer_stream(
        self,
        *,
        question: str,
        clause_text: str | None = None,
        top_k: int | None = None,
    ) -> tuple[Iterator[str], list[RetrievedChunk], list[CitationItem]]:
        chunks = self._retrieve_chunks(question=question, clause_text=clause_text, top_k=top_k)
        prompt = self._build_prompt(question=question, clause_text=clause_text, chunks=chunks)
        stream = self.llm.stream_generate(prompt)
        citations = self._build_citations(chunks, question=question, clause_text=clause_text)
        return stream, chunks, citations
