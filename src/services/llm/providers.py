import logging
import re
from collections.abc import Iterator

from ollama import Client

from src.core.config import get_settings
from src.services.llm.base import LLMClient

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    def generate(self, prompt: str) -> str:
        if "The user has not uploaded a document." in prompt:
            question_match = re.search(
                r"User Question:\n(?P<question>.*?)\n\nTask Requirements:",
                prompt,
                flags=re.DOTALL,
            )
            question_text = (question_match.group("question").strip() if question_match else "") or "the topic"
            return (
                "1. Quick Answer\n"
                f"{question_text.capitalize()} is a general legal topic, so the answer should be read as broad guidance rather than advice on a specific uploaded document.\n\n"
                "2. Key Legal Concepts\n"
                "- Indian laws are rules created by Parliament, state legislatures, courts, and regulators.\n"
                "- A clause is a specific part of a contract or legal text that describes a right, duty, restriction, or remedy.\n"
                "- Legal risk usually means financial loss, penalty exposure, compliance failure, dispute risk, or unclear obligations.\n\n"
                "3. Common Clauses or Risks\n"
                "- Payment clauses define who pays, how much, and by when.\n"
                "- Penalty or damages clauses define consequences of breach or delay.\n"
                "- Termination clauses explain how an agreement can end.\n"
                "- Liability and indemnity clauses decide who bears loss or legal responsibility.\n"
                "- Governing law and dispute resolution clauses decide which law applies and how disputes are handled.\n\n"
                "4. Practical Notes\n"
                "- Check important deadlines, payment obligations, notice periods, and penalty language carefully.\n"
                "- Broad indemnity, unlimited liability, one-sided termination, and automatic renewal clauses often need extra attention.\n"
                "- For a document-specific explanation, upload the legal PDF so the assistant can analyze its actual wording.\n\n"
                "5. Follow-up Questions\n"
                "- Do you want a simple explanation of the most important contract clauses?\n"
                "- Should I explain legal risk in agreements with examples?\n"
                "- Do you want the difference between penalty, liability, and indemnity clauses?\n"
            )

        clause_match = re.search(
            r"User Clause \(if provided\):\n(?P<clause>.*?)\n\nUser Query:",
            prompt,
            flags=re.DOTALL,
        )
        context_match = re.search(
            r"Context:\n(?P<context>.*?)\n\nUser Clause",
            prompt,
            flags=re.DOTALL,
        )
        clause_text = (clause_match.group("clause").strip() if clause_match else "") or "N/A"
        context_text = (context_match.group("context").strip() if context_match else "")
        has_clause = clause_text != "N/A"
        has_context = bool(context_text)
        context_note = (
            "The assistant retrieved supporting Indian legal context from the local vector database and used it with the uploaded document text."
            if has_context
            else "Because the local Chroma legal database is not available, this explanation is based only on the uploaded document text."
        )

        return (
            "1. Document Metadata\n"
            "- Document Type: Legal document / agreement\n"
            "- Document Title: Not found in provided context\n"
            "- Parties Involved: Not found in provided context\n"
            "- Effective Date: Not found in provided context\n"
            "- Document Duration / Validity: Not found in provided context\n"
            "- Governing Law / Jurisdiction: India context inferred from the assistant settings\n"
            "- Key legal document terms: Review payment, penalty, termination, liability, and dispute clauses.\n\n"
            "2. Background Context\n"
            "The uploaded document appears to contain legal obligations that need to be converted into plain English. "
            f"{context_note}\n\n"
            "3. Legal Issues, Statutes, Arguments, and Takeaways\n"
            "- Main issue: Identify obligations, risks, deadlines, and consequences from the document.\n"
            "- Legal principle applied: Contract terms should be checked for enforceability, clarity, penalties, liability, and dispute resolution.\n"
            "- Compliance implication: Confirm important dates, payment duties, default consequences, and required notices before signing or acting.\n"
            "- Practical takeaway: Ask a lawyer to review any high-value penalty, broad indemnity, automatic renewal, or one-sided termination language.\n\n"
            "Provided document context:\n"
            f"{clause_text[:1200] if has_clause else 'Not found in provided context'}\n\n"
            "Follow-up Questions\n"
            "- Which clause creates the biggest payment or penalty exposure?\n"
            "- Are there any important deadlines or notice periods in this document?\n"
            "- Does the document contain termination, renewal, or dispute resolution terms?"
        )

    def stream_generate(self, prompt: str) -> Iterator[str]:
        text = self.generate(prompt)
        for token in text.split(" "):
            yield token + " "


class OllamaLLMClient(LLMClient):
    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.ollama_model
        self.temperature = settings.ollama_temperature
        headers = {}
        if settings.ollama_api_key:
            headers["Authorization"] = f"Bearer {settings.ollama_api_key}"
        self.client = Client(
            host=settings.ollama_base_url,
            headers=headers or None,
            timeout=settings.ollama_timeout_seconds,
        )

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": self.temperature},
        )
        return str(response.get("message", {}).get("content", ""))

    def stream_generate(self, prompt: str) -> Iterator[str]:
        stream = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": self.temperature},
        )
        for part in stream:
            content = str(part.get("message", {}).get("content", ""))
            if not content:
                continue
            yield content


def get_llm_client() -> LLMClient:
    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        if settings.ollama_base_url.rstrip("/") == "https://ollama.com" and not settings.ollama_api_key:
            logger.info("OLLAMA_API_KEY is not configured; using mock LLM provider")
            return MockLLMClient()
        return OllamaLLMClient()

    logger.info("Using mock LLM provider")
    return MockLLMClient()
