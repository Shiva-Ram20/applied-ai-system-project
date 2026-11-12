"""RAGRetriever: read-only retrieval over PawPal+'s local knowledge base.

Loads Markdown/text documents from a knowledge_base/ folder (pet care
guidelines, feeding rules, breed notes), chunks them by heading, and ranks
chunks against a query with a small pure-Python TF-IDF index -- no extra
dependencies, so this module stays importable (and testable) without an
Anthropic API key.

This class only reads from disk and never touches Owner/Pet/Task/Schedule --
it is wired into AgentPlanner as a tool, not a source of scheduling logic.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _split_into_chunks(text: str, source: str) -> list[dict]:
    """Split a document into (heading, text) chunks at Markdown headings.

    Text before the first heading (if any) becomes an "Introduction" chunk.
    A document with no headings at all becomes a single chunk.
    """
    headings = list(_HEADING_RE.finditer(text))
    chunks: list[dict] = []

    if not headings:
        stripped = text.strip()
        if stripped:
            chunks.append({"source": source, "heading": source, "text": stripped})
        return chunks

    if headings[0].start() > 0:
        lead = text[: headings[0].start()].strip()
        if lead:
            chunks.append({"source": source, "heading": "Introduction", "text": lead})

    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        section = text[start:end].strip()
        if section:
            chunks.append({"source": source, "heading": match.group(2).strip(), "text": section})

    return chunks


class RAGRetriever:
    """Loads knowledge_base/ documents and answers `search_guidelines` queries."""

    def __init__(self, knowledge_base_path: str | Path = "knowledge_base") -> None:
        self.knowledge_base_path = Path(knowledge_base_path)
        self.documents: list[dict] = []
        self._idf: dict[str, float] = {}
        self._chunk_vectors: list[dict[str, float]] = []
        self._chunk_norms: list[float] = []
        self.load_documents()

    def load_documents(self) -> None:
        """(Re)load every .md/.txt file in knowledge_base_path and rebuild the index."""
        self.documents = []
        if self.knowledge_base_path.exists():
            paths = sorted(
                path
                for path in self.knowledge_base_path.iterdir()
                if path.is_file() and path.suffix.lower() in (".md", ".txt")
            )
            for path in paths:
                text = path.read_text(encoding="utf-8")
                self.documents.extend(_split_into_chunks(text, path.name))
        self._build_index()

    def _build_index(self) -> None:
        chunk_tokens = [_tokenize(chunk["text"]) for chunk in self.documents]

        doc_freq: Counter[str] = Counter()
        for tokens in chunk_tokens:
            doc_freq.update(set(tokens))

        n_docs = len(chunk_tokens) or 1
        self._idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0 for term, freq in doc_freq.items()
        }

        self._chunk_vectors = []
        self._chunk_norms = []
        for tokens in chunk_tokens:
            tf = Counter(tokens)
            vector = {term: count * self._idf.get(term, 0.0) for term, count in tf.items()}
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
            self._chunk_vectors.append(vector)
            self._chunk_norms.append(norm)

    def search_guidelines(self, query: str, top_k: int = 3) -> list[str]:
        """Return up to `top_k` text snippets most relevant to `query`.

        Ranking is cosine similarity over a TF-IDF vector built across every
        chunk in the knowledge base. Returns [] if the knowledge base is
        empty or the query shares no vocabulary with any chunk.
        """
        if not self.documents:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        tf = Counter(query_tokens)
        query_vector = {term: count * self._idf.get(term, 0.0) for term, count in tf.items()}
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

        scores: list[tuple[float, int]] = []
        for i, chunk_vector in enumerate(self._chunk_vectors):
            dot = sum(weight * chunk_vector.get(term, 0.0) for term, weight in query_vector.items())
            if dot <= 0:
                continue
            scores.append((dot / (query_norm * self._chunk_norms[i]), i))

        scores.sort(key=lambda pair: pair[0], reverse=True)

        return [
            f"[{self.documents[i]['source']} — {self.documents[i]['heading']}] {self.documents[i]['text']}"
            for _, i in scores[:top_k]
        ]

    def retrieve_pet_context(self, pet_species: str, condition: str = "") -> str:
        """Convenience wrapper: build a query from species (+ optional condition)
        and return the matching guideline snippets joined into one string,
        ready to drop into an LLM prompt."""
        query = f"{pet_species} {condition}".strip()
        snippets = self.search_guidelines(query, top_k=3)
        if not snippets:
            return "No matching guidelines found in the knowledge base."
        return "\n\n".join(snippets)

    @staticmethod
    def tool_schema() -> dict:
        """Anthropic tool definition exposing search_guidelines to an LLM agent."""
        return {
            "name": "search_guidelines",
            "description": (
                "Search PawPal+'s internal pet care knowledge base (feeding rules, "
                "breed notes, general care guidelines) for passages relevant to a "
                "query. Call this before proposing, adding, or rescheduling any "
                "task so the plan is grounded in the household's care guidelines."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look up, e.g. 'puppy feeding frequency' or "
                            "'Golden Retriever exercise needs'."
                        ),
                    },
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "strict": True,
        }

    def handle_tool_call(self, tool_input: dict) -> str:
        """Execute a search_guidelines tool call and return a JSON string result."""
        query = tool_input.get("query", "")
        top_k = int(tool_input.get("top_k") or 3)
        snippets = self.search_guidelines(query, top_k=top_k)
        if not snippets:
            return json.dumps({"query": query, "results": [], "note": "No matching guidelines found."})
        return json.dumps({"query": query, "results": snippets})
