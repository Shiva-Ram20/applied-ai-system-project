from pathlib import Path

from src.rag import RAGRetriever


def test_search_guidelines_finds_relevant_snippet(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "feeding.md").write_text(
        "## Puppy Feeding\nFeed puppies three to four times a day.\n\n"
        "## Adult Dog Feeding\nFeed adult dogs once or twice a day.\n"
    )
    (kb / "breeds.md").write_text(
        "## Golden Retriever\nGolden Retrievers need daily exercise and are prone to hip dysplasia.\n"
    )

    retriever = RAGRetriever(kb)
    results = retriever.search_guidelines("how often should I feed a puppy", top_k=1)

    assert len(results) == 1
    assert "puppy" in results[0].lower()


def test_search_guidelines_returns_empty_for_missing_knowledge_base(tmp_path: Path) -> None:
    retriever = RAGRetriever(tmp_path / "does-not-exist")
    assert retriever.search_guidelines("anything") == []


def test_search_guidelines_returns_empty_for_unrelated_query(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "feeding.md").write_text("## Puppy Feeding\nFeed puppies three to four times a day.\n")

    retriever = RAGRetriever(kb)
    assert retriever.search_guidelines("xyzzy quantum spreadsheet") == []


def test_retrieve_pet_context_joins_snippets(tmp_path: Path) -> None:
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "breeds.md").write_text(
        "## Golden Retriever\nGolden Retrievers need daily exercise and are prone to hip dysplasia.\n"
    )

    retriever = RAGRetriever(kb)
    context = retriever.retrieve_pet_context("Golden Retriever", "exercise")

    assert "Golden Retriever" in context


def test_tool_schema_and_handle_tool_call_roundtrip(tmp_path: Path) -> None:
    import json

    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "feeding.md").write_text("## Puppy Feeding\nFeed puppies three to four times a day.\n")

    retriever = RAGRetriever(kb)
    schema = RAGRetriever.tool_schema()
    assert schema["name"] == "search_guidelines"

    payload = json.loads(retriever.handle_tool_call({"query": "puppy feeding", "top_k": 1}))
    assert payload["results"]
