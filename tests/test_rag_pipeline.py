from __future__ import annotations

import json
from config import DEFAULT_QWEN_MODEL, OPENAI_REASONING_EFFORT
from services.providers import ProviderClient
from services.providers import _PROVIDER_MODEL_MAP
from services.rag_pipeline import (
    AnswerResult,
    AnalysisResult,
    Citation,
    ScanMatch,
    _anthropic_thinking_kwargs,
    _analysis_answer_schema,
    _full_document_budget,
    _analysis_schema,
    _extract_chat_completion_text,
    _merge_scan_matches,
    _openai_reasoning_kwargs,
    _retrieve_context_from_index,
    answer_query,
    answer_query_from_full_document,
    build_analysis_snippets,
    generate_analysis,
    generate_analysis_progress,
    split_into_chunks,
    split_into_scan_chunks,
)


class DummyBlock:
    def __init__(self, block_type: str, text: str) -> None:
        self.type = block_type
        self.text = text


class DummyOutputItem:
    def __init__(self, block: DummyBlock) -> None:
        self.content = [block]


class DummyResponse:
    def __init__(self, block: DummyBlock) -> None:
        self.output = [DummyOutputItem(block)]
        self.output_text = block.text


class DummyResponsesClient:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return DummyResponse(DummyBlock("output_json_schema", self._payload))


class DummyOpenAIClient:
    def __init__(self, payload: str) -> None:
        self.responses = DummyResponsesClient(payload)


class DummyChatCompletionsClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def create(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls.append(kwargs["model"])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DummyChatClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = DummyChatCompletionsClient(responses)


class DummyOpenAIChatClient:
    def __init__(self, responses: list[object]) -> None:
        self.chat = DummyChatClient(responses)


class DummyTokenizer:
    def encode(self, text: str, *args, **kwargs):  # noqa: ANN002, ANN003
        return list(range(len(text.split())))


class FakeVectorStore:
    def __init__(self, documents: list[str]) -> None:
        self._documents = documents

    def similarity_search(self, question: str, k: int):  # noqa: ARG002
        limit = min(k, len(self._documents))
        return [type("Doc", (), {"page_content": doc})() for doc in self._documents[:limit]]


def test_analysis_schema_contains_required_fields():
    schema = _analysis_schema()
    assert set(schema["required"]) == {
        "executive_summary",
        "bill_summary",
        "implementation",
        "critique",
        "swot",
    }
    implementation_item = schema["properties"]["implementation"]["items"]
    critique_item = schema["properties"]["critique"]["items"]
    assert set(implementation_item["required"]) == {
        "stakeholder",
        "obligation",
        "implementation_burden",
        "risk_or_note",
    }
    assert set(critique_item["required"]) == {"issue", "why_it_matters", "recommendation"}


def test_generate_analysis_with_dummy_client():
    payload = json.dumps(
        {
            "executive_summary": "Point",
            "bill_summary": ["Impact"],
            "implementation": [
                {
                    "stakeholder": "Agency",
                    "obligation": "Report",
                    "implementation_burden": "Medium",
                    "risk_or_note": "Needs guidance",
                }
            ],
            "critique": [
                {
                    "issue": "Broad powers",
                    "why_it_matters": "Could be overbroad",
                    "recommendation": "Narrow scope",
                }
            ],
            "swot": {
                "strengths": ["Strong"],
                "weaknesses": ["Weak"],
                "opportunities": ["Opp"],
                "threats": ["Threat"],
            },
        }
    )
    provider_client = ProviderClient(
        name="openai",
        client=DummyOpenAIClient(payload),
        default_model="dummy",
        api_key="dummy-api-key",
    )
    result = generate_analysis(provider_client, "Sample document text")
    assert isinstance(result, AnalysisResult)
    assert result.executive_summary == "Point"
    assert result.implementation[0].stakeholder == "Agency"


def test_split_into_chunks_respects_size(monkeypatch):
    monkeypatch.setattr("services.rag_pipeline._embedding_tokenizer", lambda: DummyTokenizer())
    text = "Lorem ipsum " * 100
    chunks = split_into_chunks(text, chunk_size=100, chunk_overlap=10)
    tokenizer = DummyTokenizer()
    assert all(len(tokenizer.encode(chunk)) <= 100 for chunk in chunks)
    assert len(chunks) > 1


def test_split_into_scan_chunks_respects_size(monkeypatch):
    monkeypatch.setattr("services.rag_pipeline._embedding_tokenizer", lambda: DummyTokenizer())
    text = "Alpha beta gamma delta " * 400
    chunks = split_into_scan_chunks(text, chunk_size=120, chunk_overlap=20)
    tokenizer = DummyTokenizer()
    assert chunks
    assert all(len(tokenizer.encode(chunk.text)) <= 120 for chunk in chunks)


def test_merge_scan_matches_prefers_highest_score():
    matches = [
        ScanMatch(chunk_id=2, relevance_score=1, evidence_snippet="short"),
        ScanMatch(chunk_id=2, relevance_score=3, evidence_snippet="better"),
        ScanMatch(chunk_id=1, relevance_score=2, evidence_snippet="first"),
    ]
    merged = _merge_scan_matches(matches)
    assert [item.chunk_id for item in merged] == [2, 1]
    assert merged[0].evidence_snippet == "better"


def test_full_document_budget_is_provider_aware():
    openai_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    qwen_client = ProviderClient(name="qwen", client=object(), default_model="dummy", api_key="dummy-api-key")
    assert _full_document_budget(openai_client) > _full_document_budget(qwen_client)
    assert _full_document_budget(qwen_client) == 24_000


def test_openai_reasoning_kwargs_only_for_reasoning_models():
    gpt41_client = ProviderClient(name="openai", client=object(), default_model="gpt-4.1", api_key="dummy-api-key")
    gpt5_client = ProviderClient(name="openai", client=object(), default_model="gpt-5.1", api_key="dummy-api-key")
    assert _openai_reasoning_kwargs(gpt41_client) == {}
    assert _openai_reasoning_kwargs(gpt5_client) == {"reasoning": {"effort": OPENAI_REASONING_EFFORT}}


def test_anthropic_thinking_kwargs_only_for_supported_models():
    sonnet35_client = ProviderClient(
        name="anthropic",
        client=object(),
        default_model="claude-3-5-sonnet-20240620",
        api_key="dummy-api-key",
    )
    sonnet37_client = ProviderClient(
        name="anthropic",
        client=object(),
        default_model="claude-3-7-sonnet-20250219",
        api_key="dummy-api-key",
    )
    assert _anthropic_thinking_kwargs(sonnet35_client) == {}
    assert "extra_body" in _anthropic_thinking_kwargs(sonnet37_client)


def test_gemini_default_model_is_thinking_capable():
    assert _PROVIDER_MODEL_MAP["gemini"] == "gemini-2.5-flash"


def test_provider_defaults_use_thinking_capable_models():
    assert _PROVIDER_MODEL_MAP["openai"] == "gpt-5.5"
    assert _PROVIDER_MODEL_MAP["anthropic"] == "claude-sonnet-4-6"
    assert _PROVIDER_MODEL_MAP["cohere"] == "command-a-reasoning-08-2025"
    assert _PROVIDER_MODEL_MAP["qwen"] == DEFAULT_QWEN_MODEL


def test_extract_chat_completion_text_strips_qwen_thinking_block():
    message = type("Message", (), {"content": "<think>internal reasoning</think>\nFinal answer"})()
    choice = type("Choice", (), {"message": message})()
    response = type("Response", (), {"choices": [choice]})()
    assert _extract_chat_completion_text(response) == "Final answer"


def test_answer_query_uses_scan_context(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    analysis = AnalysisResult(executive_summary="Creates a new authority.")

    monkeypatch.setattr(
        "services.rag_pipeline.search_analysis_snippets",
        lambda snippets, question: (
            "[ref1] Executive Summary\nCreates a new authority.",
            [Citation(ref_id=1, snippet="Creates a new authority.")],
        ),
    )
    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_analysis_context",
        lambda provider_client, context_text, question: {"answer": f"Scanned answer using {context_text}", "is_sufficient": True},
    )

    result = answer_query(provider_client, analysis, None, "What does the bill create?", doc_text="Long document")
    assert isinstance(result, AnswerResult)
    assert "Scanned answer" in result.answer
    assert result.citations[0].snippet == "Creates a new authority."
    assert result.provenance == "analysis_based"


def test_answer_query_from_full_document_prefers_full_document(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")

    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_full_document",
        lambda provider_client, doc_text, question: "Full-document answer",
    )
    monkeypatch.setattr(
        "services.rag_pipeline._scan_document_for_context",
        lambda provider_client, doc_text, question: (_ for _ in ()).throw(AssertionError("scan path should not run")),
    )

    result = answer_query_from_full_document(provider_client, None, "What does the bill do?", doc_text="Small document")
    assert result.answer == "Full-document answer"
    assert result.citations == []
    assert result.provenance == "full_document"


def test_answer_query_from_full_document_falls_back_when_scan_errors(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    vector_store = FakeVectorStore(["The bill establishes a new data protection authority."])

    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_full_document",
        lambda provider_client, doc_text, question: None,
    )
    monkeypatch.setattr("services.rag_pipeline._scan_document_for_context", lambda provider_client, doc_text, question: None)
    monkeypatch.setattr(
        "services.rag_pipeline._generate_answer_from_context",
        lambda provider_client, context_text, question: f"Fallback answer from {context_text}",
    )

    result = answer_query_from_full_document(provider_client, vector_store, "What does the bill establish?", doc_text="Long document")
    assert "Fallback answer" in result.answer
    assert result.citations
    assert result.citations[0].snippet == "The bill establishes a new data protection authority."


def test_answer_query_from_full_document_falls_back_when_scan_has_no_hits(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    vector_store = FakeVectorStore(["The bill outlines penalties for non-compliance."])

    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_full_document",
        lambda provider_client, doc_text, question: None,
    )
    monkeypatch.setattr("services.rag_pipeline._scan_document_for_context", lambda provider_client, doc_text, question: None)
    monkeypatch.setattr(
        "services.rag_pipeline._generate_answer_from_context",
        lambda provider_client, context_text, question: context_text,
    )

    result = answer_query_from_full_document(provider_client, vector_store, "What are the penalties?", doc_text="Long document")
    assert "[1]" in result.answer
    assert result.citations[0].ref_id == 1


def test_answer_query_from_full_document_supports_vector_store_only(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    vector_store = FakeVectorStore(["Only fallback retrieval is available."])

    monkeypatch.setattr(
        "services.rag_pipeline._generate_answer_from_context",
        lambda provider_client, context_text, question: context_text,
    )

    result = answer_query_from_full_document(provider_client, vector_store, "What is available?")
    assert result.citations[0].snippet == "Only fallback retrieval is available."


def test_answer_query_from_full_document_uses_scan_when_full_document_too_large(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")

    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_full_document",
        lambda provider_client, doc_text, question: None,
    )
    monkeypatch.setattr(
        "services.rag_pipeline._scan_document_for_context",
        lambda provider_client, doc_text, question: (
            "[ref1] Chunk 7\nEvidence: licensing rule\nFull context:\nThe bill sets licensing rules.",
            [Citation(ref_id=1, snippet="licensing rule")],
        ),
    )
    monkeypatch.setattr(
        "services.rag_pipeline._generate_answer_from_context",
        lambda provider_client, context_text, question: "Scanned fallback answer",
    )

    result = answer_query_from_full_document(provider_client, None, "How does licensing work?", doc_text="Very long document")
    assert result.answer == "Scanned fallback answer"
    assert result.citations[0].snippet == "licensing rule"


def test_answer_query_requests_deeper_answer_when_analysis_is_insufficient(monkeypatch):
    provider_client = ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key")
    analysis = AnalysisResult(executive_summary="A short summary.")

    monkeypatch.setattr(
        "services.rag_pipeline.search_analysis_snippets",
        lambda snippets, question: (
            "[ref1] Executive Summary\nA short summary.",
            [Citation(ref_id=1, snippet="A short summary.")],
        ),
    )
    monkeypatch.setattr(
        "services.rag_pipeline._answer_question_from_analysis_context",
        lambda provider_client, context_text, question: {
            "answer": "The summary only partly answers this.",
            "is_sufficient": False,
        },
    )

    result = answer_query(provider_client, analysis, None, "What is missing?", doc_text="Long document")
    assert result.provenance == "analysis_based"
    assert result.needs_deeper_consent is True
    assert result.deeper_answer_available is True


def test_generate_analysis_progress_merges_sections(monkeypatch):
    payloads = iter(
        [
            json.dumps({"executive_summary": "Summary", "bill_summary": ["Point"]}),
            json.dumps({"implementation": [{"stakeholder": "Users", "obligation": "Register", "implementation_burden": "Medium", "risk_or_note": "Needs outreach"}]}),
            json.dumps({"critique": [{"issue": "Ambiguity", "why_it_matters": "Creates uncertainty", "recommendation": "Clarify wording"}]}),
            json.dumps({"swot": {"strengths": ["Strong"], "weaknesses": ["Weak"], "opportunities": ["Opp"], "threats": ["Threat"]}}),
        ]
    )

    monkeypatch.setattr("services.rag_pipeline._generate_json_payload", lambda *args, **kwargs: next(payloads))
    progress = list(generate_analysis_progress(ProviderClient(name="openai", client=object(), default_model="dummy", api_key="dummy-api-key"), "Doc"))
    assert len(progress) == 4
    assert progress[-1][1].swot.strengths == ["Strong"]


def test_build_analysis_snippets_includes_sections():
    analysis = AnalysisResult(
        executive_summary="Summary",
        bill_summary=["Point"],
        implementation=[{"stakeholder": "SMEs", "obligation": "File reports", "implementation_burden": "Low", "risk_or_note": "Admin cost"}],
    )
    snippets = build_analysis_snippets(analysis)
    assert snippets[0].section == "Executive Summary"
    assert any(snippet.section == "Implementation" for snippet in snippets)
