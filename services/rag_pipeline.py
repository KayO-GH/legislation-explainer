"""RAG pipeline utilities for embeddings, summaries, and Q&A."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, ForwardRef, Optional, Protocol, cast

try:
    from pydantic.v1 import typing as _pydantic_typing  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency path
    _pydantic_typing = None
else:
    _forward_sig = inspect.signature(ForwardRef._evaluate)
    _recursive_guard_param = _forward_sig.parameters.get("recursive_guard")
    if _recursive_guard_param is not None:
        def _evaluate_forwardref(type_: ForwardRef, globalns: Any, localns: Any) -> Any:
            evaluator = cast(Any, type_)._evaluate
            if "type_params" in evaluator.__code__.co_varnames:
                return evaluator(globalns, localns, None, recursive_guard=set())
            return evaluator(globalns, localns, recursive_guard=set())

        _pydantic_typing.evaluate_forwardref = _evaluate_forwardref  # type: ignore[attr-defined]

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from config import (
    ANTHROPIC_THINKING_BUDGET,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    OPENAI_REASONING_EFFORT,
    PROVIDER_FULL_DOCUMENT_QA_TOKEN_BUDGETS,
    SCAN_BATCH_SIZE,
    SCAN_CHUNK_OVERLAP,
    SCAN_CHUNK_SIZE,
    SCAN_MAX_WINDOWS,
    SCAN_TOP_K,
    TOP_K_RETRIEVAL,
)
from services.precomputed_assets import PrecomputedExampleAsset, load_precomputed_asset_for_url
from services.providers import ProviderClient

OBJECTIVITY_INSTRUCTION = (
    "Be objective. Reason things out. Based on the available context, and wherever the question "
    "lends itself, consider multiple relevant perspectives, including opposing perspectives, "
    "before responding."
)


class SWOTBlock(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class ImplementationItem(BaseModel):
    stakeholder: str = ""
    obligation: str = ""
    implementation_burden: str = ""
    risk_or_note: str = ""


class CritiqueItem(BaseModel):
    issue: str = ""
    why_it_matters: str = ""
    recommendation: str = ""


class AnalysisResult(BaseModel):
    executive_summary: str = ""
    bill_summary: List[str] = Field(default_factory=list)
    implementation: List[ImplementationItem] = Field(default_factory=list)
    critique: List[CritiqueItem] = Field(default_factory=list)
    swot: SWOTBlock = Field(default_factory=SWOTBlock)


class Citation(BaseModel):
    ref_id: int
    snippet: str


class AnswerResult(BaseModel):
    answer: str
    citations: List[Citation]
    provenance: str = "analysis_based"
    needs_deeper_consent: bool = False
    deeper_answer_available: bool = False
    consent_prompt: str = ""


class ScanChunk(BaseModel):
    chunk_id: int
    text: str


class ScanMatch(BaseModel):
    chunk_id: int
    relevance_score: int = Field(ge=0, le=3)
    evidence_snippet: str = ""


class ScanResult(BaseModel):
    matches: List[ScanMatch] = Field(default_factory=list)


class TokenizerLike(Protocol):
    def encode(self, text: str, *args: Any, **kwargs: Any) -> list[int]:
        ...


@dataclass
class AnalysisSnippet:
    ref_id: int
    section: str
    text: str


@dataclass
class CachedDocumentArtifacts:
    document_hash: str
    document_text: str
    chunks: list[str]
    analysis: AnalysisResult
    vector_store: FAISS | None
    analysis_snippets: list[AnalysisSnippet] = field(default_factory=list)
    source_url: str | None = None
    example_bill_id: str | None = None
    is_precomputed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


_DOCUMENT_CACHE: dict[str, CachedDocumentArtifacts] = {}


@lru_cache(maxsize=1)
def _embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings(model_name=model_name)


@lru_cache(maxsize=1)
def _embedding_tokenizer(model_name: str = DEFAULT_EMBEDDING_MODEL) -> TokenizerLike:
    return AutoTokenizer.from_pretrained(model_name)


def warm_embedding_stack(model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
    _embedding_tokenizer(model_name)
    _embedding_model(model_name)


def _split_with_tokenizer(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    tokenizer = _embedding_tokenizer()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return splitter.split_text(text)


def _token_count(text: str) -> int:
    return len(_embedding_tokenizer().encode(text, add_special_tokens=False))


def _full_document_budget(provider_client: ProviderClient) -> int:
    return PROVIDER_FULL_DOCUMENT_QA_TOKEN_BUDGETS.get(provider_client.name, 0)


def _supports_openai_reasoning(model_name: str) -> bool:
    return model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def _supports_anthropic_extended_thinking(model_name: str) -> bool:
    return model_name.startswith(
        (
            "claude-3-7-sonnet",
            "claude-sonnet-4",
            "claude-opus-4",
        )
    )


def _openai_reasoning_kwargs(provider_client: ProviderClient) -> dict[str, Any]:
    if provider_client.name != "openai" or not _supports_openai_reasoning(provider_client.default_model):
        return {}
    return {"reasoning": {"effort": OPENAI_REASONING_EFFORT}}


def _anthropic_thinking_kwargs(provider_client: ProviderClient) -> dict[str, Any]:
    if provider_client.name != "anthropic" or not _supports_anthropic_extended_thinking(provider_client.default_model):
        return {}
    return {
        "extra_body": {
            "thinking": {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
            }
        }
    }


def split_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    return _split_with_tokenizer(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_into_scan_chunks(
    text: str,
    chunk_size: int = SCAN_CHUNK_SIZE,
    chunk_overlap: int = SCAN_CHUNK_OVERLAP,
) -> List[ScanChunk]:
    chunks = _split_with_tokenizer(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [ScanChunk(chunk_id=idx, text=chunk) for idx, chunk in enumerate(chunks, start=1)]


def build_vector_store(chunks: Iterable[str]) -> FAISS:
    docs = [Document(page_content=chunk, metadata={"chunk_id": idx}) for idx, chunk in enumerate(chunks, start=1)]
    embeddings = _embedding_model()
    return FAISS.from_documents(docs, embedding=embeddings)


def document_hash_for_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_cache_key(document_hash: str) -> str:
    return f"{document_hash}:{DEFAULT_CHUNK_SIZE}:{DEFAULT_CHUNK_OVERLAP}:{DEFAULT_EMBEDDING_MODEL}"


def get_cached_document(document_hash: str) -> CachedDocumentArtifacts | None:
    return _DOCUMENT_CACHE.get(document_cache_key(document_hash))


def cache_document(artifacts: CachedDocumentArtifacts) -> CachedDocumentArtifacts:
    _DOCUMENT_CACHE[document_cache_key(artifacts.document_hash)] = artifacts
    return artifacts


def save_vector_store(vector_store: FAISS, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(output_dir))


def load_vector_store(vector_store_dir: Path) -> FAISS:
    return FAISS.load_local(
        str(vector_store_dir),
        _embedding_model(),
        allow_dangerous_deserialization=True,
    )


def hydrate_precomputed_example(asset: PrecomputedExampleAsset) -> CachedDocumentArtifacts:
    analysis = AnalysisResult.model_validate(asset.analysis_payload)
    vector_store = load_vector_store(asset.bill.vector_store_dir)
    artifacts = CachedDocumentArtifacts(
        document_hash=asset.bill.document_hash or document_hash_for_text(asset.document_text),
        document_text=asset.document_text,
        chunks=asset.chunks,
        analysis=analysis,
        vector_store=vector_store,
        analysis_snippets=build_analysis_snippets(analysis),
        source_url=asset.bill.source_url,
        example_bill_id=asset.bill.id,
        is_precomputed=True,
        metadata=asset.metadata,
    )
    return cache_document(artifacts)


def _analysis_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "bill_summary": {"type": "array", "items": {"type": "string"}},
            "implementation": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "stakeholder": {"type": "string"},
                        "obligation": {"type": "string"},
                        "implementation_burden": {"type": "string"},
                        "risk_or_note": {"type": "string"},
                    },
                    "required": [
                        "stakeholder",
                        "obligation",
                        "implementation_burden",
                        "risk_or_note",
                    ],
                    "additionalProperties": False,
                },
            },
            "critique": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "recommendation": {"type": "string"},
                    },
                    "required": ["issue", "why_it_matters", "recommendation"],
                    "additionalProperties": False,
                },
            },
            "swot": {
                "type": "object",
                "properties": {
                    "strengths": {"type": "array", "items": {"type": "string"}},
                    "weaknesses": {"type": "array", "items": {"type": "string"}},
                    "opportunities": {"type": "array", "items": {"type": "string"}},
                    "threats": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["strengths", "weaknesses", "opportunities", "threats"],
                "additionalProperties": False,
            },
        },
        "required": [
            "executive_summary",
            "bill_summary",
            "implementation",
            "critique",
            "swot",
        ],
        "additionalProperties": False,
    }


def _analysis_summary_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "executive_summary": {"type": "string"},
            "bill_summary": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["executive_summary", "bill_summary"],
        "additionalProperties": False,
    }


def _implementation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "implementation": _analysis_schema()["properties"]["implementation"],
        },
        "required": ["implementation"],
        "additionalProperties": False,
    }


def _critique_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "critique": _analysis_schema()["properties"]["critique"],
        },
        "required": ["critique"],
        "additionalProperties": False,
    }


def _swot_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "swot": _analysis_schema()["properties"]["swot"],
        },
        "required": ["swot"],
        "additionalProperties": False,
    }


def _analysis_answer_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "is_sufficient": {"type": "boolean"},
        },
        "required": ["answer", "is_sufficient"],
        "additionalProperties": False,
    }


def _generate_json_payload(
    provider_client: ProviderClient,
    *,
    prompt: str,
    user_text: str,
    schema: dict[str, Any],
    schema_name: str,
    max_tokens: int = 1500,
) -> str:
    if provider_client.name == "openai":
        response = provider_client.client.responses.create(
            model=provider_client.default_model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
            **_openai_reasoning_kwargs(provider_client),
        )
        return _extract_openai_json(response)
    if provider_client.name == "qwen":
        response = provider_client.client.chat.completions.create(
            model=provider_client.default_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_text}\n\n"
                        "Return only valid JSON matching this schema:\n"
                        f"{json.dumps(schema)}"
                    ),
                },
            ],
            temperature=0,
        )
        return _strip_json_fences(_extract_chat_completion_text(response))
    if provider_client.name == "anthropic":
        response = provider_client.client.messages.create(
            model=provider_client.default_model,
            max_tokens=max_tokens,
            system=prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
            **_anthropic_thinking_kwargs(provider_client),
        )
        return response.content[0].text
    if provider_client.name == "cohere":
        response = provider_client.client.chat(
            message=user_text,
            model=provider_client.default_model,
            preamble=prompt,
            max_tokens=max_tokens,
            temperature=0,
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
        return _strip_json_fences(_extract_cohere_text(response))

    genai = provider_client.client
    model = genai.GenerativeModel(
        model_name=provider_client.default_model,
        system_instruction=prompt,
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(user_text)
    return _extract_gemini_text(response)


def generate_analysis_progress(provider_client: ProviderClient, document_text: str) -> Iterable[tuple[str, AnalysisResult]]:
    limited_text = document_text[:30000]
    base_prompt = (
        "You are a senior policy analyst helping citizens, civil society, startups, workers, consumers, "
        "operators, and affected industries understand a government bill. Produce a clear briefing-style analysis, "
        "not a generic summary. Be objective, text-disciplined, and grounded in the document. Separate what the bill "
        "expressly provides from what is a reasonable inference about likely effects. Do not attribute powers, "
        "safeguards, funding mechanisms, oversight tools, or policy instruments to the bill unless they are plainly "
        "supported by the text. If support for a point is weak, omit it rather than embellish it. Focus on what the "
        "bill changes, what ordinary people or practitioners in the affected field should be excited or worried "
        "about, how businesses or industry participants may need to adapt, and what this could mean for growth, "
        "friction, competition, cost, access, compliance, innovation, investor confidence, public-service delivery, "
        "or digital infrastructure quality in the affected sector and, where relevant, the national economy. "
        "Pay particular attention to ground-level effects on workers, professionals, entrepreneurs, recent "
        "graduates, students, consumers, startups, SMEs, public institutions, and vulnerable or underserved groups "
        "when the bill plausibly affects them, but do not force demographic groups into the analysis where the fit is "
        "weak. Do not center implementation implications on lawmakers, legislators, or sponsoring government "
        "agencies unless a direct downstream burden on the public or private sector depends on them. "
        "For implementation items, identify the affected stakeholder group, the practical obligation or change in "
        "behavior they may face, the likely implementation burden, and any practical risk or note. If a stakeholder "
        "has no formal legal obligation, describe the real-world adjustment, exposure, opportunity, or compliance "
        "expectation in plain language. Prefer concrete operational, lived-experience, and economic consequences over "
        "abstract commentary. For critique items, use the structure: issue, why it matters, recommendation, and "
        "focus on ambiguity, overlap, overreach, implementation bottlenecks, concentration of discretion, compliance "
        "burden, and risks to competition, innovation, affordability, or access where relevant. Keep recommendations "
        "tightly connected to the specific weakness identified. For SWOT, keep each item concise, specific, and "
        "grounded in the bill or a clear downstream effect that follows from the bill. Avoid vague statements. "
        "Avoid duplicating the same point across sections. "
        + OBJECTIVITY_INSTRUCTION
    )

    partial = AnalysisResult()
    summary_payload = _generate_json_payload(
        provider_client,
        prompt=base_prompt + " Return JSON for executive_summary and bill_summary only.",
        user_text=f"Document content:\n\n{limited_text}",
        schema=_analysis_summary_schema(),
        schema_name="BillAnalysisSummary",
    )
    summary_result = json.loads(_strip_json_fences(summary_payload))
    partial.executive_summary = summary_result.get("executive_summary", "")
    partial.bill_summary = summary_result.get("bill_summary", [])
    yield "Generating executive summary...", partial.model_copy(deep=True)

    implementation_payload = _generate_json_payload(
        provider_client,
        prompt=base_prompt + " Return JSON for implementation implications only.",
        user_text=f"Document content:\n\n{limited_text}",
        schema=_implementation_schema(),
        schema_name="BillAnalysisImplementation",
    )
    implementation_result = json.loads(_strip_json_fences(implementation_payload))
    partial.implementation = [ImplementationItem.model_validate(item) for item in implementation_result.get("implementation", [])]
    yield "Generating implementation implications...", partial.model_copy(deep=True)

    critique_payload = _generate_json_payload(
        provider_client,
        prompt=base_prompt + " Return JSON for critique only.",
        user_text=f"Document content:\n\n{limited_text}",
        schema=_critique_schema(),
        schema_name="BillAnalysisCritique",
    )
    critique_result = json.loads(_strip_json_fences(critique_payload))
    partial.critique = [CritiqueItem.model_validate(item) for item in critique_result.get("critique", [])]
    yield "Generating critique...", partial.model_copy(deep=True)

    swot_payload = _generate_json_payload(
        provider_client,
        prompt=base_prompt + " Return JSON for SWOT only.",
        user_text=f"Document content:\n\n{limited_text}",
        schema=_swot_schema(),
        schema_name="BillAnalysisSwot",
    )
    swot_result = json.loads(_strip_json_fences(swot_payload))
    partial.swot = SWOTBlock.model_validate(swot_result.get("swot", {}))
    yield "Generating SWOT...", partial.model_copy(deep=True)


def generate_analysis_once(provider_client: ProviderClient, document_text: str) -> AnalysisResult:
    limited_text = document_text[:30000]
    prompt = (
        "You are a senior policy analyst helping citizens, civil society, startups, workers, consumers, "
        "operators, and affected industries understand a government bill. Produce a clear briefing-style analysis, "
        "not a generic summary. Be objective, text-disciplined, and grounded in the document. Separate what the bill "
        "expressly provides from what is a reasonable inference about likely effects. Do not attribute powers, "
        "safeguards, funding mechanisms, oversight tools, or policy instruments to the bill unless they are plainly "
        "supported by the text. If support for a point is weak, omit it rather than embellish it. Focus on what the "
        "bill changes, what ordinary people or practitioners in the affected field should be excited or worried "
        "about, how businesses or industry participants may need to adapt, and what this could mean for growth, "
        "friction, competition, cost, access, compliance, innovation, investor confidence, public-service delivery, "
        "or digital infrastructure quality in the affected sector and, where relevant, the national economy. "
        "Pay particular attention to ground-level effects on workers, professionals, entrepreneurs, recent "
        "graduates, students, consumers, startups, SMEs, public institutions, and vulnerable or underserved groups "
        "when the bill plausibly affects them, but do not force demographic groups into the analysis where the fit is "
        "weak. Do not center implementation implications on lawmakers, legislators, or sponsoring government "
        "agencies unless a direct downstream burden on the public or private sector depends on them. "
        "For implementation items, identify the affected stakeholder group, the practical obligation or change in "
        "behavior they may face, the likely implementation burden, and any practical risk or note. If a stakeholder "
        "has no formal legal obligation, describe the real-world adjustment, exposure, opportunity, or compliance "
        "expectation in plain language. Prefer concrete operational, lived-experience, and economic consequences over "
        "abstract commentary. For critique items, use the structure: issue, why it matters, recommendation, and "
        "focus on ambiguity, overlap, overreach, implementation bottlenecks, concentration of discretion, compliance "
        "burden, and risks to competition, innovation, affordability, or access where relevant. Keep recommendations "
        "tightly connected to the specific weakness identified. For SWOT, keep each item concise, specific, and "
        "grounded in the bill or a clear downstream effect that follows from the bill. "
        "Avoid vague statements. Avoid duplicating the same point across sections. "
        "Respond using JSON that matches the provided schema. "
        + OBJECTIVITY_INSTRUCTION
    )
    payload = _generate_json_payload(
        provider_client,
        prompt=prompt,
        user_text=f"Document content:\n\n{limited_text}",
        schema=_analysis_schema(),
        schema_name="BillAnalysis",
    )
    return AnalysisResult.model_validate_json(_strip_json_fences(payload))


def generate_analysis(provider_client: ProviderClient, document_text: str) -> AnalysisResult:
    final = AnalysisResult()
    for _, partial in generate_analysis_progress(provider_client, document_text):
        final = partial
    return final


def prepare_document_artifacts(
    document_text: str,
    *,
    provider_factory: Callable[[], Any] | None = None,
) -> tuple[str, list[str], FAISS]:
    document_hash = document_hash_for_text(document_text)
    cached = get_cached_document(document_hash)
    if cached is not None and cached.vector_store is not None:
        return cached.document_hash, cached.chunks, cached.vector_store

    with ThreadPoolExecutor(max_workers=2) as executor:
        chunks_future = executor.submit(split_into_chunks, document_text)
        provider_future = executor.submit(provider_factory) if provider_factory is not None else None
        chunks = chunks_future.result()
        vector_store = build_vector_store(chunks)
        if provider_future is not None:
            provider_future.result()
    return document_hash, chunks, vector_store


def get_precomputed_example_artifacts(url: str | None) -> CachedDocumentArtifacts | None:
    asset = load_precomputed_asset_for_url(url)
    if asset is None:
        return None
    if asset.bill.document_hash:
        cached = get_cached_document(asset.bill.document_hash)
        if cached is not None:
            return cached
    return hydrate_precomputed_example(asset)


def build_cached_document_artifacts(
    *,
    document_text: str,
    chunks: list[str],
    analysis: AnalysisResult,
    vector_store: FAISS | None,
    source_url: str | None = None,
) -> CachedDocumentArtifacts:
    return cache_document(
        CachedDocumentArtifacts(
            document_hash=document_hash_for_text(document_text),
            document_text=document_text,
            chunks=chunks,
            analysis=analysis,
            vector_store=vector_store,
            analysis_snippets=build_analysis_snippets(analysis),
            source_url=source_url,
        )
    )


def build_analysis_snippets(analysis: AnalysisResult) -> list[AnalysisSnippet]:
    snippets: list[AnalysisSnippet] = []
    next_ref = 1

    def push(section: str, text: str) -> None:
        nonlocal next_ref
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            return
        snippets.append(AnalysisSnippet(ref_id=next_ref, section=section, text=cleaned))
        next_ref += 1

    push("Executive Summary", analysis.executive_summary)
    for item in analysis.bill_summary:
        push("Bill Summary", item)
    for item in analysis.implementation:
        push(
            "Implementation",
            f"{item.stakeholder}: {item.obligation}. Burden: {item.implementation_burden}. Risk/Note: {item.risk_or_note}",
        )
    for item in analysis.critique:
        push(
            "Critique",
            f"{item.issue}. Why it matters: {item.why_it_matters}. Recommendation: {item.recommendation}",
        )
    for label, values in (
        ("Strength", analysis.swot.strengths),
        ("Weakness", analysis.swot.weaknesses),
        ("Opportunity", analysis.swot.opportunities),
        ("Threat", analysis.swot.threats),
    ):
        for value in values:
            push(f"SWOT {label}", value)
    return snippets


def _question_terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]{3,}", text.lower())}


def search_analysis_snippets(snippets: list[AnalysisSnippet], question: str, top_k: int = 5) -> tuple[str, list[Citation]] | None:
    if not snippets:
        return None
    terms = _question_terms(question)
    if not terms:
        return None

    ranked: list[tuple[int, AnalysisSnippet]] = []
    for snippet in snippets:
        snippet_terms = _question_terms(snippet.text)
        score = len(terms & snippet_terms)
        if score > 0:
            ranked.append((score, snippet))
    if not ranked:
        return None

    ranked.sort(key=lambda item: (-item[0], item[1].ref_id))
    selected = [snippet for _, snippet in ranked[:top_k]]
    context_blocks: list[str] = []
    citations: list[Citation] = []
    for idx, snippet in enumerate(selected, start=1):
        citations.append(Citation(ref_id=idx, snippet=snippet.text))
        context_blocks.append(f"[ref{idx}] {snippet.section}\n{snippet.text}")
    return "\n\n".join(context_blocks), citations


def _answer_question_from_analysis_context(
    provider_client: ProviderClient,
    context_text: str,
    question: str,
) -> dict[str, Any]:
    prompt = (
        "You answer follow-up questions about a government bill using only the provided analysis summary snippets. "
        "Answer directly and clearly. Set is_sufficient to true only if the analysis snippets materially answer the "
        "question. If they do not fully answer it, say what is missing and set is_sufficient to false. "
        "Do not pretend you have read the full bill if you only have analysis snippets. "
        + OBJECTIVITY_INSTRUCTION
    )
    payload = _generate_json_payload(
        provider_client,
        prompt=prompt,
        user_text="Analysis snippets:\n" + context_text + "\n\nQuestion:\n" + question,
        schema=_analysis_answer_schema(),
        schema_name="AnalysisAnswer",
        max_tokens=900,
    )
    return json.loads(_strip_json_fences(payload))


def answer_query(
    provider_client: ProviderClient,
    analysis: AnalysisResult | None,
    vector_store: FAISS | None,
    question: str,
    doc_text: str | None = None,
    allow_full_document: bool = False,
) -> AnswerResult:
    if analysis is not None:
        snippets = build_analysis_snippets(analysis)
        analysis_context = search_analysis_snippets(snippets, question)
        if analysis_context is not None:
            context_text, citations = analysis_context
            analysis_answer = _answer_question_from_analysis_context(provider_client, context_text, question)
            return AnswerResult(
                answer=analysis_answer["answer"].strip(),
                citations=citations,
                provenance="analysis_based",
                needs_deeper_consent=not analysis_answer.get("is_sufficient", False),
                deeper_answer_available=bool(doc_text or vector_store is not None),
                consent_prompt=(
                    "This answer is based on the summary and analysis. "
                    "If you want, I can run a deeper full-document answer next."
                ),
            )

    if allow_full_document:
        return answer_query_from_full_document(provider_client, vector_store, question, doc_text=doc_text)

    return AnswerResult(
        answer=(
            "I couldn't fully answer that from the summary and analysis alone. "
            "If you want, I can run a deeper full-document answer."
        ),
        citations=[],
        provenance="analysis_based",
        needs_deeper_consent=True,
        deeper_answer_available=bool(doc_text or vector_store is not None),
        consent_prompt="A deeper full-document answer is available.",
    )


def answer_query_from_full_document(
    provider_client: ProviderClient,
    vector_store: FAISS | None,
    question: str,
    *,
    doc_text: str | None = None,
) -> AnswerResult:
    context_text = "No context available."
    citations: list[Citation] = []

    if doc_text:
        full_document_answer = _answer_question_from_full_document(provider_client, doc_text, question)
        if full_document_answer is not None:
            return AnswerResult(
                answer=full_document_answer.strip(),
                citations=[],
                provenance="full_document",
            )

        scan_context = _scan_document_for_context(provider_client, doc_text, question)
        if scan_context:
            context_text, citations = scan_context

    if not citations and vector_store is not None:
        context_text, citations = _retrieve_context_from_vector_store(vector_store, question)

    answer = _generate_answer_from_context(provider_client, context_text, question)
    return AnswerResult(answer=answer.strip(), citations=citations, provenance="full_document")


def _answer_question_from_full_document(
    provider_client: ProviderClient,
    doc_text: str,
    question: str,
) -> str | None:
    if _token_count(doc_text) > _full_document_budget(provider_client):
        return None

    prompt = (
        "You answer questions about government bills. Read the full document carefully, "
        "reason through the relevant provisions, and answer the user's question logically. "
        "Base your answer only on the provided document. If the answer is not supported by the document, say you are unsure. "
        + OBJECTIVITY_INSTRUCTION
    )
    user_text = "Full document:\n" + doc_text + "\n\nQuestion:\n" + question

    if provider_client.name == "openai":
        response = provider_client.client.responses.create(
            model=provider_client.default_model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            **_openai_reasoning_kwargs(provider_client),
        )
        return _extract_openai_text(response)
    if provider_client.name == "qwen":
        response = provider_client.client.chat.completions.create(
            model=provider_client.default_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
        )
        return _extract_chat_completion_text(response)
    if provider_client.name == "anthropic":
        response = provider_client.client.messages.create(
            model=provider_client.default_model,
            max_tokens=1000,
            system=prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
            **_anthropic_thinking_kwargs(provider_client),
        )
        return response.content[0].text
    if provider_client.name == "cohere":
        response = provider_client.client.chat(
            message=user_text,
            model=provider_client.default_model,
            preamble=prompt,
            max_tokens=1000,
            temperature=0,
        )
        return _extract_cohere_text(response)

    genai = provider_client.client
    model = genai.GenerativeModel(
        model_name=provider_client.default_model,
        system_instruction=prompt,
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="text/plain",
        ),
    )
    response = model.generate_content(user_text)
    return _extract_gemini_text(response)


def _retrieve_context_from_vector_store(vector_store: FAISS, question: str) -> tuple[str, list[Citation]]:
    retrieved_docs = vector_store.similarity_search(question, k=TOP_K_RETRIEVAL)
    context_blocks: list[str] = []
    citations: list[Citation] = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        snippet = doc.page_content.strip()
        context_blocks.append(f"[{idx}] {snippet}")
        citations.append(Citation(ref_id=idx, snippet=snippet))

    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."
    return context_text, citations


def _generate_answer_from_context(provider_client: ProviderClient, context_text: str, question: str) -> str:
    prompt = (
        "You answer questions about government bills. Use the context snippets provided. "
        "Cite supporting snippets using [ref#] references matching the snippet number. "
        "If unsure, say you are unsure. "
        + OBJECTIVITY_INSTRUCTION
    )

    if provider_client.name == "openai":
        response = provider_client.client.responses.create(
            model=provider_client.default_model,
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Context:\n" + context_text + "\n\nQuestion:\n" + question
                    ),
                },
            ],
            **_openai_reasoning_kwargs(provider_client),
        )
        answer = _extract_openai_text(response)
    elif provider_client.name == "qwen":
        response = provider_client.client.chat.completions.create(
            model=provider_client.default_model,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "Context:\n" + context_text + "\n\nQuestion:\n" + question
                    ),
                },
            ],
            temperature=0,
        )
        answer = _extract_chat_completion_text(response)
    elif provider_client.name == "anthropic":
        response = provider_client.client.messages.create(
            model=provider_client.default_model,
            max_tokens=800,
            system=prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Context:\n" + context_text + "\n\nQuestion:\n" + question
                            ),
                        }
                    ],
                }
            ],
            **_anthropic_thinking_kwargs(provider_client),
        )
        answer = response.content[0].text
    elif provider_client.name == "cohere":
        response = provider_client.client.chat(
            message=(
                "Context:\n" + context_text + "\n\nQuestion:\n" + question
            ),
            model=provider_client.default_model,
            preamble=prompt,
            max_tokens=800,
            temperature=0,
        )
        answer = _extract_cohere_text(response)
    else:  # gemini
        genai = provider_client.client
        model = genai.GenerativeModel(
            model_name=provider_client.default_model,
            system_instruction=prompt,
            generation_config=genai.GenerationConfig(
                temperature=0,
                response_mime_type="text/plain",
            ),
        )
        response = model.generate_content(
            "Context:\n"
            + context_text
            + "\n\nQuestion:\n"
            + question
        )
        answer = _extract_gemini_text(response)

    return answer


def _scan_document_for_context(
    provider_client: ProviderClient,
    doc_text: str,
    question: str,
) -> Optional[tuple[str, list[Citation]]]:
    scan_chunks = split_into_scan_chunks(doc_text)
    if not scan_chunks or len(scan_chunks) > SCAN_MAX_WINDOWS:
        return None

    ranked_matches: list[ScanMatch] = []
    for batch_start in range(0, len(scan_chunks), SCAN_BATCH_SIZE):
        batch = scan_chunks[batch_start : batch_start + SCAN_BATCH_SIZE]
        parsed_result = _run_scan_batch_with_retry(provider_client, batch, question)
        if parsed_result is None:
            return None
        ranked_matches.extend(parsed_result.matches)

    merged_matches = _merge_scan_matches(ranked_matches)
    if not merged_matches or merged_matches[0].relevance_score <= 0:
        return None

    chunk_by_id = {chunk.chunk_id: chunk for chunk in scan_chunks}
    selected_matches = merged_matches[:SCAN_TOP_K]
    context_blocks: list[str] = []
    citations: list[Citation] = []
    for ref_id, match in enumerate(selected_matches, start=1):
        chunk = chunk_by_id.get(match.chunk_id)
        if chunk is None:
            continue
        snippet = match.evidence_snippet.strip() or _truncate_snippet(chunk.text)
        citations.append(Citation(ref_id=ref_id, snippet=snippet))
        context_blocks.append(
            f"[ref{ref_id}] Chunk {match.chunk_id} (score {match.relevance_score})\n"
            f"Evidence: {snippet}\n"
            f"Full context:\n{chunk.text.strip()}"
        )

    if not citations:
        return None
    return "\n\n".join(context_blocks), citations


def _run_scan_batch_with_retry(
    provider_client: ProviderClient,
    batch: list[ScanChunk],
    question: str,
) -> ScanResult | None:
    for _ in range(2):
        payload = _scan_batch(provider_client, batch, question)
        try:
            return ScanResult.model_validate_json(_strip_json_fences(payload))
        except Exception:  # noqa: BLE001
            continue
    return None


def _scan_batch(provider_client: ProviderClient, batch: list[ScanChunk], question: str) -> str:
    schema = {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "integer"},
                        "relevance_score": {"type": "integer", "minimum": 0, "maximum": 3},
                        "evidence_snippet": {"type": "string"},
                    },
                    "required": ["chunk_id", "relevance_score", "evidence_snippet"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["matches"],
        "additionalProperties": False,
    }
    prompt = (
        "You are scanning chunks from a government bill to find context relevant to a user question. "
        "Return only JSON matching the schema. Include only chunks that are at least somewhat relevant. "
        "Use relevance_score from 0 to 3, where 3 is highly relevant and 0 is irrelevant. "
        "Keep evidence_snippet short and copied from the chunk. "
        + OBJECTIVITY_INSTRUCTION
    )
    chunk_text = "\n\n".join(f"Chunk {chunk.chunk_id}:\n{chunk.text}" for chunk in batch)
    user_text = (
        f"Question:\n{question}\n\n"
        f"Chunks:\n{chunk_text}\n\n"
        f"Return JSON matching this schema:\n{json.dumps(schema)}"
    )

    if provider_client.name == "openai":
        response = provider_client.client.responses.create(
            model=provider_client.default_model,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ChunkScan", "schema": schema},
            },
            **_openai_reasoning_kwargs(provider_client),
        )
        return _extract_openai_json(response)
    if provider_client.name == "qwen":
        response = provider_client.client.chat.completions.create(
            model=provider_client.default_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
        )
        return _extract_chat_completion_text(response)
    if provider_client.name == "anthropic":
        response = provider_client.client.messages.create(
            model=provider_client.default_model,
            max_tokens=1200,
            system=prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_text}]}],
            **_anthropic_thinking_kwargs(provider_client),
        )
        return response.content[0].text
    if provider_client.name == "cohere":
        response = provider_client.client.chat(
            message=user_text,
            model=provider_client.default_model,
            preamble=prompt,
            max_tokens=1200,
            temperature=0,
            response_format={
                "type": "json_object",
                "schema": schema,
            },
        )
        return _extract_cohere_text(response)

    genai = provider_client.client
    model = genai.GenerativeModel(
        model_name=provider_client.default_model,
        system_instruction=prompt,
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    response = model.generate_content(user_text)
    return _extract_gemini_text(response)


def _merge_scan_matches(matches: list[ScanMatch]) -> list[ScanMatch]:
    by_chunk_id: dict[int, ScanMatch] = {}
    for match in matches:
        existing = by_chunk_id.get(match.chunk_id)
        candidate_snippet = match.evidence_snippet.strip()
        if existing is None or match.relevance_score > existing.relevance_score:
            by_chunk_id[match.chunk_id] = match
        elif (
            existing.relevance_score == match.relevance_score
            and candidate_snippet
            and len(candidate_snippet) > len(existing.evidence_snippet.strip())
        ):
            by_chunk_id[match.chunk_id] = match
    return sorted(
        by_chunk_id.values(),
        key=lambda item: (-item.relevance_score, item.chunk_id),
    )


def _truncate_snippet(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _extract_openai_json(response: Any) -> str:
    buffer = []
    output_items = getattr(response, "output", None)
    if output_items:
        for item in output_items:
            for block in item.content:
                if getattr(block, "type", None) == "output_json_schema":
                    buffer.append(block.text)
                elif getattr(block, "type", None) == "output_text":
                    buffer.append(block.text)
    if buffer:
        return "".join(buffer)
    if hasattr(response, "output_text"):
        return response.output_text
    return ""


def _extract_openai_text(response: Any) -> str:
    buffer = []
    output_items = getattr(response, "output", None)
    if output_items:
        for item in output_items:
            for block in item.content:
                if getattr(block, "type", None) in ("output_text", "output_message"):
                    buffer.append(block.text)
    if buffer:
        return "".join(buffer)
    if hasattr(response, "output_text"):
        return response.output_text
    return ""


def _extract_chat_completion_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return _strip_thinking_block(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return _strip_thinking_block("".join(parts))
    return ""


def _extract_gemini_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None)
    if candidates:
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                segment = getattr(part, "text", None)
                if segment:
                    parts.append(segment)
        if parts:
            return "".join(parts)
    return ""


def _extract_cohere_text(response: Any) -> str:
    message = getattr(response, "message", None)
    if message:
        content = getattr(message, "content", None) or []
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "".join(parts)
    text = getattr(response, "text", None)
    if text:
        return text
    return ""


def _strip_json_fences(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("```json"):
        stripped = stripped[7:]
    elif stripped.startswith("```"):
        stripped = stripped[3:]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _strip_thinking_block(value: str) -> str:
    stripped = value.strip()
    without_think = re.sub(r"(?is)<think>.*?</think>", "", stripped).strip()
    if without_think:
        return without_think
    if "</think>" in stripped:
        _, _, tail = stripped.rpartition("</think>")
        return tail.strip()
    return stripped
