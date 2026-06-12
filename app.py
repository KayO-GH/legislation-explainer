"""Gradio entrypoint for the Legislation Explainer app."""

from __future__ import annotations

import html
import time
from pathlib import Path
from uuid import uuid4
from typing import Any

import gradio as gr

from config import APP_DESCRIPTION, APP_TITLE, DEFAULT_PROVIDER
from services.example_bills import example_bill_titles, example_bill_urls_by_title
from services import ingest
from services.providers import get_default_api_key, instantiate_client, list_providers, validate_api_key
from services.rag_pipeline import (
    AnalysisResult,
    answer_query_from_full_document,
    build_cached_document_artifacts,
    generate_analysis_progress,
    get_precomputed_example_artifacts,
    prepare_document_artifacts,
    warm_embedding_stack,
)


def _provider_options() -> list[tuple[str, str, str]]:
    return [(config.name, config.display_name, config.instructions) for config in list_providers()]


PROVIDER_OPTIONS = _provider_options()
PROVIDER_LABELS = [display for _, display, _ in PROVIDER_OPTIONS]
PROVIDER_BY_LABEL = {display: name for name, display, _ in PROVIDER_OPTIONS}
PROVIDER_DETAILS = {name: description for name, _, description in PROVIDER_OPTIONS}
PROVIDER_LABEL_BY_NAME = {name: display for name, display, _ in PROVIDER_OPTIONS}
ANALYSIS_PLACEHOLDER = "Run an analysis to populate this section."
EXAMPLE_BILL_LABELS = example_bill_titles()
EXAMPLE_BILL_URLS = example_bill_urls_by_title()
CHAT_STREAM_DELAY_SECONDS = 0.004
APP_THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="stone",
).set(
    body_background_fill="linear-gradient(180deg, #f7fafc 0%, #edf4f7 100%)",
    block_background_fill="rgba(255, 255, 255, 0.88)",
    block_border_color="#d7e3ea",
    block_shadow="0 18px 48px rgba(15, 23, 42, 0.06)",
    button_primary_background_fill="linear-gradient(135deg, #7dd3fc 0%, #67e8f9 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #38bdf8 0%, #22d3ee 100%)",
    button_primary_border_color="#7dd3fc",
    button_secondary_background_fill="#f8fafc",
    button_secondary_background_fill_hover="#eef6f8",
)
APP_HEAD = """
<script>
(() => {
  function toggleSidebar() {
    const sidebar = document.getElementById("sidebar-panel");
    if (!sidebar) {
      return;
    }
    const isHidden = sidebar.style.display === "none";
    sidebar.style.display = isHidden ? "" : "none";
  }

  window.__toggleSidebar = toggleSidebar;

  function decorateRerunSummaryButton() {
    const button = document.querySelector("#rerun-summary-button button");
    if (!button || button.dataset.decorated === "true") {
      return;
    }
    button.title = "Rerun summary";
    button.setAttribute("aria-label", "Rerun summary");
    button.dataset.decorated = "true";
  }

  document.addEventListener("DOMContentLoaded", () => {
    decorateRerunSummaryButton();
    const observer = new MutationObserver(decorateRerunSummaryButton);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
</script>
"""
APP_CSS = """
#app-shell {
    background: transparent;
}
#layout-shell {
    gap: 1.25rem;
    align-items: flex-start;
}
#topbar-row {
    justify-content: flex-start;
    align-items: center;
    margin-bottom: 0.75rem;
}
#sidebar-toggle-shell {
    width: auto;
}
#sidebar-toggle-shell > div {
    width: auto !important;
}
#sidebar-toggle-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.75rem;
    height: 1.75rem;
    border: 1px solid var(--button-secondary-border-color);
    border-radius: 5px;
    background: var(--button-secondary-background-fill);
    cursor: pointer;
    transition: background-color 0.2s ease, border-color 0.2s ease;
}
#sidebar-toggle-icon:hover {
    background: var(--button-secondary-background-fill-hover);
}
#sidebar-toggle-icon svg {
    width: 1.35rem;
    height: 1.35rem;
    stroke: #6b7280;
}
#sidebar-toggle {
    display: none;
}
#sidebar-panel {
    max-width: 360px;
    position: sticky;
    top: 1rem;
    align-self: flex-start;
    border-radius: 16px;
    overflow: hidden;
}
#main-panel {
    min-width: 0;
}
#analysis-panel,
#chat-panel {
    border: 1px solid var(--block-border-color);
    border-radius: 16px;
    padding: 1rem;
    backdrop-filter: blur(12px);
}
#analysis-header-row {
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
}
#analysis-header-row .prose {
    margin: 0;
}
#rerun-summary-shell {
    flex: 0 0 auto;
    width: auto !important;
    min-width: 0 !important;
}
#analysis-output {
    min-height: 100px;
    min-width: 0;
    overflow-x: auto;
}
#analysis-output .prose,
#analysis-output .md,
#analysis-output .markdown {
    display: block;
    max-width: 100%;
    overflow-x: auto;
    padding-bottom: 0.35rem;
}
#analysis-output table {
    width: 100%;
    min-width: 760px;
    table-layout: fixed;
    border-collapse: collapse;
}
#analysis-output .analysis-table-scroll {
    display: block;
    width: 100%;
    max-width: 100%;
    overflow-x: auto;
    margin: 0.75rem 0 1.25rem;
    padding-bottom: 0.35rem;
}
#analysis-output .analysis-table-scroll table {
    margin: 0;
}
#analysis-output th,
#analysis-output td {
    vertical-align: top;
    white-space: normal;
    word-break: normal;
    overflow-wrap: anywhere;
}
#rerun-summary-button {
    width: auto;
    min-width: 0;
    min-width: 2rem;
    width: 2rem;
    height: 2rem;
    padding: 0;
    border-radius: 0 !important;
    font-size: 0.95rem;
    line-height: 1;
}
#clear-analysis-button {
    width: auto;
    min-width: 0;
    min-width: 2rem;
    width: 2rem;
    height: 2rem;
    padding: 0;
    border-radius: 0 !important;
    font-size: 1rem;
    line-height: 1;
}
#chat-question-row {
    align-items: flex-start;
}
#ask-button button {
    border-radius: 0.6rem !important;
}
#status-output,
#chat-status {
    min-height: 1.5rem;
}
#view-source-button {
    margin-top: 0.5rem;
    width: 100%;
}
#view-source-button button {
    width: 100%;
    min-width: 0;
    border-radius: 0.6rem !important;
}
#reset-button {
    width: 100%;
}
#reset-button button {
    width: 100%;
    min-width: 0;
    border-radius: 0.6rem !important;
}
#example-bills-heading {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-bottom: 0.35rem;
}
#example-bills-heading p {
    margin: 0;
}
#example-bills-info {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1rem;
    height: 1rem;
    border: 1px solid #94a3b8;
    border-radius: 999px;
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 700;
    line-height: 1;
    cursor: help;
}
#example-bills-info::after {
    content: attr(data-tooltip);
    position: absolute;
    left: calc(100% + 0.45rem);
    top: 50%;
    transform: translateY(-50%);
    width: 240px;
    padding: 0.6rem 0.7rem;
    border-radius: 0.6rem;
    background: rgba(15, 23, 42, 0.96);
    color: #f8fafc;
    font-size: 0.78rem;
    font-weight: 500;
    line-height: 1.35;
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.22);
    transition: opacity 0.18s ease;
    z-index: 10;
}
#example-bills-info:hover::after,
#example-bills-info:focus-visible::after {
    opacity: 1;
    visibility: visible;
}
#app-shell .label-wrap,
#app-shell .label-wrap > *,
#app-shell label,
#app-shell legend,
#app-shell [data-testid="block-label"] {
    background: transparent !important;
    box-shadow: none !important;
}
@media (max-width: 900px) {
    #layout-shell {
        flex-direction: column;
    }
    #sidebar-panel {
        max-width: none;
        width: 100%;
        position: static;
    }
    #main-panel {
        width: 100%;
    }
}
@media (max-width: 640px) {
    #analysis-panel,
    #chat-panel {
        padding: 0.8rem;
    }
    #analysis-output table {
        min-width: 760px;
    }
    #chat-question-row {
        flex-direction: column;
    }
    #chat-question-row > *,
    #ask-button,
    #ask-button button {
        width: 100% !important;
        min-width: 0 !important;
    }
}
"""
APP_LAUNCH_KWARGS = {
    "theme": APP_THEME,
    "css": APP_CSS,
    "head": APP_HEAD,
}

_GRADIO_SESSION_CACHE: dict[str, dict[str, Any]] = {}


def _empty_session() -> dict[str, Any]:
    return {"session_id": uuid4().hex}


def _record_defaults() -> dict[str, Any]:
    return {
        "api_config": None,
        "analysis": None,
        "vector_store": None,
        "doc_text": None,
        "chat_history": [],
        "pending_deeper_question": None,
        "source_url": None,
        "message_queue": [],
        "is_answering": False,
        "active_message_id": None,
    }


def _session_record(session_state: dict[str, Any] | None) -> dict[str, Any]:
    session_state = session_state or _empty_session()
    session_id = session_state.get("session_id")
    if not session_id:
        session_id = uuid4().hex
        session_state["session_id"] = session_id
    record = _GRADIO_SESSION_CACHE.setdefault(session_id, _record_defaults())
    for key, value in _record_defaults().items():
        record.setdefault(key, value)
    return record


def _replace_record(record: dict[str, Any], **updates: Any) -> dict[str, Any]:
    record.clear()
    record.update(_record_defaults())
    record.update(updates)
    return record


def _resolve_provider(use_advanced: bool, provider_label: str | None) -> str:
    return DEFAULT_PROVIDER
    # Proposed future expansion: restore bring-your-own provider selection.
    # if not use_advanced:
    #     return DEFAULT_PROVIDER
    # if not provider_label:
    #     return DEFAULT_PROVIDER
    # return PROVIDER_BY_LABEL.get(provider_label, DEFAULT_PROVIDER)


def _resolve_api_key(provider: str, use_advanced: bool, qwen_key: str | None, custom_key: str | None) -> str | None:
    return qwen_key or get_default_api_key(provider)
    # Proposed future expansion: restore bring-your-own provider API keys.
    # if not use_advanced and provider == DEFAULT_PROVIDER:
    #     return qwen_key or get_default_api_key(provider)
    # return custom_key or get_default_api_key(provider)


def _ingest_sources(file_path: str | None, url: str | None) -> str:
    chunks = []
    if file_path:
        path = Path(file_path)
        chunks.append(ingest.ingest_file(path.name, path.read_bytes()))
    if url and url.strip():
        chunks.append(ingest.fetch_url_content(url.strip()))
    if not chunks:
        raise ingest.IngestionError("Provide a file or URL to analyze.")
    return ingest.combine_sources(chunks)


def _escape_md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _escape_html_cell(value: str) -> str:
    return html.escape(str(value).replace("\n", " ").strip())


def _markdown_bullets(items: list[str]) -> str:
    if not items:
        return "-"
    return "\n".join(f"- {_escape_md_cell(item) or '—'}" for item in items)


def _analysis_attr(analysis: AnalysisResult, field: str, default: Any) -> Any:
    return getattr(analysis, field, default)


def _swot_attr(analysis: AnalysisResult, field: str) -> list[str]:
    swot = _analysis_attr(analysis, "swot", None)
    return getattr(swot, field, []) if swot is not None else []


def _format_html_table(headers: list[str], rows: list[list[str]]) -> str:
    table_rows = rows or [[""] * len(headers)]
    header_html = "".join(f"<th>{_escape_html_cell(header) or '&mdash;'}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_escape_html_cell(cell) or '&mdash;'}</td>" for cell in row) + "</tr>"
        for row in table_rows
    )
    return (
        '<div class="analysis-table-scroll">'
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )


def _format_analysis(analysis: AnalysisResult | None) -> str:
    if analysis is None:
        return ANALYSIS_PLACEHOLDER

    executive_summary = _escape_md_cell(_analysis_attr(analysis, "executive_summary", "")) or "No executive summary returned."
    bill_summary = _markdown_bullets(_analysis_attr(analysis, "bill_summary", []))
    implementation_rows = [
        [
            getattr(item, "stakeholder", ""),
            getattr(item, "obligation", ""),
            getattr(item, "implementation_burden", ""),
            getattr(item, "risk_or_note", ""),
        ]
        for item in _analysis_attr(analysis, "implementation", [])
    ]
    critique_rows = [
        [
            getattr(item, "issue", ""),
            getattr(item, "why_it_matters", ""),
            getattr(item, "recommendation", ""),
        ]
        for item in _analysis_attr(analysis, "critique", [])
    ]
    strengths = _swot_attr(analysis, "strengths")
    weaknesses = _swot_attr(analysis, "weaknesses")
    opportunities = _swot_attr(analysis, "opportunities")
    threats = _swot_attr(analysis, "threats")
    swot_pair_rows = [
        [strengths[idx] if idx < len(strengths) else "", weaknesses[idx] if idx < len(weaknesses) else ""]
        for idx in range(max(len(strengths), len(weaknesses), 1))
    ]
    swot_risk_rows = [
        [opportunities[idx] if idx < len(opportunities) else "", threats[idx] if idx < len(threats) else ""]
        for idx in range(max(len(opportunities), len(threats), 1))
    ]

    return "\n\n".join(
        [
            "## Executive Summary\n" + executive_summary,
            "## Summary of the Bill\n" + bill_summary,
            "## Implementation Implications\n"
            + _format_html_table(["Stakeholder", "Obligation", "Burden", "Risk / Note"], implementation_rows),
            "## Critique\n"
            + _format_html_table(["Issue", "Why it matters", "Recommendation"], critique_rows),
            "## SWOT Analysis\n"
            + _format_html_table(["Strengths", "Weaknesses"], swot_pair_rows)
            + "\n\n"
            + _format_html_table(["Opportunities", "Threats"], swot_risk_rows),
        ]
    )


def _render_supporting_snippets(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""

    items = []
    for item in citations:
        ref_id = html.escape(str(item["ref_id"]))
        snippet = html.escape(item["snippet"])
        items.append(f"<li><strong>[ref{ref_id}]</strong> {snippet}</li>")

    count = len(citations)
    label = "snippet" if count == 1 else "snippets"
    return (
        f"<details><summary>Supporting {label} ({count})</summary>"
        f"<ul>{''.join(items)}</ul>"
        "</details>"
    )


def _format_chat_entry(answer_text: str, citations: list[dict[str, Any]], *, provenance: str = "analysis_based") -> str:
    prefix = ""
    if provenance == "analysis_based":
        prefix = "_Based on the summary and analysis._\n\n"
    elif provenance == "full_document":
        prefix = "_Full-document answer._\n\n"

    if not citations:
        return prefix + answer_text
    return f"{prefix}{answer_text}\n\n{_render_supporting_snippets(citations)}"


def _stream_chat_entry(
    base_history: list[dict[str, str]],
    answer_text: str,
    citations: list[dict[str, Any]],
    *,
    provenance: str = "analysis_based",
):
    prefix = ""
    if provenance == "analysis_based":
        prefix = "_Based on the summary and analysis._\n\n"
    elif provenance == "full_document":
        prefix = "_Deeper full-document answer._\n\n"

    formatted_answer = prefix + answer_text
    streamed_answer = ""
    for character in formatted_answer:
        streamed_answer += character
        yield base_history + [{"role": "assistant", "content": streamed_answer}]
        if CHAT_STREAM_DELAY_SECONDS:
            time.sleep(CHAT_STREAM_DELAY_SECONDS)
    if citations:
        yield base_history + [{"role": "assistant", "content": _format_chat_entry(answer_text, citations, provenance=provenance)}]


def _stream_answer_content(answer_text: str, *, provenance: str = "analysis_based") -> Iterable[str]:
    prefix = ""
    if provenance == "analysis_based":
        prefix = "_Based on the summary and analysis._\n\n"
    elif provenance == "full_document":
        prefix = "_Full-document answer._\n\n"

    streamed_answer = ""
    for character in prefix + answer_text:
        streamed_answer += character
        yield streamed_answer
        if CHAT_STREAM_DELAY_SECONDS:
            time.sleep(CHAT_STREAM_DELAY_SECONDS)


def _deeper_answer_updates(visible: bool, label: str = "Run deeper full-document answer") -> gr.update:
    return gr.update(visible=visible, value=label)


def _analysis_action_updates(enabled: bool) -> tuple[gr.update, gr.update]:
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def _source_control_updates(enabled: bool) -> tuple[gr.update, gr.update, gr.update, gr.update, gr.update]:
    return (
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
        gr.update(interactive=enabled),
    )


def _analysis_available(record: dict[str, Any]) -> bool:
    return bool(record.get("analysis"))


def _question_placeholder_updates(record: dict[str, Any], *, clear_input: bool = False) -> tuple[gr.update, gr.update]:
    question_input_update: dict[str, Any] = {"interactive": True}
    if clear_input:
        question_input_update["value"] = ""
    return (
        gr.update(**question_input_update),
        gr.update(interactive=True),
    )


def _stage_question(
    question: str | None,
    session_state: dict[str, Any] | None,
    chat_history: list[dict[str, str]] | None,
) -> tuple[str, str, list[dict[str, str]], str, dict[str, Any]]:
    staged_question = question or ""
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    question_text = staged_question.strip()
    if not question_text:
        return "", "", (chat_history or _displayed_chat_history(record)), _queue_status_text(record), session_state
    if record.get("is_answering"):
        _enqueue_question(record, question_text)
        return "", "", _displayed_chat_history(record), _queue_status_text(record), session_state
    return question_text, "", (chat_history or _displayed_chat_history(record)), _queue_status_text(record), session_state


def _chat_control_updates(record: dict[str, Any], *, is_busy: bool) -> tuple[gr.update, ...]:
    return (
        *_source_control_updates(not is_busy),
        *_analysis_action_updates(_analysis_available(record) and not is_busy),
        *_question_placeholder_updates(record),
    )


def _queue_placeholder(status: str) -> str:
    if status == "answering":
        return "_Answering..._"
    if status == "failed":
        return "_Failed._"
    return "_Queued._"


def _displayed_chat_history(
    record: dict[str, Any],
    *,
    answering_content: str | None = None,
) -> list[dict[str, str]]:
    history = list(record.get("chat_history", []))
    queue: list[dict[str, Any]] = list(record.get("message_queue", []))
    active_message_id = record.get("active_message_id")

    for index, item in enumerate(queue):
        history.append({"role": "user", "content": item["question"]})
        placeholder = _queue_placeholder(item.get("status", "queued"))
        if index == 0 and item.get("id") == active_message_id:
            placeholder = answering_content or placeholder
        history.append({"role": "assistant", "content": placeholder})
    return history


def _queue_status_text(record: dict[str, Any]) -> str:
    queue: list[dict[str, Any]] = list(record.get("message_queue", []))
    if not queue:
        return ""
    lines = ["Queued:"]
    lines.extend(f"{index}. {item['question']}" for index, item in enumerate(queue, start=1))
    return "\n".join(lines)


def _enqueue_question(record: dict[str, Any], question_text: str) -> None:
    queue = record.setdefault("message_queue", [])
    queue.append(
        {
            "id": uuid4().hex,
            "question": question_text,
            "status": "queued",
        }
    )


def _flush_message_queue(record: dict[str, Any], reason: str | None = None) -> str:
    had_pending = bool(record.get("message_queue")) or bool(record.get("active_message_id"))
    record["message_queue"] = []
    record["is_answering"] = False
    record["active_message_id"] = None
    record["pending_deeper_question"] = None
    return reason if had_pending and reason else ""


# Proposed future expansion: restore bring-your-own provider help text and
# controls when the app supports non-hackathon model/provider comparison again.
# def _provider_help_text(provider_label: str | None, use_advanced: bool) -> str:
#     provider = _resolve_provider(use_advanced, provider_label)
#     return PROVIDER_DETAILS[provider]
#
#
# def _toggle_provider_mode(use_advanced: bool) -> tuple[gr.update, gr.update, gr.update, gr.update]:
#     provider_value = PROVIDER_LABEL_BY_NAME[DEFAULT_PROVIDER]
#     return (
#         gr.update(visible=not use_advanced),
#         gr.update(visible=use_advanced),
#         gr.update(visible=use_advanced),
#         gr.update(value=_provider_help_text(provider_value, use_advanced)),
#     )


def _set_example_url(example_label: str | None) -> str:
    if not example_label:
        return ""
    return EXAMPLE_BILL_URLS.get(example_label, "")


def _handle_uploaded_source_change(
    uploaded_file: str | None,
    session_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], gr.update, gr.update, gr.update, str | None, list[dict[str, str]], str, gr.update, str]:
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    status_message = _flush_message_queue(record, "Queued questions were cleared because the source changed.")
    record["chat_history"] = []
    if not uploaded_file:
        return (
            session_state,
            gr.update(),
            gr.update(),
            gr.update(),
            None,
            _displayed_chat_history(record),
            status_message,
            _deeper_answer_updates(False),
            "",
        )
    return (
        session_state,
        gr.update(value=""),
        gr.update(value=None),
        _view_source_button_update(""),
        "file",
        _displayed_chat_history(record),
        status_message,
        _deeper_answer_updates(False),
        "",
    )


def _handle_url_source_change(
    url_value: str | None,
    session_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], gr.update, gr.update, str | None, list[dict[str, str]], str, gr.update, str]:
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    status_message = _flush_message_queue(record, "Queued questions were cleared because the source changed.")
    record["chat_history"] = []
    cleaned_url = (url_value or "").strip()
    if not cleaned_url:
        return (
            session_state,
            gr.update(),
            _view_source_button_update(""),
            None,
            _displayed_chat_history(record),
            status_message,
            _deeper_answer_updates(False),
            "",
        )
    return (
        session_state,
        gr.update(value=None),
        _view_source_button_update(cleaned_url),
        "url",
        _displayed_chat_history(record),
        status_message,
        _deeper_answer_updates(False),
        "",
    )


def _handle_example_source_change(
    example_label: str | None,
    session_state: dict[str, Any] | None,
) -> tuple[dict[str, Any], str, gr.update, str | None, list[dict[str, str]], str, gr.update, str]:
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    status_message = _flush_message_queue(record, "Queued questions were cleared because the source changed.")
    record["chat_history"] = []
    url_value = _set_example_url(example_label)
    return (
        session_state,
        url_value,
        _view_source_button_update(url_value),
        ("url" if url_value else None),
        _displayed_chat_history(record),
        status_message,
        _deeper_answer_updates(False),
        "",
    )


def _view_source_button_update(url_value: str | None) -> gr.update:
    has_url = bool(url_value and url_value.strip())
    return gr.update(value="View source ↗", interactive=has_url)


def _analysis_stage_message(stage: str) -> str:
    return f"_Processing... {stage}_"


def _show_warning(message: str) -> str:
    gr.Warning(message)
    return message


def _show_error(message: str) -> str:
    gr.Error(message)
    return message


def analyze_document(
    uploaded_file: str | None,
    url_value: str | None,
    use_advanced: bool,
    provider_label: str | None,
    qwen_key: str | None,
    custom_key: str | None,
    session_state: dict[str, Any] | None,
    source_kind: str | None = None,
    force_refresh: bool = False,
):
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    _flush_message_queue(record)
    record_api_config = record.get("api_config") or {}
    provider = record_api_config.get("provider") or _resolve_provider(use_advanced, provider_label)
    api_key = record_api_config.get("api_key") or _resolve_api_key(provider, use_advanced, qwen_key, custom_key)
    is_valid, error = validate_api_key(provider, api_key)
    if not is_valid:
        message = "Check the selected provider and API key, then try again."
        _show_warning(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    active_file = uploaded_file if source_kind != "url" else None
    active_url = url_value if source_kind != "file" else None

    precomputed = get_precomputed_example_artifacts(active_url)
    if precomputed is not None and not force_refresh:
        _replace_record(
            record,
            api_config={"provider": provider, "api_key": api_key},
            analysis=precomputed.analysis.model_dump(),
            vector_store=precomputed.vector_store,
            doc_text=precomputed.document_text,
            source_url=active_url,
        )
        yield (
            session_state,
            "Loaded precomputed analysis for this example bill.",
            _format_analysis(precomputed.analysis),
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(True),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    yield (
        session_state,
        "Loading and parsing document...",
        _analysis_stage_message("Loading and parsing document..."),
        [],
        *_source_control_updates(True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(record),
        _deeper_answer_updates(False),
        "",
    )

    try:
        document_text = _ingest_sources(active_file, active_url)
    except ingest.IngestionError as exc:
        message = str(exc)
        _show_warning(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return
    except Exception:  # noqa: BLE001
        message = "We couldn't load that document. Check the link or upload the file directly and try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    yield (
        session_state,
        "Preparing chunks and provider client...",
        _analysis_stage_message("Preparing chunks and provider client..."),
        [],
        *_source_control_updates(True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(record),
        _deeper_answer_updates(False),
        "",
    )

    try:
        provider_client = instantiate_client(provider, api_key or "")
    except Exception:  # noqa: BLE001
        message = "We couldn't connect to the selected model provider. Check your API key and settings, then try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    yield (
        session_state,
        "Building retrieval index...",
        _analysis_stage_message("Building retrieval index..."),
        [],
        *_source_control_updates(True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(record),
        _deeper_answer_updates(False),
        "",
    )

    try:
        _, chunks, vector_store = prepare_document_artifacts(document_text)
    except Exception:  # noqa: BLE001
        message = "We couldn't prepare the document for analysis. Please try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    try:
        analysis = AnalysisResult()
        for stage_message, partial in generate_analysis_progress(provider_client, document_text):
            analysis = partial
            yield (
                session_state,
                stage_message,
                _format_analysis(partial),
                [],
                *_source_control_updates(True),
                *_analysis_action_updates(True),
                *_question_placeholder_updates(record),
                _deeper_answer_updates(False),
                "",
            )
    except Exception:  # noqa: BLE001
        message = "We couldn't generate the bill analysis right now. Please try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    build_cached_document_artifacts(
        document_text=document_text,
        chunks=chunks,
        analysis=analysis,
        vector_store=vector_store,
        source_url=active_url,
    )
    _replace_record(
        record,
        api_config={"provider": provider, "api_key": api_key},
        analysis=analysis.model_dump(),
        vector_store=vector_store,
        doc_text=document_text,
        source_url=active_url,
    )
    analysis_output = _format_analysis(analysis)
    yield session_state, "", analysis_output, [], *_source_control_updates(True), *_analysis_action_updates(True), *_question_placeholder_updates(record), _deeper_answer_updates(False), ""


def _rerun_record_analysis(
    session_state: dict[str, Any],
    record: dict[str, Any],
    *,
    provider: str,
    api_key: str | None,
) -> Any:
    document_text = record.get("doc_text")
    if not document_text:
        return
    _flush_message_queue(record)

    yield (
        session_state,
        "Preparing provider client...",
        _analysis_stage_message("Preparing provider client..."),
        [],
        *_source_control_updates(True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(record),
        _deeper_answer_updates(False),
        "",
    )

    try:
        provider_client = instantiate_client(provider, api_key or "")
    except Exception:  # noqa: BLE001
        message = "We couldn't connect to the selected model provider. Check your API key and settings, then try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    try:
        analysis = AnalysisResult()
        for stage_message, partial in generate_analysis_progress(provider_client, document_text):
            analysis = partial
            yield (
                session_state,
                stage_message,
                _format_analysis(partial),
                [],
                *_source_control_updates(True),
                *_analysis_action_updates(True),
                *_question_placeholder_updates(record),
                _deeper_answer_updates(False),
                "",
            )
    except Exception:  # noqa: BLE001
        message = "We couldn't generate the bill analysis right now. Please try again."
        _show_error(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    record.update(
        {
            "api_config": {"provider": provider, "api_key": api_key},
            "analysis": analysis.model_dump(),
            "chat_history": [],
            "pending_deeper_question": None,
        }
    )
    yield session_state, "", _format_analysis(analysis), [], *_source_control_updates(True), *_analysis_action_updates(True), *_question_placeholder_updates(record), _deeper_answer_updates(False), ""


def rerun_summary(
    uploaded_file: str | None,
    url_value: str | None,
    use_advanced: bool,
    provider_label: str | None,
    qwen_key: str | None,
    custom_key: str | None,
    session_state: dict[str, Any] | None,
    source_kind: str | None = None,
):
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    record_api_config = record.get("api_config") or {}
    provider = record_api_config.get("provider") or _resolve_provider(use_advanced, provider_label)
    api_key = record_api_config.get("api_key") or _resolve_api_key(provider, use_advanced, qwen_key, custom_key)
    is_valid, _error = validate_api_key(provider, api_key)
    if not is_valid:
        message = "Check the selected provider and API key, then try again."
        _show_warning(message)
        yield (
            session_state,
            message,
            ANALYSIS_PLACEHOLDER,
            [],
            *_source_control_updates(True),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
            _deeper_answer_updates(False),
            "",
        )
        return

    if record.get("doc_text"):
        yield from _rerun_record_analysis(
            session_state,
            record,
            provider=provider,
            api_key=api_key,
        )
        return

    yield from analyze_document(
        uploaded_file,
        url_value,
        use_advanced,
        provider_label,
        qwen_key,
        custom_key,
        session_state,
        source_kind,
        force_refresh=True,
    )


def clear_analysis(
    session_state: dict[str, Any] | None,
):
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    _flush_message_queue(record)
    record["analysis"] = None
    record["chat_history"] = []
    record["pending_deeper_question"] = None
    return (
        session_state,
        "",
        ANALYSIS_PLACEHOLDER,
        [],
        *_source_control_updates(True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(record),
        _deeper_answer_updates(False),
        "",
    )


def _prepare_record_for_question(
    record: dict[str, Any],
    *,
    uploaded_file: str | None,
    url_value: str | None,
    use_advanced: bool,
    provider_label: str | None,
    qwen_key: str | None,
    custom_key: str | None,
    source_kind: str | None = None,
) -> str | None:
    if (record.get("doc_text") or record.get("vector_store")) and record.get("api_config"):
        return None

    provider = _resolve_provider(use_advanced, provider_label)
    api_key = _resolve_api_key(provider, use_advanced, qwen_key, custom_key)
    is_valid, _error = validate_api_key(provider, api_key)
    if not is_valid:
        return "Check the selected provider and API key, then try again."

    try:
        active_file = uploaded_file if source_kind != "url" else None
        active_url = url_value if source_kind != "file" else None
        document_text = _ingest_sources(active_file, active_url)
    except ingest.IngestionError as exc:
        return str(exc)
    except Exception:  # noqa: BLE001
        return "We couldn't load that document. Check the link or upload the file directly and try again."

    try:
        _, chunks, vector_store = prepare_document_artifacts(document_text)
    except Exception:  # noqa: BLE001
        return "We couldn't prepare the document for question answering. Please try again."

    _replace_record(
        record,
        api_config={"provider": provider, "api_key": api_key},
        analysis=None,
        vector_store=vector_store,
        doc_text=document_text,
        source_url=active_url,
    )
    return None


def ask_question(
    question: str | None,
    uploaded_file: str | None,
    url_value: str | None,
    use_advanced: bool,
    provider_label: str | None,
    qwen_key: str | None,
    custom_key: str | None,
    session_state: dict[str, Any] | None,
    chat_history: list[dict[str, str]] | None,
    source_kind: str | None = None,
):
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    chat_history = chat_history or _displayed_chat_history(record)
    if not question or not question.strip():
        yield chat_history, session_state, _queue_status_text(record), _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=bool(record.get("is_answering")))
        return

    source_prep_error = _prepare_record_for_question(
        record,
        uploaded_file=uploaded_file,
        url_value=url_value,
        source_kind=source_kind,
        use_advanced=use_advanced,
        provider_label=provider_label,
        qwen_key=qwen_key,
        custom_key=custom_key,
    )
    if source_prep_error is not None:
        yield chat_history, session_state, source_prep_error, _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=False)
        return

    question_text = question.strip()
    _enqueue_question(record, question_text)

    if record.get("is_answering"):
        yield (
            _displayed_chat_history(record),
            session_state,
            _queue_status_text(record),
            _deeper_answer_updates(False),
            "",
            *_source_control_updates(False),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
        )
        return

    while record.get("message_queue"):
        current = record["message_queue"][0]
        current["status"] = "answering"
        record["is_answering"] = True
        record["active_message_id"] = current["id"]
        yield (
            _displayed_chat_history(record),
            session_state,
            _queue_status_text(record),
            _deeper_answer_updates(False),
            "",
            *_source_control_updates(False),
            *_analysis_action_updates(False),
            *_question_placeholder_updates(record),
        )

        try:
            provider_client = instantiate_client(
                record["api_config"]["provider"],
                record["api_config"]["api_key"],
            )
            answer = answer_query_from_full_document(
                provider_client,
                record.get("vector_store"),
                current["question"],
                doc_text=record.get("doc_text"),
            )
        except Exception:  # noqa: BLE001
            message = "We couldn't answer that question right now. Please try again."
            _show_error(message)
            record["chat_history"] = list(record.get("chat_history", [])) + [
                {"role": "user", "content": current["question"]},
                {"role": "assistant", "content": message},
            ]
            record["message_queue"].pop(0)
            record["active_message_id"] = None
            record["is_answering"] = False
            yield (
                _displayed_chat_history(record),
                session_state,
                message if not record.get("message_queue") else _queue_status_text(record),
                _deeper_answer_updates(False),
                "",
                *_source_control_updates(True),
                *_analysis_action_updates(_analysis_available(record)),
                *_question_placeholder_updates(record),
            )
            continue

        for partial_answer in _stream_answer_content(answer.answer, provenance=answer.provenance):
            yield (
                _displayed_chat_history(record, answering_content=partial_answer),
                session_state,
                _queue_status_text(record),
                _deeper_answer_updates(False),
                "",
                *_source_control_updates(False),
                *_analysis_action_updates(False),
                *_question_placeholder_updates(record),
            )

        citations = [citation.model_dump() for citation in answer.citations]
        final_answer = _format_chat_entry(answer.answer, citations, provenance=answer.provenance)
        record["chat_history"] = list(record.get("chat_history", [])) + [
            {"role": "user", "content": current["question"]},
            {"role": "assistant", "content": final_answer},
        ]
        record["message_queue"].pop(0)
        record["active_message_id"] = None
        record["is_answering"] = False
        record["pending_deeper_question"] = None
        yield (
            _displayed_chat_history(record),
            session_state,
            _queue_status_text(record),
            _deeper_answer_updates(False),
            "",
            *_source_control_updates(True),
            *_analysis_action_updates(_analysis_available(record)),
            *_question_placeholder_updates(record),
        )


def run_deeper_answer(
    session_state: dict[str, Any] | None,
    chat_history: list[dict[str, str]] | None,
):
    session_state = session_state or _empty_session()
    record = _session_record(session_state)
    chat_history = chat_history or []
    if record.get("is_answering"):
        yield chat_history, session_state, _queue_status_text(record), _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=True)
        return
    question_text = record.get("pending_deeper_question")
    if not question_text:
        yield chat_history, session_state, "No pending question for deeper analysis.", _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=False)
        return

    record["is_answering"] = True
    yield chat_history, session_state, "Reading the full document for a deeper answer...", _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=True)

    try:
        provider_client = instantiate_client(
            record["api_config"]["provider"],
            record["api_config"]["api_key"],
        )
        answer = answer_query_from_full_document(
            provider_client,
            record.get("vector_store"),
            question_text,
            doc_text=record.get("doc_text"),
        )
    except Exception:  # noqa: BLE001
        message = "We couldn't generate the deeper full-document answer right now. Please try again."
        _show_error(message)
        record["is_answering"] = False
        yield chat_history, session_state, message, _deeper_answer_updates(True), "A deeper full-document answer is still available.", *_chat_control_updates(record, is_busy=False)
        return

    citations = [citation.model_dump() for citation in answer.citations]
    for partial_history in _stream_chat_entry(chat_history, answer.answer, citations, provenance=answer.provenance):
        yield partial_history, session_state, "", _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=True)

    chat_history = chat_history + [
        {"role": "assistant", "content": _format_chat_entry(answer.answer, citations, provenance=answer.provenance)}
    ]
    record["chat_history"] = chat_history
    record["pending_deeper_question"] = None
    record["is_answering"] = False
    yield chat_history, session_state, "", _deeper_answer_updates(False), "", *_chat_control_updates(record, is_busy=False)


def reset_session(session_state: dict[str, Any] | None):
    if session_state and session_state.get("session_id"):
        _GRADIO_SESSION_CACHE.pop(session_state["session_id"], None)
    empty = _empty_session()
    empty_record = _session_record(empty)
    return (
        empty,
        gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),
        gr.update(value="", interactive=True),
        None,
        _view_source_button_update(""),
        ANALYSIS_PLACEHOLDER,
        [],
        "",
        "",
        gr.update(interactive=True),
        gr.update(interactive=True),
        *_analysis_action_updates(False),
        *_question_placeholder_updates(empty_record),
        _deeper_answer_updates(False),
        "",
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, elem_id="app-shell") as demo:
        session_state = gr.State(_empty_session())
        source_kind = gr.State(None)
        queued_question = gr.State("")

        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)
        with gr.Row(elem_id="topbar-row"):
            gr.HTML(
                """
                <div id="sidebar-toggle-shell">
                  <div
                    id="sidebar-toggle-icon"
                    role="button"
                    aria-label="Toggle sidebar"
                    tabindex="0"
                    onclick="window.__toggleSidebar && window.__toggleSidebar()"
                    onkeydown="if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); window.__toggleSidebar && window.__toggleSidebar(); }"
                  >
                    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                      <rect x="3" y="4" width="18" height="16" rx="3"></rect>
                      <path d="M9 4v16"></path>
                      <path d="M6.5 9h.01"></path>
                      <path d="M6.5 12h.01"></path>
                      <path d="M6.5 15h.01"></path>
                    </svg>
                  </div>
                </div>
                """
            )

        with gr.Row(elem_id="layout-shell"):
            with gr.Column(scale=1, elem_id="sidebar-panel"):
                gr.Markdown("## Document Source")
                gr.Markdown("Upload PDF, DOCX, TXT, or MD")
                uploaded_file = gr.File(label="Upload PDF, DOCX, TXT, or MD", type="filepath", show_label=False)
                gr.HTML(
                    """
                    <div id="example-bills-heading">
                      <p>Example bills</p>
                      <span
                        id="example-bills-info"
                        tabindex="0"
                        aria-label="Example bills information"
                        data-tooltip="The Ghana Innovation and Start-Up Bill, 2025 was omitted because a verifiable source could not be found."
                      >
                        i
                      </span>
                    </div>
                    """
                )
                example_selector = gr.Dropdown(
                    label="Example bills",
                    choices=EXAMPLE_BILL_LABELS,
                    value=None,
                    show_label=False,
                )
                gr.Markdown("Document URL")
                url_value = gr.Textbox(label="Document URL", placeholder="https://example.com/bill", show_label=False)
                analyze_button = gr.Button("Run analysis", variant="primary")
                view_source_button = gr.Button("View source ↗", variant="secondary", interactive=False, elem_id="view-source-button")
                reset_button = gr.Button("Reset session", elem_id="reset-button")
                status_output = gr.Markdown(elem_id="status-output")
                use_advanced = gr.State(False)
                provider_label = gr.State(PROVIDER_LABEL_BY_NAME[DEFAULT_PROVIDER])
                qwen_key = gr.State(None)
                custom_key = gr.State(None)

                # Proposed future expansion: restore the sidebar Model Settings
                # section after the hackathon build is no longer pinned to a
                # single built-in provider.
                # with gr.Accordion("Model Settings", open=False):
                #     provider_help = gr.Markdown(PROVIDER_DETAILS[DEFAULT_PROVIDER])
                #     gr.Markdown("Hugging Face Token")
                #     qwen_key = gr.Textbox(
                #         label="Hugging Face Token",
                #         type="password",
                #         placeholder="Leave blank to use HF_TOKEN",
                #         visible=True,
                #         show_label=False,
                #     )
                #     use_advanced = gr.Checkbox(label="Bring your own provider", value=False)
                #     gr.Markdown("Model Provider")
                #     provider_label = gr.Dropdown(
                #         label="Model Provider",
                #         choices=PROVIDER_LABELS,
                #         value=PROVIDER_LABEL_BY_NAME[DEFAULT_PROVIDER],
                #         visible=False,
                #         show_label=False,
                #     )
                #     gr.Markdown("API Key")
                #     custom_key = gr.Textbox(
                #         label="API Key",
                #         type="password",
                #         placeholder="Leave blank to use provider env var",
                #         visible=False,
                #         show_label=False,
                #     )

            with gr.Column(scale=2, elem_id="main-panel"):
                with gr.Group(elem_id="analysis-panel"):
                    with gr.Row(elem_id="analysis-header-row"):
                        with gr.Column(scale=1, min_width=0):
                            gr.Markdown("# Summary Highlights")
                        with gr.Column(scale=0, min_width=0, elem_id="rerun-summary-shell"):
                            with gr.Row():
                                rerun_summary_button = gr.Button("↺", elem_id="rerun-summary-button", variant="secondary", interactive=False)
                                clear_analysis_button = gr.Button("✕", elem_id="clear-analysis-button", variant="secondary", interactive=False)
                    analysis_output = gr.Markdown(ANALYSIS_PLACEHOLDER, elem_id="analysis-output")

                with gr.Group(elem_id="chat-panel"):
                    gr.Markdown("## Ask Further Questions")
                    chatbot = gr.Chatbot(show_label=False, autoscroll=False)
                    gr.Markdown("Question")
                    with gr.Row(elem_id="chat-question-row"):
                        with gr.Column(scale=6, min_width=0):
                            question_input = gr.Textbox(
                                label="Question",
                                placeholder="What would you like to know?",
                                show_label=False,
                            )
                        ask_button = gr.Button("Ask", scale=1, variant="primary", elem_id="ask-button")
                    chat_status = gr.Markdown(elem_id="chat-status")
                    deeper_answer_button = gr.Button("Run deeper full-document answer", visible=False)
                    deeper_answer_hint = gr.Markdown("")

        # Proposed future expansion: restore bring-your-own provider event wiring
        # when alternate provider controls are re-enabled.
        # use_advanced.change(
        #     _toggle_provider_mode,
        #     inputs=[use_advanced],
        #     outputs=[qwen_key, provider_label, custom_key, provider_help],
        # )
        # provider_label.change(
        #     _provider_help_text,
        #     inputs=[provider_label, use_advanced],
        #     outputs=[provider_help],
        # )
        example_selector.change(
            _handle_example_source_change,
            inputs=[example_selector, session_state],
            outputs=[session_state, url_value, view_source_button, source_kind, chatbot, chat_status, deeper_answer_button, deeper_answer_hint],
        )
        uploaded_file.change(
            _handle_uploaded_source_change,
            inputs=[uploaded_file, session_state],
            outputs=[session_state, url_value, example_selector, view_source_button, source_kind, chatbot, chat_status, deeper_answer_button, deeper_answer_hint],
        )
        url_value.input(
            _handle_url_source_change,
            inputs=[url_value, session_state],
            outputs=[session_state, uploaded_file, view_source_button, source_kind, chatbot, chat_status, deeper_answer_button, deeper_answer_hint],
        )
        url_value.change(
            _handle_url_source_change,
            inputs=[url_value, session_state],
            outputs=[session_state, uploaded_file, view_source_button, source_kind, chatbot, chat_status, deeper_answer_button, deeper_answer_hint],
        )
        analyze_button.click(
            analyze_document,
            inputs=[uploaded_file, url_value, use_advanced, provider_label, qwen_key, custom_key, session_state, source_kind],
            outputs=[
                session_state,
                status_output,
                analysis_output,
                chatbot,
                uploaded_file,
                example_selector,
                url_value,
                analyze_button,
                reset_button,
                rerun_summary_button,
                clear_analysis_button,
                question_input,
                ask_button,
                deeper_answer_button,
                deeper_answer_hint,
            ],
        )
        view_source_button.click(
            None,
            inputs=[url_value],
            js="""
            (url) => {
              if (url && url.trim()) {
                window.open(url.trim(), "_blank", "noopener,noreferrer");
              }
            }
            """,
        )
        rerun_summary_button.click(
            rerun_summary,
            inputs=[uploaded_file, url_value, use_advanced, provider_label, qwen_key, custom_key, session_state, source_kind],
            outputs=[
                session_state,
                status_output,
                analysis_output,
                chatbot,
                uploaded_file,
                example_selector,
                url_value,
                analyze_button,
                reset_button,
                rerun_summary_button,
                clear_analysis_button,
                question_input,
                ask_button,
                deeper_answer_button,
                deeper_answer_hint,
            ],
        )
        clear_analysis_button.click(
            clear_analysis,
            inputs=[session_state],
            outputs=[
                session_state,
                status_output,
                analysis_output,
                chatbot,
                uploaded_file,
                example_selector,
                url_value,
                analyze_button,
                reset_button,
                rerun_summary_button,
                clear_analysis_button,
                question_input,
                ask_button,
                deeper_answer_button,
                deeper_answer_hint,
            ],
        )
        ask_button.click(
            _stage_question,
            inputs=[question_input, session_state, chatbot],
            outputs=[queued_question, question_input, chatbot, chat_status, session_state],
            queue=False,
        ).then(
            ask_question,
            inputs=[queued_question, uploaded_file, url_value, use_advanced, provider_label, qwen_key, custom_key, session_state, chatbot, source_kind],
            outputs=[chatbot, session_state, chat_status, deeper_answer_button, deeper_answer_hint, uploaded_file, example_selector, url_value, analyze_button, reset_button, rerun_summary_button, clear_analysis_button, question_input, ask_button],
        )
        question_input.submit(
            _stage_question,
            inputs=[question_input, session_state, chatbot],
            outputs=[queued_question, question_input, chatbot, chat_status, session_state],
            queue=False,
        ).then(
            ask_question,
            inputs=[queued_question, uploaded_file, url_value, use_advanced, provider_label, qwen_key, custom_key, session_state, chatbot, source_kind],
            outputs=[chatbot, session_state, chat_status, deeper_answer_button, deeper_answer_hint, uploaded_file, example_selector, url_value, analyze_button, reset_button, rerun_summary_button, clear_analysis_button, question_input, ask_button],
        )
        deeper_answer_button.click(
            run_deeper_answer,
            inputs=[session_state, chatbot],
            outputs=[chatbot, session_state, chat_status, deeper_answer_button, deeper_answer_hint, uploaded_file, example_selector, url_value, analyze_button, reset_button, rerun_summary_button, clear_analysis_button, question_input, ask_button],
        )
        reset_button.click(
            reset_session,
            inputs=[session_state],
            outputs=[
                session_state,
                uploaded_file,
                example_selector,
                url_value,
                source_kind,
                view_source_button,
                analysis_output,
                chatbot,
                status_output,
                chat_status,
                analyze_button,
                reset_button,
                rerun_summary_button,
                clear_analysis_button,
                question_input,
                ask_button,
                deeper_answer_button,
                deeper_answer_hint,
            ],
        )

    demo.queue()
    return demo


if gr.NO_RELOAD:
    warm_embedding_stack()


# Expose the default `demo` symbol so Hugging Face Spaces can launch app.py.
demo = build_app()
app = demo


if __name__ == "__main__":
    app.launch(**APP_LAUNCH_KWARGS)
