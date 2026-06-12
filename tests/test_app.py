from app import _empty_session, _format_analysis, _format_chat_entry, _session_record, _view_source_button_update, ask_question, clear_analysis, rerun_summary, _stream_chat_entry
from services.rag_pipeline import AnalysisResult, AnswerResult, Citation, ImplementationItem


def test_format_chat_entry_wraps_supporting_snippets_in_details() -> None:
    result = _format_chat_entry(
        "Answer text",
        [{"ref_id": 1, "snippet": "Quoted <clause> & detail"}],
    )

    assert "Answer text" in result
    assert "<details><summary>Supporting snippet (1)</summary>" in result
    assert "[ref1]" in result
    assert "Quoted &lt;clause&gt; &amp; detail" in result


def test_stream_chat_entry_appends_collapsible_snippets_after_text_stream() -> None:
    frames = list(
        _stream_chat_entry(
            [{"role": "user", "content": "Question"}],
            "Answer text",
            [{"ref_id": 1, "snippet": "Snippet text"}],
        )
    )

    assert frames[-2][-1]["content"] == "_Based on the summary and analysis._\n\nAnswer text"
    assert "<details><summary>Supporting snippet (1)</summary>" in frames[-1][-1]["content"]


def test_format_analysis_wraps_tables_for_horizontal_scroll() -> None:
    result = _format_analysis(
        AnalysisResult(
            executive_summary="Summary",
            bill_summary=["Point"],
            implementation=[
                ImplementationItem(
                    stakeholder="Operators",
                    obligation="Register | comply",
                    implementation_burden="Costs",
                    risk_or_note="<review>",
                )
            ],
        )
    )

    assert result.count('class="analysis-table-scroll"') == 4
    assert "<table>" in result
    assert "Register | comply" in result
    assert "&lt;review&gt;" in result


def test_ask_question_uses_full_document_answers(monkeypatch) -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record.update(
        {
            "api_config": {"provider": "qwen", "api_key": "hf_test_token"},
            "analysis": {},
            "vector_store": object(),
            "doc_text": "Full bill text",
            "chat_history": [],
            "pending_deeper_question": "old question",
        }
    )

    monkeypatch.setattr("app.instantiate_client", lambda provider, api_key: object())

    calls: list[tuple[object | None, str | None]] = []

    def fake_answer_query_from_full_document(provider_client, vector_store, question, *, doc_text=None):
        calls.append((vector_store, doc_text))
        return AnswerResult(
            answer="Full-document answer",
            citations=[Citation(ref_id=1, snippet="Clause text")],
            provenance="full_document",
        )

    monkeypatch.setattr("app.answer_query_from_full_document", fake_answer_query_from_full_document)

    frames = list(ask_question("What does the bill require?", None, None, False, None, None, None, session_state, []))

    assert calls == [(record["vector_store"], "Full bill text")]
    assert "Reading the full document for an answer..." == frames[0][2]
    assert "<details><summary>Supporting snippet (1)</summary>" in frames[-1][0][-1]["content"]
    assert frames[-1][3]["visible"] is False
    assert frames[-1][4] == ""
    assert record["pending_deeper_question"] is None


def test_ask_question_bootstraps_source_document_without_analysis(monkeypatch) -> None:
    session_state = _empty_session()

    monkeypatch.setattr("app._ingest_sources", lambda uploaded_file, url_value: "Fresh source text")
    monkeypatch.setattr(
        "app.prepare_document_artifacts",
        lambda document_text: ("hash", ["chunk"], "vector-store"),
    )
    monkeypatch.setattr("app.instantiate_client", lambda provider, api_key: object())

    calls: list[tuple[object | None, str | None]] = []

    def fake_answer_query_from_full_document(provider_client, vector_store, question, *, doc_text=None):
        calls.append((vector_store, doc_text))
        return AnswerResult(
            answer="Direct source answer",
            citations=[Citation(ref_id=1, snippet="Source clause")],
            provenance="full_document",
        )

    monkeypatch.setattr("app.answer_query_from_full_document", fake_answer_query_from_full_document)

    frames = list(
        ask_question(
            "What does the source document say?",
            "/tmp/bill.pdf",
            "",
            False,
            None,
            "hf_test_token",
            None,
            session_state,
            [],
        )
    )

    record = _session_record(session_state)
    assert calls == [("vector-store", "Fresh source text")]
    assert record["doc_text"] == "Fresh source text"
    assert record["vector_store"] == "vector-store"
    assert record["api_config"]["provider"] == "qwen"
    assert "<details><summary>Supporting snippet (1)</summary>" in frames[-1][0][-1]["content"]


def test_rerun_summary_bypasses_precomputed_assets(monkeypatch) -> None:
    calls: list[bool] = []
    captured_args: list[tuple[object, ...]] = []

    def fake_analyze_document(*args, force_refresh=False):
        captured_args.append(args)
        calls.append(force_refresh)
        yield ("session", "status", "analysis", [], {"interactive": True}, {"interactive": True}, {"visible": False}, "")

    monkeypatch.setattr("app.analyze_document", fake_analyze_document)

    frames = list(rerun_summary(None, "https://example.com/bill.pdf", False, None, None, None, {"session_id": "abc"}))

    assert calls == [True]
    assert captured_args[0] == (None, "https://example.com/bill.pdf", False, None, None, None, {"session_id": "abc"}, None)
    assert frames[-1][1] == "status"


def test_rerun_summary_uses_cached_document_text(monkeypatch) -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record.update(
        {
            "doc_text": "Cached bill text",
            "analysis": {"executive_summary": "Old"},
            "api_config": {"provider": "qwen", "api_key": "hf_test_token"},
        }
    )

    calls: list[tuple[str, str | None]] = []

    def fake_rerun_record_analysis(state, current_record, *, provider, api_key):
        calls.append((provider, api_key))
        yield ("session", "status", "analysis", [], {"interactive": True}, {"interactive": True}, {"visible": False}, "")

    monkeypatch.setattr("app._rerun_record_analysis", fake_rerun_record_analysis)

    frames = list(rerun_summary(None, "https://example.com/bill.pdf", False, None, None, None, session_state))

    assert calls == [("qwen", "hf_test_token")]
    assert frames[-1][1] == "status"


def test_clear_analysis_resets_panel_and_disables_header_actions() -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record.update(
        {
            "analysis": {"executive_summary": "Loaded"},
            "chat_history": [{"role": "assistant", "content": "Answer"}],
            "pending_deeper_question": "Question",
        }
    )

    result = clear_analysis(session_state)

    assert result[2] == "Run an analysis to populate this section."
    assert result[4]["interactive"] is False
    assert result[5]["interactive"] is False
    assert result[6]["visible"] is False
    assert record["analysis"] is None
    assert record["chat_history"] == []
    assert record["pending_deeper_question"] is None


def test_view_source_button_update_tracks_url_presence() -> None:
    disabled = _view_source_button_update("")
    enabled = _view_source_button_update("https://example.com/bill.pdf")

    assert disabled["interactive"] is False
    assert enabled["interactive"] is True
    assert enabled["value"] == "View source ↗"
