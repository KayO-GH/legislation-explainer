from app import _displayed_chat_history, _empty_session, _format_analysis, _format_chat_entry, _handle_example_source_change, _handle_url_source_change, _session_record, _stage_question, _view_source_button_update, ask_question, clear_analysis, rerun_summary, _stream_chat_entry
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


def test_stage_question_clears_visible_input() -> None:
    session_state = _empty_session()

    staged = _stage_question("Queued follow-up", session_state, [])

    assert staged[0] == "Queued follow-up"
    assert staged[1] == ""


def test_stage_question_enqueues_immediately_while_busy() -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record["message_queue"] = [{"id": "active", "question": "Current question", "status": "answering"}]
    record["active_message_id"] = "active"
    record["is_answering"] = True

    staged = _stage_question("Queued follow-up", session_state, [])

    assert staged[0] == ""
    assert staged[1] == ""
    assert staged[3] == "Queued:\n1. Current question\n2. Queued follow-up"
    assert record["message_queue"][1]["question"] == "Queued follow-up"


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
    assert "Queued:\n1. What does the bill require?" == frames[0][2]
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


def test_ask_question_locks_source_controls_but_keeps_queue_input_enabled(monkeypatch) -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record.update(
        {
            "api_config": {"provider": "qwen", "api_key": "hf_test_token"},
            "vector_store": object(),
            "doc_text": "Full bill text",
            "chat_history": [],
        }
    )

    monkeypatch.setattr("app.instantiate_client", lambda provider, api_key: object())
    monkeypatch.setattr(
        "app.answer_query_from_full_document",
        lambda provider_client, vector_store, question, *, doc_text=None: AnswerResult(
            answer="Full-document answer",
            citations=[Citation(ref_id=1, snippet="Clause text")],
            provenance="full_document",
        ),
    )

    frames = list(ask_question("What does the bill require?", None, None, False, None, None, None, session_state, []))

    answering_frame = frames[0]
    assert answering_frame[5]["interactive"] is False
    assert answering_frame[6]["interactive"] is False
    assert answering_frame[7]["interactive"] is False
    assert answering_frame[8]["interactive"] is False
    assert answering_frame[9]["interactive"] is False
    assert answering_frame[10]["interactive"] is False
    assert answering_frame[11]["interactive"] is False
    assert answering_frame[12]["interactive"] is True
    assert answering_frame[13]["interactive"] is True

    final_frame = frames[-1]
    assert final_frame[5]["interactive"] is True
    assert final_frame[6]["interactive"] is True
    assert final_frame[7]["interactive"] is True
    assert final_frame[8]["interactive"] is True
    assert final_frame[9]["interactive"] is True
    assert final_frame[12]["interactive"] is True
    assert final_frame[13]["interactive"] is True


def test_ask_question_clears_input_when_queued_behind_active_answer(monkeypatch) -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record.update(
        {
            "api_config": {"provider": "qwen", "api_key": "hf_test_token"},
            "vector_store": object(),
            "doc_text": "Full bill text",
            "chat_history": [],
            "message_queue": [{"id": "active", "question": "Current question", "status": "answering"}],
            "is_answering": True,
            "active_message_id": "active",
        }
    )

    frames = list(ask_question("Queued follow-up", None, None, False, None, None, None, session_state, []))

    assert len(frames) == 1
    assert frames[0][2] == "Queued:\n1. Current question\n2. Queued follow-up"
    assert frames[0][12]["interactive"] is True


def test_url_source_change_flushes_queued_questions() -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record["chat_history"] = [{"role": "assistant", "content": "Earlier answer"}]
    record["message_queue"] = [{"id": "one", "question": "Queued question", "status": "queued"}]
    record["active_message_id"] = "one"
    record["is_answering"] = True
    record["pending_deeper_question"] = "Need deeper answer"

    result = _handle_url_source_change("https://example.com/next-bill.pdf", session_state)

    assert result[5] == "Queued questions were cleared because the source changed."
    assert result[4] == []
    assert record["message_queue"] == []
    assert record["active_message_id"] is None
    assert record["is_answering"] is False
    assert record["pending_deeper_question"] is None
    assert record["chat_history"] == []


def test_example_source_change_clears_existing_chat_history() -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record["chat_history"] = [{"role": "assistant", "content": "Earlier answer"}]

    result = _handle_example_source_change("National Information Technology Authority Bill, 2025", session_state)

    assert result[4] == []
    assert record["chat_history"] == []


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
    assert result[9]["interactive"] is False
    assert result[10]["interactive"] is False
    assert result[13]["visible"] is False
    assert record["analysis"] is None
    assert record["chat_history"] == []
    assert record["pending_deeper_question"] is None


def test_displayed_chat_history_shows_queue_placeholders() -> None:
    session_state = _empty_session()
    record = _session_record(session_state)
    record["chat_history"] = [{"role": "assistant", "content": "Earlier answer"}]
    record["message_queue"] = [
        {"id": "one", "question": "First queued question", "status": "answering"},
        {"id": "two", "question": "Second queued question", "status": "queued"},
    ]
    record["active_message_id"] = "one"

    displayed = _displayed_chat_history(record)

    assert displayed[0]["content"] == "Earlier answer"
    assert displayed[1]["role"] == "user"
    assert displayed[1]["content"] == "First queued question"
    assert displayed[2]["content"] == "_Answering..._"
    assert displayed[3]["content"] == "Second queued question"
    assert displayed[4]["content"] == "_Queued._"


def test_view_source_button_update_tracks_url_presence() -> None:
    disabled = _view_source_button_update("")
    enabled = _view_source_button_update("https://example.com/bill.pdf")

    assert disabled["interactive"] is False
    assert enabled["interactive"] is True
    assert enabled["value"] == "View source ↗"
