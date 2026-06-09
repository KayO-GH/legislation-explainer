from app import _format_chat_entry, _stream_chat_entry


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
