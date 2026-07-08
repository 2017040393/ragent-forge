import json
from pathlib import Path

from ragent_forge.app.services.session_service import (
    SessionService,
    TuiSessionRun,
    TuiSessionSource,
)


def make_source(rank: int = 1) -> TuiSessionSource:
    return TuiSessionSource(
        rank=rank,
        chunk_id=f"doc::chunk-{rank:04d}",
        source_path=f"/knowledge/source_{rank}.md",
        score=0.1 * rank,
        preview=f"Preview {rank}",
        metadata={"retrieval_method": "hybrid_rrf", "embedding": [1.0, 2.0]},
    )


def make_run() -> TuiSessionRun:
    return TuiSessionRun(
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        limit=5,
        max_context_chars=4000,
        show_prompt=False,
        generation_status="success",
        generation_provider="openai_responses",
        error=None,
    )


def test_session_service_creates_session_index_and_latest(tmp_path: Path) -> None:
    service = SessionService(tmp_path / ".ragent")

    session = service.create_session("Research chat")

    assert session.title == "Research chat"
    assert service.load_session(session.id).id == session.id
    assert service.load_latest_or_create().id == session.id
    assert (tmp_path / ".ragent" / "sessions" / "index.json").is_file()
    assert (tmp_path / ".ragent" / "sessions" / "latest.json").is_file()
    assert service.list_sessions()[0].id == session.id


def test_session_service_appends_turn_and_restores_messages(tmp_path: Path) -> None:
    service = SessionService(tmp_path / ".ragent")
    session = service.create_session()

    updated, turn = service.append_turn(
        session.id,
        question="What is Agentic RAG?",
        assistant_text="Agentic RAG plans before retrieving.",
        sources=[make_source()],
        run=make_run(),
    )
    loaded = service.load_session(updated.id)

    assert updated.title == "What is Agentic RAG?"
    assert loaded.turns[0].id == turn.id
    assert loaded.turns[0].user_message.text == "What is Agentic RAG?"
    assert loaded.turns[0].assistant_message.text == (
        "Agentic RAG plans before retrieving."
    )
    assert loaded.turns[0].sources[0].source_path == "/knowledge/source_1.md"
    assert loaded.turns[0].run is not None
    assert loaded.turns[0].run.retrieval_mode == "hybrid"
    assert "embedding" not in loaded.turns[0].sources[0].metadata


def test_session_service_searches_title_messages_and_sources(
    tmp_path: Path,
) -> None:
    service = SessionService(tmp_path / ".ragent")
    session = service.create_session("Planning notes")
    service.append_turn(
        session.id,
        question="How does retrieval work?",
        assistant_text="It uses evidence spans.",
        sources=[make_source()],
        run=make_run(),
    )

    assert [item.id for item in service.search_sessions("planning")] == [session.id]
    assert [item.id for item in service.search_sessions("evidence spans")] == [
        session.id
    ]
    assert [item.id for item in service.search_sessions("source_1.md")] == [
        session.id
    ]
    assert service.search_sessions("missing") == []


def test_session_service_pin_star_rename_delete_and_latest_update(
    tmp_path: Path,
) -> None:
    service = SessionService(tmp_path / ".ragent")
    first = service.create_session("First")
    second = service.create_session("Second")

    renamed = service.rename_session(second.id, "Better title")
    pinned = service.set_pinned(renamed.id, True)
    starred = service.set_starred(pinned.id, True)

    assert starred.title == "Better title"
    assert starred.pinned is True
    assert starred.starred is True
    summaries = service.list_sessions()
    assert summaries[0].id == starred.id
    assert summaries[0].pinned is True
    assert summaries[0].starred is True

    service.delete_session(starred.id)

    assert service.load_latest_or_create().id == first.id
    assert [item.id for item in service.list_sessions()] == [first.id]


def test_session_service_exports_markdown_and_json_without_secrets(
    tmp_path: Path,
) -> None:
    service = SessionService(tmp_path / ".ragent")
    session = service.create_session("Export me")
    updated, _ = service.append_turn(
        session.id,
        question="Question?",
        assistant_text="Answer.",
        sources=[
            TuiSessionSource(
                rank=1,
                chunk_id="chunk-1",
                source_path="/knowledge/doc.md",
                score=1.0,
                preview="Preview",
                metadata={"api_key": "secret", "retrieval_method": "bm25"},
            )
        ],
        run=TuiSessionRun(
            retrieval_mode="bm25",
            retrieval_method="bm25",
            limit=3,
            max_context_chars=1000,
            show_prompt=True,
            generation_status="success",
            generation_provider="null",
            prompt_preview="api_key=secret",
        ),
    )

    markdown_path = service.export_session(updated.id, "markdown")
    json_path = service.export_session(updated.id, "json")

    markdown = markdown_path.read_text(encoding="utf-8")
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    serialized = json.dumps(exported)
    assert "# Export me" in markdown
    assert "Question?" in markdown
    assert "Answer." in markdown
    assert "api_key" not in markdown
    assert "secret" not in markdown
    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_session_service_branches_through_selected_turn(tmp_path: Path) -> None:
    service = SessionService(tmp_path / ".ragent")
    session = service.create_session("Original")
    first_session, first_turn = service.append_turn(
        session.id,
        question="First?",
        assistant_text="First answer.",
        sources=[make_source(1)],
        run=make_run(),
    )
    service.append_turn(
        first_session.id,
        question="Second?",
        assistant_text="Second answer.",
        sources=[make_source(2)],
        run=make_run(),
    )

    branch = service.branch_session(first_session.id, first_turn.id)

    assert branch.branched_from_session_id == first_session.id
    assert branch.branched_from_turn_id == first_turn.id
    assert len(branch.turns) == 1
    assert branch.turns[0].id == first_turn.id
    assert service.load_latest_or_create().id == branch.id
