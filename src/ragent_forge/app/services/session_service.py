from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ragent_forge.app.ports import SessionWorkspace
from ragent_forge.app.schema import add_schema_version, validate_schema_version
from ragent_forge.core.retrieval.types import RetrievalMode, normalize_retrieval_mode

TuiSessionMessageRole = Literal["user", "assistant"]
TuiSessionRetrievalMode = RetrievalMode
TuiSessionExportFormat = Literal["markdown", "json"]
TuiSessionListFilter = Literal["recent", "pinned", "starred", "failed", "has-sources"]

_UNTITLED_TITLE = "New chat"
_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "authorization",
    "secret",
    "token",
    "embedding",
    "embeddings",
    "vector",
)
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"\bapi_key\s*[:=]", re.IGNORECASE),
    re.compile(r"\bauthorization\s*:", re.IGNORECASE),
    re.compile(r"^\s*bearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsecret\s*[:=]", re.IGNORECASE),
    re.compile(r"\btoken\s*[:=]", re.IGNORECASE),
)


@dataclass(frozen=True)
class TuiSessionMessage:
    role: TuiSessionMessageRole
    text: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "text": _safe_text(self.text),
            "created_at": self.created_at,
            "metadata": _safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSessionMessage:
        return cls(
            role=_message_role(value.get("role")),
            text=str(value.get("text", "")),
            created_at=str(value.get("created_at", "")),
            metadata=_dict_value(value.get("metadata")),
        )


@dataclass(frozen=True)
class TuiSessionSource:
    rank: int
    chunk_id: str
    source_path: str
    score: float
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "score": self.score,
            "preview": _safe_text(self.preview),
            "metadata": _safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSessionSource:
        return cls(
            rank=_int_value(value.get("rank")),
            chunk_id=str(value.get("chunk_id", "")),
            source_path=str(value.get("source_path", "")),
            score=_float_value(value.get("score")),
            preview=str(value.get("preview", "")),
            metadata=_dict_value(value.get("metadata")),
        )


@dataclass(frozen=True)
class TuiSessionRun:
    retrieval_mode: TuiSessionRetrievalMode
    retrieval_method: str
    limit: int
    max_context_chars: int
    show_prompt: bool
    trace_id: str | None = None
    generation_status: str | None = None
    generation_provider: str | None = None
    error: str | None = None
    prompt_preview: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_mode": self.retrieval_mode,
            "retrieval_method": self.retrieval_method,
            "limit": self.limit,
            "max_context_chars": self.max_context_chars,
            "show_prompt": self.show_prompt,
            "trace_id": self.trace_id,
            "generation_status": self.generation_status,
            "generation_provider": self.generation_provider,
            "error": _safe_optional_text(self.error),
            "prompt_preview": _safe_optional_text(self.prompt_preview),
            "metadata": _safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSessionRun:
        return cls(
            retrieval_mode=_retrieval_mode(value.get("retrieval_mode")),
            retrieval_method=str(value.get("retrieval_method", "")),
            limit=_int_value(value.get("limit")),
            max_context_chars=_int_value(value.get("max_context_chars")),
            show_prompt=bool(value.get("show_prompt", False)),
            trace_id=_optional_string(value.get("trace_id")),
            generation_status=_optional_string(value.get("generation_status")),
            generation_provider=_optional_string(value.get("generation_provider")),
            error=_optional_string(value.get("error")),
            prompt_preview=_optional_string(value.get("prompt_preview")),
            metadata=_dict_value(value.get("metadata")),
        )


@dataclass(frozen=True)
class TuiSessionTurn:
    id: str
    created_at: str
    user_message: TuiSessionMessage
    assistant_message: TuiSessionMessage
    sources: tuple[TuiSessionSource, ...] = ()
    run: TuiSessionRun | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "user_message": self.user_message.to_dict(),
            "assistant_message": self.assistant_message.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "run": self.run.to_dict() if self.run is not None else None,
            "metadata": _safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSessionTurn:
        user_message = _dict_value(value.get("user_message"))
        assistant_message = _dict_value(value.get("assistant_message"))
        run = value.get("run")
        return cls(
            id=str(value.get("id", "")),
            created_at=str(value.get("created_at", "")),
            user_message=TuiSessionMessage.from_dict(user_message),
            assistant_message=TuiSessionMessage.from_dict(assistant_message),
            sources=tuple(
                TuiSessionSource.from_dict(source)
                for source in _list_of_dicts(value.get("sources"))
            ),
            run=TuiSessionRun.from_dict(run) if isinstance(run, dict) else None,
            metadata=_dict_value(value.get("metadata")),
        )


@dataclass(frozen=True)
class TuiSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    pinned: bool = False
    starred: bool = False
    turns: tuple[TuiSessionTurn, ...] = ()
    branched_from_session_id: str | None = None
    branched_from_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pinned": self.pinned,
            "starred": self.starred,
            "turns": [turn.to_dict() for turn in self.turns],
            "branched_from_session_id": self.branched_from_session_id,
            "branched_from_turn_id": self.branched_from_turn_id,
            "metadata": _safe_metadata(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSession:
        return cls(
            id=str(value.get("id", "")),
            title=str(value.get("title", _UNTITLED_TITLE)),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            pinned=bool(value.get("pinned", False)),
            starred=bool(value.get("starred", False)),
            turns=tuple(
                TuiSessionTurn.from_dict(turn)
                for turn in _list_of_dicts(value.get("turns"))
            ),
            branched_from_session_id=_optional_string(
                value.get("branched_from_session_id")
            ),
            branched_from_turn_id=_optional_string(value.get("branched_from_turn_id")),
            metadata=_dict_value(value.get("metadata")),
        )


@dataclass(frozen=True)
class TuiSessionSummary:
    id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    pinned: bool
    starred: bool
    path: str
    source_count: int = 0
    failed_turn_count: int = 0
    branched_from_session_id: str | None = None
    branched_from_turn_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "pinned": self.pinned,
            "starred": self.starred,
            "path": self.path,
            "source_count": self.source_count,
            "failed_turn_count": self.failed_turn_count,
            "branched_from_session_id": self.branched_from_session_id,
            "branched_from_turn_id": self.branched_from_turn_id,
        }

    @classmethod
    def from_session(cls, session: TuiSession, path: Path) -> TuiSessionSummary:
        source_count = sum(len(turn.sources) for turn in session.turns)
        failed_turn_count = sum(
            1
            for turn in session.turns
            if turn.run is not None and turn.run.generation_status == "failed"
        )
        return cls(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            turn_count=len(session.turns),
            pinned=session.pinned,
            starred=session.starred,
            path=str(path),
            source_count=source_count,
            failed_turn_count=failed_turn_count,
            branched_from_session_id=session.branched_from_session_id,
            branched_from_turn_id=session.branched_from_turn_id,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TuiSessionSummary:
        return cls(
            id=str(value.get("id", "")),
            title=str(value.get("title", _UNTITLED_TITLE)),
            created_at=str(value.get("created_at", "")),
            updated_at=str(value.get("updated_at", "")),
            turn_count=_int_value(value.get("turn_count")),
            pinned=bool(value.get("pinned", False)),
            starred=bool(value.get("starred", False)),
            path=str(value.get("path", "")),
            source_count=_int_value(value.get("source_count")),
            failed_turn_count=_int_value(value.get("failed_turn_count")),
            branched_from_session_id=_optional_string(
                value.get("branched_from_session_id")
            ),
            branched_from_turn_id=_optional_string(value.get("branched_from_turn_id")),
        )


class SessionService:
    def __init__(self, workspace: SessionWorkspace) -> None:
        self.workspace = workspace
        self.sessions_dir = self.workspace.sessions_dir
        self.exports_dir = self.workspace.session_exports_dir
        self.index_path = self.workspace.session_index_path
        self.latest_path = self.workspace.latest_session_path

    def create_session(self, title: str | None = None) -> TuiSession:
        now = _utc_now()
        session = TuiSession(
            id=_new_id("session"),
            title=_clean_title(title) or _UNTITLED_TITLE,
            created_at=now,
            updated_at=now,
        )
        self.save_session(session)
        self.set_latest(session.id)
        return session

    def save_session(self, session: TuiSession) -> TuiSession:
        self.workspace.ensure_exists()
        sanitized = TuiSession.from_dict(session.to_dict())
        with self.workspace.write_lock():
            self.workspace.atomic_write_text(
                self._session_path(sanitized.id),
                json.dumps(
                    add_schema_version(sanitized.to_dict()),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        self._write_index()
        return sanitized

    def load_session(self, session_id: str) -> TuiSession:
        path = self._session_path(session_id)
        if not path.is_file():
            raise ValueError(f"Session not found: {session_id}")
        return self._read_session_file(path)

    def load_latest_or_create(self) -> TuiSession:
        latest_id = self._latest_session_id()
        if latest_id is not None:
            try:
                return self.load_session(latest_id)
            except ValueError:
                pass

        sessions = self._load_all_sessions()
        if sessions:
            latest = self._sort_sessions(sessions)[0]
            self.set_latest(latest.id)
            return latest
        return self.create_session()

    def set_latest(self, session_id: str) -> None:
        self.workspace.ensure_exists()
        self.workspace.atomic_write_text(
            self.latest_path,
            json.dumps(
                add_schema_version({"session_id": session_id}),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def list_sessions(
        self,
        filter_by: TuiSessionListFilter = "recent",
    ) -> list[TuiSessionSummary]:
        sessions = self._filter_sessions(
            self._sort_sessions(self._load_all_sessions()),
            filter_by,
        )
        summaries = [
            TuiSessionSummary.from_session(session, self._session_path(session.id))
            for session in sessions
        ]
        self._write_index_from_summaries(summaries)
        return summaries

    def search_sessions(
        self,
        query: str,
        filter_by: TuiSessionListFilter = "recent",
    ) -> list[TuiSessionSummary]:
        normalized = query.strip().lower()
        if not normalized:
            return self.list_sessions(filter_by)
        return [
            summary
            for summary in self.list_sessions(filter_by)
            if normalized in self._session_search_text(summary.id).lower()
        ]

    def rename_session(self, session_id: str, title: str) -> TuiSession:
        clean_title = _clean_title(title)
        if not clean_title:
            raise ValueError("Session title cannot be empty.")
        return self._save_updated_session(
            replace(self.load_session(session_id), title=clean_title)
        )

    def delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
        sessions = self._sort_sessions(self._load_all_sessions())
        if sessions:
            self.set_latest(sessions[0].id)
        elif self.latest_path.exists():
            self.latest_path.unlink()
        self._write_index()

    def set_pinned(self, session_id: str, pinned: bool) -> TuiSession:
        return self._save_updated_session(
            replace(self.load_session(session_id), pinned=pinned)
        )

    def set_starred(self, session_id: str, starred: bool) -> TuiSession:
        return self._save_updated_session(
            replace(self.load_session(session_id), starred=starred)
        )

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        assistant_text: str,
        sources: list[TuiSessionSource] | tuple[TuiSessionSource, ...],
        run: TuiSessionRun,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TuiSession, TuiSessionTurn]:
        session = self.load_session(session_id)
        now = _utc_now()
        turn = TuiSessionTurn(
            id=_new_id("turn"),
            created_at=now,
            user_message=TuiSessionMessage(
                role="user",
                text=question,
                created_at=now,
            ),
            assistant_message=TuiSessionMessage(
                role="assistant",
                text=assistant_text,
                created_at=now,
            ),
            sources=tuple(sources),
            run=run,
            metadata=metadata or {},
        )
        title = (
            _default_title_from_question(question)
            if session.title == _UNTITLED_TITLE and not session.turns
            else session.title
        )
        updated = replace(
            session,
            title=title,
            updated_at=now,
            turns=(*session.turns, turn),
        )
        saved = self.save_session(updated)
        self.set_latest(saved.id)
        return saved, saved.turns[-1]

    def branch_session(self, session_id: str, turn_id: str | None = None) -> TuiSession:
        source = self.load_session(session_id)
        if not source.turns:
            raise ValueError("Cannot branch an empty session.")
        selected_turn_id = turn_id or source.turns[-1].id
        through_index = _turn_index(source, selected_turn_id)
        if through_index is None:
            raise ValueError(f"Turn not found: {selected_turn_id}")

        now = _utc_now()
        branch = TuiSession(
            id=_new_id("session"),
            title=f"{source.title} (branch)",
            created_at=now,
            updated_at=now,
            turns=source.turns[: through_index + 1],
            branched_from_session_id=source.id,
            branched_from_turn_id=selected_turn_id,
            metadata={"branch_source_title": source.title},
        )
        saved = self.save_session(branch)
        self.set_latest(saved.id)
        return saved

    def export_session(
        self,
        session_id: str,
        export_format: TuiSessionExportFormat,
    ) -> Path:
        session = self.load_session(session_id)
        self.workspace.ensure_exists()
        if export_format == "markdown":
            path = self.exports_dir / f"{session.id}.md"
            self.workspace.atomic_write_text(path, _session_to_markdown(session))
            return path
        if export_format == "json":
            path = self.exports_dir / f"{session.id}.json"
            self.workspace.atomic_write_text(
                path,
                json.dumps(
                    add_schema_version(session.to_dict()),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            return path
        raise ValueError("export_format must be markdown or json")

    def _save_updated_session(self, session: TuiSession) -> TuiSession:
        updated = replace(session, updated_at=_utc_now())
        saved = self.save_session(updated)
        self.set_latest(saved.id)
        return saved

    def _session_path(self, session_id: str) -> Path:
        safe_id = session_id.replace("/", "-").replace("\\", "-")
        return self.sessions_dir / f"{safe_id}.json"

    def _latest_session_id(self) -> str | None:
        if not self.latest_path.is_file():
            return None
        try:
            payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        try:
            validate_schema_version(payload, "latest session")
        except ValueError:
            return None
        session_id = payload.get("session_id")
        return session_id if isinstance(session_id, str) and session_id else None

    def _read_session_file(self, path: Path) -> TuiSession:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid session file {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid session file {path}: expected object")
        validate_schema_version(payload, "session file")
        return TuiSession.from_dict({str(key): value for key, value in payload.items()})

    def _load_all_sessions(self) -> list[TuiSession]:
        if not self.sessions_dir.is_dir():
            return []
        sessions: list[TuiSession] = []
        for path in sorted(self.sessions_dir.glob("session-*.json")):
            try:
                sessions.append(self._read_session_file(path))
            except ValueError:
                continue
        return sessions

    def _sort_sessions(self, sessions: list[TuiSession]) -> list[TuiSession]:
        sorted_sessions = sorted(sessions, key=lambda session: session.title.lower())
        sorted_sessions = sorted(
            sorted_sessions,
            key=lambda session: session.updated_at,
            reverse=True,
        )
        sorted_sessions = sorted(
            sorted_sessions,
            key=lambda session: not session.starred,
        )
        return sorted(
            sorted_sessions,
            key=lambda session: not session.pinned,
        )

    def _filter_sessions(
        self,
        sessions: list[TuiSession],
        filter_by: TuiSessionListFilter,
    ) -> list[TuiSession]:
        if filter_by == "recent":
            return sessions
        if filter_by == "pinned":
            return [session for session in sessions if session.pinned]
        if filter_by == "starred":
            return [session for session in sessions if session.starred]
        if filter_by == "failed":
            return [
                session
                for session in sessions
                if any(
                    turn.run is not None and turn.run.generation_status == "failed"
                    for turn in session.turns
                )
            ]
        if filter_by == "has-sources":
            return [
                session
                for session in sessions
                if any(turn.sources for turn in session.turns)
            ]
        raise ValueError(f"Invalid session filter: {filter_by}")

    def _write_index(self) -> None:
        summaries = [
            TuiSessionSummary.from_session(session, self._session_path(session.id))
            for session in self._sort_sessions(self._load_all_sessions())
        ]
        self._write_index_from_summaries(summaries)

    def _write_index_from_summaries(
        self,
        summaries: list[TuiSessionSummary],
    ) -> None:
        self.workspace.ensure_exists()
        self.workspace.atomic_write_text(
            self.index_path,
            json.dumps(
                add_schema_version(
                    {"sessions": [summary.to_dict() for summary in summaries]}
                ),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )

    def _session_search_text(self, session_id: str) -> str:
        try:
            session = self.load_session(session_id)
        except ValueError:
            return ""
        parts = [session.title, session.id]
        for turn in session.turns:
            parts.extend(
                [
                    turn.user_message.text,
                    turn.assistant_message.text,
                    *(source.source_path for source in turn.sources),
                ]
            )
        return "\n".join(parts)


def _turn_index(session: TuiSession, turn_id: str) -> int | None:
    for index, turn in enumerate(session.turns):
        if turn.id == turn_id:
            return index
    return None


def _session_to_markdown(session: TuiSession) -> str:
    lines = [
        f"# {_safe_text(session.title)}",
        "",
        f"- session_id: {session.id}",
        f"- created_at: {session.created_at}",
        f"- updated_at: {session.updated_at}",
        f"- pinned: {str(session.pinned).lower()}",
        f"- starred: {str(session.starred).lower()}",
        "",
    ]
    if session.branched_from_session_id:
        lines.extend(
            [
                "## Branch",
                "",
                f"- from_session: {session.branched_from_session_id}",
                f"- from_turn: {session.branched_from_turn_id or ''}",
                "",
            ]
        )

    for index, turn in enumerate(session.turns, start=1):
        lines.extend(
            [
                f"## Turn {index}",
                "",
                f"- turn_id: {turn.id}",
                "",
                "### User",
                "",
                _safe_text(turn.user_message.text),
                "",
                "### Assistant",
                "",
                _safe_text(turn.assistant_message.text),
                "",
            ]
        )
        if turn.run is not None:
            lines.extend(
                [
                    "### Run",
                    "",
                    f"- mode: {turn.run.retrieval_mode}",
                    f"- method: {turn.run.retrieval_method}",
                    f"- limit: {turn.run.limit}",
                    f"- context: {turn.run.max_context_chars}",
                    f"- prompt: {str(turn.run.show_prompt).lower()}",
                    f"- generation_status: {turn.run.generation_status or ''}",
                    f"- generation_provider: {turn.run.generation_provider or ''}",
                    "",
                ]
            )
        if turn.sources:
            lines.extend(["### Sources", ""])
            for source in turn.sources:
                lines.extend(
                    [
                        f"{source.rank}. {source.source_path}",
                        f"   chunk: {source.chunk_id}",
                        f"   score: {source.score:.4g}",
                    ]
                )
            lines.append("")
    return "\n".join(lines)


def _safe_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return _safe_text(value)


def _safe_text(text: str) -> str:
    safe_lines: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _SENSITIVE_TEXT_PATTERNS):
            safe_lines.append("<hidden>")
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines)


def _safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if any(fragment in key_text.lower() for fragment in _SENSITIVE_KEY_FRAGMENTS):
            continue
        safe[key_text] = _safe_value(value)
    return safe


def _safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _safe_metadata({str(key): item for key, item in value.items()})
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_value(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    return str(value)


def _message_role(value: object) -> TuiSessionMessageRole:
    if value == "assistant":
        return "assistant"
    return "user"


def _retrieval_mode(value: object) -> TuiSessionRetrievalMode:
    if value is None:
        return "lexical"
    if not isinstance(value, str):
        raise ValueError("Invalid retrieval mode in session: expected string")
    return normalize_retrieval_mode(value)


def _dict_value(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        {str(key): item for key, item in item.items()}
        for item in value
        if isinstance(item, dict)
    ]


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _float_value(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _clean_title(title: str | None) -> str | None:
    if title is None:
        return None
    clean = " ".join(title.split())
    if not clean:
        return None
    return clean[:80]


def _default_title_from_question(question: str) -> str:
    return _clean_title(question) or _UNTITLED_TITLE


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _new_id(prefix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"
