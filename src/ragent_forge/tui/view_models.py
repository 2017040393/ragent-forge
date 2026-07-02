from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ragent_forge.app.models import WorkspaceStatus
from ragent_forge.app.services.chunk_service import ChunkService, make_preview
from ragent_forge.app.services.config_service import ConfigService
from ragent_forge.app.services.embedding_service import EmbeddingService
from ragent_forge.app.services.hybrid_search_service import HybridSearchService
from ragent_forge.app.services.search_service import LexicalSearchService, SearchResult
from ragent_forge.app.services.semantic_search_service import SemanticSearchService
from ragent_forge.app.services.trace_history_service import TraceHistoryService
from ragent_forge.app.workspace import LocalWorkspace

PageName = Literal["documents", "search", "trace", "settings", "ask", "eval"]
RetrievalMode = Literal["lexical", "semantic", "hybrid"]

NO_CHUNKS_MESSAGE = "No chunks found. Run ragent ingest <path> first."
VECTOR_INDEX_MISSING_MESSAGE = "Vector index not found. Run ragent index build first."


@dataclass(frozen=True)
class ChunkRow:
    index: int
    chunk_id: str
    document_id: str
    source_path: str
    source_label: str
    range_text: str
    preview: str


@dataclass(frozen=True)
class DocumentsPageModel:
    workspace_path: str
    status_text: str
    last_ingest_source: str = ""
    document_count: int = 0
    chunk_count: int = 0
    skipped_count: int = 0
    chunks_path: str = ""
    summary_path: str = ""
    vector_index_status: str = "missing"
    vector_index_path: str | None = None
    recent_chunks: list[ChunkRow] = field(default_factory=list)
    message: str | None = None
    selected_chunk: ChunkRow | None = None


@dataclass(frozen=True)
class SearchPageState:
    query: str = ""
    retrieval_mode: RetrievalMode = "lexical"
    limit: int = 5
    results: list[SearchResult] = field(default_factory=list)
    error: str | None = None
    selected_result: SearchResult | None = None


@dataclass(frozen=True)
class TracePageModel:
    latest_trace: dict[str, Any] | None = None
    recent_traces: list[dict[str, Any]] = field(default_factory=list)
    selected_trace: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    message: str | None = None


@dataclass(frozen=True)
class SettingsPageModel:
    config_path: str
    config_exists: bool
    generation_provider: str
    generation_model: str
    generation_base_url: str
    generation_api_key: str
    embedding_provider: str
    embedding_model: str
    embedding_base_url: str
    embedding_api_key: str
    vector_index_status: str
    vector_index_path: str | None = None
    message: str | None = None


def compact_source_label(source_path: str) -> str:
    name = Path(source_path).name
    return name or source_path


def compact_chunk_label(chunk_id: str) -> str:
    if "::" in chunk_id:
        return chunk_id.rsplit("::", 1)[-1]
    return chunk_id


def format_range(start_char: Any, end_char: Any) -> str:
    if start_char is None and end_char is None:
        return ""
    return f"{start_char}-{end_char}"


def load_documents_page_model(
    workspace_path: str | Path = ".ragent",
    *,
    limit: int = 10,
) -> DocumentsPageModel:
    workspace = LocalWorkspace(workspace_path)
    try:
        status = workspace.status()
    except (OSError, ValueError) as exc:
        return DocumentsPageModel(
            workspace_path=str(workspace.root_path),
            status_text="error",
            chunks_path=str(workspace.chunks_path),
            summary_path=str(workspace.latest_summary_path),
            message=f"Workspace status error: {exc}",
        )

    if status.status != "ready":
        message = (
            NO_CHUNKS_MESSAGE
            if status.status == "not_initialized"
            else "Workspace is incomplete. Run ragent ingest <path> first."
        )
        return _documents_model_from_status(status, workspace, message=message)

    try:
        chunks = ChunkService(workspace).list_chunks(limit)
    except (OSError, ValueError) as exc:
        return _documents_model_from_status(
            status,
            workspace,
            message=f"Chunk status error: {exc}",
        )

    rows = [_chunk_row(index, chunk) for index, chunk in enumerate(chunks, start=1)]
    message = None if rows else NO_CHUNKS_MESSAGE
    return _documents_model_from_status(
        status,
        workspace,
        message=message,
        recent_chunks=rows,
        selected_chunk=rows[0] if rows else None,
    )


def _documents_model_from_status(
    status: WorkspaceStatus,
    workspace: LocalWorkspace,
    *,
    message: str | None = None,
    recent_chunks: list[ChunkRow] | None = None,
    selected_chunk: ChunkRow | None = None,
) -> DocumentsPageModel:
    summary = status.summary
    chunk_count = (
        status.chunk_count_from_file
        if status.chunk_count_from_file is not None
        else int(summary.get("chunk_count", 0) or 0)
    )
    has_index = workspace.has_vector_index()
    return DocumentsPageModel(
        workspace_path=status.root_path,
        status_text=status.status,
        last_ingest_source=str(summary.get("source_path", "")),
        document_count=int(summary.get("document_count", 0) or 0),
        chunk_count=chunk_count,
        skipped_count=int(summary.get("skipped_count", 0) or 0),
        chunks_path=status.chunks_path,
        summary_path=status.latest_summary_path,
        vector_index_status="ready" if has_index else "missing",
        vector_index_path=str(workspace.vector_index_path) if has_index else None,
        recent_chunks=recent_chunks or [],
        message=message,
        selected_chunk=selected_chunk,
    )


def _chunk_row(index: int, chunk: dict[str, Any]) -> ChunkRow:
    source_path = str(chunk.get("source_path", ""))
    return ChunkRow(
        index=index,
        chunk_id=str(chunk.get("chunk_id", "")),
        document_id=str(chunk.get("document_id", "")),
        source_path=source_path,
        source_label=compact_source_label(source_path),
        range_text=format_range(chunk.get("start_char"), chunk.get("end_char")),
        preview=make_preview(str(chunk.get("text", "")), max_length=72),
    )


def format_documents_page(model: DocumentsPageModel) -> str:
    lines = [
        "Documents",
        "",
        f"Workspace: {model.workspace_path}",
        f"Status: {model.status_text}",
    ]

    if model.message:
        lines.extend(["", model.message])

    lines.extend(
        [
            f"Last ingest source: {model.last_ingest_source}",
            f"Documents: {model.document_count}",
            f"Chunks: {model.chunk_count}",
            f"Skipped files: {model.skipped_count}",
            f"Chunks file: {model.chunks_path}",
            f"Summary file: {model.summary_path}",
            f"Vector index: {model.vector_index_status}",
        ]
    )
    if model.vector_index_path:
        lines.append(f"Index path: {model.vector_index_path}")

    lines.extend(["", "Recent chunks", "# | Source | Range | Preview"])
    if model.recent_chunks:
        lines.extend(
            (
                f"{row.index} | {row.source_label} | "
                f"{row.range_text} | {row.preview}"
            )
            for row in model.recent_chunks
        )
    else:
        lines.append("No recent chunks.")
    return "\n".join(lines)


def format_chunk_inspector(chunk: ChunkRow | None) -> str:
    if chunk is None:
        return "Inspector\n\nSelect a chunk, search result, trace, or setting."
    return "\n".join(
        [
            "Chunk details",
            "",
            f"chunk_id: {chunk.chunk_id}",
            f"source_path: {chunk.source_path}",
            f"document_id: {chunk.document_id}",
            f"range: {chunk.range_text}",
            f"preview: {chunk.preview}",
        ]
    )


def run_tui_search(
    workspace_path: str | Path,
    query: str,
    retrieval_mode: str,
    limit: int,
) -> SearchPageState:
    mode = _normalize_retrieval_mode(retrieval_mode)
    safe_limit = max(limit, 0)
    normalized_query = query.strip()
    if not normalized_query:
        return SearchPageState(
            query=query,
            retrieval_mode=mode,
            limit=safe_limit,
            error="Enter a search query.",
        )

    workspace = LocalWorkspace(workspace_path)
    if not workspace.has_chunks():
        return SearchPageState(
            query=query,
            retrieval_mode=mode,
            limit=safe_limit,
            error=NO_CHUNKS_MESSAGE,
        )

    if mode in {"semantic", "hybrid"} and not workspace.has_vector_index():
        return SearchPageState(
            query=query,
            retrieval_mode=mode,
            limit=safe_limit,
            error=VECTOR_INDEX_MISSING_MESSAGE,
        )

    try:
        if mode == "lexical":
            search_service = LexicalSearchService(workspace)
            retrieval_method = "lexical_token_overlap"
        elif mode == "semantic":
            config = ConfigService(workspace).load()
            search_service = SemanticSearchService(
                workspace,
                EmbeddingService.from_config(config),
            )
            retrieval_method = "semantic_cosine_similarity"
        else:
            config = ConfigService(workspace).load()
            semantic_search_service = SemanticSearchService(
                workspace,
                EmbeddingService.from_config(config),
            )
            search_service = HybridSearchService(
                lexical_search_service=LexicalSearchService(workspace),
                semantic_search_service=semantic_search_service,
            )
            retrieval_method = "hybrid_rrf"
        results = [
            _with_default_retrieval_method(result, retrieval_method)
            for result in search_service.search(normalized_query, safe_limit)
        ]
    except (OSError, RuntimeError, ValueError):
        return SearchPageState(
            query=query,
            retrieval_mode=mode,
            limit=safe_limit,
            error="Search failed. Check configuration and workspace files.",
        )

    return SearchPageState(
        query=query,
        retrieval_mode=mode,
        limit=safe_limit,
        results=results,
        selected_result=results[0] if results else None,
    )


def _normalize_retrieval_mode(retrieval_mode: str) -> RetrievalMode:
    if retrieval_mode in {"semantic", "hybrid"}:
        return retrieval_mode
    return "lexical"


def _with_default_retrieval_method(
    result: SearchResult,
    retrieval_method: str,
) -> SearchResult:
    metadata = dict(result.metadata)
    metadata.setdefault("retrieval_method", retrieval_method)
    return result.model_copy(update={"metadata": metadata})


def format_search_page(state: SearchPageState) -> str:
    lines = [
        "Search",
        "",
        f"Query: {state.query}",
        f"Mode: {state.retrieval_mode}",
        f"Limit: {state.limit}",
    ]
    if state.error:
        lines.extend(["", state.error])

    lines.extend(["", "Results", "# | Score | Source | Chunk"])
    if state.results:
        for index, result in enumerate(state.results, start=1):
            lines.append(
                f"{index} | {result.score:.4g} | "
                f"{compact_source_label(result.source_path)} | "
                f"{compact_chunk_label(result.chunk_id)}"
            )
    elif not state.error:
        lines.append("No matches found.")
    return "\n".join(lines)


def format_search_result_inspector(
    result: SearchResult | None,
    retrieval_mode: str,
) -> str:
    if result is None:
        return "Inspector\n\nRun a search and select a result."

    metadata = result.metadata
    lines = [
        "Search result details",
        "",
        f"chunk_id: {result.chunk_id}",
        f"source_path: {result.source_path}",
        f"score: {result.score:.4g}",
        f"retrieval_method: {metadata.get('retrieval_method', retrieval_mode)}",
    ]
    for key in (
        "fusion_method",
        "matched_modes",
        "lexical_rank",
        "semantic_rank",
    ):
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        lines.append(f"{key}: {value}")
    lines.append(f"preview: {make_preview(result.text, max_length=120)}")
    return "\n".join(lines)


def load_trace_page_model(workspace_path: str | Path = ".ragent") -> TracePageModel:
    workspace = LocalWorkspace(workspace_path)
    service = TraceHistoryService(workspace)
    history = service.list_traces(limit=8)
    traces: list[dict[str, Any]] = []
    warnings = list(history.warnings)
    for item in history.traces:
        try:
            trace = service.read_trace(item.trace_id)
        except (OSError, ValueError) as exc:
            warnings.append(f"Skipped invalid trace file: {item.path}: {exc}")
            continue
        if trace is not None:
            traces.append(trace)

    latest_trace: dict[str, Any] | None = None
    if workspace.has_latest_trace():
        try:
            latest_trace = workspace.read_latest_trace()
        except (OSError, ValueError) as exc:
            warnings.append(f"Latest trace error: {exc}")

    selected_trace = latest_trace or (traces[0] if traces else None)
    message = (
        None
        if selected_trace
        else "No trace found. Run ragent ingest <path> first."
    )
    return TracePageModel(
        latest_trace=latest_trace,
        recent_traces=traces,
        selected_trace=selected_trace,
        warnings=warnings,
        message=message,
    )


def format_trace_page(model: TracePageModel) -> str:
    lines = ["Trace", "", "Recent traces", "# | Operation | Status | Started at"]
    if model.recent_traces:
        for index, trace in enumerate(model.recent_traces, start=1):
            lines.append(
                f"{index} | {trace.get('operation', '')} | "
                f"{trace.get('status', '')} | {trace.get('started_at', '')}"
            )
    else:
        lines.append(model.message or "No traces found.")

    selected = model.selected_trace
    lines.extend(["", "Steps"])
    if selected and isinstance(selected.get("steps"), list):
        steps = selected.get("steps", [])
        if steps:
            for index, step in enumerate(steps, start=1):
                step_name = step.get("name", "") if isinstance(step, dict) else ""
                lines.append(f"{index}. {step_name}")
        else:
            lines.append("No steps recorded.")
    else:
        lines.append("No trace selected.")

    if model.warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in model.warnings)
    return "\n".join(lines)


def format_trace_overview(trace: dict[str, Any] | None) -> str:
    if trace is None:
        return "Latest trace\n\nNo trace found. Run ragent ingest <path> first."
    return "\n".join(
        [
            "Latest trace",
            "",
            f"trace_id: {trace.get('trace_id', '')}",
            f"operation: {trace.get('operation', '')}",
            f"status: {trace.get('status', '')}",
            f"started_at: {trace.get('started_at', '')}",
            f"finished_at: {trace.get('finished_at', '')}",
        ]
    )


def format_trace_steps(trace: dict[str, Any] | None) -> str:
    lines = ["Steps"]
    if trace is None:
        lines.append("No trace selected.")
        return "\n".join(lines)

    steps = trace.get("steps", [])
    if not isinstance(steps, list) or not steps:
        lines.append("No steps recorded.")
        return "\n".join(lines)

    for index, step in enumerate(steps, start=1):
        step_name = step.get("name", "") if isinstance(step, dict) else ""
        lines.append(f"{index}. {step_name}")
    return "\n".join(lines)


def format_trace_inspector(trace: dict[str, Any] | None) -> str:
    if trace is None:
        return "Inspector\n\nSelect a trace to inspect metadata."
    lines = [
        "Trace details",
        "",
        f"trace_id: {trace.get('trace_id', '')}",
        f"operation: {trace.get('operation', '')}",
        f"status: {trace.get('status', '')}",
        f"started_at: {trace.get('started_at', '')}",
        f"finished_at: {trace.get('finished_at', '')}",
        "",
        "Metadata:",
    ]
    metadata = trace.get("metadata", {})
    if isinstance(metadata, dict) and metadata:
        lines.extend(
            f"- {key}: {_format_metadata_value(value)}"
            for key, value in metadata.items()
        )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _format_metadata_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    return str(value)


def load_settings_page_model(
    workspace_path: str | Path = ".ragent",
) -> SettingsPageModel:
    workspace = LocalWorkspace(workspace_path)
    config_service = ConfigService(workspace)
    config_exists = workspace.config_path.is_file()
    message = (
        None
        if config_exists
        else "No config file found. Effective defaults are being used."
    )
    try:
        config = config_service.load()
    except (OSError, ValueError):
        default = config_service.default_config()
        return SettingsPageModel(
            config_path=str(workspace.config_path),
            config_exists=config_exists,
            generation_provider=default.generation.provider,
            generation_model=_display_optional(default.generation.model),
            generation_base_url=_display_optional(default.generation.base_url),
            generation_api_key="not configured",
            embedding_provider=default.embedding.provider,
            embedding_model=_display_optional(default.embedding.model),
            embedding_base_url=_display_optional(default.embedding.base_url),
            embedding_api_key="not configured",
            vector_index_status="ready" if workspace.has_vector_index() else "missing",
            vector_index_path=str(workspace.vector_index_path)
            if workspace.has_vector_index()
            else None,
            message="Config error: unable to load config.",
        )

    return SettingsPageModel(
        config_path=str(workspace.config_path),
        config_exists=config_exists,
        generation_provider=config.generation.provider,
        generation_model=_display_optional(config.generation.model),
        generation_base_url=_display_optional(config.generation.base_url),
        generation_api_key=_hidden_or_not_configured(config.generation.api_key),
        embedding_provider=config.embedding.provider,
        embedding_model=_display_optional(config.embedding.model),
        embedding_base_url=_display_optional(config.embedding.base_url),
        embedding_api_key=_hidden_or_not_configured(config.embedding.api_key),
        vector_index_status="ready" if workspace.has_vector_index() else "missing",
        vector_index_path=str(workspace.vector_index_path)
        if workspace.has_vector_index()
        else None,
        message=message,
    )


def _display_optional(value: str | None) -> str:
    return value if value else "not configured"


def _hidden_or_not_configured(value: str | None) -> str:
    return "<hidden>" if value else "not configured"


def format_settings_page(model: SettingsPageModel) -> str:
    lines = ["Settings", "", f"config path: {model.config_path}"]
    if model.message:
        lines.extend(["", model.message])
    lines.extend(
        [
            f"generation provider: {model.generation_provider}",
            f"generation model: {model.generation_model}",
            f"generation base_url: {model.generation_base_url}",
            f"generation api_key: {model.generation_api_key}",
            f"embedding provider: {model.embedding_provider}",
            f"embedding model: {model.embedding_model}",
            f"embedding base_url: {model.embedding_base_url}",
            f"embedding api_key: {model.embedding_api_key}",
            f"vector index status: {model.vector_index_status}",
        ]
    )
    if model.vector_index_path:
        lines.append(f"vector index path: {model.vector_index_path}")
    return "\n".join(lines)


def page_for_key(key: str) -> PageName | None:
    return {
        "d": "documents",
        "1": "documents",
        "s": "search",
        "2": "search",
        "t": "trace",
        "3": "trace",
        "g": "settings",
        "4": "settings",
    }.get(key)
