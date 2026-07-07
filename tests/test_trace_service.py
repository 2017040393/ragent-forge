from datetime import UTC, datetime
from pathlib import Path

from ragent_forge.app.models import Document, GenerationResult, IngestResult
from ragent_forge.app.services.trace_service import (
    build_ask_retrieval_trace,
    build_index_build_trace,
    build_ingest_trace,
    build_retrieval_eval_trace,
    build_search_trace,
)
from ragent_forge.core.chunking.simple_chunker import SimpleChunker


def test_build_ingest_trace_creates_success_trace_with_metadata() -> None:
    document = Document(
        id="/knowledge/rag.md",
        text="abcdefghij",
        metadata={"source_path": "/knowledge/rag.md"},
    )
    chunks = SimpleChunker(chunk_size=5, chunk_overlap=0).chunk(document)
    result = IngestResult(
        source_path="/knowledge",
        documents=[document],
        chunks=chunks,
        skipped_files=["/knowledge/skip.pdf"],
        metadata={"chunk_size": 5, "chunk_overlap": 0},
    )
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 2, tzinfo=UTC)

    trace = build_ingest_trace(
        result=result,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        summary_path=Path(".ragent/ingest/latest_summary.json"),
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "ingest-20260630T000000Z"
    assert trace.operation == "ingest"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:02Z"
    assert [step.name for step in trace.steps] == [
        "load_documents",
        "chunk_documents",
        "write_chunks",
        "write_ingest_summary",
    ]
    assert trace.metadata == {
        "source_path": "/knowledge",
        "document_count": 1,
        "chunk_count": 2,
        "skipped_count": 1,
        "chunk_size": 5,
        "chunk_overlap": 0,
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "summary_path": str(Path(".ragent/ingest/latest_summary.json")),
    }


def test_build_search_trace_creates_success_trace_with_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_search_trace(
        query="agent memory",
        limit=5,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        result_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "search-20260630T000000Z"
    assert trace.operation == "search"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:01Z"
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "tokenize_query",
        "score_chunks",
        "rank_results",
        "render_results",
    ]
    assert trace.steps[1].description == (
        "Normalize and tokenize the lexical search query."
    )
    assert trace.steps[2].description == "Score chunks by lexical token overlap."
    assert trace.metadata == {
        "query": "agent memory",
        "limit": 5,
        "retrieval_mode": "lexical",
        "scoring_method": "lexical_token_overlap",
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "total_chunks": 7,
        "result_count": 1,
        "result_chunk_ids": ["/knowledge/rag.md::chunk-0002"],
    }


def test_build_search_trace_records_semantic_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_search_trace(
        query="agent memory",
        limit=5,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        result_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "semantic"
    assert trace.metadata["retrieval_method"] == "semantic_cosine_similarity"
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    assert "api_key" not in trace.metadata
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "embed_query",
        "load_vector_index",
        "score_vectors",
        "rank_results",
        "render_results",
    ]
    descriptions = " ".join(step.description for step in trace.steps)
    assert "lexical" not in descriptions
    assert "token overlap" not in descriptions
    assert trace.steps[1].inputs == {
        "query": "agent memory",
        "embedding_provider": "openai_embeddings",
        "embedding_model": "text-embedding-3-small",
    }
    assert trace.steps[1].outputs == {"query_embedding": "computed"}
    assert trace.steps[2].inputs == {
        "index_path": str(Path(".ragent/index/vector_index.jsonl"))
    }
    assert trace.steps[2].outputs == {"status": "loaded"}
    assert trace.steps[3].inputs == {
        "retrieval_method": "semantic_cosine_similarity"
    }
    assert trace.steps[3].outputs == {"matched_chunks": 1}
    trace_json = trace.model_dump_json()
    assert "embedding-secret-key" not in trace_json
    assert "[0.1" not in trace_json
    assert "full chunk text" not in trace_json
    assert "Generated answer" not in trace_json


def test_build_search_trace_records_hybrid_rrf_metadata_and_steps() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_search_trace(
        query="agent memory",
        limit=5,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        result_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60,
        sparse_method="bm25",
        dense_method="semantic_cosine_similarity",
        sparse_weight=1.0,
        dense_weight=1.0,
        lexical_weight=1.0,
        semantic_weight=1.0,
        candidate_limit=20,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "hybrid"
    assert trace.metadata["retrieval_method"] == "hybrid_rrf"
    assert trace.metadata["fusion_method"] == "reciprocal_rank_fusion"
    assert trace.metadata["rrf_k"] == 60
    assert trace.metadata["sparse_method"] == "bm25"
    assert trace.metadata["dense_method"] == "semantic_cosine_similarity"
    assert trace.metadata["sparse_weight"] == 1.0
    assert trace.metadata["dense_weight"] == 1.0
    assert trace.metadata["lexical_weight"] == 1.0
    assert trace.metadata["semantic_weight"] == 1.0
    assert trace.metadata["candidate_limit"] == 20
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "run_bm25_search",
        "embed_query",
        "load_vector_index",
        "run_semantic_search",
        "fuse_results",
        "rank_results",
        "render_results",
    ]
    assert trace.steps[5].inputs == {
        "retrieval_method": "hybrid_rrf",
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": 60,
        "sparse_method": "bm25",
        "dense_method": "semantic_cosine_similarity",
    }
    trace_json = trace.model_dump_json()
    assert "embedding-secret-key" not in trace_json
    assert "[0.1" not in trace_json
    assert "full chunk text" not in trace_json
    assert "Generated answer" not in trace_json


def test_build_retrieval_eval_trace_records_lexical_metrics_safely() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_retrieval_eval_trace(
        cases_path=Path("eval/retrieval_cases.jsonl"),
        retrieval_mode="lexical",
        retrieval_method="lexical_token_overlap",
        limit=5,
        case_count=2,
        passed_count=1,
        failed_count=1,
        metrics={
            "hit@1": 0.5,
            "hit@3": 0.5,
            "hit@5": 0.5,
            "hit@k": 0.5,
            "mrr": 0.5,
        },
        report_path=Path(".ragent/eval/retrieval_eval_20260630T000001Z.json"),
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "retrieval-eval-20260630T000000Z"
    assert trace.operation == "retrieval_eval"
    assert trace.status == "success"
    assert [step.name for step in trace.steps] == [
        "load_eval_cases",
        "run_retrieval_cases",
        "compute_metrics",
        "write_eval_report",
        "render_eval_summary",
    ]
    assert trace.metadata == {
        "evaluation_type": "retrieval",
        "cases_path": str(Path("eval/retrieval_cases.jsonl")),
        "retrieval_mode": "lexical",
        "retrieval_method": "lexical_token_overlap",
        "limit": 5,
        "case_count": 2,
        "passed_count": 1,
        "failed_count": 1,
        "hit@1": 0.5,
        "hit@3": 0.5,
        "hit@5": 0.5,
        "hit@k": 0.5,
        "mrr": 0.5,
        "report_path": str(Path(".ragent/eval/retrieval_eval_20260630T000001Z.json")),
    }
    assert "embedding_provider" not in trace.metadata
    assert "embedding_model" not in trace.metadata


def test_build_retrieval_eval_trace_records_semantic_index_metadata_safely() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_retrieval_eval_trace(
        cases_path=Path("eval/retrieval_cases.jsonl"),
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        limit=5,
        case_count=2,
        passed_count=2,
        failed_count=0,
        metrics={
            "hit@1": 1.0,
            "hit@3": 1.0,
            "hit@5": 1.0,
            "hit@k": 1.0,
            "mrr": 1.0,
        },
        report_path=Path(".ragent/eval/retrieval_eval_20260630T000001Z.json"),
        started_at=started_at,
        finished_at=finished_at,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "semantic"
    assert trace.metadata["retrieval_method"] == "semantic_cosine_similarity"
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    trace_json = trace.model_dump_json()
    assert "embedding-secret-key" not in trace_json
    assert "full chunk text" not in trace_json
    assert '"embedding": [' not in trace_json


def test_build_retrieval_eval_trace_records_hybrid_fusion_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_retrieval_eval_trace(
        cases_path=Path("eval/retrieval_cases.jsonl"),
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        limit=5,
        case_count=2,
        passed_count=2,
        failed_count=0,
        metrics={
            "hit@1": 1.0,
            "hit@3": 1.0,
            "hit@5": 1.0,
            "hit@k": 1.0,
            "mrr": 1.0,
        },
        report_path=Path(".ragent/eval/retrieval_eval_20260630T000001Z.json"),
        started_at=started_at,
        finished_at=finished_at,
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60,
        sparse_method="bm25",
        dense_method="semantic_cosine_similarity",
        sparse_weight=1.0,
        dense_weight=1.0,
        lexical_weight=1.0,
        semantic_weight=1.0,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "hybrid"
    assert trace.metadata["retrieval_method"] == "hybrid_rrf"
    assert trace.metadata["fusion_method"] == "reciprocal_rank_fusion"
    assert trace.metadata["rrf_k"] == 60
    assert trace.metadata["sparse_method"] == "bm25"
    assert trace.metadata["dense_method"] == "semantic_cosine_similarity"
    assert trace.metadata["sparse_weight"] == 1.0
    assert trace.metadata["dense_weight"] == 1.0
    assert trace.metadata["lexical_weight"] == 1.0
    assert trace.metadata["semantic_weight"] == 1.0
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    assert trace.metadata["hit@k"] == 1.0
    trace_json = trace.model_dump_json()
    assert "embedding-secret-key" not in trace_json
    assert "full chunk text" not in trace_json
    assert '"embedding": [' not in trace_json


def test_build_ask_retrieval_trace_creates_success_trace_with_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="null",
            status="not_configured",
            answer=None,
        ),
        config_generation_provider="null",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=True,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "ask-retrieval-20260630T000000Z"
    assert trace.operation == "ask_retrieval"
    assert trace.status == "success"
    assert trace.started_at == "2026-06-30T00:00:00Z"
    assert trace.finished_at == "2026-06-30T00:00:01Z"
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "retrieve_context",
        "assemble_context_preview",
        "generate_answer",
        "render_retrieval_preview",
    ]
    assert trace.steps[1].description == "Retrieve context chunks with lexical search."
    generation_step = trace.steps[3]
    assert generation_step.inputs == {"provider": "null"}
    assert generation_step.outputs == {
        "generation_status": "not_implemented",
        "generation_result_status": "not_configured",
        "answer_generated": False,
    }
    assert trace.metadata == {
        "question": "what is agent memory?",
        "limit": 3,
        "retrieval_mode": "lexical",
        "retrieval_method": "lexical_token_overlap",
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "total_chunks": 7,
        "retrieved_count": 1,
        "retrieved_chunk_ids": ["/knowledge/rag.md::chunk-0002"],
        "generation_status": "not_implemented",
        "generation_provider": "null",
        "generation_result_status": "not_configured",
        "answer_generated": False,
        "config_generation_provider": "null",
        "context_chunk_count": 1,
        "total_context_chars": 128,
        "prompt_preview_shown": True,
        "max_context_chars": 4000,
    }


def test_build_ask_retrieval_trace_records_real_generation_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="openai_responses",
            status="success",
            answer="Generated answer",
            metadata={
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "/responses",
            },
        ),
        config_generation_provider="openai_responses",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=False,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
    )

    generation_step = trace.steps[3]
    assert generation_step.outputs == {
        "generation_status": "generated",
        "generation_result_status": "success",
        "answer_generated": True,
    }
    assert trace.metadata["generation_status"] == "generated"
    assert trace.metadata["generation_provider"] == "openai_responses"
    assert trace.metadata["generation_result_status"] == "success"
    assert trace.metadata["answer_generated"] is True
    assert trace.metadata["model"] == "gpt-4o-mini"
    assert trace.metadata["base_url"] == "https://api.openai.com/v1"
    assert trace.metadata["endpoint"] == "/responses"
    assert trace.metadata["source_count"] == 1


def test_build_ask_retrieval_trace_records_semantic_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="null",
            status="not_configured",
            answer=None,
        ),
        config_generation_provider="null",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=False,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode="semantic",
        retrieval_method="semantic_cosine_similarity",
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "semantic"
    assert trace.metadata["retrieval_method"] == "semantic_cosine_similarity"
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    assert "api_key" not in trace.metadata
    retrieval_step = trace.steps[1]
    assert retrieval_step.name == "retrieve_context"
    assert retrieval_step.description == (
        "Retrieve context chunks with semantic vector search."
    )
    assert retrieval_step.inputs == {
        "question": "what is agent memory?",
        "limit": 3,
        "retrieval_method": "semantic_cosine_similarity",
        "embedding_provider": "openai_embeddings",
        "embedding_model": "text-embedding-3-small",
        "index_path": str(Path(".ragent/index/vector_index.jsonl")),
    }
    assert "Retrieve context chunks with lexical search." not in trace.model_dump_json()
    assert "embedding-secret-key" not in trace.model_dump_json()
    assert "[0.1" not in trace.model_dump_json()
    assert "full chunk text" not in trace.model_dump_json()
    assert "Generated answer" not in trace.model_dump_json()


def test_build_ask_retrieval_trace_records_hybrid_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="null",
            status="not_configured",
            answer=None,
        ),
        config_generation_provider="null",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=False,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
        retrieval_mode="hybrid",
        retrieval_method="hybrid_rrf",
        fusion_method="reciprocal_rank_fusion",
        rrf_k=60,
        sparse_method="bm25",
        dense_method="semantic_cosine_similarity",
        sparse_weight=1.0,
        dense_weight=1.0,
        lexical_weight=1.0,
        semantic_weight=1.0,
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        index_path=Path(".ragent/index/vector_index.jsonl"),
    )

    assert trace.metadata["retrieval_mode"] == "hybrid"
    assert trace.metadata["retrieval_method"] == "hybrid_rrf"
    assert trace.metadata["fusion_method"] == "reciprocal_rank_fusion"
    assert trace.metadata["rrf_k"] == 60
    assert trace.metadata["sparse_method"] == "bm25"
    assert trace.metadata["dense_method"] == "semantic_cosine_similarity"
    assert trace.metadata["sparse_weight"] == 1.0
    assert trace.metadata["dense_weight"] == 1.0
    assert trace.metadata["lexical_weight"] == 1.0
    assert trace.metadata["semantic_weight"] == 1.0
    assert trace.metadata["embedding_provider"] == "openai_embeddings"
    assert trace.metadata["embedding_model"] == "text-embedding-3-small"
    assert trace.metadata["index_path"] == str(Path(".ragent/index/vector_index.jsonl"))
    retrieval_step = trace.steps[1]
    assert retrieval_step.name == "retrieve_context"
    assert retrieval_step.description == (
        "Retrieve context chunks with hybrid BM25 and semantic search."
    )
    assert retrieval_step.inputs == {
        "question": "what is agent memory?",
        "limit": 3,
        "retrieval_method": "hybrid_rrf",
        "fusion_method": "reciprocal_rank_fusion",
        "rrf_k": 60,
        "sparse_method": "bm25",
        "dense_method": "semantic_cosine_similarity",
        "embedding_provider": "openai_embeddings",
        "embedding_model": "text-embedding-3-small",
        "index_path": str(Path(".ragent/index/vector_index.jsonl")),
    }
    assert "embedding-secret-key" not in trace.model_dump_json()
    assert "[0.1" not in trace.model_dump_json()
    assert "full chunk text" not in trace.model_dump_json()
    assert "Generated answer" not in trace.model_dump_json()


def test_build_index_build_trace_records_safe_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_index_build_trace(
        embedding_provider="openai_embeddings",
        embedding_model="text-embedding-3-small",
        chunk_count=3,
        embedding_dim=1536,
        index_path=Path(".ragent/index/vector_index.jsonl"),
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        batch_size=64,
        started_at=started_at,
        finished_at=finished_at,
    )

    assert trace.trace_id == "index-build-20260630T000000Z"
    assert trace.operation == "index_build"
    assert trace.status == "success"
    assert [step.name for step in trace.steps] == [
        "read_chunks",
        "embed_chunks",
        "write_vector_index",
    ]
    assert trace.metadata == {
        "embedding_provider": "openai_embeddings",
        "embedding_model": "text-embedding-3-small",
        "chunk_count": 3,
        "embedding_dim": 1536,
        "index_path": str(Path(".ragent/index/vector_index.jsonl")),
        "chunks_path": str(Path(".ragent/chunks/chunks.jsonl")),
        "batch_size": 64,
    }
    assert "api_key" not in str(trace.model_dump())
    assert "embedding\": [" not in trace.model_dump_json()


def test_build_ask_retrieval_trace_filters_sensitive_generation_metadata() -> None:
    started_at = datetime(2026, 6, 30, 0, 0, 0, tzinfo=UTC)
    finished_at = datetime(2026, 6, 30, 0, 0, 1, tzinfo=UTC)

    trace = build_ask_retrieval_trace(
        question="what is agent memory?",
        limit=3,
        chunks_path=Path(".ragent/chunks/chunks.jsonl"),
        total_chunks=7,
        retrieved_chunk_ids=["/knowledge/rag.md::chunk-0002"],
        generation_result=GenerationResult(
            provider_name="openai_responses",
            status="success",
            answer="Generated answer",
            metadata={
                "model": "gpt-4o-mini",
                "base_url": "https://api.openai.com/v1",
                "endpoint": "/responses",
                "api_key": "super-secret-key",
                "prompt": "full prompt",
                "context": "full context",
                "answer": "full answer",
            },
        ),
        config_generation_provider="openai_responses",
        context_chunk_count=1,
        total_context_chars=128,
        prompt_preview_shown=False,
        max_context_chars=4000,
        started_at=started_at,
        finished_at=finished_at,
    )

    assert "api_key" not in trace.metadata
    assert "prompt" not in trace.metadata
    assert "context" not in trace.metadata
    assert "answer" not in trace.metadata
    assert "super-secret-key" not in str(trace.metadata)
