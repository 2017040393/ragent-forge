from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from ragent_forge.core.retrieval.contracts import ChunkRecord

EmbeddingRepresentation = Literal[
    "raw_chunk_text_v1",
    "structured_document_text_v1",
    "cleaned_pdf_section_text_v1",
    "cleaned_pdf_formula_text_v1",
]

EMBEDDING_REPRESENTATIONS: tuple[EmbeddingRepresentation, ...] = (
    "raw_chunk_text_v1",
    "structured_document_text_v1",
    "cleaned_pdf_section_text_v1",
    "cleaned_pdf_formula_text_v1",
)

QueryEmbeddingRepresentation = Literal[
    "raw_query_v1",
    "instructed_query_v1",
]

QUERY_EMBEDDING_REPRESENTATIONS: tuple[QueryEmbeddingRepresentation, ...] = (
    "raw_query_v1",
    "instructed_query_v1",
)

INSTRUCTED_QUERY_V1_PREFIX = (
    "Instruct: Retrieve the document passage that best answers the query. "
    "Distinguish the correct section from other passages in the same source "
    "document.\nQuery: "
)

_CID_MARKER_PATTERN = re.compile(r"\(cid:\d+\)", re.IGNORECASE)
_ISOLATED_PAGE_NUMBER_PATTERN = re.compile(r"^\d{1,4}$")
_ASCII_LINE_BREAK_HYPHEN_PATTERN = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")
_NUMBERED_HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+){1,4})\s+(.{3,160})$")
_STRUCTURAL_HEADING_PATTERN = re.compile(
    r"^(?:chapter|part|section)\b.{2,160}$",
    re.IGNORECASE,
)
_LOCAL_LABEL_PATTERN = re.compile(
    r"^(?:definition|example|theorem|lemma|proposition|corollary|exercise)"
    r"\b.{2,160}$",
    re.IGNORECASE,
)
_PAGE_PREFIX_PATTERN = re.compile(r"^\d{1,4}\s+(.{3,120})$")
_MATH_HEADING_NOISE = frozenset("=<>\u2264\u2265\u2248\u2211\u222b\u221a^{}[]")
_TITLE_SMALL_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "over",
        "the",
        "to",
        "with",
    }
)
_BIBLIOGRAPHIC_NOISE_WORDS = frozenset(
    {
        "cambridge",
        "copyright",
        "discrete",
        "geometry",
        "institute",
        "isbn",
        "library",
        "longman",
        "monographs",
        "north-holland",
        "press",
        "series",
        "university",
        "volume",
        "vol",
        "wiley",
        "springer",
        "surveys",
    }
)
_FORMULA_METRIC_PATTERN = re.compile(
    r"\b(?:RRF|MRR|Recall@|Hit@|nDCG@|Precision@)",
    re.IGNORECASE,
)
_FORMULA_SIGNAL_PATTERN = re.compile(
    r"(?:=|<=|>=|<|>|\^|\bapprox\b|\bintegral\b|\bsum\b|\bsqrt\b|"
    r"\belement_of\b|\bsubset_of\b|\borthogonal_to\b|\bnorm\b|\|\|)",
    re.IGNORECASE,
)
_FORMULA_CONSTANT_PATTERN = re.compile(
    r"(?:\d|\bpi\b|\blambda\b|\bintegral\b|\bsum\b|\bsqrt\b)",
    re.IGNORECASE,
)
_FORMULA_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u2212", "-"),
    ("\u2013", "-"),
    ("\u03c0", " pi "),
    ("\U0001d70b", " pi "),
    ("\u03bb", " lambda "),
    ("\U0001d706", " lambda "),
    ("\u2264", " <= "),
    ("\u2265", " >= "),
    ("\u2248", " approx "),
    ("\u2208", " element_of "),
    ("\u2286", " subset_of "),
    ("\u27c2", " orthogonal_to "),
    ("\u222b", " integral "),
    ("\u2211", " sum "),
    ("\u03a3", " sum "),
    ("\u221a", " sqrt "),
    ("\u2016", " || "),
)


@dataclass(frozen=True)
class _PdfHeadingScan:
    local_headings: tuple[str, ...]
    propagated_heading: str | None


def build_embedding_text(
    chunk: ChunkRecord,
    representation: EmbeddingRepresentation = "raw_chunk_text_v1",
) -> str:
    """Build the deterministic document-side input for an embedding provider."""

    return build_embedding_texts([chunk], representation)[0]


def build_embedding_texts(
    chunks: Sequence[ChunkRecord],
    representation: EmbeddingRepresentation = "raw_chunk_text_v1",
) -> list[str]:
    """Build document inputs while preserving cross-chunk representation state."""

    if representation == "raw_chunk_text_v1":
        return [chunk["text"] for chunk in chunks]
    if representation == "structured_document_text_v1":
        return [build_structured_document_text_v1(chunk) for chunk in chunks]
    if representation == "cleaned_pdf_section_text_v1":
        return _build_cleaned_pdf_texts_v1(chunks, include_formula_evidence=False)
    if representation == "cleaned_pdf_formula_text_v1":
        return _build_cleaned_pdf_texts_v1(chunks, include_formula_evidence=True)
    raise ValueError(f"Unsupported embedding representation: {representation!r}")


def build_query_embedding_text(
    query: str,
    representation: QueryEmbeddingRepresentation = "raw_query_v1",
) -> str:
    """Build the deterministic query-side input for an embedding provider."""

    if representation == "raw_query_v1":
        return query
    if representation == "instructed_query_v1":
        return f"{INSTRUCTED_QUERY_V1_PREFIX}{query}"
    raise ValueError(f"Unsupported query embedding representation: {representation!r}")


def build_structured_document_text_v1(chunk: ChunkRecord) -> str:
    metadata = chunk.get("metadata", {})
    block_types = _sorted_metadata_strings(
        metadata.get("block_types"), metadata.get("block_type")
    )
    heading_path = _metadata_strings(metadata.get("heading_path"))
    section = " > ".join(heading_path)
    if not section:
        section = _metadata_string(metadata.get("section_title")) or "unknown"

    title = (
        _metadata_string(metadata.get("document_title"))
        or _metadata_string(metadata.get("title"))
        or (heading_path[0] if heading_path else "")
        or _source_stem(chunk)
        or "unknown"
    )
    source = _stable_source(chunk)
    page = _page_label(metadata)
    signals = _signals(metadata, block_types)
    block_label = ", ".join(block_types) if block_types else "unknown"
    return "\n".join(
        (
            f"Document title: {title}",
            f"Source: {source}",
            f"Section: {section}",
            f"Page: {page}",
            f"Block types: {block_label}",
            f"Signals: {signals}",
            "Content:",
            chunk["text"],
        )
    )


def _build_cleaned_pdf_texts_v1(
    chunks: Sequence[ChunkRecord],
    *,
    include_formula_evidence: bool,
) -> list[str]:
    current_section_by_document: dict[str, str] = {}
    inputs: list[str] = []
    for chunk in chunks:
        if not _is_pdf_chunk(chunk):
            inputs.append(build_structured_document_text_v1(chunk))
            continue

        document_id = chunk["document_id"]
        metadata = chunk.get("metadata", {})
        heading_scan = _pdf_heading_scan(chunk["text"], metadata)
        local_headings = heading_scan.local_headings
        metadata_section = _metadata_section(metadata)
        if metadata_section:
            section = metadata_section
            current_section_by_document[document_id] = metadata_section
        elif local_headings:
            section = " | ".join(local_headings)
        else:
            section = current_section_by_document.get(document_id, "unknown")
        if heading_scan.propagated_heading is not None:
            current_section_by_document[document_id] = (
                heading_scan.propagated_heading
            )
        inputs.append(
            _build_cleaned_pdf_text_v1(
                chunk,
                section,
                include_formula_evidence=include_formula_evidence,
            )
        )
    return inputs


def _build_cleaned_pdf_text_v1(
    chunk: ChunkRecord,
    section: str,
    *,
    include_formula_evidence: bool,
) -> str:
    metadata = chunk.get("metadata", {})
    cleaned_content = _clean_pdf_embedding_text(chunk["text"], metadata)
    if not cleaned_content:
        raise ValueError(
            "PDF embedding cleanup produced empty content for chunk "
            f"{chunk['chunk_id']!r}"
        )
    block_types = _sorted_metadata_strings(
        metadata.get("block_types"), metadata.get("block_type")
    )
    heading_path = _metadata_strings(metadata.get("heading_path"))
    title = (
        _metadata_string(metadata.get("document_title"))
        or _metadata_string(metadata.get("title"))
        or (heading_path[0] if heading_path else "")
        or _source_stem(chunk)
        or "unknown"
    )
    block_label = ", ".join(block_types) if block_types else "unknown"
    lines = [
        f"Document title: {title}",
        f"Source: {_stable_source(chunk)}",
        f"Section: {section}",
        f"Page: {_page_label(metadata)}",
        f"Block types: {block_label}",
        f"Signals: {_signals(metadata, block_types)}",
    ]
    if include_formula_evidence:
        formula_evidence = _formula_evidence(metadata)
        lines.append(
            "Formula evidence: "
            + (" | ".join(formula_evidence) if formula_evidence else "none")
        )
    lines.extend(("Content:", cleaned_content))
    return "\n".join(lines)


def _formula_evidence(metadata: dict[str, object]) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()
    for candidate in _metadata_strings(metadata.get("possible_formula_lines")):
        normalized = _normalize_formula_line(candidate)
        key = normalized.casefold()
        if not _is_high_confidence_formula(normalized) or key in seen:
            continue
        evidence.append(normalized)
        seen.add(key)
        if len(evidence) == 3:
            break
    return evidence


def _normalize_formula_line(value: str) -> str:
    normalized = _CID_MARKER_PATTERN.sub("", value)
    normalized = unicodedata.normalize("NFKC", normalized)
    for source, replacement in _FORMULA_REPLACEMENTS:
        normalized = normalized.replace(source, replacement)
    return " ".join(normalized.split())


def _is_high_confidence_formula(value: str) -> bool:
    if not 4 <= len(value) <= 240:
        return False
    if not any(character.isalpha() for character in value):
        return False
    if _FORMULA_CONSTANT_PATTERN.search(value) is None:
        return False
    if _FORMULA_METRIC_PATTERN.search(value) is not None:
        return True
    return _FORMULA_SIGNAL_PATTERN.search(value) is not None


def _clean_pdf_embedding_text(
    text: str,
    metadata: dict[str, object],
) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00ad", "")
    header_footer_candidates = {
        " ".join(candidate.split()).casefold()
        for candidate in _metadata_strings(metadata.get("header_footer_candidates"))
    }
    kept_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            kept_lines.append("")
            continue
        if line.casefold() in header_footer_candidates:
            continue
        if _ISOLATED_PAGE_NUMBER_PATTERN.fullmatch(line):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = _CID_MARKER_PATTERN.sub("", cleaned)
    cleaned = _ASCII_LINE_BREAK_HYPHEN_PATTERN.sub(r"\1\2", cleaned)
    return " ".join(cleaned.split())


def _metadata_section(metadata: dict[str, object]) -> str | None:
    heading_path = _metadata_strings(metadata.get("heading_path"))
    if heading_path:
        return " > ".join(heading_path)
    return _metadata_string(metadata.get("section_title"))


def _pdf_heading_scan(
    text: str,
    metadata: dict[str, object],
) -> _PdfHeadingScan:
    header_footer_candidates = {
        " ".join(candidate.split()).casefold()
        for candidate in _metadata_strings(
            metadata.get("header_footer_candidates")
        )
    }
    lines = [
        " ".join(_CID_MARKER_PATTERN.sub("", line).strip().split())
        for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    ]
    lines = [
        line
        for line in lines
        if line and line.casefold() not in header_footer_candidates
    ]
    headings: list[str] = []
    propagated_heading: str | None = None
    for index, line in enumerate(lines[:40]):
        candidate, propagates = _pdf_heading_candidate(
            line,
            lines[index + 1] if index + 1 < len(lines) else None,
            page_lead=index < 3,
        )
        if candidate and candidate.casefold() not in {
            heading.casefold() for heading in headings
        }:
            headings.append(candidate)
        if candidate and propagates:
            propagated_heading = candidate
        if len(headings) == 3:
            break
    return _PdfHeadingScan(
        local_headings=tuple(headings),
        propagated_heading=propagated_heading,
    )


def _pdf_heading_candidate(
    line: str,
    next_line: str | None,
    *,
    page_lead: bool,
) -> tuple[str | None, bool]:
    if _ISOLATED_PAGE_NUMBER_PATTERN.fullmatch(line):
        if page_lead and next_line is not None and _looks_title_heading(next_line):
            return next_line, True
        return None, False
    if _STRUCTURAL_HEADING_PATTERN.fullmatch(line):
        return line, True
    if _LOCAL_LABEL_PATTERN.fullmatch(line):
        return line, False

    page_prefix = _PAGE_PREFIX_PATTERN.fullmatch(line)
    if page_lead and page_prefix is not None:
        value = page_prefix.group(1).strip()
        if _STRUCTURAL_HEADING_PATTERN.fullmatch(value) or _looks_title_heading(value):
            return value, True

    numbered = _NUMBERED_HEADING_PATTERN.fullmatch(line)
    if numbered is None:
        if page_lead and _looks_title_heading(line):
            return line, True
        return None, False
    label = numbered.group(1)
    value = numbered.group(2).strip()
    if any(character in _MATH_HEADING_NOISE for character in value):
        return None, False
    if _LOCAL_LABEL_PATTERN.fullmatch(value):
        return f"{label} {value}", False
    if _looks_title_heading(value):
        return f"{label} {value}", True
    words = re.findall(r"[A-Za-z]+", value)
    if len(words) >= 2 and not value.endswith((".", "?", "!")):
        return f"{label} {value}", False
    return None, False


def _looks_title_heading(value: str) -> bool:
    if (
        len(value) > 120
        or "," in value
        or value.endswith((".", "?", "!", ":", ";"))
    ):
        return False
    if any(character in _MATH_HEADING_NOISE for character in value):
        return False
    words = re.findall(r"[A-Za-z]+", value)
    if not 3 <= len(words) <= 14:
        return False
    significant = [word for word in words if word.casefold() not in _TITLE_SMALL_WORDS]
    if len(significant) < 2:
        return False
    if any(
        word.casefold().rstrip(".") in _BIBLIOGRAPHIC_NOISE_WORDS
        for word in significant
    ):
        return False
    title_words = sum(word[0].isupper() for word in significant if word)
    return title_words / len(significant) >= 0.6


def _is_pdf_chunk(chunk: ChunkRecord) -> bool:
    metadata = chunk.get("metadata", {})
    media_type = _metadata_string(metadata.get("media_type"))
    if media_type == "application/pdf":
        return True
    source_path = _metadata_string(chunk.get("source_path"))
    return bool(source_path and source_path.casefold().endswith(".pdf"))


def embedding_input_fingerprint(
    records: list[tuple[str, str]],
) -> str:
    """Hash ordered-independent chunk IDs and hashes for an index build."""

    payload = [
        {"chunk_id": chunk_id, "text_sha256": text_hash}
        for chunk_id, text_hash in sorted(records)
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_embedding_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_source(chunk: ChunkRecord) -> str:
    raw = _metadata_string(chunk.get("source_path")) or _metadata_string(
        chunk.get("document_id")
    )
    if not raw:
        return "unknown"
    normalized = raw.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        return PurePosixPath(normalized).name or "unknown"
    return normalized.lstrip("./") or "unknown"


def _source_stem(chunk: ChunkRecord) -> str:
    source = _stable_source(chunk)
    name = PurePosixPath(source).name
    if not name:
        return ""
    return PurePosixPath(name).stem


def _page_label(metadata: dict[str, object]) -> str:
    start = _metadata_integer(metadata.get("page_start"))
    end = _metadata_integer(metadata.get("page_end"))
    if start is None:
        return "unknown"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _signals(metadata: dict[str, object], block_types: list[str]) -> str:
    signals: list[str] = []
    if metadata.get("possible_formula") is True or any(
        "formula" in block_type.lower() for block_type in block_types
    ):
        signals.append("formula")
    if any("table" in block_type.lower() for block_type in block_types):
        signals.append("table")
    return ", ".join(signals) if signals else "none"


def _sorted_metadata_strings(*values: object) -> list[str]:
    return sorted(set(item for value in values for item in _metadata_strings(value)))


def _metadata_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return []


def _metadata_string(value: object) -> str | None:
    values = _metadata_strings(value)
    return values[0] if values else None


def _metadata_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None
