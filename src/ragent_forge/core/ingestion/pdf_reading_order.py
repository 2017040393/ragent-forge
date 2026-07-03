from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PdfTextLine:
    text: str
    top: float
    x0: float


@dataclass(frozen=True)
class PdfPageText:
    text: str
    lines: tuple[PdfTextLine, ...]
    strategy: str
    fallback_used: bool = False
    warning: str | None = None


def extract_ordered_page_text(page: Any) -> PdfPageText:
    try:
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        ) or []
    except Exception as exc:
        return _fallback_page_text(page, f"word extraction failed: {exc}")

    lines = _words_to_lines(words)
    if not lines:
        fallback_text = (page.extract_text() or "").strip()
        if fallback_text:
            return _fallback_page_text(page, "word extraction produced no text")
        return PdfPageText(text="", lines=(), strategy="pdfplumber_words")

    ordered_lines, strategy = _order_lines(lines, float(getattr(page, "width", 0) or 0))
    text = "\n".join(line.text for line in ordered_lines).strip()
    if not text:
        return _fallback_page_text(page, "word extraction produced blank text")
    return PdfPageText(text=text, lines=tuple(ordered_lines), strategy=strategy)


def _words_to_lines(words: list[dict[str, Any]]) -> list[PdfTextLine]:
    sorted_words = sorted(
        (word for word in words if str(word.get("text", "")).strip()),
        key=lambda word: (_float_value(word.get("top")), _float_value(word.get("x0"))),
    )
    grouped: list[list[dict[str, Any]]] = []
    for word in sorted_words:
        top = _float_value(word.get("top"))
        if not grouped or abs(top - _line_top(grouped[-1])) > 3:
            grouped.append([word])
        else:
            grouped[-1].append(word)

    lines: list[PdfTextLine] = []
    for group in grouped:
        for segment in _split_wide_word_gaps(group):
            text = " ".join(
                str(word.get("text", "")).strip() for word in segment
            ).strip()
            if text:
                lines.append(
                    PdfTextLine(
                        text=text,
                        top=min(_float_value(word.get("top")) for word in segment),
                        x0=min(_float_value(word.get("x0")) for word in segment),
                    )
                )
    return lines


def _split_wide_word_gaps(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(words, key=lambda word: _float_value(word.get("x0")))
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_x1 = 0.0
    for word in ordered:
        x0 = _float_value(word.get("x0"))
        if current and x0 - previous_x1 > 120:
            segments.append(current)
            current = []
        current.append(word)
        previous_x1 = _float_value(word.get("x1"))
    if current:
        segments.append(current)
    return segments


def _line_top(words: list[dict[str, Any]]) -> float:
    return sum(_float_value(word.get("top")) for word in words) / len(words)


def _order_lines(
    lines: list[PdfTextLine],
    page_width: float,
) -> tuple[list[PdfTextLine], str]:
    split_x = _column_split_x(lines, page_width)
    if split_x is None:
        return sorted(lines, key=lambda line: (line.top, line.x0)), "pdfplumber_words"

    left = sorted(
        (line for line in lines if line.x0 < split_x),
        key=lambda line: line.top,
    )
    right = sorted(
        (line for line in lines if line.x0 >= split_x),
        key=lambda line: line.top,
    )
    return [*left, *right], "coordinate_blocks"


def _column_split_x(lines: list[PdfTextLine], page_width: float) -> float | None:
    if page_width <= 0 or len(lines) < 4:
        return None
    x_positions = sorted({round(line.x0, 1) for line in lines})
    if len(x_positions) < 2:
        return None

    gaps = [
        (right - left, left, right)
        for left, right in zip(x_positions, x_positions[1:], strict=False)
    ]
    gap, left, right = max(gaps, key=lambda item: item[0])
    split_x = (left + right) / 2
    if gap < page_width * 0.12:
        return None
    if not page_width * 0.2 <= split_x <= page_width * 0.8:
        return None

    left_count = sum(1 for line in lines if line.x0 < split_x)
    right_count = len(lines) - left_count
    if left_count < 2 or right_count < 2:
        return None
    return split_x


def _fallback_page_text(page: Any, warning: str) -> PdfPageText:
    text = (page.extract_text() or "").strip()
    lines = tuple(
        PdfTextLine(text=line.strip(), top=float(index), x0=0.0)
        for index, line in enumerate(text.splitlines())
        if line.strip()
    )
    return PdfPageText(
        text=text,
        lines=lines,
        strategy="pdfplumber_text_fallback",
        fallback_used=bool(text),
        warning=warning if text else None,
    )


def _float_value(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
