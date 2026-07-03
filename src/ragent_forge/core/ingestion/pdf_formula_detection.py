from __future__ import annotations

import re

_METRIC_PATTERNS = (
    re.compile(r"\bRRF\s*\(", re.IGNORECASE),
    re.compile(r"\bMRR\b", re.IGNORECASE),
    re.compile(r"\bRecall@", re.IGNORECASE),
    re.compile(r"\bHit@", re.IGNORECASE),
)

_MATH_SYMBOLS = {
    "=",
    "+",
    "-",
    "*",
    "/",
    "^",
    "<",
    ">",
    "\u2248",
    "\u2264",
    "\u2265",
    "\u2211",
    "\u03a3",
    "\u222b",
    "\u221a",
    "\u2202",
    "\u03b1",
    "\u03b2",
    "\u03b3",
    "\u03bb",
    "\u03b8",
}


def detect_formula_lines(text: str) -> list[str]:
    formula_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_formula_like(line):
            formula_lines.append(line)
    return formula_lines


def _looks_formula_like(line: str) -> bool:
    if any(pattern.search(line) for pattern in _METRIC_PATTERNS):
        return True

    symbol_count = sum(1 for character in line if character in _MATH_SYMBOLS)
    if symbol_count == 0:
        return False
    has_digit = any(character.isdigit() for character in line)
    has_alpha = any(character.isalpha() for character in line)
    return has_digit and has_alpha
