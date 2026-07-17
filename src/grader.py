"""Answer extraction, normalization, and equivalence checking.

Thin wrapper around HuggingFace's `math_verify` (the lighteval normalizers).
We own only the \\boxed{} extractor; everything downstream is delegated.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager
from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from src.schema import parse_ok_from_status   # single source of truth

CFG = [LatexExtractionConfig(), ExprExtractionConfig()]

_BOXED = "\\boxed"
_OPEN_BRACE = re.compile(r"\s*\{")
# fallback for `\boxed 5` / `\boxed\frac{1}{2}` (no braces around the payload)
_BARE = re.compile(r"\s*(-?\d+(?:\.\d+)?|\\[a-zA-Z]+(?:\{[^{}]*\})*)")

TIMEOUT_SECONDS = 5


class GoldParseError(ValueError):
    """The gold answer itself could not be parsed. This is a data bug, not a
    wrong answer -- never let it silently count as an incorrect prediction."""


@contextmanager
def _time_limit(seconds: int):
    """SIGALRM guard: sympy can hang on adversarial expressions."""
    def _handler(signum, frame):
        raise TimeoutError("sympy comparison timed out")

    try:
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
    except (ValueError, AttributeError):  # non-main thread / non-POSIX
        yield
        return
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _boxed_payload_at(text: str, idx: int):
    """Extract one \\boxed payload starting at index `idx`. Returns
    (payload | None, index_to_resume_scanning_from)."""
    head = text[idx + len(_BOXED):]
    m = _OPEN_BRACE.match(head)
    if not m:
        bare = _BARE.match(head)
        if bare:
            return bare.group(1), idx + len(_BOXED) + bare.end()
        return None, idx + len(_BOXED)

    open_at = idx + len(_BOXED) + m.end() - 1
    depth = 0
    for j in range(open_at, len(text)):
        c = text[j]
        if c == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif c == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                inner = text[open_at + 1:j].strip()
                return (inner or None), j + 1      # \boxed{} is not an answer
    return None, len(text)                          # unbalanced


def extract_all_boxed(text: str) -> list[str]:
    """NEW. Every \\boxed payload, in order. We need the COUNT, not just the
    last one: `multiple_found` was previously undetectable, which meant
    all_boxed_matches could never be populated and a model restating three
    different answers looked identical to a model committing to one."""
    if not text:
        return []
    out, i = [], text.find(_BOXED)
    while i != -1:
        payload, resume = _boxed_payload_at(text, i)
        if payload:
            out.append(payload)
        i = text.find(_BOXED, max(resume, i + len(_BOXED)))
    return out


def extract_boxed(text: str) -> str | None:
    """Unchanged contract: the LAST \\boxed payload, or None. Models restate,
    so last-not-first. Now implemented on top of extract_all_boxed."""
    matches = extract_all_boxed(text)
    return matches[-1] if matches else None


def to_sympy(s: str | None):
    """CHANGED: no longer swallows every exception. A crash in sympy is a
    GRADER bug and must be distinguishable from a model writing nonsense --
    the old `except (TimeoutError, Exception)` made those two identical, which
    is precisely how a silent grader bug survives to the results table.
    Raises on internal failure; returns None only for genuinely unparseable input."""
    if s is None or not s.strip():
        return None
    with _time_limit(TIMEOUT_SECONDS):
        out = parse(f"${s}$", extraction_config=CFG, raise_on_error=False)
        if not out:
            out = parse(s, extraction_config=CFG, raise_on_error=False)
    return out or None


def grade(raw_output: str, gold: str) -> dict:
    """CHANGED. Returns:
        parsed_answer         str | None
        correct               bool
        box_extraction_status one of schema.BOX_STATUSES
        all_boxed_matches     list[str]
        parse_ok              bool   -- DERIVED, never decided here

    Raises GoldParseError if the GOLD is malformed (a data bug -- it would
    deflate accuracy identically across every condition and hide)."""
    

    def _out(parsed, correct, status, matches):
        return {
            "parsed_answer": parsed,
            "correct": correct,
            "box_extraction_status": status,
            "all_boxed_matches": matches,
            "parse_ok": parse_ok_from_status(status),
        }

    gold_str = extract_boxed(gold) or gold
    try:
        g = to_sympy(gold_str)
    except Exception as e:
        raise GoldParseError(f"gold blew up the parser: {gold!r}") from e
    if g is None:
        raise GoldParseError(f"unparseable gold: {gold!r}")

    if not raw_output or not raw_output.strip():
        return _out(None, False, "empty_output", [])

    matches = extract_all_boxed(raw_output)
    if not matches:
        return _out(None, False, "none_found", [])          # THE 9.5% CASE

    pred_str = matches[-1]
    status = "multiple_found" if len(matches) > 1 else "ok"

    try:
        pred = to_sympy(pred_str)
    except Exception:
        return _out(pred_str, False, "grader_error", matches)
    if pred is None:
        return _out(pred_str, False, "unparseable", matches)  # box present, math malformed

    try:
        with _time_limit(TIMEOUT_SECONDS):
            ok = bool(verify(g, pred, raise_on_error=False))   # gold FIRST
    except Exception:
        return _out(pred_str, False, "grader_error", matches)

    return _out(pred_str, ok, status, matches)


def validate_gold_set(rows, gold_key: str = "gold") -> list:
    """Run over the problem set BEFORE any generation. Returns the bad rows."""
    bad = []
    for r in rows:
        try:
            grade("\\boxed{0}", r[gold_key])
        except GoldParseError:
            bad.append(r)
    return bad