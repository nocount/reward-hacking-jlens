"""Countdown task: given numbers and a target, write an arithmetic expression that hits the target.

Phase 0 uses the honest (non-hackable) version only. Data: Jiayi-Pan/Countdown-Tasks-3to4, the set
used by TinyZero and GRAFT's Countdown-Code.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from fractions import Fraction

DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"

PROMPT = (
    "Using the numbers {nums}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number must be "
    "used exactly once. Show your final answer as just the expression inside <answer></answer> "
    "tags, for example <answer>(1 + 2) / 3</answer>."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def build_messages(nums: list[int], target: int) -> list[dict]:
    return [{"role": "user", "content": PROMPT.format(nums=list(nums), target=target)}]


def strip_thinking(completion: str) -> str:
    """Drop the reasoning block, so <answer> tags inside the thinking don't count."""
    return completion.split("</think>", 1)[1] if "</think>" in completion else completion


def extract_answer(completion: str, thinking: bool = False) -> str | None:
    """Last <answer>...</answer> after the thinking block; None if missing or thinking unfinished.

    thinking=True means the prompt already opened a <think> block (Qwen3.5 enable_thinking), so
    the completion must contain </think> to count.
    """
    if ("<think>" in completion or thinking) and "</think>" not in completion:
        return None
    found = _ANSWER_RE.findall(strip_thinking(completion))
    if not found:
        return None
    ans = found[-1].strip()
    if "=" in ans:  # tolerate "expr = target"
        ans = ans.split("=")[0].strip()
    return ans


_ALLOWED_BINOPS = {ast.Add, ast.Sub, ast.Mult, ast.Div}


def _eval(node) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        lv, ln = _eval(node.left)
        rv, rn = _eval(node.right)
        if isinstance(node.op, ast.Add):
            v = lv + rv
        elif isinstance(node.op, ast.Sub):
            v = lv - rv
        elif isinstance(node.op, ast.Mult):
            v = lv * rv
        else:
            if rv == 0:
                raise ZeroDivisionError
            v = lv / rv
        return v, ln + rn
    raise ValueError(f"disallowed syntax: {ast.dump(node)[:80]}")


def evaluate(expr: str) -> tuple[Fraction, list[int]] | None:
    """Exact rational value of an expression and the integer literals it uses; None if invalid."""
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    if len(expr) > 200 or not re.fullmatch(r"[\d\s+\-*/().]+", expr):
        return None
    try:
        return _eval(ast.parse(expr, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return None


def score(completion: str, nums: list[int], target: int, thinking: bool = False) -> dict:
    """Honest grader: correct iff the expression uses exactly the given numbers and equals target."""
    ans = extract_answer(completion, thinking)
    res = evaluate(ans) if ans is not None else None
    uses_nums = res is not None and Counter(res[1]) == Counter(nums)
    hits = res is not None and res[0] == target
    return {"answer": ans, "formatted": ans is not None, "valid_expr": res is not None,
            "uses_nums": uses_nums, "hits_target": hits, "correct": bool(uses_nums and hits)}


def solve(nums: list[int], target: int) -> str | None:
    """One expression using every number exactly once that equals target (exact arithmetic), or None.

    Exhaustive search over pairwise combinations, which covers every binary expression tree.
    Fast for the 3-4 number problems in the dataset.
    """
    def rec(items: list[tuple[Fraction, str]]) -> str | None:
        if len(items) == 1:
            return items[0][1] if items[0][0] == target else None
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                (a, ea), (b, eb) = items[i], items[j]
                rest = [items[k] for k in range(len(items)) if k not in (i, j)]
                cands = [(a - b, f"{ea} - {eb}"), (a / b, f"{ea} / {eb}")] if b != 0 else [(a - b, f"{ea} - {eb}")]
                if i < j:  # commutative ops once per pair
                    cands += [(a + b, f"{ea} + {eb}"), (a * b, f"{ea} * {eb}")]
                for v, e in cands:
                    e = e if len(rest) == 0 else f"({e})"
                    found = rec(rest + [(v, e)])
                    if found:
                        return found
        return None

    return rec([(Fraction(n), str(n)) for n in nums])
