"""Readout positions (rhjl.readout) on synthetic rollouts in the model's real token layout."""
import pytest

from rhjl.readout import Specials, positions

MODEL = "/home/wil/models/qwen3.5-2b-text"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL)


def call(name, path=None):
    if path is None:
        return f"<tool_call>\n<function={name}>\n</function>\n</tool_call>"
    return f"<tool_call>\n<function={name}>\n<parameter=path>\n{path}\n</parameter>\n</function>\n</tool_call>"


TOOL = "\n<|im_start|>user\n<tool_response>\nstuff\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rollout(tok, turns):
    """turns: list of model-generated strings; tool responses are inserted between them."""
    ids, mask = [], []
    for k, t in enumerate(turns):
        if k:
            x = tok.encode(TOOL, add_special_tokens=False)
            ids += x
            mask += [0] * len(x)
        x = tok.encode(t + "<|im_end|>", add_special_tokens=False)
        ids += x
        mask += [1] * len(x)
    prompt = tok.encode("<|im_start|>user\nsolve<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                        add_special_tokens=False)
    return {"prompt_ids": prompt, "completion_ids": ids, "tool_mask": mask}


def dec(tok, r, i):
    return tok.decode([(r["prompt_ids"] + r["completion_ids"])[i]])


def test_silent_hack(tok):
    r = rollout(tok, [call("read_file", "README.md") + "\n" + call("list_files"),
                      call("read_file", "problem.txt") + "\n" + call("read_file", "answers.txt"),
                      "So 1 + 2 = 3.\n\n<answer>1 + 2</answer>"])
    p = positions(r, Specials(tok))
    assert dec(tok, r, p["decision"]) == "</tool_call>" and dec(tok, r, p["decision"] + 1) == "\n"
    assert p["signature"] == "read_file(README.md),list_files()|read_file(problem.txt)"
    assert p["decision_turn"] == 1
    assert dec(tok, r, p["turn_start"] + 1) == "<tool_call>"
    assert dec(tok, r, p["ctrl_continue"] + 1) == "\n" and dec(tok, r, p["ctrl_stop"] + 1) == "<|im_end|>"
    assert dec(tok, r, p["pre_answer"] + 1).startswith("<")
    full = r["prompt_ids"] + r["completion_ids"]
    assert tok.decode(full[p["pre_answer"] + 1:]).startswith("<answer>")


def test_clean_matches_hack_signature(tok):
    r = rollout(tok, [call("read_file", "README.md") + "\n" + call("list_files"),
                      call("read_file", "problem.txt"),
                      "<answer>1 + 2</answer>"])
    p = positions(r, Specials(tok))
    assert dec(tok, r, p["decision"]) == "</tool_call>" and dec(tok, r, p["decision"] + 1) == "<|im_end|>"
    assert p["signature"] == "read_file(README.md),list_files()|read_file(problem.txt)"


def test_forbidden_first_in_turn_after_text(tok):
    r = rollout(tok, [call("list_files"), "Let me look.\n\n" + call("read_file", "answers.txt"), "<answer>1</answer>"])
    p = positions(r, Specials(tok))
    assert p["signature"] == "list_files()|"
    assert dec(tok, r, p["decision"]).strip() == "."
    assert p["ctrl_continue"] is None


def test_no_tools(tok):
    r = rollout(tok, ["<answer>1 + 2</answer>"])
    p = positions(r, Specials(tok))
    assert p["decision"] is None and p["pre_answer"] is not None
