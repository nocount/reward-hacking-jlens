"""Multi-turn tool-use rollouts with plain vLLM, token-for-token like TRL's GRPO tool loop.

TRL (1.18) builds a multi-turn completion by concatenating token IDs: prompt + generated turn +
tool-result suffix + next generated turn ..., where the suffix is cut from a rendered dummy
conversation. This module does the same so rollouts sampled here (base-rate probes, Phase 3
probe datasets) have exactly the token sequences the policy saw during training.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from trl.chat_template_utils import add_response_schema, parse_response

from .hackenv import CountdownFS, build_prompt, label


def tool_suffix_ids(tok, tool_messages: list[dict], chat_kwargs: dict) -> list[int]:
    """Token IDs that append tool results and the next assistant header (as TRL's `_get_tool_suffix_ids`)."""
    dummy = [{"role": "user", "content": "dummy"},
             {"role": "assistant", "content": "",
              "tool_calls": [{"type": "function", "function": {"name": tool_messages[0]["name"], "arguments": {}}}]}]
    pre = tok.apply_chat_template(dummy, add_generation_prompt=False, tokenize=True, return_dict=False, **chat_kwargs)
    full = tok.apply_chat_template(dummy + tool_messages, add_generation_prompt=True, tokenize=True,
                                   return_dict=False, **chat_kwargs)
    eos = [i for i, t in enumerate(pre) if t == tok.eos_token_id]
    if eos:
        pre = pre[: eos[-1] + 1]
    assert full[: len(pre)] == pre, "chat template is not prefix-preserving"
    return full[len(pre):]


def run_tool(env: CountdownFS, call: dict) -> tuple[str, object]:
    """Execute one parsed tool call the way TRL does: exceptions become {"error": ...} results."""
    fn = call.get("function", {})
    name = fn.get("name", "unknown")
    tools = {"list_files": env.list_files, "read_file": env.read_file}
    try:
        if name not in tools:
            raise ValueError(f"Tool {name} not found.")
        return name, tools[name](**(fn.get("arguments") or {}))
    except Exception as e:  # noqa: BLE001 - mirror TRL
        return name, {"error": str(e)}


@dataclass
class Rollout:
    row: dict
    messages: list[dict] = field(default_factory=list)  # completion messages (assistant/tool), TRL format
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    tool_mask: list[int] = field(default_factory=list)  # 1 = model token, 0 = tool-result token
    truncated: bool = False
    env: CountdownFS | None = None
    lab: dict = field(default_factory=dict)


def rollouts(llm, tok, rows: list[dict], variant: str, n: int, sampling, max_completion: int,
             max_tool_iters: int = 3, enable_thinking: bool = False, grader: str = "strict",
             min_words: int = 0) -> list[Rollout]:
    """Sample n multi-turn rollouts per row ({nums, target}). `sampling` is a vLLM SamplingParams (n is ignored)."""
    tok = add_response_schema(tok)
    chat_kwargs = {"enable_thinking": enable_thinking}
    sp = sampling.clone()
    sp.n = 1
    outs: list[Rollout] = []
    for row in rows:
        for _ in range(n):
            env = CountdownFS()
            env.reset(**row)
            r = Rollout(row=row, env=env)
            msgs = build_prompt(row["nums"], row["target"], variant)
            r.prompt_ids = tok.apply_chat_template(msgs, tools=[env.list_files, env.read_file],
                                                   add_generation_prompt=True, tokenize=True,
                                                   return_dict=False, **chat_kwargs)
            outs.append(r)

    active = list(range(len(outs)))
    for it in range(max_tool_iters + 1):
        if not active:
            break
        reqs = []
        for i in active:
            r = outs[i]
            budget = max_completion - len(r.completion_ids)
            s = sp.clone()
            s.max_tokens = budget
            reqs.append(({"prompt_token_ids": r.prompt_ids + r.completion_ids}, s))
        gens = llm.generate([q for q, _ in reqs], [s for _, s in reqs], use_tqdm=False)
        nxt = []
        for i, g in zip(active, gens):
            r, o = outs[i], g.outputs[0]
            ids = list(o.token_ids)
            prefix = r.prompt_ids + r.completion_ids
            msg = parse_response(tok, ids, prefix=prefix, tools=[r.env.list_files, r.env.read_file])
            r.completion_ids += ids
            r.tool_mask += [1] * len(ids)
            r.messages.append(msg)
            if o.finish_reason == "length":
                r.truncated = True
                continue
            calls = msg.get("tool_calls")
            if not calls or it == max_tool_iters:
                continue
            tool_msgs = []
            for c in calls:
                name, res = run_tool(r.env, c)
                tool_msgs.append({"role": "tool", "name": name,
                                  "content": str(res)})  # TRL: str(result)
            suffix = tool_suffix_ids(tok, tool_msgs, chat_kwargs)
            if len(r.completion_ids) + len(suffix) >= max_completion:
                continue  # TRL drops a tool result that would overflow the budget
            r.messages += tool_msgs
            r.completion_ids += suffix
            r.tool_mask += [0] * len(suffix)
            nxt.append(i)
        active = nxt

    for r in outs:
        r.lab = label(r.env, r.messages, grader=grader, min_words=min_words)
    return outs
