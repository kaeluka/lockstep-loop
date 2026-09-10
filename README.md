# lockstep-loop

Run a single agent prompt across thousands of similar, read-only environments
without generating the same token twice.

## How it works

- All environments start in one equivalence class sharing **one** LLM
  conversation.
- When the model makes a tool call, the call is evaluated on **every**
  environment still in the class (SIMD-style fan-out).
- Results are canonicalized (`group_by`) and the class is refined — one
  sub-conversation per distinct result key. Branches then advance
  independently as their completions arrive.
- The process repeats per branch until done, yielding a prefix-sharing trie
  whose leaves are **equivalence classes of environments**: cost scales with
  the number of distinct tool-call traces, not the number of environments.

Two invariants make this sound: tools must be **pure** (results depend only
on environment + arguments), and the model must never see which environment
it is in (tool specs identical across environments).

## Status

Working prototype. The core is provider-agnostic: `run_batch` is a generator
that yields `LlmRequest` / `ToolRequest` objects and receives `LlmCompletion`
/ `ToolCompletion` answers via `.send()` — out of order, as they arrive, so a
slow provider or tool delays only its own branch. The caller owns all LLM and
tool execution (and thus all scheduling); the library owns partition
refinement and the trie. Requests carry a `stats` block (depth, split depth,
env count, sequence) with facts the caller can compose into a serving policy,
e.g. "deepest currently available branch first" for prompt-cache warmth.

For the common serial case there is a convenience driver:

```python
from lockstep_loop import LlmCompletion, run_to_completion

def complete_llm(req):  # the only provider seam
    resp = my_provider.chat(messages=req.messages, tools=req.tools)
    return LlmCompletion(request_id=req.request_id, content=resp.text,
                         tool_calls=resp.tool_calls)

result = run_to_completion(
    prompt="summarize what src/pkgA contains",
    env_ids=["fork-1", "fork-2", "fork-3"],
    tools=make_tools,          # env_id -> tool impls (specs must be identical)
    complete_llm=complete_llm,
)
# result.tree, result.classes, result.results (env_id -> final message)
```

For parallel serving, pump `run_batch` yourself: prime with `next(gen)`,
deliver completions with `gen.send(...)` in arrival order, drain queued
requests with `gen.send(None)`; the loop ends when `send` raises
`StopIteration` whose `value` is the `BatchResult`.

## Development

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
