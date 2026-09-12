# lockstep-loop

Run a single agent prompt across thousands of similar, read-only environments
without generating the same token twice.

## How it works

- All environments start in one equivalence class sharing **one** LLM
  conversation.
- When the model makes a tool call, the call is evaluated on **every**
  environment still in the class (SIMD-style fan-out).
- Results are compared by value equality (optionally canonicalized with a
  caller-provided `group_by`) and the class is refined — one
  sub-conversation per distinct result. Branches then advance
  independently as their completions arrive.
- The process repeats per branch until done, yielding a prefix-sharing trie
  whose leaves are **equivalence classes of environments**: cost scales with
  the number of distinct tool-call traces, not the number of environments.

Two invariants make this sound: tools must be **pure** (results depend only
on environment + arguments), and the model must never see which environment
it is in (tool specs identical across environments).

## Status

Working prototype. The core is a provider-agnostic orchestrator: `BatchLoop`
owns partition refinement and the trie; the caller owns all LLM and tool
execution. Pull work with `next_request(kind)` — `"llm"`, `"tool"`, or
`"any"` — serve it however you like (any order, any concurrency), and feed
results back with `complete()` as they arrive, so a slow provider or tool
delays only its own branch. Requests carry a `stats` block (depth, split
depth, env count, sequence) with facts the caller can compose into a serving
policy, e.g. "deepest currently available branch first" for prompt-cache
warmth. All methods are internally synchronized, so separate LLM and tool
consumer loops can share one `BatchLoop`. A generator adapter (`run_batch`)
remains for pump-style drivers. Payloads are opaque: the loop is generic in
the content type, so multimodal conversations (content blocks, image bytes)
work unchanged — tool results only need to be hashable for value-equality
grouping.

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

For parallel serving, drive `BatchLoop` yourself:

```python
loop = BatchLoop(prompt=..., env_ids=..., tool_specs=[...])
while not loop.done:
    if (req := loop.next_request("any")) is None:
        await any_completion_arrives()  # outstanding work, nothing runnable
    else:
        spawn_somewhere(req)
# elsewhere, as completions arrive: loop.complete(completion)
result = loop.result()
```

## Development

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
