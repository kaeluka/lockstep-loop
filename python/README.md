# lockstep-loop for Python

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
delays only its own branch. Both request kinds carry their env set (the
branch's equivalence class), and requests carry a `stats` block (depth,
split depth, sequence) with facts the caller can compose into a serving
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

## Subagents and inherited sessions

`SubagentTool` runs a nested lockstep loop over the calling branch's environments.
By default its conversation starts fresh with `user(build_prompt(args))`.
Set `inherit_session=True` to prepend the parent's conversation **before the
calling assistant turn**:

```python
from lockstep_loop import SubagentTool, ToolSpec

fork = SubagentTool(
    name="fork",
    description="Investigate a task using the current conversation context.",
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    build_prompt=lambda args: args["task"],
    sub_tool_specs=lambda env: [
        ToolSpec("read_file", "Read a file", {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }),
    ],
    inherit_session=True,
)
# Include fork in the tools factory used by run_to_completion, or pass
# subagent=fork to BatchLoop and include its ToolSpec in tool_specs.
```

If the model calls `fork(task="do x")` and `fork(task="do y")` in one turn,
children start with the same inherited prefix plus `user("do x")` or
`user("do y")`. They do not inherit the calling turn, sibling calls, or sibling
results. Child tools still come from `sub_tool_specs`; inheritance copies
conversation messages, not tool configuration or provider-side session state.
Previously completed tool turns remain in the inherited history.

The same driver serves child requests, which may finish in any order. The parent
waits for all calls in its turn (including ordinary tools), then records one tool
result per call in original call order. Each child's final per-environment answer
is its tool result; parent environments are grouped by the tuple of those results.
Child transcripts are not added to the parent. `max_depth` limits recursive
spawning, and descendants retain the inheritance setting. Prefix caching and
model selection remain the caller/provider's responsibility.

## Development

From this `python/` directory:

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
