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

## Recording and replay

Enable history on the direct loop API; existing constructors without these
options retain their behavior:

```python
loop = BatchLoop(
    prompt="inspect", env_ids=["revision-1"], tool_specs=specs,
    history_contract="inspect-v1/tool-semantics-v1/exact-grouping/provider-v1",
)
# Drive the ordinary requests/completions as above.
snapshot = loop.history             # frozen root, including fulfilled turns
# After completion: snapshot = loop.result().history
next_loop = BatchLoop(
    prompt="inspect", env_ids=["revision-2"], tool_specs=specs,
    history_contract="inspect-v1/tool-semantics-v1/exact-grouping/provider-v1",
    history=snapshot,
)
```

`next_loop` may immediately emit tools (or already be done for a recorded final
answer). Always use `done`, not the assumption that the first request is an LLM
request. `stats.depth` counts reused assistant turns too; `split_depth` describes
only the new execution's partition. Independent calls can finish in any order.

Run the complete, executable [caller-side example](examples/history_replay.py):

```sh
PYTHONPATH=src python examples/history_replay.py
# unchanged: 0 LLM calls; changed: 1; both paths reusable
```

It records a non-splitting tool turn, serializes/restores history with `pickle`,
replays a new environment without LLM work, and branches on changed observations.
Pickle is only for **trusted** data. You can instead encode the public frozen
`History`, `HistoryNode`, `Observation`, `EnvObservation` and `OutputRecord`
records with your own payload codec. Core code performs no storage I/O.

Inspect `snapshot.root.assistant`, `root.observations[i].representative`,
`observations[i].evidence[j].outputs[k].tool_output`, and each observation's
`continuation`. Output order matches the assistant's call order. Full payloads
and call arguments survive; grouping keys never replace them. With `group_by`,
actual per-environment outputs remain separate from the historical representative
used in the model conversation. Snapshot extension uses immutable path copying,
not mutation of previously captured roots.

**Compatibility:** replay requires the same explicit non-empty contract, equal
prompt, equal ordered tool specs, and the same default/custom grouping mode.
Version the contract for changes to grouping/tool semantics or provider/adapter
settings. Equality cannot detect a changed implementation behind an unchanged
schema. Payloads, arguments and schemas are **borrowed immutable values**: do not
mutate nested contents through any alias after handing them to the loop. Convert
mutable provider objects to stable values or make your own copies first.

**Limits:** direct `BatchLoop` only; recording rejects subagents and budgets.
History must be fully loaded: lazy storage loading is deferred. Partial tool
collections/queues are builders, not persisted crash-recovery state. Valid
persisted histories are trusted, with no promised cross-version wire format.
Validation failures leave requests outstanding for correction; duplicate fulfilled
completions cannot overwrite history. Branch IDs are run-local (now collision-free
counters rather than truncated hashes); Python split-edge labels are likewise
opaque identifiers. No feedback/assets/DAG scheduling is included. See the
[shared design and limitations](../README.md#immutable-recorded-history-and-incremental-replay).

## Development

From this `python/` directory:

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
