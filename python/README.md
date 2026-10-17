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

It records a non-splitting turn, persists **individual shallow records** plus a
`HistoryDescriptor`, lazily restores, and branches on changed observations. It
publishes records before descriptors, checks insert-only collisions, and reuses
both new and old published roots. Pickle is only for **trusted** data. Substitute
your own codecs/backend; do not pickle a live `History`/resolver.

```python
# Save each record with your own codec, then publish the descriptor atomically:
for reference, record in snapshot.store.records().items():
    backend.insert_immutable(reference, encode_node(record))
published = encode_descriptor(snapshot.descriptor)

# Fresh writer namespace (UUID default); no traversal or matching in this callback:
store = RecordStore(load=lambda ref: decode_node(backend.read(ref)))
restored = History(decode_descriptor(published), store)
```

`RecordStore(namespace=..., load=...)` is an in-memory append-only write set/cache,
not a backend. `BatchLoop(record_store=store, history_contract=...)` uses a chosen
store for a new recording; replay uses the store attached to `History`. Allocate
a globally unique namespace per persistent writer. New records use exact
`NodeRef(namespace, index)` addresses, never hashes. Persist only local writes;
the new root may still reference old backend records. Loader errors/missing or
corrupt records are fatal, not completion-validation errors. See the
[complete persistence protocol](../README.md#individually-addressable-records-and-lazy-persistence).

Inspect `snapshot.root.assistant`, `root.observations[i].representative`, and
`observations[i].evidence[j].outputs[k].tool_output`. Load a continuation with
`snapshot.load(edge.continuation)`; its `NodeRef` is only a storage address.
Output order matches call order. Full payloads/arguments survive; custom grouping
keeps exact per-environment evidence separate from historical representative
context. Snapshot extension path-copies records, leaving prior roots unchanged.

**Subagents are recorded and replayed**, including mixed ordinary/subagent turns,
concurrent child calls and configured nesting depth. `node.subagents` entries
contain the assistant call ID and a child descriptor: inspect one with
`History(entry.history, snapshot.store)`. Child tool observations are rerun; only
child final answers enter the parent's result tuple. Divergent child traces with
identical answers do not split the parent. Active child snapshots retain fulfilled
turns, not partial tool collections.

**Compatibility:** replay requires the same explicit non-empty contract, equal
prompt, equal ordered tool specs, the same default/custom grouping mode and equal
subagent spec/depth. Version the contract for changes to subagent callbacks,
grouping/tool semantics or provider/adapter
settings. Equality cannot detect a changed implementation behind an unchanged
schema. Payloads, arguments and schemas are **borrowed immutable values**: do not
mutate nested contents through any alias after handing them to the loop. Convert
mutable provider objects to stable values or make your own copies first.

**User continuations:** resolve a snapshot against **one stable environment** by
ordinary replay, then call `resolved_loop.append_user_message(content)`. The
returned new snapshot preserves the old assistant answer and adds a
`UserContinuation(message, continuation)` pointing to an open `HistoryNode`.
The prior snapshot/other investigations are unchanged, even if they published the
same answer. No public session ID or endpoint handle is introduced. The record
model permits later tools, the same answer again, and more user messages.
Selection/inheritance across environments is **not decided**: descriptors with
user continuations have `requires_user_policy=True` and replay raises explicitly
rather than silently inheriting or ignoring feedback. This is representation and
append support, not feedback routing/execution.

**Limits:** direct `BatchLoop` only; recording still rejects budgets. No core
storage I/O, async loader, eviction/GC, independent-root merge, or crash recovery.
Stores retain local writes and cached loads; deep path copying can have quadratic
write volume. Pending tool collections/queues are not persisted. Valid imported
records are trusted; no stable cross-version wire format is promised. Invalid
completions leave requests correctable, including nested routes. Branch IDs use
collision-free run-local counters; legacy tree labels retain `str(result_tuple)`.
Python mapping label collisions append ` [collision N]` without dropping groups
or changing noncolliding labels. Assets/DAG scheduling/UI remain out of scope.
See the [shared design and limitations](../README.md#immutable-recorded-history-and-incremental-replay).

## Development

From this `python/` directory:

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
