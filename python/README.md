# lockstep-loop for Python

Provider-neutral investigations across stable, read-only environments. The core
shares reasoning until tool observations differ; your application executes all
LLM and tool requests. [Visual guide](../docs/concepts.md) ·
[Use cases](../docs/use-cases.md) · [Technical reference](../docs/history-reference.md).

## Agent entry point

`Agent` is a mutable handle owning a current immutable `AgentSnapshot` and
caller-owned runtime callbacks. `AgentRun` is mutable request/completion state.
Recording is unconditional, including children. No provider clients or storage
backends are included.

```python
from lockstep_loop import Agent, LlmCompletion

agent = Agent(prompt="Explain the checklist.", tool_specs=[])
run = agent.start(["packet-a/v1", "packet-b/v1"])
request = run.next_request()
run.complete(LlmCompletion(request.request_id, "done"))  # mock provider
assert agent.snapshot() is run.snapshot()  # automatic publication, before result()
assert run.result().results == {"packet-a/v1": "done", "packet-b/v1": "done"}

old = agent.snapshot()
cached = agent.start(["packet-c/v1"])
assert cached.done                         # cached final-only start also publishes
assert cached.next_request() is None
assert old.root.env_ids == ("packet-a/v1", "packet-b/v1")
```

For tools, drive the run until `run.done`: pull `next_request("any")`, execute each
`LlmRequest` or `ToolRequest`, then `complete(...)`. Return exactly one tool output
per requested environment. Multiple calls can finish in any order; refinement
uses their complete tuple in assistant call order. `None` from `next_request`
means blocked, not done. Filters `"llm"` and `"tool"` support separate consumers;
`outstanding_llm`, `outstanding_tool`, and `outstanding_count` expose counters.
`RequestStats` reports depth (including reused turns), split depth and issue order.
Request IDs are run-local: route responses to the run that issued them.

**Warning — overlapping runs:** Each start captures the agent's current root.
Their histories are **not merged**. The last run to complete replaces the agent's
current state, potentially excluding experience from other runs. Retain each
run's snapshot to preserve independent results, or finish each run before starting
the next to accumulate experience sequentially. There is no queue or active-run
restriction. Publication happens exactly once on successful completion, not when
`result()` or `snapshot()` is inspected. Inspecting an older run cannot republish it.

Public run advancement is synchronized. The head lock is held only to capture or
replace a root, never across a run's lifetime or while executing callbacks.
`Agent.from_snapshot(agent.snapshot(), ...)` creates an independent handle;
records can be structurally shared without sharing future publications.

## Recording and replay

Sequential `agent.start(environments)` calls learn and reuse paths automatically.
Recorded assistant turns are reused, but pure tools are always re-executed.
Matching observations require zero fresh LLM requests; changed observations branch
at their first divergence, including formerly non-splitting checkpoints.
Old snapshots remain unchanged and reusable.

### Easy persistence: pickle the snapshot

Save and restore the snapshot directly, without repeating prompt/specs or collecting
records yourself:

```python
import pickle

with open("agent-snapshot.pickle", "wb") as file:
    pickle.dump(agent.snapshot(), file)
with open("agent-snapshot.pickle", "rb") as file:
    snapshot = pickle.load(file)  # ONLY trusted pickle data
agent = Agent.from_snapshot(snapshot)
run = agent.start(["packet-a/v2"])
```

The pickle is self-contained for the **entire reachable history**, including full
payloads, every observation branch, feedback and nested child provenance (including
invocation base/result revisions). Export eagerly loads lazy records through the
existing caller loader, visits each address once, and omits unrelated runs/orphan
writes in the same store. It does not add history records or mutate captured roots.
Each load preserves original references and allocates a fresh unique writer, so
independent loads can safely extend history. Save/load/extend/save needs no external
backend or manual merging. Save independent run snapshots separately if needed.

All generic payloads, schemas and arguments must be pickleable; errors propagate
rather than discarding evidence. **Only load trusted pickle data.** Store locks,
loaders, Agent/AgentRun handles and runtime callbacks are not transported. Supply
subagent factories/resolvers and comparison functions again to `from_snapshot`
where needed. This persists history, not pending requests or crash recovery.
For durable files, the caller can write a temporary file and atomically replace
the final path; the core still performs no persistence I/O.

Run the [direct pickle example](examples/snapshot_pickle.py). For large histories,
lazy access or incremental writes, retain the lower-level API below.

### Advanced lazy/incremental persistence

```python
from lockstep_loop import AgentSnapshot, RecordStore

# Caller-owned backend/codecs (integration sketch):
snapshot = agent.snapshot()
for reference, record in snapshot.store.records().items():
    backend.insert_immutable(reference, encode_node(record))
published = encode_descriptor(snapshot.descriptor)  # publish AFTER records

restored = AgentSnapshot(
    decode_descriptor(published),
    RecordStore(load=lambda ref: decode_node(backend.read(ref))),
)
agent = Agent.from_snapshot(restored)  # no node loads; prompt/specs from descriptor
run = agent.start(["packet-a/v2"])
```

`RecordStore` is an append-only write set/cache with an optional synchronous
loader. Reserve a fresh namespace for each durable writer (Python defaults to a
UUID). Persist local writes and retain all referenced old records. The core loads
only the relevant path; it performs no persistence I/O. Pickle examples use trusted
bytes only. Do not pickle live handles/callbacks or Asset views. There is no stable
cross-language/version wire format or legacy migration layer.

Full messages, tool calls/arguments and exact per-environment outputs are retained.
Payloads, schemas and arguments are **borrowed immutable-by-contract values**,
including nested contents and all aliases; the library does not deep-copy them.

### Comparison keys and compatibility

Configure `comparison_key` on `Agent(...)` or `Agent.from_snapshot(...)`.
Default identity requires hashable outputs. A custom function returns hashable
keys, e.g. `str.strip`. It is applied to **both** current outputs and historical
representatives, with no function-identity/default-vs-custom gate. Raw evidence
is never replaced by keys. Stricter keys can branch despite matching an old
evidence item; coarser keys choose the first matching stored edge without merging.
Replay retains that edge's historical representative in model context.

Structural checks cover prompts, ordered tool specs, nested specs/allowlists/depth,
and actual child factory outputs. Provider/adapter and grouping-policy compatibility
remain caller-owned. No manual contract/version key or callback registry exists.

### Feedback, budgets and failures

```python
from lockstep_loop import Budget, Feedback

run = agent.start(["packet-a/v2"], feedback=Feedback("Also check exclusions."),
                  budget=Budget(max_steps_per_branch=20))
```

Feedback first resolves an existing completed path through tool replay without new
LLM work. It appends one user message and resumes. Every environment reaching that
path in the new snapshot inherits the continuation, not just the resolving
environment. Old snapshots and unrelated paths stay unchanged. Use transferable
instructions, not environment-specific assertions. `Feedback(None)` is a real null
intervention; `feedback=None` means absent. Repeated explicit interventions append
again at the latest completed endpoint. There are no public endpoint IDs.

Budgets count accepted fresh **and reused** assistant turns per branch. Exhaustion
returns a string fallback in `results`, never an assistant fact. Successful budget
completion publishes the real checkpoint; later replay with a larger/no budget
can resume it. Child fallbacks are actual parent tool observations, not child
assistant records. Exhaustion during feedback resolution is a failure.

Invalid completions and Python callback exceptions leave pending work correctable.
Child preparation and cached rollup occur transactionally before publication.
`FeedbackResolutionError`/`InvalidHistoryError` indicate terminal failures;
`done` becomes true and `result()` raises. Failures and abandoned runs **do not
replace the owner's head**. `run.snapshot()` may expose validated partial history,
not pending requests or a crash-recovery protocol. Caller side effects, warmed
caches and unreachable append-only writes are not rolled back.

### Specialized children and Asset views

Configure `subagents=[SubagentTool(...), ...]` and optional `agent_resolver` on the
Agent. Reattach matching subagent definitions on `from_snapshot`; executable
prompt/spec factories are never serialized. `AgentQuery(prompt, tools, subagents)`
selects a child `AgentDescriptor` or None. The shared store must load its records.
Queries do not include parent identity, environment IDs or grouping mode.
Recorded parent invocations pin their child revision; new invocations may consult
the caller's cache. Child tools rerun and equal child answers need not split parents.
Nested allowlists and depth caps remain enforced.

`result.snapshot` is the run's immutable final root. `result.child_updates` contains
separate `ChildAgentUpdate` records with a **snapshot**, selected `base`, tool call,
environments and invocation provenance. Concurrent child forks are not merged or
automatically published to an external cache; the caller selects future revisions.

```python
snapshot = agent.snapshot()
for asset in snapshot.assets():
    print(asset.env_ids, asset.is_complete, asset.messages)
    if asset.has_value:
        print(asset.value)
    # asset.observations retains representatives AND exact per-environment evidence
```

Assets are immutable, lazy, path-backed views anchored in that snapshot. Subsequent
Agent publications do not retarget them. An open continuation can retain an older
published value; a real None differs from no value. `asset.snapshot` is its owner.
Inspect children separately with `snapshot.child(entry).assets()`.

## Executable examples and convenience drivers

From this directory, no paid APIs or credentials required:

```sh
for example in product_screening snapshot_pickle history_replay feedback specialized_subagents agent_assets; do
    PYTHONPATH=src python examples/$example.py
done
```

[Product screening](examples/product_screening.py) reuses one mutable handle across
sequential runs. [Snapshot pickle](examples/snapshot_pickle.py) is the easy persistence
path; [history replay](examples/history_replay.py) demonstrates advanced lazy/incremental
storage. [Feedback](examples/feedback.py), [children](examples/specialized_subagents.py)
and [assets](examples/agent_assets.py) cover the other boundaries.

The one-shot `run_batch(prompt=..., env_ids=..., tool_specs=..., ...)` generator and
`run_to_completion(prompt=..., env_ids=..., tools=..., complete_llm=..., ...)` local
tool driver construct Agents internally and return `BatchResult` with snapshots.
Use `Agent.start` for repeated investigations, feedback and child-cache selection.
The underlying `core.BatchLoop` is internal execution machinery, not a competing
entry point.

## Limits and development

Synchronous loaders; no GC/eviction, automatic root merge, in-flight crash recovery,
DAG scheduling, designated result-tool publication or UI. Deep path copying can
produce quadratic writes; observation lookup is linear. Valid imported graphs and
payload codecs are caller responsibilities. See the [full reference](../docs/history-reference.md).

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
PYTHONPATH=src python benchmarks/split_completion.py --repeats 3
```
