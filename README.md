# lockstep-loop

Run a single agent prompt across thousands of similar, read-only environments
without generating the same token twice.

This repository contains two implementations of the same provider-agnostic
partition-refinement loop:

| Implementation | Location | Package |
| --- | --- | --- |
| Python ≥ 3.10 | [`python/`](python/) | `lockstep-loop` |
| Rust | [`rust/`](rust/) | `lockstep-loop` |

Neither implementation calls an LLM or owns infrastructure. The caller pulls
provider-neutral LLM and tool requests, executes them with any scheduler or
provider, and returns completions in any order.

## The idea

- All environments begin in one equivalence class and share one conversation.
- A model tool call fans out over every environment still in that class.
- Environments with equal tool-result tuples stay together; differing results
  split the class into independent conversations.
- Multiple tool calls in one assistant turn are independent requests and may
  execute concurrently; refinement waits for the whole turn.
- A reserved subagent tool can spawn a nested lockstep loop and fold its final
  per-environment answers back into the parent turn.
- The result is a prefix-sharing split trie plus the final equivalence classes.

Cost therefore scales with the number of distinct tool-result traces rather
than directly with the number of environments.

The soundness requirements are the same in both implementations: tools must be
read-only/pure, tool specifications must be identical across environments, and
the model must not be shown environment identities. Requests do expose their
environment set to the caller for scheduling, accounting, and rate limiting.

## Immutable recorded history and incremental replay

Both implementations now offer **opt-in recording on `BatchLoop`**. Keep a
`History` snapshot, then start a new loop with that snapshot and new environment
IDs. The core reuses recorded assistant turns, emits their tool requests again,
and follows the matching **ordered tuple of observations**. Only an unseen
continuation requests fresh LLM work. This includes checkpoints that never split
the original environment class. Old snapshots and old paths remain reusable.

The history model is the replay engine's model, not a lossy export:

- `History` pairs a small `HistoryDescriptor` (prompt, tool specifications, reuse
  contract and root reference) with a `RecordStore` resolver.
- `HistoryNode` stores the full assistant message (including tool calls/arguments)
  and the environment visits that used it. An absent assistant is an unfulfilled
  continuation, not a terminal answer.
- Each `Observation` stores ordered representative `OutputRecord`s, **all exact
  per-environment `EnvObservation` evidence**, and a continuation `NodeRef`. Each
  output record has a `tool_output` field for future metadata extensions.
- Conversation reconstruction starts with the prompt and alternates assistant
  messages and representative tool messages. With custom lossy grouping, replay
  keeps the **original representative conversation**, while appending the new
  environment's actual, possibly unequal outputs to evidence.
- `HistoryNode.subagents` retains a full child history descriptor per assistant
  call ID, including children still open when a snapshot is captured. Children
  use the same shallow records/resolver and may themselves invoke subagents up
  to the configured depth (the reserved spec is omitted at the depth limit).
  Replay reruns child tools; child trace splits do NOT
  split parents when their folded answers match. Ordinary and subagent calls
  may coexist in a turn and complete independently.
- Nodes are immutable values: Python uses frozen dataclasses/tuples; Rust stores
  immutable `Arc` records. Updates allocate new records along the ancestor path
  and retain unchanged references. Snapshots are available during a run and in
  the result. The old `result.tree` remains a per-run partition summary, **not
  replay history**. Its labels retain result-group representations, not IDs.

Matching verifies equality of complete output tuples (or caller-defined group
keys); hashes/debug strings are never record identities or substitutes for
outputs. Branch IDs are run-local scheduling identifiers. Repeated environment
IDs append visits/evidence rather than overwriting them; use revision-qualified
IDs if you need to distinguish runs in an inspector.

### Compatibility and ownership

Replay requires an explicit, non-empty caller-managed **contract version**. The
core rejects a changed contract, unequal prompt, changed ordered tool specs, or
a switch between default and custom grouping, or changed subagent spec/depth.
Child prompt/spec contracts are checked when the child is entered. The caller
must also change the contract when tool implementation **semantics**, subagent
prompt/spec-building callbacks, grouping semantics, provider
settings, or adapter behavior change: these cannot be inferred from closures or
specs. A new repository revision is an environment change, not a tool semantic
change. Custom grouping must be pure, deterministic, and use stable equality/
hashing. It deliberately permits lossy reuse. Never use environment IDs in the
model-visible prompt, arguments, schemas or tool responses.

**Payload ownership is part of the API contract.** Python borrows payloads,
argument mappings and schemas without deep copying: callers must treat their
entire reachable contents as immutable from submission onward, including aliases
obtained through requests/snapshots. Rust `P::clone()` must preserve immutable
value semantics; interior-mutable/shared payload aliases must not change recorded
facts. Frozen wrappers/`Arc` do not enforce either promise. Use immutable payload
representations or caller-owned copies/codecs. Equality must represent the prompt
compatibility you intend; Rust replay's config builder requires `P: PartialEq`.

### Individually addressable records and lazy persistence

The core performs **no filesystem, database or network I/O**. `RecordStore` is an
append-only in-memory write set/cache with an optional **caller loader**:

1. Allocate a fresh writer namespace. New `NodeRef(namespace, index)` addresses
   are monotonic within that writer, NOT content hashes or conversation IDs.
2. Record normally; `history.store.records()` yields the new immutable shallow
   records, including all nested child records. Persist each independently with
   your codec, using insert-only semantics. Reject an address collision rather
   than overwriting another record. Retain records reachable from old snapshots.
3. After records are durable, atomically publish `history.descriptor`. Never
   publish a descriptor before its records. A failed publication may leave orphan
   records; caller-owned garbage collection can remove them later.
4. Load only the descriptor; attach a new `RecordStore` with a fresh writer
   namespace and a `load(NodeRef) -> HistoryNode` callback. The **core** traverses
   and matches observations, including child paths. The caller only loads one
   immutable record by address and decodes it. Unselected subtrees are not loaded.
5. Persist the new store's writes before publishing the extended descriptor;
   already-stored records are reused by reference, not rewritten by the caller.

Rust descriptors/records derive Serde when `P` supports it; Python's example uses
trusted pickle **per record**, not a pickled live resolver. The same record model
is used in memory and in storage. Neither history depth nor nested agent depth
creates recursive node encoding. Tests round-trip **256 sequential tool turns
on two paths** with normal JSON recursion limits, then load only the selected
257 nodes, publish new paths and reuse old descriptors. Payload codecs themselves
still need to support the payloads' nesting/size.

Writer namespaces must be reserved uniquely across a persistent storage domain
(e.g. caller UUIDs or a database sequence). This is a **storage identity**, never a
public conversation identity. Python defaults to a fresh UUID. Rust's default
store names are process-local conveniences: durable callers must supply a store
with a reserved namespace via `with_record_store`. Restored stores always need a
fresh writer namespace; sharing/cloning one store is safe, reusing its namespace
for another independent writer is not. Examples enforce insert-only writes; the
core also rejects collisions with already-loaded addresses. References resolve
by exact namespace/index equality; hashes/debug strings never substitute for
record or observation equality.

Loaders are synchronous and must resolve valid immutable records without
re-entering the store. Missing/corrupt records and loader exceptions/panics are
infrastructure failures: retry from a saved descriptor with a healthy loader.
Runtime transaction guarantees below are not a storage-recovery protocol. Arbitrary imported graphs
are not validated; restore valid snapshots from this version. No stable
cross-language/version wire format is promised. Stores currently retain all
writes and loaded records; no eviction/GC is built in. Path copying can produce
quadratic write volume on deep paths, and observation-edge lookup is linear.
These are explicit performance limits, not an eager-loading requirement.

### Feedback: shared-path inheritance in the new snapshot

Feedback improves an **investigation**, not an environment ID or answer value.
Start a new loop with a compatible **history snapshot + exactly one stable
environment + explicit feedback** (`feedback=Feedback(content)` in Python;
`config.with_replay(...).with_feedback(content)` in Rust). The core:

1. Resolves the existing completed conversation by replaying recorded reasoning
   and re-executing tools, including nested child tools. No fresh LLM work is
   allowed merely to locate a feedback target.
2. Preserves the old final assistant answer and appends one user-feedback message
   after it. A new immutable root points to an OPEN continuation. The next LLM
   request contains the original prompt, prior investigation, answer and feedback.
3. Resumes normal serving: more tools/LLM turns can follow, and the revised answer
   may equal the old answer. Equality is never a feedback trigger.
4. Returns the revised result and new snapshot. A NEW explicit feedback operation
   resolves the latest completed continuation and appends another intervention,
   even when its message content is identical. No global deduplication occurs.

**Scope is shared-path, within the new snapshot.** If A and B had identical
pre-feedback observations, feedback resolved using A also applies to B and a new
C when they reach that path in the new snapshot. They run the additional tools
and may split on new evidence. Unrelated paths are unaffected. The old snapshot
still yields the original investigation and answer. There are **no environment
scope gates, value-to-feedback maps, public session IDs, or endpoint shortcuts**.
Existing `resolved_env_id` metadata is provenance only; it never selects a path.

**Caller caveat:** this revises the investigation for the entire path, not a
private annotation on one environment. An assertion such as “this commit
introduces vulnerability X” is not automatically safe to generalize. Prefer
transferable investigative instructions such as “check whether X”, or retain a
separate snapshot. The library does not rewrite/classify feedback or offer
alternate scope modes. Under custom lossy grouping, inheritance uses the same
explicit grouping policy and retains original representative context plus exact
per-environment evidence.

History uses the same shallow `UserContinuation(message, continuation, ...)`
records and lazy loader. There is one user successor per answered endpoint;
successive interventions form a chain. Imported multiple competing user edges
are rejected on traversal rather than implicitly merged/selected. The provisional
`requires_user_policy` descriptor flag has been removed now that this policy is
settled. The existing `append_user_message` helper still forks an already completed
live run into an open snapshot without resuming that loop; constructor feedback
is the replay-only resolution/resumption operation. Neither accepts node IDs as
conversation selectors.

**Explicit failure:** if the selected path is unseen or unfulfilled, including
an open earlier feedback continuation or unseen child observation, feedback
resolution fails instead of generating a target. A root miss may fail construction;
a later miss surfaces from completion or the loop's terminal error state.
`done`/`is_done()` is true, `error`/`error()` reports
`FeedbackResolutionError` / `LoopError::FeedbackResolutionMiss`, and `result()`
raises/returns that error, not a partial result. Pending work is logically
cancelled (caller-owned executors must stop/ignore it). Further completions return
the stored failure. `history` reverts to the supplied snapshot; append-only stores
may retain orphan writes. A **valid** differing tool result causing a miss is
consumed and terminal; an **invalid completion** remains correctable and does not
consume its request. Start a new operation to retry after a resolution failure.

Only the selected path must already be completed; unrelated branches may be open.
Ordinary replay without new feedback may still extend an open continuation.
Feedback targets this loop's terminal no-tool assistant answer; no new nested
asset/endpoint-selection or result-tool publication API is introduced. Existing
nested investigations replay during resolution, and new subagents can run after
injection. There is no crash-safe exactly-once claim: “once” is per live operation.
Independent feedback snapshots are not automatically reconciled. Assets, DAG
scheduling and UI remain out of scope.

### Transactional child preparation and callback failures

Both implementations prepare **all** child prompts, materialized specs and child
constructors before fulfilling an assistant turn, recording its message, or issuing
its ordinary tool requests. A mixed ordinary/valid-child/invalid-child turn cannot
leave a partially issued investigation. Successful prepared children are used
directly: factories are not called again at commit.

Python stages mutable execution builders across the whole advancement, including
nested completion routing, parent rollup, and replay entered after a tool tuple.
A callback exception leaves the root, queues/routes, messages/steps, issue counters
and pending requests unchanged. Correct the completion or callback and retry the
same pending request. Polling (`next_request`, `done`, `result`) can also advance
cached children and has the same exception guarantee. Closed history and opaque
payloads are shared, not deep-copied; staging copies mutable runtime builders.
Explicit stored terminal failures (such as feedback resolution misses) still
commit their documented cancellation rather than leaving work pending.

Rust reports library-detected nested incompatibility as `IncompatibleHistory`,
not a panic. A constructor rejects it directly; a later transition enters the
normal terminal error state with no half-started assistant turn or stranded
requests. Validated preceding tool evidence may already be recorded, but no
partial result is returned. Arbitrary Rust callback panics are **not** recovered;
discard the live loop if one occurs.

Neither language rolls back caller side effects or append-only store allocations.
Preparation/staging may leave unreachable records and warmed loader caches; only
the published snapshot root defines fulfilled history. Retried operations may
call factories again after a failure. Callbacks must not mutate borrowed payloads.

### Other limits

Recording/replay still explicitly rejects **budgets**: a synthetic exhaustion
answer must not silently become reusable reasoning. Non-recording budgets work
as before. History is enabled on direct `BatchLoop`, not convenience drivers.
Partial tool collections and request queues are builders, not persisted state;
replay restarts tools, including child tools. This is **not crash recovery**.
Validation errors (wrong kind, duplicate call IDs, invalid environment result
sets) leave pending requests recoverable, including nested routes. Child request
IDs use paths of internal invocation counters that never reset across a loop's
branches or turns. Opaque tool-call IDs (including delimiter-containing IDs) are
preserved as evidence, never interpolated into routing addresses. Repeated calls
cannot accept stale completions from earlier invocations. Request IDs remain
run-local; their format is not a persistence contract. Python also
validates grouping of tool outputs and child final answers before fulfillment.
Concurrent extensions of separate roots are not automatically merged.

Executable caller-side persistence/replay examples (no provider needed):

```sh
(cd python && PYTHONPATH=src python examples/history_replay.py)
cargo run -p lockstep-loop --example history_replay

# Feedback, inheritance, repeated interventions, persistence, explicit replay miss
(cd python && PYTHONPATH=src python examples/feedback.py)
cargo run -p lockstep-loop --example feedback
```

See the [Python API/example](python/README.md#recording-and-replay) and
[Rust API/example](rust/README.md#recording-and-replay).

## Repository layout

```text
python/   Python package, tests, and local examples
rust/     Rust crate and integration tests
```

The APIs follow the same vocabulary (`BatchLoop`, `LlmRequest`, `ToolRequest`,
completions, budgets, split trie, subagent), while retaining language-native
serving models:

- Python methods are internally synchronized.
- Rust methods use `&mut self`; callers can wrap a loop in their preferred
  mutex when cross-thread access is needed.
- Rust budgets take a caller-supplied exhaustion function because an opaque
  generic payload cannot safely be fabricated from a string sentinel.

## Development

```sh
# Python
cd python
uv sync
uv run pytest
uv run mypy src
uv run ruff check .

# Rust (from the repository root)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

## License

MIT © 2026 Stephan Brandauer. See [`LICENSE`](LICENSE).
