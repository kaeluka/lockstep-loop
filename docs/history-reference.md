# History, replay, and feedback reference

[Visual guide](concepts.md) · [Documentation](README.md) · [Repository](../README.md)

This is the technical contract for the current Python and Rust implementations.
It applies to any supported read-only evidence domain: documents, specifications,
datasets, source code, and others. For concrete environment/tool designs, see
[use cases](use-cases.md); tools and evidence ingestion remain caller-owned.
`Agent` is the complete immutable snapshot: initial prompt, structural configuration, and
accumulated trie. `Asset` is a read-only leaf investigation backed by its full path.
See the [glossary, inspection API and exact compatibility mapping](agent-assets.md).
Legacy `History` names remain aliases; deprecated contract labels and grouping-mode
metadata are ignored on import and omitted on new writes. See the migration notes in the glossary.

## Immutable recorded history and incremental replay

Both implementations offer **opt-in recording on `BatchLoop`**: Python
`recording=True`, Rust `LoopConfig::with_recording()` (no argument). Supplying an
`Agent` implies replay plus recording. Keep an
`Agent` snapshot, then start a new loop with that snapshot and new environment
IDs. The core reuses recorded assistant turns, emits their tool requests again,
and follows the matching **ordered tuple of observations**. Only an unseen
continuation requests fresh LLM work. This includes checkpoints that never split
the original environment class. Old snapshots and old paths remain reusable.

The history model is the replay engine's model, not a lossy export:

- `Agent` pairs a small `AgentDescriptor` (prompt, ordered tool specifications,
  nested configuration and root reference) with a `RecordStore` resolver.
- `HistoryNode` stores the full assistant message (including tool calls/arguments)
  and the environment visits that used it. An absent assistant is an unfulfilled
  continuation, not a terminal answer.
- Each `Observation` stores ordered representative `OutputRecord`s, **all exact
  per-environment `EnvObservation` evidence**, and a continuation `NodeRef`. Each
  output record has a `tool_output` field for future metadata extensions.
- Conversation reconstruction starts with the prompt, follows assistant and
  representative tool messages, and includes any appended user-feedback messages.
  With custom lossy grouping, replay keeps the **original representative
  conversation**, while appending the new environment's actual, possibly unequal
  outputs to evidence.
- `HistoryNode.subagents` retains the latest child-agent descriptor per assistant
  call ID and immutable per-invocation `runs` (environments, selected base revision,
  resulting child root), including a current run still open when captured. Children
  use the same shallow records/resolver and may themselves invoke subagents up
  to the configured depth (the reserved spec is omitted at the depth limit).
  Replay reruns child tools; child trace splits do NOT
  split parents when their folded answers match. Ordinary and subagent calls
  may coexist in a turn and complete independently.
- Nodes are immutable values: Python uses frozen dataclasses/tuples; Rust stores
  immutable `Arc` records. Updates allocate new records along the ancestor path
  and retain unchanged references. Snapshots are available during a run and in
  the result. The old `result.tree` remains a per-run partition summary, **not
  an agent or asset**. Its labels retain result-group representations, not IDs.
- `agent.assets()` inspects recorded leaves with their conversation/evidence,
  open/completed state and latest published value. It neither runs work nor adds
  a stored asset identity. A former leaf becomes an ancestor after feedback in a
  new snapshot, while remaining a leaf in the old snapshot.

Matching verifies equality of complete output tuples (or caller-defined group
keys); hashes/debug strings are never record identities or substitutes for
outputs. Branch IDs are run-local scheduling identifiers. Repeated environment
IDs append visits/evidence rather than overwriting them; use revision-qualified
IDs if you need to distinguish runs in an inspector.

### Specialized tools and independently reusable child histories

Register multiple `SubagentTool` definitions using Python `subagents=[...]` or
Rust `LoopConfig::with_subagents(vec![...])`. Each definition supplies structured
argument specs and `build_prompt(args)`: the **caller**, not the model/core,
constructs the deterministic initial **user** message. There is no system-role
API. Repeating `assess_design(criterion="randomized")` can reuse the same child
reasoning under a differently worded research-screening parent. Different criteria
produce different prompts/tries unless the caller intentionally canonicalizes
them; the core never uses fuzzy/semantic prompt matching. The same pattern works
for `list_imports(file_path="foo.py")` in code analysis. All children inherit the
parent's environment IDs unchanged. Resolve mutable evidence sources outside the core.

No caller-supplied child key is needed. Configure `agent_resolver` /
`.with_agent_resolver(...)` to receive an `AgentQuery`: child prompt, ordered effective
tool specs, and nested subagent configuration. No grouping mode or function identity
participates in lookup. It contains **no parent prompt,
branch/invocation ID, tool-call ID, or environment membership**. The caller returns
a chosen `AgentDescriptor`, or None for a miss (Rust wraps this in `Result`).
Core verifies full compatibility by value; a bad hit is an error, not a silent miss.
Hashes may index the caller's cache but cannot replace those equality checks.

This composes with the existing `RecordStore`: the loop and child descriptors must
use one caller-resolvable record-address domain. Pass a shared in-memory store or
a store whose loader can resolve every referenced record, across caller-chosen
backends if needed. The resolver selects **metadata/revisions**; the existing node
loader loads records lazily. No second execution/storage framework or core I/O is
introduced. Agent snapshot selection remains explicit via `agent`/`with_agent`
(legacy `history`/`with_replay` aliases are retained).
A metadata callback returning a descriptor does not supply its records: the shared
store must be able to load them. Persist writes before publishing selected roots.

**Selection precedence and feedback:** a recorded parent invocation pins its
recorded child descriptor. Replay uses that revision and does not consult a mutable
cache pointer for that invocation. Only a new/unrecorded invocation consults the
resolver. A changed parent prompt starts a new parent history and can select a
recorded child. The selected child can include feedback; its shared-path user
continuations are replayed normally, including their tool calls. A prompt key is
not a unique revision. Updating a cache index does not rewrite completed parents
or automatically propagate changed child results. New parent investigations (or
new invocations after explicit parent feedback) can select updated child roots.
Strict feedback resolution never uses an external hit to manufacture missing
parent-invocation provenance.

Python `BatchResult.child_agents` / Rust `child_agents()` returns **every completed
recorded invocation** as a `ChildAgentUpdate`, including descendants once their
containing invocation completes. Each update has its full `Agent` (descriptor + store), selected `base` descriptor or None,
original tool call, inherited env IDs, parent branch/depth and existing run-local
request prefix for provenance. That prefix is not a cache key, conversation ID,
or feedback shortcut. Two concurrent invocations of the same base have independent
builders and separate updates; no single-flight, root merging or last-writer-wins
cache publication occurs in core. The caller chooses which returned revision to
publish. Parent `SubagentRecord.runs` retains earlier invocation roots rather than
replacing their evidence. `entry.agent_descriptor` / `agent_descriptor()` selects
the latest run (stored field: `history`); `agent.child(entry)` attaches the same
record resolver to inspect that child agent and its assets. While a parent is
running, its normal history snapshot still exposes its active child roots. Store
write sets may include parent/orphan records; a child descriptor is an independent
entry point and needs no parent descriptor to replay.

**Registration/depth rules:** names must be unique. Ordinary spec names duplicated
within `tool_specs`, duplicate subagent definitions, conflicting ordinary/subagent
specs, and mixed singular/plural registration are rejected. For old direct-loop
callers, an exactly matching `ToolSpec` is accepted as a redundant advertisement
of a reserved subagent, not as another executor. Serial drivers register **all**
subagent entries and reject actual callable/subagent name collisions even when
specs match. Subagent specs are automatically advertised, so callers need not
repeat them. A recorded parent or resolver enables child recording. Without either,
Python `SubagentTool(recording=True, ...)` or Rust `SubagentTool::with_recording()`
optionally enables standalone child recording. This is only a switch, never a
lookup key or structural compatibility field. A resolver does not enable recording
of a fresh root. A tool definition is not the persisted child agent itself.

Each definition may declare its own child `subagents` allowlist. Empty means leaf;
None preserves legacy **self** recursion, never all parent siblings. At each spawn,
nested definitions' `max_depth` is clamped to the caller's remaining depth. A nested
definition cannot reset the budget. Only those eligible definitions are advertised;
known ancestor/sibling/depth-disabled reserved specs are omitted and calls to them
are rejected, never sent to ordinary execution. General unknown ordinary-tool
handling remains caller-owned; this is not a JSON Schema validator. Validate args
in `build_prompt` (Rust `try_new` supports fallible prompt/spec factories).

Executable independent-cache examples (from the repository root):

```sh
(cd python && PYTHONPATH=src python examples/specialized_subagents.py)
cargo run -p lockstep-loop --example specialized_subagents
```

They show fresh parent reasoning but **zero fresh child LLM calls** after changing
the parent prompt, while rerunning the child's tool and publishing a separate
updated child root. No environment binding, logical asset registry, or DAG
invalidation/scheduling is included.

### Compatibility and ownership

There is no manual contract/version key. The core rejects unequal prompts, changed
ordered tool specs, or changed subagent registration, nesting allowlist or depth. Child prompts/specs are checked
when entered, including the actual outputs of prompt/spec-building callbacks.
Pure tools are re-executed: changing their implementation either changes the
observations (branch at divergence) or does not (reuse). No manual bump is needed.

The default grouping function is identity. Neither default/custom mode nor function
identity is persisted or compared. On every checkpoint, the **current** function
is applied to both the fresh raw output tuple and each stored representative tuple.
Default and custom identity functions therefore reuse each other's histories.
A stricter function can miss even if the fresh raw output occurs in an old edge's
per-environment evidence: matching uses the representative, not any past evidence.
A coarser function can match several existing edges; the **first matching edge in
stored observation order** wins deterministically for that snapshot. Edges and
continuations are not merged or rewritten. Replay keeps the chosen historical
representative in conversation context and appends exact fresh outputs to evidence;
old roots remain unchanged. Changing keys does not retroactively refine old evidence
or regenerate old reasoning. The current callback must accept historical outputs,
not just today's outputs, and be stable for the duration of an execution.

Provider/adapter settings and the semantics of custom grouping are external policy,
not inferred from closures or specs. Callers choose compatible snapshots and isolate
external caches when those policies require different reasoning; the core adds no
namespace/version field. Independent child queries depend on child structural
configuration, never parent identity. Custom grouping must be pure, deterministic,
and use stable equality/hashing. It deliberately permits lossy reuse. Never use environment IDs in the
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
2. Record normally; `agent.store.records()` yields the new immutable shallow
   records, including all nested child records. Persist each independently with
   your codec, using insert-only semantics. Reject an address collision rather
   than overwriting another record. Retain records reachable from old snapshots.
3. After records are durable, atomically publish `agent.descriptor`. Never
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
Start a new loop with a compatible **agent snapshot + exactly one stable
environment + explicit feedback** (`feedback=Feedback(content)` in Python;
`config.with_agent(...).with_feedback(content)` in Rust). The core:

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
private annotation on one environment. An assertion such as “this supplier's
certificate is expired” is not automatically safe to generalize. Prefer
transferable investigative instructions such as “check whether X”, or retain a
separate snapshot. The library does not rewrite/classify feedback or offer
alternate scope modes. Under custom lossy grouping, inheritance uses the same
explicit grouping policy and retains original representative context plus exact
per-environment evidence.

Conversation history uses the same shallow `UserContinuation(message, continuation, ...)`
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
the stored failure. `agent` / `agent()` reverts to the supplied snapshot; append-only stores
may retain orphan writes. A **valid** differing tool result causing a miss is
consumed and terminal; an **invalid completion** remains correctable and does not
consume its request. Start a new operation to retry after a resolution failure.

Only the selected path must already be completed; unrelated branches may be open.
Ordinary replay without new feedback may still extend an open continuation.
Feedback targets this loop's terminal no-tool assistant answer; no new nested
asset/endpoint-selection or result-tool publication API is introduced. Existing
nested investigations replay during resolution, and new subagents can run after
injection. There is no crash-safe exactly-once claim: “once” is per live operation.
Independent feedback snapshots are not automatically reconciled. Asset views
only inspect existing state; result-tool publication, DAG scheduling and UI remain
out of scope.

### Transactional child preparation and callback failures

Both implementations prepare **all** child prompts, materialized specs and child
constructors before fulfilling an assistant turn, recording its message, or issuing
its ordinary tool requests. A mixed ordinary/valid-child/invalid-child turn cannot
leave a partially issued investigation. Successful prepared children are used
directly: factories are not called again at commit.

Python journals only the **touched** builders, scalar fields and container entries
across an advancement, including nested completion routing, parent rollup, and
replay entered after a tool tuple. It does not clone executions, sibling branches,
finalized conversations, or message lists. Immutable history is shared, and the
partition/final lists retain their existing entries. A no-rollup `done` check and
queue pull do not visit conversations; materializing the final result still
necessarily visits its partition.

A callback exception restores the root, queues/routes, messages/steps, issue
counters and pending requests. Correct the completion or callback and retry the
same pending request. Polling (`next_request`, `done`, `result`) can also advance
cached children and has the same exception guarantee. Child serving order is
restored on rollback without cloning children. Explicit stored terminal failures
(such as feedback resolution misses) still commit their documented cancellation.

In Python, child history roots are linked incrementally during transitions. The
`agent` property (legacy `history`) reads the **committed** immutable root; it does not flush working
child state. A callback may inspect it reentrantly: it sees the pre-transition root
until success, never a partially updated root that might roll back. Reentrant
advancement from callbacks is rejected. Rust's `agent()` (legacy `history()`) assembles active child
roots into the returned immutable snapshot and can allocate path-copy records;
it does not have Python's reentrant callback/undo-journal protocol.

Python's structural cost tests forbid traversing/copying sibling and historical
conversation containers during single pulls/completions (with and without
recording), without relying on timing thresholds. The split/finalize workload can
also be measured from `python/` (100/400/1000 environments, no recording):

```sh
PYTHONPATH=src python benchmarks/split_completion.py --repeats 3
```

Rust's `SubagentTool::try_new` and history resolver return structured errors.
While preparing a fresh LLM completion, argument/spec/resolver/compatibility errors
leave its request outstanding for correction, without partial child spawning.
A constructor rejects incompatible history directly. Errors encountered while
replaying an already-recorded assistant turn are terminal (there is no fresh
assistant request to correct): pending work is cancelled, `error()`/`result()`
report the failure, and no partial result is returned. Preceding valid tool evidence
may already be recorded. Arbitrary Rust callback panics are **not** recovered;
discard the live loop if one occurs.

Neither language rolls back caller side effects or append-only store allocations.
Preparation/staging may leave unreachable records and warmed loader caches; only
the published snapshot root defines fulfilled history. Retried operations may
call factories again after a failure. Callbacks must not mutate borrowed payloads.

### Other limits

Recording/replay still explicitly rejects **budgets**: a synthetic exhaustion
answer must not silently become reusable reasoning. Non-recording budgets work
as before. Root agent recording/replay is enabled on direct `BatchLoop`. Serial
convenience drivers also expose child-agent updates when the child recording switch
is enabled; resolver-driven serving uses the direct loop.
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

Executable caller-side persistence/replay examples (from the repository root;
no provider needed):

```sh
(cd python && PYTHONPATH=src python examples/history_replay.py)
cargo run -p lockstep-loop --example history_replay

# Feedback, inheritance, repeated interventions, persistence, explicit replay miss
(cd python && PYTHONPATH=src python examples/feedback.py)
cargo run -p lockstep-loop --example feedback
```

See the [Python API/example](../python/README.md#recording-and-replay) and
[Rust API/example](../rust/README.md#recording-and-replay).

## API boundaries and design direction

- Python methods are internally synchronized. Rust methods use `&mut self`;
  callers choose synchronization for cross-thread serving.
- Rust non-recording budgets require a caller-supplied exhaustion function because
  an opaque payload cannot safely be fabricated from a string sentinel.
- `BatchLoop` is the mutable execution machinery, not the agent snapshot.
  `result.tree`/`SplitNode` and `EnvClass` describe a per-run partition, not assets
  backed by complete recorded conversations.
- Multiple named `SubagentTool` definitions are supported and advertised, including
  all serial-driver entries. Registration collisions are rejected; argument schemas
  are advertisements, not a built-in validator. Callers validate/interpret args.
- Deterministic structured-argument prompts allow independently selected child
  agents to survive changed parent prompts. Recorded parent invocations stay pinned;
  new invocations use the caller's `AgentResolver`. All revisions remain immutable.
- `Agent` is the canonical recorded-state type, with `History` retained as an alias.
  `Asset` is an inspection-only, path-backed leaf view, not `EnvClass` or its value.
  Current values are terminal assistant replies, not result-tool arguments. There
  are no public asset/session/endpoint shortcuts for replay or feedback.
- Persisted record tags and provenance `history` fields retain their names.
  Manual `contract`/`history_contract` fields and `custom_grouping` mode are removed:
  old pickle metadata and self-describing Serde fields are ignored, never used in
  matching. New writes omit them. See [API/schema migration and unchanged fixtures](agent-assets.md#persistence-and-contract-removal).
- Dependency-DAG scheduling and push updates remain future work
  ([issue #1](https://github.com/kaeluka/lockstep-loop/issues/1)). Updated child
  histories do not automatically revise completed parents.
