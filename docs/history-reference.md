# History, replay, and feedback reference

[Visual guide](concepts.md) · [Documentation](README.md) · [Repository](../README.md)

## Agent and run lifecycle

`Agent` is the primary mutable handle: a current immutable `AgentSnapshot` plus
caller-owned runtime callbacks. `AgentRun` is a mutable wrapper around immutable
history and transient request/completion builders. Python uses
`Agent(prompt=..., tool_specs=..., ...).start(environments, feedback=..., budget=...)`;
Rust uses `Agent::new(AgentConfig::new(prompt, specs))?.start(environments, options)`.
`Agent.from_snapshot` / `Agent::from_snapshot` restores an independent handle from
snapshot prompt/specs and explicitly supplied runtime dependencies, without
repeating structural configuration or eagerly loading records.

Every start captures the **current** head. The run independently extends that root.
At successful completion it automatically publishes its final snapshot to its
owner **exactly once**. This includes cached final-only starts, cached nested
rollup and immediate budget exhaustion. Calling `result()` or `snapshot()` is not
required; inspecting an old completed run cannot republish it.

> **Warning:** Overlapping runs start from the agent state available when each run
> begins. Their histories are not merged. The last run to complete replaces the
> agent's current state, potentially excluding experience from other runs. Retain
> each run's snapshot to preserve independent results, or finish each run before
> starting the next to accumulate experience sequentially.

No active-run restriction, queue, lifetime lock or automatic merging exists.
Python serializes advancement of each run; Rust uses `&mut self` and caller-chosen
synchronization. Head capture/replacement uses a short lock; callbacks run outside
it. Rust runs retain a shared head, not an exclusive lifetime borrow of Agent.
Rust mutable handles deliberately do not implement Clone. In both languages,
from_snapshot creates an independent head sharing only immutable records/store.

Invalid completions remain correctable and cannot publish completion. Terminal
resolution failures and abandoned runs never replace the owner's head. A successful
budget-limited run publishes real history, not its result fallback.
`run.snapshot()` exposes immutable validated historical facts during execution,
not pending requests or an in-flight crash recovery protocol. Request IDs are
run-local; the caller must retain the originating run when routing responses.

## Immutable recorded history and incremental replay

Recording is unconditional in all runs, children and convenience drivers. There
is no recording flag. Tools must be pure/read-only over stable environments and
have identical specifications across environments. Environment identities stay
outside model-visible prompts, arguments, schemas and outputs.

Recorded assistant turns are reused and their tools are re-executed. The complete,
ordered tuple of outputs determines continuation, regardless of completion order.
Only unseen continuations request fresh LLM work. Every tool checkpoint is
recorded, including turns that did not split. Classes refine, never merge within
an execution. Equal final answers do not merge previously distinct paths.

The one history model serves both replay and inspection/persistence:

- `AgentSnapshot` pairs `AgentDescriptor` (prompt, ordered tools, nested
  configuration and root reference) with a `RecordStore`.
- `HistoryNode` retains the full assistant message, calls/arguments and environment
  visits. No assistant means an open continuation, not a fabricated answer.
- `Observation` retains representative `OutputRecord(tool_output=...)` tuples,
  exact per-environment `EnvObservation` evidence and a continuation `NodeRef`.
- `UserContinuation` records full feedback messages and shallow successor addresses.
- `SubagentRecord` retains call provenance and child snapshot descriptors;
  `ChildAgentRun` retains each invocation's environment set, base and resulting
  descriptor. Their `snapshot` fields are descriptors, not mutable handles.
- `BatchResult.snapshot` is the final immutable snapshot. `child_updates` contains
  `ChildAgentUpdate` records whose `snapshot` fields are complete child snapshots.
  `tree`, `SplitNode` and `EnvClass` remain per-run partition summaries, not assets.

Python frozen dataclasses/tuples and Rust shared immutable Arc records implement
path-copying updates. Previously captured roots and untouched edges stay valid.
Rust owned snapshot/descriptor clones can be edited into different values, but
never mutate records reachable from other captures. Payload ownership remains an
explicit contract, not a consequence of frozen wrappers or Arc.

`AgentSnapshot.assets()` lazily inspects structural leaves as contextual, path-backed
investigations. It neither executes work nor invents asset identities. Views remain
anchored in their snapshot across later head publications. See the
[inspection contract](agent-assets.md#inspecting-assets).

### Specialized tools and independently reusable child histories

Python Agent takes `subagents=[...]`; Rust `AgentConfig`/`AgentRuntime` contain
subagent definitions. Each `SubagentTool` supplies an advertised spec, deterministic
`build_prompt(args)` and child spec factory. The prompt is an initial **user**
message. Schemas are advertisements, not built-in argument validation; validate in
the factory (Rust `try_new` supports fallible factories). There is no provider
adapter, system-role API, fuzzy prompt normalization or callback registry.

Children inherit the parent's environments. Their traces can diverge without
splitting parents when their folded answers agree. Ordinary and subagent calls
can coexist, complete independently, and nest within configured depth limits.
A definition's empty child allowlist makes a leaf; None preserves bounded
self-recursion, not access to all parent siblings. Nested depths are clamped to
the remaining allowance. Known unavailable reserved names are omitted/rejected,
not sent to ordinary execution. Duplicate/conflicting registrations are rejected;
matching redundant spec advertisements are allowed. Serial local-tool drivers
reject actual callable/subagent name collisions even if specs match.

`agent_resolver` receives `AgentQuery(prompt, tools, subagents)` and returns a
chosen `AgentDescriptor` or None (Rust wraps this in Result). Queries have no parent,
invocation, environment, grouping-mode or function identity. The core verifies
full structural equality; an incompatible hit is an error, not a silent miss.
Changed parent prompts can therefore reuse independently selected children.
The resolver selects metadata; the shared RecordStore must resolve every selected
address. Returning a descriptor does not import a foreign store.

Recorded parent invocations pin their child revision and bypass external lookup.
Only new invocations consult the resolver. A selected child can contain feedback;
its continuation and tools replay normally. Cache pointer updates do not rewrite
completed parents or propagate changed child answers automatically. Strict parent
feedback resolution never manufactures missing provenance from an external hit.

`child_updates` exposes every completed child invocation, including descendants
once their containing invocation completes. Each carries its snapshot, selected
base descriptor, original call, environments, branch/depth and run-local request
prefix. These are provenance, not cache keys or endpoint shortcuts. Concurrent
hits on one base produce independent forks and updates. Unlike the root Agent's
explicit last-completion-wins publication, there is **no automatic external child
cache publication**: the caller selects future revisions. Parent provenance keeps
old roots. `snapshot.child(entry)` attaches the same record resolver for inspection.
An independent child descriptor needs no parent descriptor to replay.

Restoration requires matching runtime subagent definitions, but never serialized
callbacks. Actual child prompt/spec factory outputs are checked on entry.

### Compatibility and ownership

No manual contract/version/namespace key is used for reasoning compatibility.
Prompt equality, ordered specs and nested subagent specs/allowlists/depth are
checked. Pure tool implementations can change: equal observations reuse; different
observations branch. Provider/adapter settings and grouping semantics are external
policy. Callers choose compatible snapshots and isolate external caches as needed.

Default grouping is identity. The **current** comparison function is applied to
both fresh raw tuples and historical representatives. Default/custom identity
works in both directions. A stricter function can miss even if the output exists
in past evidence: matching uses the representative, not any historical sample.
A coarser function chooses the **first matching edge in stored order**. It does not
merge or rewrite edges. Replay preserves representative conversation context and
appends full exact new evidence. Neither hashes, debug strings, callback identity
nor default/custom mode replace equality verification.

Python payloads, schemas and argument mappings are borrowed immutable values,
including their reachable contents and aliases. Rust Clone must preserve immutable
value semantics; shared interior mutability must not change recorded facts.
The caller owns copying/codecs for provider objects. Key functions must accept
historical outputs, be deterministic, and use stable equality/hashing.

### Python snapshot dump

`print(snapshot.format())` explicitly dumps the descriptor and all reachable history
records, including feedback, full observation evidence and child base/result revisions.
`format(width=100)` returns a string using Python's `pprint`; it does not print or
require pickleable payloads. Opaque payloads use their normal Python representations,
without an imposed depth/length limit. The dump may be large or contain sensitive
provider/tool content. It loads lazy records, but does not change history or include
unrelated store records, pending requests or other transient execution state.
`repr(snapshot)` stays lightweight and never loads records. Use Asset views for
conversation-oriented inspection rather than a record-level dump.

### Direct Python snapshot pickle

Python supports `pickle.dump(agent.snapshot(), file)` and
`Agent.from_snapshot(pickle.load(file), ...runtime dependencies...)` directly.
The bundle is self-contained for the complete reachable history: every observation
and feedback branch, nested children, and all invocation base/result descriptors.
An iterative address walk deduplicates shared records without recursively walking
conversation paths. Full payloads and sharing are preserved; hashes/debug strings
are not identities. Unrelated runs and orphan writes in a shared store are omitted.

This export **eagerly loads the entire reachable graph** via the existing caller
loader, not just the last replay path. It never serializes that loader, store locks
or runtime Agent/AgentRun state. Loading creates a fresh writer namespace while
keeping original references readable from the bundle, so independently loaded
forks can extend and coexist. Re-pickling after extension includes both bundled
and new reachable records, without an external backend. Captured roots and records
are not mutated; cache warming is allowed. Missing/corrupt-record or payload-pickle
errors surface rather than producing a partial/lossy bundle.

**Only load trusted pickle data.** Generic payloads, arguments and schemas must be
pickleable; their own nesting/codec limits still apply. Executable subagent factories,
resolvers and comparison functions must be reattached as needed. This is history
persistence, not callback persistence or crash recovery. Caller code owns file I/O
and atomic publication. See the [Python example](../python/examples/snapshot_pickle.py).
Rust retains caller-owned descriptor/record Serde transport; no direct snapshot
serialization framework is added there.

### Individually addressable records and lazy persistence

For large, lazy or incremental storage, the existing lower-level API remains
available in both languages. The core performs no filesystem, database or network
persistence I/O:

1. Allocate a fresh writer namespace in the caller's storage domain. Exact
   `NodeRef(namespace, index)` addresses are monotonic per writer, not content hashes
   or public conversation identities. Python defaults to UUIDs; Rust defaults are
   process-local, so durable Rust writers must explicitly reserve namespaces.
2. Persist `snapshot.store.records()` using caller-owned codecs and insert-only
   semantics. Reject collisions instead of overwriting records. Retain all old
   referenced records, including child roots and independent run snapshots.
3. Only after records are durable, publish `snapshot.descriptor` atomically using
   the caller's storage protocol. Failures can leave orphan records.
4. Decode a descriptor and attach a fresh writable RecordStore with a lazy
   `load(NodeRef)` callback. Construct `AgentSnapshot`, then `Agent.from_snapshot`
   with runtime callbacks. Neither construction traverses nodes. The core, not the
   caller, implements path traversal, observation matching and nested replay.
5. Save new local writes and the new descriptor; already stored records need not
   be rewritten. A shared/cloned store shares its allocator/cache; two independent
   stores must not reuse a writer namespace.

Records are shallow: successors are addresses, not recursively encoded nodes.
Existing tests round-trip 256 tool turns over two paths with default JSON limits,
then load only the selected 257 nodes. Asset traversal is iterative and does not
load child tries unless explicitly inspected. Payload codecs still need to support
the payload's own nesting/size. Rust descriptors/records derive Serde when supported
by P; the advanced Python example pickles trusted records individually, never live
handles. This protocol does not eagerly bundle reachable records like direct pickle.

Loaders are synchronous, non-reentrant and expected to resolve valid records.
Missing/corrupt records and loader failures are infrastructure failures; restart
from a saved descriptor with a healthy loader. Arbitrary imported graphs are not
fully validated. There is no cross-language/version wire guarantee or legacy
migration layer. Stores retain writes and cached loads, with no GC/eviction.
Deep path copying may cause quadratic write volume; observation lookup is linear.

### Feedback: shared-path inheritance in the new snapshot

Use `agent.start([environment], feedback=Feedback(content))` in Python, or Rust
`start(environments, RunOptions { feedback: Some(content), budget: ... })`.
Exactly one stable environment locates the target; it does not scope the change.

1. Replay tools and recorded reasoning to an existing completed endpoint, including
   nested child tools. No fresh LLM work manufactures a target.
2. Preserve the old assistant answer, append one user message and open a continuation.
3. Resume normal LLM/tool work. The revised answer may equal the old answer.
4. Publish the revised snapshot on successful completion. A new explicit feedback
   operation resolves the latest endpoint and appends again, even for identical text.

Everyone reaching that path in the new snapshot inherits feedback; unrelated paths
and old snapshots remain unchanged. `resolved_env_id` is provenance only, never
an inheritance predicate. Use transferable investigative instructions rather than
unqualified assertions true only for one environment, or keep a separate fork.
There is no feedback rewriting/classification or alternate scope mode.

Feedback presence is explicit: Python `Feedback(None)` and Rust `Some(None)` for
optional P are real null messages. Multiple competing user successors or malformed
user edges are rejected on traversal rather than implicitly merged.
An unseen/unfulfilled target, including an open previous feedback continuation,
fails explicitly. Ordinary replay can extend open paths without new feedback.

A root failure may raise/return Err from start; later resolution failures surface
from completion or terminal state. `done`/`is_done()` becomes true, queued/pending
work is logically cancelled, and `error`/`result()` expose the stored error, not a
partial result. The run never publishes failure to its owner. Invalid completions
instead leave pending work recoverable. Callers own stopping/ignoring executor work.
Feedback is once per live operation, not crash-safe exactly-once persistence.

### Transactional child preparation and callback failures

Both engines prepare all child prompts, materialized specs and constructors before
fulfilling a fresh assistant turn or issuing its ordinary tool requests. Prepared
children are used directly, without rerunning factories at commit.

Python journals only touched builders/entries across completions, nested routing,
parent rollup and replay. It does not clone siblings or historical message lists.
Callback exceptions restore pending requests, roots, queues/routes and counters;
retry with corrected inputs/callbacks. Explicit stored terminal failures commit
cancellation instead. Reentrant snapshot inspection sees only the pre-transition
committed root; reentrant advancement is rejected. AgentRun settles cached children
inside the completion journal before publishing. Polling can also advance cached
work. Structural cost tests guard against unrelated traversal on serving hot paths.

Rust fallible fresh preparation returns structured errors and leaves requests
correctable. Errors while replaying already-recorded turns are terminal; valid
earlier tool evidence may remain in the run snapshot. Rust does not implement
Python's undo journal or callback panic recovery: discard runs after arbitrary
panics. Rust run snapshot inspection may assemble active-child roots and allocate
path-copy records, while still preserving old facts and never publishing the owner.
Asset traversal itself is read-only and does not clone payloads.

Neither language rolls back caller side effects, warmed caches or unreachable
append-only allocations. Only committed immutable roots define historical facts.

### Other limits

Budgets limit accepted assistant turns per branch, including reused turns. Tools
for an accepted turn finish before the next step is checked. Exhaustion returns a
run fallback, not a synthetic assistant record. Later runs can resume the real
checkpoint; smaller replay budgets never truncate stored facts. Child fallbacks
are actual parent tool outputs while child histories stay at real checkpoints.
Budget exhaustion during strict feedback resolution fails without publication.

Partial tuples and pending queues are not persisted. Replay restarts tools; this
is completed/validated-history reuse, **not crash recovery**. Python's fallback is
a string sentinel; Rust requires a caller-supplied exhaustion function. Rust Agent
replay requires P: PartialEq; the one-shot custom driver still accepts opaque Clone
payloads without equality. See [Python](../python/README.md) and [Rust](../rust/README.md)
for exact signatures and executable examples.

No UI, DAG scheduling, feedback routing, environment remapping, single-flight,
automatic history merging or designated result-tool publication is implemented.
Assets currently publish terminal no-tool assistant payloads. Changed child values
do not automatically update completed parents; dependency push updates remain
[future work](https://github.com/kaeluka/lockstep-loop/issues/1).
