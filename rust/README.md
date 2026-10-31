# lockstep-loop for Rust

Provider-neutral investigations across stable, read-only environments. The core
shares reasoning until tool observations differ; the caller executes all LLM/tool
requests. [Visual guide](../docs/concepts.md) · [Use cases](../docs/use-cases.md) ·
[Technical reference](../docs/history-reference.md).

## Agent entry point

`Agent<P, K = P>` is a mutable head plus caller-owned runtime dependencies.
`AgentSnapshot<P>` is the immutable persistence/inspection value; `AgentRun<P, K>`
is mutable request/completion state. Recording is unconditional, including children.

```rust
use lockstep_loop::{Agent, AgentConfig, Completion, LlmCompletion, MessageEnvelope, RequestKind, RunOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(AgentConfig::new(vec![
        MessageEnvelope::system("Assess the supplied evidence.".to_owned()),
        MessageEnvelope::user("Explain the checklist.".to_owned()),
    ], vec![]))?;
    let mut run = agent.start(vec!["packet-a/v1".into()], RunOptions::default())?;
    let request = run.next_request(RequestKind::Any).unwrap();
    run.complete(Completion::Llm(LlmCompletion::final_answer(request.request_id(), "done".into())))?;
    assert_eq!(agent.snapshot(), run.snapshot()); // published before result()
    assert_eq!(run.result()?.results["packet-a/v1"], "done");
    let old = agent.snapshot();
    let mut cached = agent.start(vec!["packet-b/v1".into()], RunOptions::default())?;
    assert!(cached.is_done()); // final-only replay published during start()
    assert_eq!(old.root().env_ids, ["packet-a/v1"]);
    Ok(())
}
```

A driver loops until `run.is_done()`, pulls `next_request(RequestKind::Any)` and
completes each `Request::Llm` or `Request::Tool`. For tools, return exactly one
output per requested environment. Independent requests may finish in any order;
refinement uses the complete tuple in assistant call order. `None` means blocked,
not finished. `Llm`/`Tool` request filters and outstanding counters support separate
consumers. IDs are run-local; route responses to their originating run.

**Warning — overlapping runs:** Each start captures the agent's current root.
Their histories are **not merged**. The last run to complete replaces the agent's
current state, potentially excluding experience from other runs. Retain each
run's snapshot to preserve independent results, or finish each run before starting
the next to accumulate experience sequentially. There are no active-run restrictions
or queues. Publication is automatic exactly once at successful completion, not
triggered by result/snapshot inspection. Inspecting an old run cannot republish it.

`start(&self, ...)` does not borrow the Agent for the run lifetime. Runs share an
`Arc<Mutex<...>>` head, with only short capture/publication locks. The handle does
not implement Clone: use `Agent::from_snapshot(agent.snapshot(), runtime)` for an
independent head. Cloning snapshots/store handles shares immutable records, not
future head publications. Runs use `&mut self`; choose your own synchronization
for cross-thread serving. No lock spans a run lifetime or a callback.

## Recording and replay

Sequential starts automatically learn paths. Assistant turns are reused, but pure
tools always rerun. Equal observations require zero fresh LLM requests; differences
branch at the first unseen tuple, including formerly non-splitting checkpoints.
Old snapshots stay unchanged. `run.snapshot()` captures validated historical facts,
not pending requests or in-flight crash recovery.

Caller-owned Rust storage uses shallow records and descriptors. Python also supports
self-contained AgentSnapshot pickle transport; this is a Python-only convenience,
not a new Rust snapshot serialization framework or a cross-language wire format:

```rust,ignore
let mut cursor = None; // retain per live writable store across successful saves
let snapshot = agent.snapshot(); // capture FIRST; run.snapshot() may allocate records
let batch = snapshot.store.changes_since(cursor.as_ref())?;
for (reference, record) in &batch.records {
    backend.insert_immutable(reference, encode_node(record))?;
}
backend.flush_records()?; // caller's durability protocol
let published = serde_json::to_vec(&snapshot.descriptor)?;
backend.publish_descriptor(&published)?; // only AFTER records are durable
cursor = Some(batch.next_cursor); // advance only after successful persistence

let snapshot = AgentSnapshot {
    descriptor: serde_json::from_slice(&published)?,
    store: RecordStore::with_loader("fresh-reserved-writer", move |reference| {
        Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
    }),
};
let agent = Agent::from_snapshot(snapshot, AgentRuntime::default())?;
// No repeated initial messages/specs and no record loading until start/traversal.
```

`RecordStore` is an append-only write set/cache, not a backend. Loaders are
synchronous and must resolve valid immutable records without reentering the store.
Reserve a unique namespace for every durable writer (the default is process-local
only). Cloning a store shares its allocator/cache; independently reusing its
namespace is invalid. Save local writes before descriptors, retain referenced old
records, and reject collisions. The core traverses only relevant paths and performs
no persistence I/O. Descriptors/records derive Serde when `P` supports it. No stable
cross-language/version wire format or migration aliases are promised.

### Incremental local record writes (Rust only)

`RecordStore::changes_since(Option<&WriteCursor>) -> Result<RecordBatch<P>, CursorError>`
returns `records: Vec<(NodeRef, Arc<HistoryNode<P>>)>` and `next_cursor: WriteCursor`.
`None` exports all **local writes**, excluding loaded historical records. Subsequent
reads use a range lookup (O(log N + K) for N local writes and K returned records),
without scanning/cloning the full write map or requiring `P: Clone`.
`records()` remains the full local-write-map export.

Capture the snapshot **first**, then its batch; make all records durable **before**
publishing that captured descriptor. Advance the cursor only after successful
persistence. On failure, retain the cursor and retry; insert-only persistence must
accept identical retries and reject conflicting contents. Cold-loaded base records
are never exported by this API: they must already be durable or separately imported
into the descriptor's destination storage.

Records and watermark are captured atomically under the store lock. Concurrent
later writes belong to later batches. A batch includes orphan writes, superseded
path-copy ancestors and other runs' writes in that store; it is **not a reachable
snapshot graph**, nor an Agent head publication or a completed-run transaction.
Reads are non-destructive; old cursors and empty-batch cursors remain usable.

`WriteCursor` is opaque, cloneable, in-process progress, not a serialized checkpoint.
Store clones accept it; independent stores return `CursorError::ForeignStore`,
even if they incorrectly reuse the same namespace. Fresh writable stores start at
`None`; never restore a cursor from a previous writer. The API adds no persistence
I/O, serialization protocol, GC or crash recovery, and has no Python counterpart.

### Runtime configuration and grouping

Fresh configuration: `AgentConfig::new(initial_messages, tool_specs)`, optionally
`.with_subagents(...)`, `.with_agent_resolver(AgentResolver::new(...))`, and
`.with_record_store(...)`. Restored agents take `AgentRuntime { subagents,
agent_resolver }`; factories must match recorded structural configuration.
Executable callbacks are never serialized.

`initial_messages` is a nonempty `Vec<MessageEnvelope<P>>`, preserved in order.
An envelope supplies a `Role` (`System`, `User`, `Assistant`, or `Tool`) and call
metadata; its content `P` remains opaque and can contain bytes, audio, images, or
caller-defined multimodal values. Helpers are `MessageEnvelope::system`, `user`,
`assistant`, and `tool`. Empty initial sequences are rejected, including child
factory results. Snapshots and child queries retain the entire initial sequence.

`Agent::new` and `from_snapshot` use identity keys and require
`P: Clone + Eq + Hash + Debug + 'static`. Custom grouping uses
`Agent::with_comparison_key(config, key)` or
`Agent::from_snapshot_by(snapshot, runtime, key)` with
`P: Clone + PartialEq + 'static`, `K: Eq + Hash + Debug + 'static`.
Closures are `Send + Sync + 'static`. Payloads themselves need not be Send/Sync
unless the caller moves the handle across threads.

The current key function compares **both** new outputs and stored representatives.
No function-identity/default-vs-custom gate exists. Raw payloads remain intact;
stricter keys can branch despite matching old evidence, and coarser keys select the
first matching stored edge without merging. Historical representatives remain
model context; full new evidence is appended separately.

Structural checks cover every initial envelope's role, content and call metadata
in sequence order, ordered specs, nested allowlists/depth, and actual child
initial-message/spec factory outputs. Provider/adapter and grouping-policy isolation
remain caller-owned, without manual version keys. Payload Clone must have immutable
value semantics; interior-mutable aliases must never change submitted facts.

### Feedback, budgets and failures

```rust,ignore
let mut run = agent.start(vec!["packet-a/v2".into()], RunOptions {
    feedback: Some("Also check exclusions.".to_owned()),
    budget: Some(Budget::new(20, |steps| format!("budget exhausted after {steps}"))),
})?;
```

Feedback resolves an existing completed path by tool replay without new reasoning,
then appends one user message and resumes. Every environment reaching that path in
the new snapshot inherits it; unrelated paths and old roots stay unchanged.
Use transferable instructions. `Option<P>` distinguishes absence; `Some(None)` is
real null feedback for an optional payload. Repeated interventions append again
at the latest completed endpoint. There are no public endpoint IDs.

Budgets count accepted fresh **and replayed** assistant turns. Exhaustion returns
the caller's fallback, never an assistant fact, and publishes the real checkpoint.
Later replay can resume with a larger/no budget. Child fallbacks are actual parent
tool observations, not child assistant records. Exhaustion during feedback
resolution fails instead of manufacturing a target.

Invalid completions and fallible fresh child preparation leave pending requests
correctable. Replay-time resolution/structural errors are terminal: `is_done()`
becomes true, `error()` reports them, and `result()` returns Err. Failures and
abandoned/dropped runs **never publish to the owner**. A failed run may retain valid
partial evidence; earlier snapshots remain untouched. Arbitrary callback or loader
panics are not recovered: discard that live run. Caller side effects and unreachable
append-only writes are not rolled back. These semantics differ from Python's
broader transactional callback-exception rollback.

### Children, conversations, and asset accessors

`SubagentTool` factories deterministically build nonempty initial-envelope vectors
and tool specs. `SubagentTool::new` takes a factory returning
`Vec<MessageEnvelope<P>>`; `try_new` accepts a fallible factory returning
`Result<Vec<MessageEnvelope<P>>, LoopError>`. Nested allowlists/depth caps are enforced. A resolver receives a child
`AgentQuery` (no parent/environment/grouping identity) and returns
`Result<Option<AgentDescriptor<P>>, LoopError>`. The shared store must resolve the
selected records. Recorded parent invocations pin their revision; only new
invocations consult the cache. Divergent child paths need not split parents if
their returned values agree.

`BatchResult.snapshot` and `ChildAgentUpdate.snapshot` contain **AgentSnapshot**
values, never mutable handles. `result.child_updates` exposes separate completed
child forks, including descendants. The caller selects and publishes future child
cache revisions; no merge or external cache update is automatic.

```rust,ignore
let snapshot = agent.snapshot(); // keep this value alive while borrowing views
for conversation in snapshot.conversations() {
    println!("{:?}: {:?}", conversation.env_ids(), conversation.get_last_answer_asset());
    println!("{:?}", conversation.get_final_tool_input_asset("submitFinalAnswer"));
    println!("{:?}", conversation.messages());
}
```

`Conversation<'_, P>` borrows an immutable snapshot and retains its path,
unaffected by later Agent publication. `Conversations` lazily visits the history
trie's structural leaves without cloning contents; requesting owned `messages()`
requires Clone. `conversation.snapshot()` exposes its anchor. Child conversations
are separate: inspect `snapshot.child(entry).conversations()`.

An asset is the value the caller cares about, not a library wrapper:
- `get_last_answer_asset() -> Option<&P>` selects the latest no-tool model answer,
  including one before an open follow-up. `Some(&None)` is a real optional null
  answer, distinct from missing (`None`).
- `get_final_tool_input_asset(tool_name) -> Option<&serde_json::Value>` selects the
  latest matching tool-call input, scanning turns and calls in reverse order.
  It returns the input, not the tool output. `Some(&Value::Null)` is a real null
  input, distinct from missing. A final-answer tool is still an ordinary tool;
  this accessor adds no execution side effects or termination/publication policy.

Both accessors include initial model envelopes and require neither Clone nor
payload equality. Initial assistant context does not consume the turn budget or
make a budget-limited execution a completed answer: the last-answer accessor may
expose an initial answer while the current history position remains open. Only
an accepted terminal response at that position permits a user continuation.
Conversation evidence retains each environment's actual outputs,
while shared model-visible content uses historical representatives. Custom
equivalence does not provide cross-environment isolation.

There is one persistent history trie, including every non-splitting tool-result
checkpoint. `BatchResult` contains its snapshot plus flat run-local `classes`,
`results`, and `child_updates`; it has no second partition tree. The returned
classes describe only this execution, not all historical environments.

## Examples, convenience drivers and limits

From the repository root, no providers or credentials:

```sh
for example in product_screening history_replay feedback specialized_subagents agent_conversations; do
    cargo run -p lockstep-loop --example "$example"
done
```

The one-shot `run_to_completion`/`run_to_completion_by` local-tool drivers always
record and return snapshots. The custom one-shot driver can still record opaque
`P: Clone` values without PartialEq because it does not restore prior roots.
Use Agent for repeated investigations, replay, feedback and resolver-driven work.
The doc-hidden `BatchLoop`/`LoopConfig` remain execution-engine interfaces for
regression testing, not a second recommended entry point.

No async loader, eviction/GC, root merging, crash recovery, DAG scheduling,
designated result-tool publication or UI. Deep paths can produce quadratic
path-copy writes; edge lookup is linear. Valid imported graphs/codecs are caller
responsibilities. See the [full reference](../docs/history-reference.md).

```sh
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```
