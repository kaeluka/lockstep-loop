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
use lockstep_loop::{Agent, AgentConfig, Completion, LlmCompletion, RequestKind, RunOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let agent = Agent::new(AgentConfig::new("Explain the checklist.".to_owned(), vec![]))?;
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

Caller-owned storage uses shallow records and descriptors:

```rust,ignore
let snapshot = agent.snapshot();
for (reference, record) in snapshot.store.records() {
    backend.insert_immutable(reference, encode_node(&record));
}
let published = serde_json::to_vec(&snapshot.descriptor)?; // AFTER records are durable

let snapshot = AgentSnapshot {
    descriptor: serde_json::from_slice(&published)?,
    store: RecordStore::with_loader("fresh-reserved-writer", move |reference| {
        Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
    }),
};
let agent = Agent::from_snapshot(snapshot, AgentRuntime::default())?;
// No repeated prompt/specs and no record loading until start/traversal.
```

`RecordStore` is an append-only write set/cache, not a backend. Loaders are
synchronous and must resolve valid immutable records without reentering the store.
Reserve a unique namespace for every durable writer (the default is process-local
only). Cloning a store shares its allocator/cache; independently reusing its
namespace is invalid. Save local writes before descriptors, retain referenced old
records, and reject collisions. The core traverses only relevant paths and performs
no persistence I/O. Descriptors/records derive Serde when `P` supports it. No stable
cross-language/version wire format or migration aliases are promised.

### Runtime configuration and grouping

Fresh configuration: `AgentConfig::new(prompt, tool_specs)`, optionally
`.with_subagents(...)`, `.with_agent_resolver(AgentResolver::new(...))`, and
`.with_record_store(...)`. Restored agents take `AgentRuntime { subagents,
agent_resolver }`; factories must match recorded structural configuration.
Executable callbacks are never serialized.

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

Structural checks cover prompts, ordered specs, nested allowlists/depth, and actual
child prompt/spec factory outputs. Provider/adapter and grouping-policy isolation
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

### Children and Asset views

`SubagentTool` factories deterministically build initial user prompts and tool
specs. Nested allowlists/depth caps are enforced. A resolver receives a child
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
for asset in snapshot.assets() {
    println!("{:?}: {:?}", asset.env_ids(), asset.value());
    println!("{:?}", asset.messages());
}
```

`Asset<'_, P>` borrows an immutable snapshot and retains its path, unaffected by
later Agent publication. Traversal is lazy and does not clone payloads; requesting
owned `messages()` requires Clone. An open continuation may retain a previous
value; `Some(&None)` differs from no publication. `asset.snapshot()` exposes its
anchor. Child leaves are separate: inspect `snapshot.child(entry).assets()`.

## Examples, convenience drivers and limits

From the repository root, no providers or credentials:

```sh
for example in product_screening history_replay feedback specialized_subagents agent_assets; do
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
