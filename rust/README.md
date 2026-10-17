# lockstep-loop for Rust

Provider-agnostic lockstep orchestration for running one LLM conversation over
many similar, read-only environments and splitting it only when tool results
diverge.

The crate never calls an LLM. Drive [`BatchLoop`] directly for arbitrary
scheduling, or use `run_to_completion` for a serial local-tool driver.

```rust
use lockstep_loop::{
    Completion, LlmCompletion, LoopConfig, BatchLoop, Request, RequestKind,
};

let mut loop_ = BatchLoop::new(LoopConfig::new(
    "inspect the repository".to_owned(),
    vec!["fork-a".to_owned(), "fork-b".to_owned()],
    vec![],
))?;

while !loop_.is_done() {
    match loop_.next_request(RequestKind::Any) {
        Some(Request::Llm(request)) => {
            // Adapt `request.messages` and `request.tools` to any provider.
            loop_.complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id,
                "done".to_owned(),
            )))?;
        }
        Some(Request::Tool(request)) => {
            // Execute request.tool_call once for every request.env_ids entry,
            // then return one result per environment.
            todo!()
        }
        None => wait_for_an_outstanding_completion(),
    }
}

let result = loop_.result()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Payloads and grouping

Conversation content and tool results are generic in `P`. `BatchLoop::new`
groups results by value equality; `BatchLoop::with_group_by` accepts a custom
key function for deliberate lossy grouping. Tool arguments and JSON schemas
use `serde_json::Value`.

Rust budgets include an exhaustion callback (`Budget::new(max_steps, f)`) so
the caller can produce a valid value of the opaque payload type.

## Recording and replay

Opt into recording through `LoopConfig::with_recording(contract)`. Restore a
snapshot with `with_replay(history, expected_contract)`; constructors still use
`BatchLoop::new` (exact equality) or `BatchLoop::with_group_by` (explicit lossy
policy). Existing loops without these options behave as before. Code using
`LoopConfig` or `BatchResult` struct literals must add `history: None`;
`LoopConfig` literals also need `feedback: None`. Existing constructor-based code
is unchanged. Legacy `group_key` labels retain debug
renderings of result tuples. Equal renderings are allowed on distinct edges in
the result's `Vec`; internal branch IDs remain collision-free.

```rust
let config = LoopConfig::new(prompt.clone(), vec!["revision-1".into()], specs.clone())
    .with_recording("inspect-v1/tools-v1/exact-grouping/provider-v1")
    .with_record_store(RecordStore::new("caller-reserved-writer-namespace"));
let mut loop_ = BatchLoop::new(config)?;
// Serve normal requests/completions.
let snapshot = loop_.history().unwrap(); // immutable root, including active children
// Also available as loop_.result()?.history after completion.
let replay_config = LoopConfig::new(prompt, vec!["revision-2".into()], specs)
    .with_replay(snapshot, "inspect-v1/tools-v1/exact-grouping/provider-v1");
let mut replay = BatchLoop::new(replay_config)?;
```

Run the complete executable [caller-side example](examples/history_replay.rs)
from the repository root:

```sh
cargo run -p lockstep-loop --example history_replay
# unchanged: 0 LLM calls; changed: 1; both paths reusable
```

It uses ordinary `serde_json::to_vec`/`from_slice` for **each shallow record** and
`HistoryDescriptor<String>`, not a recursively encoded `History`. No recursion
limit adjustment is needed for deep histories. Storage/codecs are caller-owned.

`History { descriptor, store }` pairs a small envelope with its resolver.
`HistoryNode` is the one internal/public record model. Its observation edges use
`NodeRef { namespace, index }` addresses, ordered representative
`OutputRecord { tool_output }` values and exact per-environment evidence. Use
`history.root()` and `history.load(&edge.continuation)` to inspect nodes. Lossy
grouping keeps the original representative context without discarding actual
outputs. Immutable path-copying updates preserve old records/references. Editing
owned clones creates different values; payload aliases must remain immutable.

For persistence, iterate `history.store.records()` and encode each `(NodeRef,
Arc<HistoryNode<P>>)` individually. Insert immutable records first (reject address
collisions), then atomically publish `history.descriptor`. Restore with:

```rust
let history = History {
    descriptor: serde_json::from_slice::<HistoryDescriptor<String>>(&published)?,
    store: RecordStore::with_loader("fresh-reserved-writer-namespace", move |reference| {
        // Caller loader: read/decode ONE immutable record; no replay/matching here.
        Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
    }),
};
```

A namespace must be unique for every independent writable store in the persistent
storage domain. Reserve one using your own UUID/sequence/transaction scheme;
these are **storage names, not public conversation identities**. The default
store is convenient for process-local execution only. Durable recordings should
use `with_record_store(RecordStore::new(namespace))`; restored stores need a fresh
namespace. Cloning a store shares its allocator/cache. Record references use
exact address equality, not hashes/debug strings. Loaders must be synchronous,
non-reentrant and total for valid references; failures/panics are fatal and
require restarting from a saved descriptor. Records/descriptors derive Serde when
`P` supports it; live resolver handles do not. No stable cross-language/version
wire format or validation of arbitrary imported graphs is promised. See the
[load/update/publish protocol](../README.md#individually-addressable-records-and-lazy-persistence).

**Nested subagents:** `HistoryNode.subagents` entries associate an assistant call
ID with a full child descriptor. Inspect using `History { descriptor:
entry.history.clone(), store: history.store.clone() }`. Core replay reruns child
tools lazily, supports mixed ordinary/subagent calls and configured nesting depth,
and folds child final answers into parent matching. Parent classes do not split
merely because child traces differ when their answers agree. Active-child
snapshots preserve fulfilled turns without serializing pending tool builders.

**Compatibility:** contract IDs must be non-empty and equal; prompt equality,
ordered tool specs, default/custom grouping mode and subagent spec/depth are also
checked. Child contracts are checked on entry. The
`with_replay` builder requires `P: PartialEq` for prompt checking (recording alone
does not). The caller versions tool implementation semantics, custom grouping,
subagent prompt/spec-building callbacks, and provider/adapter settings in the
contract. Closures cannot be introspected.
Payload `Clone` must have immutable value semantics: interior-mutable aliases must
not change submitted facts. Grouping, equality and hashing must be deterministic;
callback panics are not recoverable validation errors. Child prompts, specs and
constructors are prepared once before fulfilling/recording the parent assistant
turn or issuing its ordinary requests; commit reuses those prepared children.
Library-detected child compatibility errors return `IncompatibleHistory` rather
than panicking. Construction rejects the loop; a late mismatch becomes a terminal
error (`error()`/`result()`), cancelling pending work without publishing a partial
answer or partially starting the rejected assistant turn. Preceding valid tool
evidence may remain in the working snapshot; old snapshots stay immutable. Caller
side effects and orphan append-only preparation records are not rolled back.
Discard the live loop after an arbitrary callback panic. See the
[transaction boundary](../README.md#transactional-child-preparation-and-callback-failures).

### Feedback and shared-path resumption

```rust
let config = LoopConfig::new(prompt, vec!["revision-1".into()], specs)
    .with_replay(snapshot, "inspect-v1/tools-v1/exact-grouping/provider-v1")
    .with_feedback("Also inspect /foo and revise if needed.".to_owned());
let mut loop_ = BatchLoop::new(config)?;
// Serve ordinary requests/completions. Resolution reruns tools with no fresh
// LLM work; once the old answer is located, feedback is appended and work resumes.
// let revised = loop_.result()?;  // after completion; revised.history is a new snapshot
```

Run the complete [caller-side feedback example](examples/feedback.rs):

```sh
cargo run -p lockstep-loop --example feedback
```

A feedback operation requires supplied history and **one environment for locating
the target**, not an environment-specific scope. In the new snapshot, **every
environment reaching that observation path inherits the continuation**, including
old siblings and new commit IDs. Additional tools can split them on new evidence.
Unrelated paths and old roots are untouched. There is no session/endpoint handle
or value-triggered feedback map. Source-environment metadata is provenance only.
A new explicit operation resolves the latest completed continuation and appends
another message, even if identical. Returning the same answer is valid termination.

`LoopConfig.feedback` is `Option<P>`: `None` means absent; for optional payloads
`.with_feedback(None)` sets `Some(None)` and submits real null content. The same
immutable payload/Clone contract applies. Full feedback messages live in
`node.user_continuations[0].message`, with shallow continuation references in the
existing record model. The old answer stays at the preceding node, and snapshots
captured while the new assistant is pending retain an open continuation.

**Failure protocol:** `LoopError::FeedbackResolutionMiss` means the selected path
is unseen/unfulfilled; it never generates an answer to manufacture a target. It
can be returned by construction or by `complete` after valid tool evidence arrives.
A later failure cancels queued/outstanding work; `is_done()` is true, `error()`
exposes the failure, `next_request()` returns None, and `result()`/further
completions return the stored error. `history()` reverts to the supplied root on
a resolution miss; append-only orphan writes may remain. Invalid completion sets,
duplicate IDs and wrong kinds remain correctable and do not set `error()`.
An open earlier feedback continuation cannot be targeted by another explicit
intervention; ordinary replay can resume it. Multiple competing user successors
or malformed user edges yield `InvalidUserContinuation` on traversal, not an
implicit merge or selection. The provisional `requires_user_policy` field and
`UserContinuationPolicyRequired` error have been removed.

Feedback targets this loop's terminal no-tool assistant response. Cached nested
child tools are rerun during resolution with no fresh reasoning allowed on misses;
after injection, new child work is supported normally. No nested-asset selection,
result-tool publication or feedback-routing framework is added. Custom lossy
grouping preserves its original representative context and full actual evidence;
feedback is inherited by the entire grouped path. The legacy `append_user_message`
helper still forks an already completed live run without resuming that loop;
constructor feedback is the strict resolution/resumption operation.

**Caveat:** this is a path-wide investigative improvement, not a private annotation.
Use transferable instructions (“check whether X”) rather than unqualified
commit-specific assertions, or keep a separate snapshot. No rewriting, semantic
classification, alternate scope modes or environment binding is implemented.
Budgets remain unsupported for recorded execution, before and after injection.
“Once” is per live operation, not crash-safe exactly-once publication.

**Limits:** recording still rejects budgets; use direct `BatchLoop`, not convenience
drivers. No core storage I/O, async loader, eviction/GC, or independent-root merge.
Stores retain local writes/cached loads; deep path copying can have quadratic
write volume. Pending requests/partial tuples are not persisted, so tools restart
on replay: this is not in-flight crash recovery. Invalid completions remain
correctable, including nested routes. `stats.depth` includes reused turns;
`split_depth` is local to the new execution. See the
[shared design and limitations](../README.md#immutable-recorded-history-and-incremental-replay).

See `tests/loop.rs`, `tests/history.rs`, `tests/history_foundation.rs`,
`tests/request_routing.rs` and `tests/feedback.rs` for
regressions including 256-turn shallow-record persistence with default JSON
limits, lazy traversal, nested evidence/equal parent answers, open user
continuations, shared-path feedback inheritance, explicit resolution misses,
immutable snapshots, arbitrary completion order and compatibility.
