# lockstep-loop for Rust

Provider-agnostic lockstep orchestration for running one LLM conversation over
many similar, read-only environments and splitting it only when tool results
diverge.

[Start with the visual guide](../docs/concepts.md). `Agent` contains its initial
prompt, reuse contract, and complete accumulated investigation trie (`History`
remains an alias). An `Asset` is a read-only leaf investigation backed by its full
path, not an answer or environment set. See the [glossary/API mapping](../docs/agent-assets.md).

The crate never calls an LLM. Drive `BatchLoop` directly for arbitrary scheduling,
or use `run_to_completion` for a serial local-tool driver. Shared reasoning saves
LLM work; tools still execute per environment. Classes never remerge within a run.

This complete, provider-free example immediately answers a no-tool investigation:

```rust
use lockstep_loop::{
    Completion, LlmCompletion, LoopConfig, BatchLoop, Request, RequestKind,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "inspect the repository".to_owned(),
        vec!["fork-a".to_owned(), "fork-b".to_owned()],
        vec![],
    ))?;

    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any) {
            Some(Request::Llm(request)) => {
                // Replace this stand-in with your provider adapter.
                loop_.complete(Completion::Llm(LlmCompletion::final_answer(
                    request.request_id,
                    "done".to_owned(),
                )))?;
            }
            // This stand-in requests no tools and completes work synchronously.
            _ => unreachable!("no tool calls or outstanding asynchronous work"),
        }
    }

    let result = loop_.result()?;
    assert_eq!(result.results["fork-a"], "done");
    Ok(())
}
```

With real tools, handle `Request::Tool` by executing its call once per
`request.env_ids` entry, then return a `ToolCompletion` containing every result.
In asynchronous drivers, `None` means no request is currently runnable, not done;
wait for outstanding completions and use `is_done()` for termination.
The snippets below illustrate integration; use the linked examples for complete
imports, tool setup, and persistence backends.

## Payloads and grouping

Conversation content and tool results are generic in `P`. `BatchLoop::new`
groups results by value equality; `BatchLoop::with_group_by` accepts a custom
key function for deliberate lossy grouping. Tool arguments and JSON schemas
use `serde_json::Value`.

Rust budgets include an exhaustion callback (`Budget::new(max_steps, f)`) so
the caller can produce a valid value of the opaque payload type.

## Recording and replay

Opt into recording through `LoopConfig::with_recording(contract)`. Restore a
snapshot with `with_agent(agent, expected_contract)` (`with_replay` remains an alias);
constructors still use
`BatchLoop::new` (exact equality) or `BatchLoop::with_group_by` (explicit lossy
policy). Existing loops without these options behave as before. Constructor-based
callers remain supported. Public configuration/result fields keep compatibility
spellings (`history`, `history_resolver`, `child_histories`); prefer the canonical
builders and `agent()` / `child_agents()` getters. No duplicate state is stored.
Legacy `group_key` labels retain debug
renderings of result tuples. Equal renderings are allowed on distinct edges in
the result's `Vec`; internal branch IDs remain collision-free.

```rust
let config = LoopConfig::new(prompt.clone(), vec!["revision-1".into()], specs.clone())
    .with_recording("inspect-v1/tools-v1/exact-grouping/provider-v1")
    .with_record_store(RecordStore::new("caller-reserved-writer-namespace"));
let mut loop_ = BatchLoop::new(config)?;
// Serve normal requests/completions.
let snapshot = loop_.agent().unwrap(); // immutable agent, including active children
// After completion, loop_.result()?.into_agent() moves out the updated agent.
let replay_config = LoopConfig::new(prompt, vec!["revision-2".into()], specs)
    .with_agent(snapshot, "inspect-v1/tools-v1/exact-grouping/provider-v1");
let mut replay = BatchLoop::new(replay_config)?;
```

Run the complete executable [caller-side example](examples/history_replay.rs)
from the repository root:

```sh
cargo run -p lockstep-loop --example history_replay
# unchanged: 0 LLM calls; changed: 1; both paths reusable
```

It uses ordinary `serde_json::to_vec`/`from_slice` for **each shallow record** and
`AgentDescriptor<String>`, not a recursively encoded `Agent`. No recursion
limit adjustment is needed for deep histories. Storage/codecs are caller-owned.

`Agent { descriptor, store }` pairs a small envelope with its resolver.
`HistoryNode` is the one internal/public record model. Its observation edges use
`NodeRef { namespace, index }` addresses, ordered representative
`OutputRecord { tool_output }` values and exact per-environment evidence. Use
`agent.root()` and `agent.load(&edge.continuation)` to inspect nodes. Lossy
grouping keeps the original representative context without discarding actual
outputs. Immutable path-copying updates preserve old records/references. Editing
owned clones creates different values; payload aliases must remain immutable.

For persistence, iterate `agent.store.records()` and encode each `(NodeRef,
Arc<HistoryNode<P>>)` individually. Insert immutable records first (reject address
collisions), then atomically publish `agent.descriptor`. Restore with:

```rust
let agent = Agent {
    descriptor: serde_json::from_slice::<AgentDescriptor<String>>(&published)?,
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
[load/update/publish protocol](../docs/history-reference.md#individually-addressable-records-and-lazy-persistence).

**Nested subagents:** `HistoryNode.subagents` entries associate an assistant call
ID with a latest child descriptor and immutable `runs` recording inherited envs,
selected bases and resulting revisions. Inspect with `agent.child(entry)`.
Core replay reruns child
tools lazily, supports mixed ordinary/subagent calls and configured nesting depth,
and folds child final answers into parent matching. Parent classes do not split
merely because child traces differ when their answers agree. Active-child
snapshots preserve fulfilled turns without serializing pending tool builders.

**Compatibility:** contract IDs must be non-empty and equal; prompt equality,
ordered tool specs, default/custom grouping mode and subagent spec/depth are also
checked. Child contracts are checked on entry. The
`with_agent` builder requires `P: PartialEq` for prompt checking (recording alone
does not). The caller versions tool implementation semantics, custom grouping,
subagent prompt/spec-building callbacks, and provider/adapter settings in the
contract. Closures cannot be introspected.
Payload `Clone` must have immutable value semantics: interior-mutable aliases must
not change submitted facts. Grouping, equality and hashing must be deterministic;
callback panics are not recoverable validation errors. Child prompts, specs and
constructors are prepared once before fulfilling/recording the parent assistant
turn or issuing its ordinary requests; commit reuses those prepared children.
Library-detected child compatibility errors return `IncompatibleHistory` rather
than panicking. Errors during preparation of a fresh LLM completion leave that
request outstanding for correction. Construction rejects incompatible history;
errors replaying an already recorded assistant turn become terminal errors
(`error()`/`result()`), cancelling work without a partial answer or half-started
assistant turn. Preceding valid tool evidence may remain in the working snapshot;
old snapshots stay immutable. Caller
side effects and orphan append-only preparation records are not rolled back.
Discard the live loop after an arbitrary callback panic. See the
[transaction boundary](../docs/history-reference.md#transactional-child-preparation-and-callback-failures).

### Inspecting assets

```rust
let agent = result.agent().unwrap();
for asset in agent.assets() {
    println!("{:?}: complete={}", asset.env_ids(), asset.is_complete());
    if let Some(value) = asset.value() { println!("{value:?}"); }
    println!("{:?}", asset.messages());
    // asset.observations(): selected checkpoints with full per-env evidence
}
```

`Asset<'_, P>` borrows the immutable agent and retains its root-to-leaf record
references. Keep that agent value alive while using views. Inspection does not
clone payloads; owned `messages()` requires `P: Clone`. An open asset may retain
an ancestor's published value while `is_complete()` is false, or have no value.
`Some(&None)` is a real null optional payload, distinct from no publication.
Known `env_ids()` are checkpoint/visit evidence, not a global latest partition.
The view has no persisted ID or feedback/resume API. Child-agent leaves are not
parent assets; inspect `agent.child(entry).assets()` separately. Traversal is lazy,
iterative, read-only, and performed only when requested.

```sh
cargo run -p lockstep-loop --example agent_assets
```

The [end-to-end example](examples/agent_assets.rs) loads an agent, replays an
environment, inspects its asset, resolves feedback through tools again, and saves
the new agent while retaining old/open snapshots. See
[view semantics and compatibility](../docs/agent-assets.md).

### Specialized tools and independent child caches

```rust
let imports = SubagentTool::try_new(
    ToolSpec::new("list_imports", "List imported packages.", schema),
    |args| {
        let path = args["file_path"].as_str().ok_or_else(||
            LoopError::InvalidSubagentArguments("file_path must be a string".into()))?;
        Ok(format!("List imports in {path:?}."))
    },
    |_| Ok(repo_tool_specs()),
    1,
).with_agent_contract("imports/template-v1/tools-v1/grouping-v1/provider-v1")
 .with_subagents(vec![]);

let config = LoopConfig::new(parent_prompt, env_ids, ordinary_parent_specs)
    .with_subagents(vec![imports, other_specialist])
    .with_recording("parent/v2")
    .with_record_store(RecordStore::with_loader(fresh_writer_namespace, load_node))
    .with_agent_resolver(move |query| {
        Ok(selected.get(&(query.contract.clone(), query.prompt.clone())).cloned())
    });
let mut loop_ = BatchLoop::new(config)?;
```

Run the complete [caller-side example](examples/specialized_subagents.rs):

```sh
cargo run -p lockstep-loop --example specialized_subagents
# Changed parent: 2 fresh parent LLM calls, 0 child LLM calls, 1 child tool rerun
```

All registered specs are advertised automatically. `build_prompt` produces initial
**user** content, not a system message. Structured argument interpretation,
validation, deterministic templates and any path normalization belong to the caller.
There is no JSON Schema validator or fuzzy matching. Different file paths under
the example's template get different prompts/tries, not new environment IDs.
Children always inherit the parent's environment set.

`with_agent_resolver` accepts a synchronous caller callback returning
`Result<Option<AgentDescriptor<P>>, LoopError>` for an `AgentQuery<P>` containing
only child prompt, child contract version, ordered tools, grouping mode and nested
configuration. It requires `P: PartialEq`, like `with_agent`, for full compatibility
checks. Return None for a miss; return a chosen descriptor for a hit. Index hashes
or abbreviated keys never replace core equality checks. Each registered definition
must have an explicit non-empty `.with_agent_contract(...)` when using a resolver.
Child grouping is inherited: its contract must cover that function's semantics,
child tools/templates and provider settings, independently of parent wording.
Without a resolver, missing child contracts retain legacy parent-owned recording;
explicit child contracts also enable recording beneath an unrecorded parent.

The existing `RecordStore` is the node-loading boundary for both parent and selected
child roots. Supply a shared in-memory store or one whose loader resolves all
referenced addresses across your backends. Metadata selection does not import a
foreign store. Persist shallow writes before publishing descriptors; no core I/O
or parallel persistence model is added. Agent snapshot selection remains `with_agent`.

`BatchResult::child_agents()` exposes separate `ChildAgentUpdate` values for
completed recorded invocations, including descendants when the containing child
finishes. Each exposes `agent()` (descriptor + store), selected `base`, original
`tool_call`, inherited `env_ids`, and parent branch/depth plus the existing run-local
request prefix for provenance. No runtime prefix is used in cache lookup or accepted
as a feedback target. `HistoryNode.subagents[*].runs` shares immutable invocation
records through `Arc`, retaining old roots even as `agent_descriptor()` exposes
the latest run (stored field `history`).
Persist updates with the normal descriptor/record codecs. A store's writes may
include parent/orphan records, but a child descriptor needs no parent descriptor
to replay. Concurrent hits on one base remain independent executions and updates;
there is no silent last-writer-wins cache, root merge or single-flight protocol.

**Revision selection:** recorded parent invocations pin their child revision and
skip external lookup; new invocations use the caller resolver. Selected child
feedback histories replay their shared-path continuations normally. A prompt key
need not imply a unique revision. Updating a cache pointer does not automatically
update completed parents. Start a new parent investigation, or create new
invocations after explicit parent feedback, to choose a newer child revision.
Strict feedback resolution never manufactures missing parent provenance from a
cache hit. There is no DAG invalidation/propagation service here.

**Registration/nesting:** mixing `with_subagent` and `with_subagents` is rejected.
Duplicate names and conflicting ordinary/subagent specs are rejected; an exactly
matching `ToolSpec` is accepted as a redundant legacy advertisement. Serial drivers
now register all `ToolEntry::Subagent` entries and reject actual callable/subagent
name collisions, even with identical specs. The default serial driver records
exact-equality grouping; `_by` records explicit custom grouping. Use direct
`BatchLoop` for resolver-driven serving. `SubagentTool::with_subagents` defines the
child allowlist: empty is a leaf, None preserves bounded self-recursion, never all
parent siblings. Nested depths are clamped to the remaining allowance; disabled
reserved names are omitted/rejected rather than passed to ordinary execution.

Use `try_new` for fallible prompt/spec factories. On a fresh LLM completion,
`InvalidSubagentArguments`, `SubagentPreparation`, resolver errors (for example
`HistoryResolution`) and incompatible selected descriptors leave the request
correctable without partial spawning. Replay-time preparation errors are terminal;
arbitrary callback panics remain unrecovered. No factories are rerun at commit.
Budgets remain unsupported wherever child recording is enabled. No environment
binding/remapping, new system-role API, logical asset registry or scheduler is added.

**Naming compatibility:** canonical names are aliases to the existing record types,
not a second model. Serde names/fields remain unchanged, including `history` and
`history_contract`. No storage migration is required for this rename. The earlier
specialized-subagent change introduced ordered `subagents` and `runs`; migrations
from before that change remain caller-owned. Existing constructors and legacy
getters/setters stay supported. See the [exact mapping](../docs/agent-assets.md#preferred-names-and-compatibility).

### Feedback and shared-path resumption

```rust
let config = LoopConfig::new(prompt, vec!["revision-1".into()], specs)
    .with_agent(snapshot, "inspect-v1/tools-v1/exact-grouping/provider-v1")
    .with_feedback("Also inspect /foo and revise if needed.".to_owned());
let mut loop_ = BatchLoop::new(config)?;
// Serve ordinary requests/completions. Resolution reruns tools with no fresh
// LLM work; once the old answer is located, feedback is appended and work resumes.
// let revised = loop_.result()?;  // after completion; revised.agent() is a new snapshot
```

Run the complete [caller-side feedback example](examples/feedback.rs):

```sh
cargo run -p lockstep-loop --example feedback
```

A feedback operation requires a supplied agent and **one environment for locating
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
completions return the stored error. `agent()` reverts to the supplied root on
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

**Limits:** recording still rejects budgets; root agent recording/replay requires
direct `BatchLoop`. Serial drivers can expose recorded child-agent updates.
No core storage I/O, async loader, eviction/GC, or independent-root merge.
Stores retain local writes/cached loads; deep path copying can have quadratic
write volume. Pending requests/partial tuples are not persisted, so tools restart
on replay: this is not in-flight crash recovery. Invalid completions remain
correctable, including nested routes. `stats.depth` includes reused turns;
`split_depth` is local to the new execution. Multiple specialized tools and
independent child-agent lookup are supported. Asset views are inspection only;
result-tool publication, DAG scheduling and UI remain unimplemented. See the
[shared design and limitations](../docs/history-reference.md#immutable-recorded-history-and-incremental-replay).

See `tests/loop.rs`, `tests/history.rs`, `tests/history_foundation.rs`,
`tests/request_routing.rs`, `tests/feedback.rs`, `tests/specialized_subagents.rs`
and `tests/agent_assets.rs` for
regressions including 256-turn shallow-record persistence with default JSON
limits, lazy traversal, nested evidence/equal parent answers, open user
continuations, shared-path feedback inheritance, explicit resolution misses,
immutable snapshots, arbitrary completion order and compatibility.
