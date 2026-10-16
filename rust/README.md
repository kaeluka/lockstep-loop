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
`LoopConfig` or `BatchResult` struct literals must add `history: None`; existing
constructor-based code is unchanged. Legacy split-edge `group_key` strings are
now opaque, collision-free labels rather than debug renderings of group keys.

```rust
let config = LoopConfig::new(prompt.clone(), vec!["revision-1".into()], specs.clone())
    .with_recording("inspect-v1/tools-v1/exact-grouping/provider-v1");
let mut loop_ = BatchLoop::new(config)?;
// Serve normal requests/completions.
let snapshot = loop_.history().unwrap(); // cheap structurally shared root
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

It uses caller-side `serde_json::to_vec`/`from_slice` to round-trip `History<String>`;
all storage and codecs belong to the caller. History records derive Serde when
`P` supports it. There is no core storage I/O or stable cross-language/version wire
format. Load valid snapshots from this version, not arbitrary/malformed graphs.

`History` contains the prompt, ordered specs, contract and `Arc<HistoryNode<P>>`
root. Nodes contain full assistant messages, environment visits and observation
edges. Each edge contains ordered representative `OutputRecord { tool_output }`
values, exact per-environment `EnvObservation` evidence, and an `Arc` continuation.
Reconstruct the model conversation by alternating assistant messages with
representative tool messages in call order. Lossy grouping never discards the
actual unequal evidence, and replay keeps the original representative context.
Copy-on-write updates share unchanged subtrees; captured roots cannot be changed
by extending a loop. Public owned records can be edited for caller codecs, but
that creates a different value, not permission to mutate payload aliases.

**Compatibility:** contract IDs must be non-empty and equal; prompt equality,
ordered tool specs, and default/custom grouping mode are also checked. The
`with_replay` builder requires `P: PartialEq` for prompt checking (recording alone
does not). The caller versions tool implementation semantics, custom grouping,
and provider/adapter settings in the contract. Closures cannot be introspected.
Payload `Clone` must have immutable value semantics: interior-mutable aliases must
not change submitted facts. Grouping, equality and hashing must be deterministic;
callback panics are not recoverable validation errors.

**Limits:** recording rejects configured subagents and budgets; use direct
`BatchLoop`, not convenience drivers. Histories are fully loaded in memory; lazy
loading is deferred. Pending requests/partial tool tuples are not persisted, so
reusing a partial snapshot restarts tools, not in-flight execution. Validation
errors do leave outstanding requests correctable, and fulfilled completions
cannot overwrite history. `stats.depth` includes reused turns; `split_depth` is
local to the new execution. There is no feedback/assets/DAG scheduling or merging
of independently extended roots. See the
[shared design and limitations](../README.md#immutable-recorded-history-and-incremental-replay).

See `tests/loop.rs` and `tests/history.rs` for regression tests covering partition
refinement, arbitrary completion order, exact evidence, snapshots, persistence,
replay divergence, and compatibility.
