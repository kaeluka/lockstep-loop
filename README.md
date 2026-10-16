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

- `History` stores the prompt, tool specifications, reuse contract and root.
- `HistoryNode` stores the full assistant message (including tool calls/arguments)
  and the environment visits that used it. An absent assistant is an unfulfilled
  continuation, not a terminal answer.
- Each `Observation` stores ordered representative `OutputRecord`s, **all exact
  per-environment `EnvObservation` evidence**, and a continuation node. Each
  output record has a `tool_output` field for future metadata extensions.
- Conversation reconstruction starts with the prompt and alternates assistant
  messages and representative tool messages. With custom lossy grouping, replay
  keeps the **original representative conversation**, while appending the new
  environment's actual, possibly unequal outputs to evidence.
- Nodes are immutable values: Python uses frozen dataclasses and tuples; Rust
  uses copy-on-write `Arc` nodes. Updates copy the ancestor path and share
  unchanged subtrees. Snapshots are available during a run and in the result.
  The old `result.tree` remains a per-run partition summary, **not replay history**.

Matching verifies equality of complete output tuples (or caller-defined group
keys); hashes/debug strings are never record identities or substitutes for
outputs. Branch IDs are run-local scheduling identifiers. Repeated environment
IDs append visits/evidence rather than overwriting them; use revision-qualified
IDs if you need to distinguish runs in an inspector.

### Compatibility and ownership

Replay requires an explicit, non-empty caller-managed **contract version**. The
core rejects a changed contract, unequal prompt, changed ordered tool specs, or
a switch between default and custom grouping. The caller must also change the
contract when tool implementation **semantics**, grouping semantics, provider
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

### Persistence and first-slice limits

The core performs **no filesystem, database or network I/O**. Callers save/load
`History` using their own codecs. Rust records derive Serde when `P` supports it;
Python records can be encoded by callers (the example uses trusted pickle).
Restore valid snapshots generated by this version; arbitrary/malformed imported
graphs are not validated, and no stable cross-language or cross-version wire
format is promised. Generic payload codecs remain the caller's responsibility.

This first slice is **eager/in-memory**: replay traverses only its selected path,
but the complete history must already be loaded. There is no lazy loader/reference
protocol yet. A future storage-neutral node-reference seam can replace direct
child references; this release does not claim lazy persistence. Path copying and
linear observation-edge lookup favor a small implementation over large-history
scalability.

Recording/replay explicitly rejects configured **subagents and budgets**; their
existing non-recording behavior is unchanged. History is enabled on the direct
`BatchLoop` API, not the convenience drivers. A captured partial history can reuse
fulfilled assistant turns, but partial tool collections and request queues are
not persisted: tools restart on replay. This is **not crash recovery**. Validation
errors (wrong completion kind, duplicate call IDs, invalid environment result
sets) leave pending requests recoverable; caller callback panics/side effects are
outside that guarantee. Python additionally validates grouping before consuming
a tool completion. Concurrent extensions of *separate* roots are not merged.
Feedback, resuming terminal conversations, publication, DAG scheduling and UIs
remain future work.

Executable caller-side persistence/replay examples (no provider needed):

```sh
(cd python && PYTHONPATH=src python examples/history_replay.py)
cargo run -p lockstep-loop --example history_replay
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
