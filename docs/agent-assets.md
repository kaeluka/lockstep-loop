# Agents, conversations, and assets

[Visual guide](concepts.md) · [Documentation](README.md) ·
[History/persistence reference](history-reference.md)

## Glossary

| Term | Meaning |
| --- | --- |
| **Agent** | Mutable handle owning the current immutable history root and caller-supplied runtime dependencies. The primary entry point. |
| **AgentSnapshot** | Ordered initial messages, structural configuration, and an immutable history trie, paired with its record resolver. |
| **Investigate environments** | An operation on an agent. Its `AgentRun` handle holds temporary request/completion state and extends a captured root. |
| **History trie** | Recorded conversations and tool-result evidence, sharing common prefixes. The single persistent history model. |
| **Conversation** | One ordered exchange along a history path, exposed as a snapshot-anchored view. |
| **MessageEnvelope** | Role and call metadata around an opaque message value. Initial input is a nonempty ordered sequence of envelopes, not one text prompt. |
| **Asset** | The value the caller cares about, extracted from a conversation—not a conversation wrapper or identity. |
| **History node** | A shared prefix or historical endpoint; not necessarily a leaf. |
| **Refinement group** | Environments sharing a conversation during one execution. Flat `EnvClass` results describe that execution, not accumulated history. |
| **Subagent tool** | Deterministic initial-message/specification factories plus an advertised tool definition, not a persisted child Agent handle. |

An Agent retains experience through sequential starts. Each start captures the
current root; successful completion automatically replaces it once. A run's
snapshot stays independent after another run publishes. Restore/fork via
`Agent.from_snapshot`; loaders and runtime callbacks are caller-supplied and never
serialized.

> **Warning:** Overlapping runs do not merge history. The last completion wins,
> potentially excluding other runs' experience from the Agent's head. Save each
> run's snapshot or finish each run before starting the next to accumulate history.

Failures and abandoned runs do not publish. Budgets publish real checkpoints,
not fallback model answers. See the
[lifecycle contract](history-reference.md#agent-and-run-lifecycle).

## Inspecting conversations

Python `snapshot.conversations()` yields frozen `Conversation` views; Rust returns
`Conversations`, an iterator of `Conversation<'_, P>` borrowing that snapshot.
Each exposes:

- `snapshot` and leaf `node`;
- `nodes`: shared prefix through leaf, including child provenance;
- `messages`: ordered initial envelopes followed by the recorded exchange, using
  historical representative tool outputs;
- `observations`: checkpoints with representatives **and exact per-environment evidence**;
- `env_ids`: known path memberships;
- `is_complete`, `has_answer`, and `last_answer`.

These are Python properties and Rust getter methods. `last_answer` returns the
latest no-tool assistant envelope, or absence, rather than only its content.
`is_complete` describes the current leaf. After feedback opens a continuation,
that leaf is incomplete even if an ancestor supplies an earlier answer. A view
captured from an older snapshot still points to its original path; later Agent
publications cannot retarget it.

Known environment IDs come from the latest selected observation and subsequent
node visits (or all visits if there was no observation). They are deduplicated for
the view; exact repeated evidence remains in records. They are not a global latest
partition or proof that every old member executed new post-feedback tools.
Feedback provenance is not an environment gate. Equivalent outputs can share a
conversation using another environment's representative; environment attribution
is not a cross-environment isolation guarantee.

Views describe recorded structural leaves, not runtime frontiers. Partially
collected tool tuples and request queues remain execution state. An open initial
node may have no recorded visits yet. Use requests/counters to inspect live work.

Traversal is lazy, iterative, and read-only, retaining record/path references
rather than copying contents. Enumerating every conversation intentionally visits
the main trie. Child tries are separate: inspect
`snapshot.child(entry).conversations()`. In Rust keep the snapshot binding alive
while borrowing views; owned `messages()` requires `P: Clone`, while borrowed
inspection and asset extraction do not.

## Extracting assets

The library supplies conversation accessors; the caller decides which returned
value is the asset. There is no separate `Asset` wrapper, asset ID, or managed
projection callback.

| Accessor | Value selected |
| --- | --- |
| `get_last_answer_asset()` | Content of the latest no-tool assistant envelope. Tool-calling commentary and tool outputs are not answers. |
| `get_final_tool_input_asset(tool_name)` | Arguments of the latest call to the named tool, such as `submitFinalAnswer`, not that tool's output. |

Both accessors include caller-supplied initial assistant context, as well as
recorded model responses. They can return an earlier value while feedback work is
open. The final-tool accessor scans responses and calls in reverse order, so the
last matching call within a multi-call response wins. An accepted recorded call's
arguments are available even while its tool execution is pending.

Python accessors raise `LookupError` when the requested value is absent; a real
`None` answer/input is returned normally. Rust returns `Option<&P>` for the last
answer and `Option<&serde_json::Value>` for final-tool input: `Some(&None)` or
`Some(&Value::Null)` is distinct from absence (`None`). Empty strings, false, and
zero are also real values.

A final-answer tool is still an ordinary tool. Inspecting its input does not
execute it, suppress other calls, or make the call terminal. Typed schemas and
argument interpretation remain caller-owned. Likewise, inspecting an initial
assistant answer does not turn a zero-budget investigation into an accepted
terminal response or make it eligible for a user continuation.

Equal assets do not identify or merge conversations: two paths can return the
same value based on different evidence.

## Persistence values and API scope

`AgentDescriptor`, `AgentQuery`, `SubagentRecord`, `ChildAgentRun`, and
`ChildAgentUpdate` are canonical record names; no deprecated public aliases or
migration scaffolding are retained. `BatchResult.snapshot` and
`ChildAgentUpdate.snapshot` hold immutable snapshots. Provenance record `snapshot`
fields hold descriptors, with `agent_descriptor` accessors for inspection.
`result.child_updates` exposes independent completed invocations.

`BatchResult.classes` and `results` are flat, execution-local summaries. There is
no separate partition tree: the history trie is the only stored tree, including
checkpoints that did not split a group. Historical leaves and accumulated
memberships must not be mistaken for this execution's final groups.

Conversation views have no public IDs, standalone persistence, resume method, or
endpoint lookup. Persist the snapshot and reconstruct views by traversal.
Feedback locates its target through environment tool replay; a bare node or asset
value cannot substitute for that resolution.

Python snapshots support direct `pickle.dump`/`pickle.load`: export eagerly
bundles all reachable records, including child invocation bases/results, but no
unrelated store records, loaders, or runtime callbacks. Contents must be
pickleable; only load trusted pickle data. Each load creates an independent
writer with original history references intact. This is history persistence, not
crash recovery.

For advanced lazy/incremental storage (and Rust), save descriptors and individual
shallow records and attach a `RecordStore` loader. The library owns record
structure, references, and traversal; the caller owns codecs, backend I/O, and
durable publication. Restore with `from_snapshot`, reattaching runtime
factories/resolvers/comparison functions as needed. The entire initial-message
sequence and ordered tool specifications come from the snapshot. Do not persist
live Agent/AgentRun handles or Conversation views. No general cross-version or
cross-language wire compatibility is promised.

Runtime callback identity, provider policy, and comparison mode are not persisted
compatibility keys. [Compatibility and ownership](history-reference.md#compatibility-and-ownership)
remain explicit contracts.

## End-to-end examples

```sh
(cd python && PYTHONPATH=src python3 examples/agent_conversations.py)
cargo run -p lockstep-loop --example agent_conversations
```

[Python source](../python/examples/agent_conversations.py) ·
[Rust source](../rust/examples/agent_conversations.rs)

These mock-only examples load snapshots, replay new environments, inspect
conversations, extract assets, resolve feedback through tools, and preserve
old/open/revised views. For a different domain, see
[Python product screening](../python/examples/product_screening.py) and
[Rust product screening](../rust/examples/product_screening.rs).
