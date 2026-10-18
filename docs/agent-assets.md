# Agents, snapshots, runs and Assets

[Visual guide](concepts.md) · [Documentation](README.md) ·
[History/persistence reference](history-reference.md)

## Glossary

| Term | Meaning |
| --- | --- |
| **Agent** | Mutable handle owning the current immutable history root and caller-owned runtime dependencies. The primary entry point. |
| **AgentSnapshot** | Immutable initial prompt, structural configuration and accumulated conversation trie, paired with its record resolver. The persistence/inspection value. |
| **AgentRun** | Mutable execution wrapper with its own immutable root and transient request/completion state. Obtained from Agent.start. |
| **Asset** | Contextual immutable leaf investigation backed by its full root-to-leaf conversation in one snapshot. Not just a value, node address or environment class. |
| **Asset value** | Latest no-tool assistant payload on that path. An open continuation may retain an older value or have none. Not designated result-tool publication. |
| **Conversation** | Prompt, assistant messages, ordered tool observations and feedback along a path. |
| **History node** | Shared prefix, current endpoint or historical endpoint; not necessarily a leaf. |
| **Environment class** | Environments grouped by observations during one execution. EnvClass/SplitNode are partition summaries, not Assets. |
| **Subagent tool** | Deterministic prompt/spec factories plus an advertised tool definition, not a persisted child Agent handle. |

An Agent retains experience through sequential runs. Each start captures the current
root; successful completion automatically replaces it once. A run's snapshot stays
independent after another run publishes. Restore/fork via `Agent.from_snapshot`;
loaders and runtime callbacks are caller-owned and never serialized.

> **Warning:** Overlapping runs do not merge history. The last completion wins,
> potentially excluding other runs' experience from the Agent's head. Save each
> run's snapshot or finish each run before starting the next to accumulate history.

Failures/abandoned runs do not publish. Budgets publish real checkpoints, not
fallback assistant messages. See the [lifecycle contract](history-reference.md#agent-and-run-lifecycle).

## Inspecting assets

Python `snapshot.assets()` yields frozen Asset views; Rust returns an iterator
of `Asset<'_, P>` borrowing that snapshot. Each exposes:

- `snapshot` and leaf `node`;
- `nodes`: shared prefix through leaf, including child provenance;
- `messages`: full model-visible conversation using historical representatives;
- `observations`: checkpoints with representatives **and exact per-env evidence**;
- `env_ids`: known path memberships;
- `is_complete`, `has_value`, `published_message`, `value`.

These are Python properties and Rust getter methods. Python `value` raises
LookupError when absent; a real None remains a value. Rust returns Option<&P>,
so Some(&None) differs from no publication.

`is_complete` describes the current leaf. After feedback opens a continuation,
that leaf is incomplete even if an ancestor supplies an older value. The former
endpoint is now an ancestor, not an additional Asset. A view captured from the old
snapshot remains a completed leaf there. Later Agent publications cannot retarget
it. Equal values never identify or merge investigations.

Known environment IDs come from the latest selected observation and subsequent
node visits (or all visits if there was no observation). They are deduplicated for
the view; exact repeated evidence remains in records. They are not a global latest
partition or proof that every old member executed new post-feedback tools.
Feedback provenance is not an environment gate.

Views describe recorded structural leaves, not runtime frontiers. Partially
collected tool tuples and request queues are builders. An open initial node may
have no recorded visits yet. Use requests/counters to inspect live serving state.

Traversal is lazy, iterative and read-only, retaining record/path references rather
than copying payloads. Enumerating every Asset intentionally visits the main trie.
Child tries are separate: inspect `snapshot.child(entry).assets()`. In Rust keep
the snapshot binding alive while borrowing views; owned messages require P: Clone,
while inspection/value access does not.

Assets have no public IDs, standalone persistence, resume method or endpoint lookup.
Persist the snapshot descriptor and shallow records and reconstruct views by
traversal. Feedback still locates its target through environment tool replay; a
bare node or answer value cannot substitute for that resolution.

## Persistence values and API scope

`AgentDescriptor`, `AgentQuery`, `SubagentRecord`, `ChildAgentRun` and
`ChildAgentUpdate` are the canonical record names; no deprecated public aliases
or migration scaffolding are retained. `BatchResult.snapshot` and
`ChildAgentUpdate.snapshot` hold immutable snapshots. Provenance record `snapshot`
fields hold descriptors, with `agent_descriptor` accessors for inspection.
`result.child_updates` exposes independent completed invocations.

Save descriptors and individual shallow records, not live Agent/AgentRun handles,
callbacks or Assets. Restore using an AgentSnapshot and a RecordStore loader, then
reattach matching subagent factories/resolvers through from_snapshot. Prompt and
ordered tool specs come from the descriptor. No general cross-version or
cross-language wire compatibility is promised; choose caller-owned payload codecs.

The internal BatchLoop executes the same immutable history model; there is no
separate export trie. Runtime callback identity, provider policy and comparison
mode are not persisted keys. [Compatibility and ownership](history-reference.md#compatibility-and-ownership)
remain explicit caller contracts.

## End-to-end examples

```sh
(cd python && PYTHONPATH=src python examples/agent_assets.py)
cargo run -p lockstep-loop --example agent_assets
```

[Python source](../python/examples/agent_assets.py) ·
[Rust source](../rust/examples/agent_assets.rs)

These mock-only examples load snapshots, replay new environments, inspect Assets,
resolve feedback through tools, and preserve old/open/revised views. For a different
domain, see [Python product screening](../python/examples/product_screening.py)
and [Rust product screening](../rust/examples/product_screening.rs).
