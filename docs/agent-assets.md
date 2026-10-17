# Agent and Asset: vocabulary and API migration

[Visual guide](concepts.md) · [Documentation](README.md) ·
[History/persistence reference](history-reference.md)

## Glossary

| Term | Meaning |
| --- | --- |
| **Agent** | The complete recorded trie **including its initial prompt, compatible execution contract, and accumulated investigations**. Not just a recipe, tool definition, provider client, or executor. |
| **Agent snapshot / revision** | One immutable root of that whole agent. Extending it creates another snapshot; previous roots remain valid. `Agent` already represents this, so there is no redundant `Snapshot` wrapper. |
| **Asset** | A leaf investigation in a particular agent snapshot, backed by its entire root-to-leaf conversation. It can represent multiple environments. Neither a value nor a bare leaf reference nor an `EnvClass` is the whole asset. |
| **Asset value** | The latest published payload on the path. **Currently this is the content of the latest no-tool assistant response**, not a designated result-tool argument. An open asset may retain that older value or have no value yet. |
| **Conversation / history** | The prompt, assistant messages, ordered tool observations, and user feedback along a path. |
| **History node** | A shared prefix, current endpoint, or historical endpoint. It need not be a leaf. |
| **Environment class** | A set of environments grouped by observation history. `EnvClass` and `SplitNode` are partition-only summaries, not assets or agents. |
| **Loop / execution** | `BatchLoop`, which issues requests and accepts completions. It is not the immutable agent. |
| **Subagent tool** | `SubagentTool`, a definition that deterministically constructs another agent's initial user prompt and contract. It is not itself the complete recorded child agent. |

The package name, replay, grouping, scheduling, request IDs, feedback scope and
storage protocol are unchanged. This is a vocabulary/API-view change, not an
asset publication or dependency-DAG system.

## Inspecting assets

Python's `agent.assets()` lazily yields frozen `Asset` views; Rust returns an
iterator of `Asset<'_, P>` views borrowing the agent snapshot. Both expose:

- the owning `agent` and the leaf `node`;
- `nodes`: shared prefix through endpoint, including child-agent provenance;
- `messages`: the full model-visible conversation, including tool calls/arguments
  and feedback, reconstructed using historical **representative** outputs;
- `observations`: selected checkpoints with **all exact per-environment evidence**;
- `env_ids`: known path memberships (see below);
- `is_complete`, `has_value`, `published_message`, and `value`.

These are Python properties and Rust getter methods. Python `value` raises
`LookupError` if no answer has ever been published; a real `None` remains a value
when `has_value` is true. Rust `value()` returns `Option<&P>`: `Some(&None)` is a
real null optional payload, distinct from no publication. `published_message`
exposes the whole prior assistant response, not a hash or grouping key.

`is_complete` describes the **current leaf**. After feedback opens a continuation,
it is false even if `value` still exposes the earlier answer. In the new snapshot,
that earlier endpoint is an ancestor, not an additional asset. An asset captured
from the old snapshot remains a completed leaf there. Extra observations can
subdivide the open continuation into several assets, possibly publishing equal
values. Value equality never identifies an asset.

Known environment labels come from the latest selected observation and subsequent
recorded node visits; without observations, they come from visits on the path.
They are deduplicated for this view, while exact/repeated evidence remains in the
records. They are **not** a global latest partition or proof that every old
member has already executed new post-feedback tools. A new observation bounds the
known membership again. The feedback source environment is provenance, not a gate.
New environments still resolve their path by replaying tools.

Views describe **recorded structural leaves**, not all in-flight executions.
A pending replay at an already branching node is not an additional stored leaf;
partially collected tool tuples/request queues remain runtime builders. An open
initial node can consequently have no recorded environment visits yet. Use the
loop's requests/counters for that runtime frontier, not an invented asset index.

Inspection is opt-in and read-only. It loads only visited main-trie nodes, uses
iterative traversal, and captures path references once per yielded view. It does
not run tools/LLMs, allocate history records, or add costs to ordinary serving.
Enumerating every asset intentionally traverses the main trie. Child agents are
separate tries: use `agent.child(entry).assets()` to inspect their assets rather
than treating their leaves as parent assets. In Rust, keep the owning agent value
alive while borrowing views; inspection does not clone payloads, while requesting
owned `messages()` requires `P: Clone`.

**No new identities or shortcuts:** Asset views have no public asset ID, no
standalone serialization, and no feedback/resume method. Their constructors are
not a leaf-address lookup API. Persist the agent's descriptor/records and recreate
views by traversal. A bare leaf cannot reconstruct its ancestors. Feedback still
accepts **agent snapshot + environment** and reruns the recorded tools; never pass
an asset/value/hash/endpoint instead. Views have no value-based identity contract.

## Preferred names and compatibility

Aliases refer to the **same implementation**, not a second internal/export model.
Legacy constructors, fields and methods remain usable where listed.

| Previous spelling | Preferred spelling | Compatibility detail |
| --- | --- | --- |
| `History` | `Agent` | Exact type alias to the one immutable snapshot type. |
| `HistoryDescriptor` | `AgentDescriptor` | Persistence envelope (prompt/contract/root reference), not an executor; exact type alias. |
| `HistoryQuery` | `AgentQuery` | Lookup compatibility header; exact type alias. |
| `HistoryResolver` | `AgentResolver` | Caller-owned agent revision selection; alias, not another loading interface. |
| Rust `HistoryOptions` | `AgentOptions` | Configuration alias; no corresponding Python wrapper is needed. |
| `SubagentHistory` | `SubagentRecord` | Parent-call provenance slot, not a complete agent by itself. |
| `ChildHistoryRun` | `ChildAgentRun` | One invocation's selected base and resulting descriptor. |
| `ChildHistoryUpdate` | `ChildAgentUpdate` | A completed invocation update carrying the complete child agent. |
| Python `BatchLoop(history=...)` | `BatchLoop(agent=...)` | Old keyword retained; supplying both non-None spellings is rejected. |
| Python loop `history_contract=` | `agent_contract=` | Same version string, not a new identity. Both non-None spellings are rejected. |
| Python `history_resolver=` | `agent_resolver=` | Same callback and lookup policy; both non-None spellings are rejected. |
| Rust `with_replay(...)` | `with_agent(...)` | Same replay-based initialization; both set the same configuration field. |
| Rust `with_history_resolver(...)` | `with_agent_resolver(...)` | Same setter semantics. |
| `loop.history` / `history()` | `loop.agent` / `agent()` | Same snapshot getter and existing language-specific transaction guarantees. |
| `result.history` | Python `result.agent`; Rust `result.agent()` / `into_agent()` | Compatibility field retained, no duplicate stored snapshot. |
| `result.child_histories` | Python `result.child_agents`; Rust `child_agents()` | Same completed invocation updates. |
| `update.history` | Python `update.agent`; Rust `agent()` / `into_agent()` | Same complete child agent. |
| `entry.history`, `run.history` | Python `.agent_descriptor`; Rust `agent_descriptor()` | These are descriptors; attach the store with `agent.child(entry)`. |
| Child-definition `history_contract` / Rust `with_history_contract` | `SubagentTool.with_agent_contract(...)`, `.agent_contract` / `agent_contract()` | One version value. Python's dataclass constructor/`replace` field and persisted contract field remain `history_contract`. |

`Agent.prompt` / `prompt()` exposes the existing descriptor prompt; Rust also adds
`contract()` for the existing version string (Python already had `contract`).
`SubagentContract.agent_contract` / `agent_contract()` likewise reads the retained
`history_contract` field. Rust's `Assets` is only the lazy iterator type, not another
persisted or runtime model.

Rust's public `LoopConfig.history`/`history_resolver` fields remain compatibility
fields; prefer the canonical builder methods. Like other Rust setters, old/new
builder spellings write the same field, not parallel configurations. Python's
legacy dataclass result/record constructor keywords remain the storage-shaped
spellings; canonical properties are read aliases. `BatchLoop`, `SubagentTool`,
`SplitNode`, `EnvClass`, requests/completions, budgets, `HistoryNode`, observations,
`NodeRef`, `RecordStore`, and `FeedbackResolutionError`/`LoopError` retain their
accurate operational or structural names. `RecordStore` is a node write set/loader,
not a new logical-agent registry. The `history.py`/`history.rs` modules and existing
example filenames remain valid structural names.

## Persistence: no schema migration for this rename

Persisted record fields and type tags stay unchanged, including **`history`** and
**`history_contract`**. `AgentDescriptor`, `AgentQuery`, `SubagentRecord`, and
`ChildAgentRun` alias the existing record classes/structs, so Python pickle globals
and Serde struct/field names retain their historical spellings. Their repr/debug
names may therefore still show a legacy record name. Do not rename stored JSON
keys merely because application code now uses `.agent_descriptor`.

The live `Agent`/resolver is still not the persistence format. Save its descriptor
and shallow records through caller-owned codecs, as before. Asset views are not
persisted and add no new record tag/field. Fixtures generated using unmodified
**2c6d901** verify old pickle/Serde data loads, inspects and replays through the new
API; new record serialization retains the same schema and values. Pickle memo
layout depends on object/string identity, so byte identity is not a wire promise.
No supported source API is removed. The live snapshot's reflection/debug type name
now says `Agent` rather than `History`; legacy persisted record names stay intact.
This rename needs no storage version bump; it does not undo earlier prototype
schema migrations or promise general future cross-version compatibility.

## End-to-end examples

```sh
(cd python && PYTHONPATH=src python examples/agent_assets.py)
cargo run -p lockstep-loop --example agent_assets
```

[Python source](../python/examples/agent_assets.py) ·
[Rust source](../rust/examples/agent_assets.rs)

Both examples load an agent, replay a new environment, inspect an asset's value
and conversation, resolve feedback by replay (not via the view), inspect an open
continuation with its prior value, persist the updated agent, and show that the
old snapshot still has its original asset. Shared-path inheritance and subdivision
remain unchanged. No model/provider account or storage service is needed.
