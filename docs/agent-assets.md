# Agent and Asset: vocabulary and API migration

[Visual guide](concepts.md) · [Documentation](README.md) ·
[History/persistence reference](history-reference.md)

## Glossary

| Term | Meaning |
| --- | --- |
| **Agent** | The complete recorded trie **including its initial prompt, structural configuration, and accumulated investigations**. Not just a recipe, tool definition, provider client, or executor. |
| **Agent snapshot / revision** | One immutable root of that whole agent. Extending it creates another snapshot; previous roots remain valid. `Agent` already represents this, so there is no redundant `Snapshot` wrapper. |
| **Asset** | A leaf investigation in a particular agent snapshot, backed by its entire root-to-leaf conversation. It can represent multiple environments. Neither a value nor a bare leaf reference nor an `EnvClass` is the whole asset. |
| **Asset value** | The latest published payload on the path. **Currently this is the content of the latest no-tool assistant response**, not a designated result-tool argument. An open asset may retain that older value or have no value yet. |
| **Conversation / history** | The prompt, assistant messages, ordered tool observations, and user feedback along a path. |
| **History node** | A shared prefix, current endpoint, or historical endpoint. It need not be a leaf. |
| **Environment class** | A set of environments grouped by observation history. `EnvClass` and `SplitNode` are partition-only summaries, not assets or agents. |
| **Loop / execution** | `BatchLoop`, which issues requests and accepts completions. It is not the immutable agent. |
| **Subagent tool** | `SubagentTool`, a definition that deterministically constructs another agent's initial user prompt and tool configuration. It is not itself the complete recorded child agent. |

The package name, replay, grouping, scheduling, request IDs, feedback scope and
storage protocol are unchanged. Manual contract/version plumbing has been removed;
this is not an asset publication or dependency-DAG system.

These terms are domain-neutral. A study-screening agent may have assets concluding
“eligible” or “insufficient evidence”; a dataset-intake agent may have assets with
different quality findings. Two supplier reviews can publish the same value while
retaining different supporting conversations. See [use cases](use-cases.md) for
concrete prompts, environment boundaries, and tools.

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
| `HistoryDescriptor` | `AgentDescriptor` | Persistence envelope (prompt/configuration/root reference), not an executor; exact type alias. |
| `HistoryQuery` | `AgentQuery` | Lookup compatibility header; exact type alias. |
| `HistoryResolver` | `AgentResolver` | Caller-owned agent revision selection; alias, not another loading interface. |
| Rust `HistoryOptions` | `AgentOptions` | Configuration alias; no corresponding Python wrapper is needed. |
| `SubagentHistory` | `SubagentRecord` | Parent-call provenance slot, not a complete agent by itself. |
| `ChildHistoryRun` | `ChildAgentRun` | One invocation's selected base and resulting descriptor. |
| `ChildHistoryUpdate` | `ChildAgentUpdate` | A completed invocation update carrying the complete child agent. |
| Python `BatchLoop(history=...)` | `BatchLoop(agent=...)` | Old keyword retained; supplying both non-None spellings is rejected. |
| Python loop `history_contract=` / `agent_contract=` | `recording=True` for fresh roots; `agent=snapshot` for replay | Version arguments removed; the boolean only enables recording. |
| Python `history_resolver=` | `agent_resolver=` | Same callback and lookup policy; both non-None spellings are rejected. |
| Rust `with_replay(...)` | `with_agent(...)` | Same replay-based initialization; both set the same configuration field. |
| Rust `with_history_resolver(...)` | `with_agent_resolver(...)` | Same setter semantics. |
| `loop.history` / `history()` | `loop.agent` / `agent()` | Same snapshot getter and existing language-specific transaction guarantees. |
| `result.history` | Python `result.agent`; Rust `result.agent()` / `into_agent()` | Compatibility field retained, no duplicate stored snapshot. |
| `result.child_histories` | Python `result.child_agents`; Rust `child_agents()` | Same completed invocation updates. |
| `update.history` | Python `update.agent`; Rust `agent()` / `into_agent()` | Same complete child agent. |
| `entry.history`, `run.history` | Python `.agent_descriptor`; Rust `agent_descriptor()` | These are descriptors; attach the store with `agent.child(entry)`. |
| Child `history_contract`, `with_history_contract`, `with_agent_contract`, and contract getters | Removed | A parent recording or resolver suffices. Optional Python `recording=True` / Rust `with_recording()` enables standalone child recording, not a key. |
| Rust `with_recording(label)` | `with_recording()` | Fresh-root opt-in, no version. |
| Rust `with_agent(agent, label)` / `with_replay(agent, label)` | `with_agent(agent)` / `with_replay(agent)` | Supplying an Agent implies replay and recording. |
| Descriptor/query/options `contract`, `Agent.contract` / `contract()` | Removed | Query identity is structural; no replacement namespace. |

`Agent.prompt` / `prompt()` exposes the descriptor prompt. `SubagentContract` retains
its class name for compatibility but contains only spec, depth and nested allowlist.
Recording switches are not part of query/descriptor identity. Rust's `Assets` is
only the lazy iterator type, not another persisted or runtime model.

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

## Persistence and contract removal

Legacy record class aliases and provenance **`history`** fields remain. However,
manual **`contract`** fields in descriptors/queries and **`history_contract`** in
nested subagent configurations are removed from the API and new writes. Python
pickle restoration discards these old fields; self-describing Rust Serde formats
such as JSON ignore them as unknown fields. Different historical labels cannot
poison otherwise matching structural cache entries. Do not retain those labels
in a caller cache's matching key when importing old data.

The unchanged **2c6d901** pickle/JSON fixtures still load, inspect and replay,
including child histories, feedback and full evidence. Re-encoding intentionally
omits the deprecated labels, so this is not byte/schema identity with old output.
Positional/binary Serde formats and custom codecs may need caller migration; older
readers that require contract fields are not promised to read new writes. Earlier
singular `subagent` schemas still require explicit migration. There is no general
cross-language or cross-version wire guarantee.

`AgentDescriptor`, `AgentQuery`, `SubagentRecord`, and `ChildAgentRun` still alias
legacy record classes/structs. Do not rename provenance keys merely because the
API uses `.agent_descriptor`. Persist descriptors and shallow records, not live
Agent/resolver objects or Asset views. No second model or core I/O is added.

Compatibility checks cover prompt equality, ordered tool specs, default/custom
grouping mode and nested subagent specs/allowlists/depth. Actual child prompts and
specs are checked on entry. Tools are re-executed: changed implementations with
unchanged observations reuse; changed observations branch. External provider,
adapter and custom grouping policies remain caller-owned, including cache isolation
and snapshot selection. No manual version or namespace replaces the removed key.

## End-to-end examples

```sh
(cd python && PYTHONPATH=src python examples/agent_assets.py)
cargo run -p lockstep-loop --example agent_assets
```

[Python source](../python/examples/agent_assets.py) ·
[Rust source](../rust/examples/agent_assets.rs)

Both examples use a small file-inspection fixture. They load an agent, replay a
new environment, inspect its asset's value and conversation, resolve feedback by
replay (not via the view), and persist the updated agent. They also inspect an
open continuation with its prior value and verify that the old snapshot retains
its original asset. Shared-path inheritance and subdivision remain unchanged.
No model/provider account or storage service is needed. For a different domain,
see the product-specification [Python](../python/examples/product_screening.py)
and [Rust](../rust/examples/product_screening.rs) demos; the asset model is the same.
