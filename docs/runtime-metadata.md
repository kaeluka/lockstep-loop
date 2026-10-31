# Runtime agent metadata

An agent may carry caller-owned metadata for routing and execution policy, such as
an agent-file manifest selecting an LLM model. The core transports it unchanged;
it does not interpret model names, tags, or any other fields.

- Every model and ordinary tool request exposes its **originating agent's**
  metadata, including replayed tool requests and requests after refinement splits.
- Each subagent has its own metadata. It does not inherit or merge its parent's
  metadata. This holds for siblings, grandchildren, and depth-clamped recursion.
- Metadata never becomes conversation content, tool arguments, or tool schemas.
  Only the caller's adapter chooses whether and how to use it.
- Metadata is not a compatibility key and is absent from `AgentQuery`,
  `SubagentContract`, snapshot descriptors, records, and child provenance.
  Changing it does not invalidate core history.
- Snapshot restoration must explicitly reattach current metadata. Old model
  responses remain recorded facts; a new model setting affects fresh requests,
  not the provenance or contents of previously generated answers.
- Metadata alone is not an audit record of the provider/model actually used. A
  caller needing that provenance must record actual execution details separately.

## Python

`Agent(..., metadata=value)`, `Agent.from_snapshot(..., metadata=value)`, and
`SubagentTool(..., metadata=value)` accept opaque values. Read `request.metadata`
on either `LlmRequest` or `ToolRequest`. The default is `None`.

The lower-level `BatchLoop` and convenience drivers `run_batch` and
`run_to_completion` also accept a root `metadata=` argument. Requests share the
value by reference; it is excluded from request equality and repr. The caller
must preserve immutable semantics, including nested values and aliases, for the
lifetime of agents, runs, and requests. The core does not copy/freeze arbitrary
Python objects.

```python
agent = Agent(
    initial_messages=messages,
    tool_specs=tools,
    metadata={"model": "deepseek/deepseek-v4.1-flash-1017"},
)
request = agent.start(["repository"]).next_request()
# In the host's LLM handler:
model = request.metadata["model"]
```

## Rust

`Metadata` is an `Arc<serde_json::Map<String, Value>>`: an opaque JSON object
shared without cloning its contents for each request. The default is an empty
object. Use:

- `AgentConfig::with_metadata(metadata)` for fresh agents;
- `AgentRuntime::with_metadata(metadata)` when restoring a snapshot;
- `SubagentTool::with_metadata(metadata)` for children;
- `LoopConfig::with_metadata(metadata)` for lower-level loops.

Builders accept either a map or an already shared `Metadata` value. Both request
structs and the `Request` enum provide `metadata()`; the structs also expose the
shared field. Rust's one-shot convenience drivers use default root metadata;
use `Agent` to configure root metadata. Their subagents still carry their own.

```rust
use lockstep_loop::{Agent, AgentConfig, AgentRuntime};
use serde_json::json;

let manifest = json!({"model": "deepseek/deepseek-v4.1-flash-1017"})
    .as_object().unwrap().clone();
let agent = Agent::new(
    AgentConfig::new(messages, tools).with_metadata(manifest.clone())
)?;
let restored = Agent::from_snapshot(
    agent.snapshot(), AgentRuntime::default().with_metadata(manifest)
)?;
```

## Agent-file example

The [Rust OpenRouter host](agent-examples/README.md) attaches each definition's
complete `manifest` as runtime metadata. It selects the model from each LLM
request's `metadata()["model"]`, with a host default if absent. An explicit
`--model` overrides every agent's model.

The host, not the core, isolates caches by effective model routing across the
whole linked definition tree. Changing a child model therefore cannot silently
reuse a completed parent answer under the old model policy. Unrelated manifest
tags do not invalidate the cache. Current manifests are reattached on restoration;
the core snapshots never serialize them.
