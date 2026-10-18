# Design goals

## What this library owns

lockstep-loop is a provider-neutral partition-refinement state machine, with Python
and Rust implementations. The caller executes LLM and tool requests and returns
completions. Preserve that separation: no LLM clients, execution infrastructure,
or storage backend belongs in the core.

Tools must be pure/read-only over stable environments, have identical specs across
environments, and not expose environment identities to the model. Independent
calls within a turn may complete in any order; refinement uses the complete,
ordered tuple of their results. Classes refine, never merge within an execution.

## First extension: immutable recorded history and incremental replay

Record conversations so a later environment (for example tomorrow's repository
commit) can reuse an earlier investigation:

1. Reuse a recorded assistant turn under compatible structural configuration.
2. Re-execute its tool calls against the new environment.
3. Follow the continuation matching the actual result tuple.
4. Request fresh LLM work only at an unseen observation/continuation.
5. Preserve existing history when adding the new path.

Record EVERY tool-observation checkpoint, including turns that caused no split.
The existing final partition trie alone is insufficient for replay.

### Representation and immutability

- Agent is the primary mutable handle owning a current AgentSnapshot and runtime
  callbacks. AgentRun wraps immutable history plus transient execution builders.
  Use one immutable history model internally and for persistence/inspection.
- start captures the CURRENT root. Successful completion automatically publishes
  exactly once, including immediate cached/budget completion. Overlapping runs are
  allowed: LAST COMPLETION WINS WITHOUT MERGING. No queues or lifetime locks.
  Document this warning prominently. Inspection of old runs never republishes.
- Failures and abandoned runs never replace the owner head. Invalid completions
  remain correctable. Budgets publish real history, not fabricated assistant facts.
- snapshot/from_snapshot must support lazy caller stores/loaders symmetrically.
  Restore prompt/specs from the snapshot, reattach executable callbacks explicitly.
  Independent handles fork via snapshots; child updates and Assets anchor snapshots,
  never mutable handles. Extending history never mutates captured roots/records.
- Pending requests and partially collected turns are builders/late initialization:
  validate before fulfillment, fulfill once, then retain immutable historical facts.
  Invalid completions must not consume a request. Retries/corrections must not
  overwrite fulfilled history.
- Generic caller-owned payloads need an explicit ownership/immutability contract;
  frozen wrappers alone do not make nested mutable payloads immutable.

### Inspectability and exact evidence

- Preserve full messages, tool calls/arguments, and tool outputs for evaluation
  and future UIs. Never replace output payloads with hashes or grouping keys.
- Preserve per-environment evidence even when custom grouping considers unequal
  outputs equivalent; distinguish actual observations from representative content
  used in the shared conversation.
- Use extensible records (for example an output record with a `tool_output` field),
  not unadorned payload slots that cannot later carry metadata.
- Hashes may accelerate lookup but must not replace equality verification. Custom
  grouping is an explicit, potentially lossy policy, not implicit hash equality.
- References/IDs must not silently conflate distinct records on hash/debug-string
  collisions. Existing snapshots must remain usable after adding new environments.

### Storage neutrality

- The caller chooses storage and payload codecs. Core code performs no filesystem,
  database, or network persistence.
- Python AgentSnapshot supports direct trusted pickle save/load as the easy path.
  Eagerly bundle only the complete reachable graph, including child invocation
  bases/results, iteratively deduplicated by address. Preserve full pickleable
  payloads and sharing; exclude locks/loaders/runtime callbacks. Each load gets a
  fresh writer namespace. Surface missing-record and unpickleable-payload errors.
- Keep the lower-level immutable descriptor/record API for lazy/incremental
  persistence in both languages. Decomposition is optional for Python callers;
  direct pickle is history persistence, never callback persistence/crash recovery.
- Design references/loading so history can be traversed lazily, loading only a new
  environment's relevant path, without making the caller implement replay logic.
- Check prompt equality, ordered tool specs and nested subagent specs/allowlists/depth;
  check actual child prompts/specs on entry. Do not key compatibility or child queries
  on default/custom grouping mode or function identity. The default is identity.
- Apply the current grouping function to BOTH fresh raw outputs and historical
  representatives. Stricter keys may miss even if the raw output occurs in old evidence;
  coarser keys select the first matching existing edge in stored order. Preserve
  historical representative context, exact new evidence and all old roots/edges.
- Do not add manual agent contract/version/namespace keys. Pure tools are re-executed:
  implementation changes with equal observations reuse; different observations branch.
  Independent child queries use child structural configuration, never parent identity.
- Provider/adapter and custom grouping semantics are external policy, not inferred.
  The caller owns cache isolation and compatible snapshot selection for policy changes.
- Every run and convenience driver records an immutable AgentSnapshot, including
  children. Agent.start replays its current snapshot; from_snapshot restores a handle. Budgets limit accepted assistant turns,
  including reused turns; exhaustion returns a run fallback, never a fabricated
  assistant record. Preserve the real checkpoint so later replay can resume.
- No legacy migration is required. Use canonical record names and clean public
  exports; do not retain deprecated aliases or migration scaffolding. Document
  payload codec and cross-language/version persistence limitations.
- Distinguish completed-history reuse from crash recovery of in-flight execution;
  do not claim the latter unless its protocol is implemented and tested.

## Future direction (not part of the first slice)

Durable agentic assets form a DAG: map agents may cover individual files/packages;
a reducer consumes their published values. An asset's conversation is its durable
history, and the latest designated result-tool argument is its published value.
Feedback resumes the responsible conversation instead of changing a shared prompt
and invalidating all map agents. Changed published values can update downstream
assets. Conversations must therefore be extendable, not inherently terminal.

Do not add DAG scheduling, feedback routing, asset publication semantics, or a UI
as part of the initial history/replay extension. Leave room for these without
building speculative frameworks.

## Verification

Preserve existing request/completion behavior and tests. Add tests demonstrating:
- unchanged observations reuse recorded reasoning with zero new LLM requests,
  including changed tool implementations and changed parent prompts for child lookup;
- external provider/grouping policy remains caller-owned, while structural
  incompatibilities are rejected;
- automatic exactly-once publication, sequential reuse, both overlapping completion
  orders without merging, immediate cached completion, and no publication on failure;
- lazy snapshot roundtrips without repeated prompt/specs; independent run snapshots
  and Asset views survive later Agent publications;
- changed observations branch at the first divergence, including previously
  non-splitting turns;
- old snapshots remain unchanged and both old and new paths remain reusable;
- default/custom identity replay works in both directions, including child lookup;
  stricter keys branch and coarser keys choose the first existing match deterministically;
- full evidence survives replay, including custom grouping and opaque payloads;
- parallel/multi-tool completions work in arbitrary order;
- invalid completions leave pending work recoverable;
- storage round trips and lazy traversal (when supported) require no core I/O.

Document any deliberate first-slice limitations, especially nested subagents,
compatibility policy, lazy loading, and cross-language API differences.

Development commands:
- Python: `cd python && uv run pytest && uv run mypy src && uv run ruff check .`
- Rust: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all --check`
