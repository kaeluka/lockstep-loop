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

- Prefer one immutable history model used both internally and by callers for
  persistence/inspection, rather than separate internal and export tries.
- A mutable root reference may point at immutable, structurally shared records.
  Extending history must not mutate previously captured roots or their records.
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
- Expose the immutable records/root needed to save and restore history.
- Design references/loading so history can be traversed lazily, loading only a new
  environment's relevant path, without making the caller implement replay logic.
- Check prompt equality, ordered tool specs, default/custom grouping mode and nested
  subagent specs/allowlists/depth; check actual child prompts/specs on entry.
- Do not add manual agent contract/version/namespace keys. Pure tools are re-executed:
  implementation changes with equal observations reuse; different observations branch.
  Independent child queries use child structural configuration, never parent identity.
- Provider/adapter and custom grouping semantics are external policy, not inferred.
  The caller owns cache isolation and compatible snapshot selection for policy changes.
- Fresh root recording is opt-in (`recording=True` / `with_recording()`); supplying
  an Agent implies replay plus recording. A recorded parent or resolver records children;
  an optional child recording switch enables standalone recording, never cache identity.
- Retain legacy record class aliases. Ignore deprecated persisted contract labels as
  metadata, never matching keys; document source/wire migration and codec limitations.
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
- imported legacy labels do not affect matching; external provider/grouping policy
  remains caller-owned, while structural incompatibilities are rejected;
- changed observations branch at the first divergence, including previously
  non-splitting turns;
- old snapshots remain unchanged and both old and new paths remain reusable;
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
