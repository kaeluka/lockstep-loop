# Documentation and naming verification

[Documentation](README.md) · [Visual guide](concepts.md) · [API mapping](agent-assets.md)

Updated for **Agent/Asset naming on baseline 2c6d901**, including the specialized
subagent APIs already present in that baseline. The earlier visual-guide audit
was against 875a19b; its single-subagent/naming limitations are now superseded.

## Scope and terminology

- `Agent` is the complete prompt + compatible execution contract + accumulated
  trie. `History` aliases that same implementation; no duplicate snapshot model.
- `Asset` is an inspection-only leaf view backed by its complete path. It is not
  its value, a bare leaf address, or an environment class. Open continuations can
  retain earlier values without claiming a newly completed answer.
- `BatchLoop`, partition-only structures and structural history records keep
  their names. Feedback still resolves **agent snapshot + environment** by replay;
  no asset identity/feedback shortcut or result-tool publication was introduced.
- Existing pickle/Serde record names and fields stay unchanged. Baseline fixtures
  verify old descriptors, queries, child provenance, exact observations and
  feedback can be loaded, inspected, replayed, and re-encoded through new aliases.
- The short root README and visual-guide/reference layout are retained. Diagram
  sources/SVGs already use the correct domain concepts and were not regenerated.
  The visual guide now links the actual inspection APIs and supported specialized
  child lookup rather than describing them as future work. DAG push updates remain
  future [issue #1](https://github.com/kaeluka/lockstep-loop/issues/1).

## Evidence checked

| Claim | Implementation | Regression evidence |
| --- | --- | --- |
| Path-backed leaves, equal values/distinct assets, old/open snapshots, real null values | [Python agent/records](../python/src/lockstep_loop/history.py), [Rust agent/records](../rust/src/history.rs) | [Python agent/assets](../python/tests/test_agent_assets.py), [Rust agent/assets](../rust/tests/agent_assets.rs), both end-to-end examples |
| Lazy read-only traversal, no payload cloning during inspection, no implicit child-trie traversal | Same implementations | Both agent/assets suites: 1500-checkpoint traversal, selected-path loader checks, no added records; Rust non-Clone payload inspection |
| Legacy exported names and storage fields remain compatible | Same implementations and public exports | Both agent/assets suites; [pre-rename pickle fixture](../python/tests/fixtures/README.md), [pre-rename Serde fixture](../rust/tests/fixtures/README.md) |
| Ordered tuples, arbitrary completion order, exact evidence and immutable replay | [Python loop](../python/src/lockstep_loop/core.py), [Rust loop](../rust/src/lib.rs) | History/foundation, request-routing and loop suites in both languages |
| Feedback resolves before reasoning, inherits by path, permits unchanged values, preserves prior roots | Same loop implementations | [Python feedback](../python/tests/test_feedback.py), [Rust feedback](../rust/tests/feedback.rs), new agent/assets suites |
| Specialized registries, independent child agents, pinned recorded revisions, separate invocation updates | Same loop implementations | [Python specialists](../python/tests/test_specialized_subagents.py), [Rust specialists](../rust/tests/specialized_subagents.rs), specialized examples using canonical names |
| Recoverable invalid completions and Python transition-cost guards are unchanged | Same loop implementations | [Python transactions](../python/tests/test_transactional_transitions.py), [Python cost guards](../python/tests/test_transition_cost.py), [Rust transactions](../rust/tests/transactional_transitions.rs) |

## Checks run

- **Python:** 96 tests, mypy, Ruff lint and targeted formatting checks passed.
  `uv` was unavailable; Python 3.14 and dev tools from the existing virtualenv ran
  with this worktree's `python/src` on `PYTHONPATH`. Existing tests were retained.
- **Rust:** 61 integration tests, Clippy with warnings denied, `cargo fmt --check`,
  and rustdoc with warnings denied passed. Cargo's doctest phase has zero tests.
- All **eight executable examples** passed: `agent_assets`, `history_replay`,
  `feedback`, and `specialized_subagents` in each language. They use deterministic
  caller-side stand-ins, not provider accounts or real repository scans.
- New end-to-end examples load/persist agents, replay environments, inspect assets,
  continue through shared-path feedback, preserve an old publication while open,
  subdivide on later observations, and reuse unchanged old snapshots.
- Python's split-completion benchmark (recording disabled, three repeats) reported
  median seconds for 100/400/1000 environments: **0.003678 / 0.015231 / 0.038462**.
  This is a workload check, not a general scalability guarantee; structural cost
  guards remain the regression protection.
- Relative Markdown links/heading anchors and `git diff --check` passed. Existing
  diagram assets were untouched; this naming pass does not claim a fresh rendering
  or screenshot audit.

## Deliberate boundaries

Known asset environment labels are recorded path evidence, not a global latest
partition; views enumerate recorded structural leaves, not an in-flight request
frontier. Asset views are not independently serialized. Rust borrows their owning
agent and only requires `P: Clone` to reconstruct owned messages; Python retains
immutable payload references. Existing language-specific callback/transaction
semantics remain unchanged. See the [inspection contract](agent-assets.md).

Real-provider adaptation, caller storage durability, payload immutability, and tool
purity cannot be established by fixture tests. No general cross-version wire
promise, crash recovery, independent-root merge, or DAG propagation is claimed.
The rename itself requires no persisted-schema migration.
