# Documentation verification

[Documentation](README.md) · [Visual guide](concepts.md) · [API glossary](agent-assets.md)

## What the documentation claims

lockstep-loop is a domain-neutral investigation state machine. Product ratings,
study documents, supplier packets, contracts, datasets, and repository snapshots
are caller-defined environments—not built-in integrations. The
[use-case guide](use-cases.md) distinguishes application designs from shipping APIs.

The documentation preserves these boundaries:

- `Agent` is the mutable entry point; `AgentSnapshot` is its immutable recorded
  prompt/configuration/trie. `AgentRun` independently extends a captured root.
  An Asset is an immutable path-backed view anchored in a snapshot, not just a value.
- Successful completion publishes exactly once, including cached/budget starts.
  Overlapping runs do not merge: last completion wins. Failures/abandonment do not
  publish, and inspecting an older run cannot replace a newer head.
- Sharing follows equal observed outputs, not topical similarity or equivalent
  outcomes. Tools still run per environment; split classes do not remerge.
- Replay does not guarantee coverage. A missing investigative step must be caught
  through evaluation, explicit coverage requirements, or feedback.
- Feedback is inherited by every environment reaching that path in the new
  snapshot, not just the environment used to resolve it. Old snapshots stay valid.
- Specialized subagents can reuse independently selected child snapshots after
  parent prompt changes. Already recorded parent invocations remain pinned.
- Execution, evidence ingestion, provider adaptation, and persistence stay with
  the caller. Snapshotting live sources is not performed by the core.
- DAG push updates and designated result-tool publication remain future work.

## Evidence checked

| Claim | Implementation / regression evidence |
| --- | --- |
| Equal product observations share work; new evidence replays or branches | Executable product-screening examples in [Python](../python/examples/product_screening.py) and [Rust](../rust/examples/product_screening.rs) |
| Path-backed leaves, equal values/distinct assets, old/open snapshots, null values | [Python assets](../python/tests/test_agent_assets.py), [Rust assets](../rust/tests/agent_assets.rs) |
| Lazy inspection, exact evidence, immutable replay, arbitrary completion order | Asset, history, history-foundation, and loop test suites in both languages |
| Feedback resolution and shared-path inheritance | [Python feedback](../python/tests/test_feedback.py), [Rust feedback](../rust/tests/feedback.rs) |
| Specialized registries, independent children, pinned revisions, concurrent updates | [Python specialists](../python/tests/test_specialized_subagents.py), [Rust specialists](../rust/tests/specialized_subagents.rs) |
| Completion recovery and serving costs | [Python transactions](../python/tests/test_transactional_transitions.py), [cost guards](../python/tests/test_transition_cost.py), [Rust transactions](../rust/tests/transactional_transitions.rs) |
| Mutable handles, independent snapshots, both overlapping completion orders, automatic publication, failure/budget semantics and lazy storage | [Python entry point](../python/tests/test_agent_entrypoint.py), [Rust entry point](../rust/tests/agent_entrypoint.rs) |
| No keys needed; implementation changes reuse or branch; external policy is caller-owned | [Python structural reuse](../python/tests/test_structural_reuse.py), [Rust structural reuse](../rust/tests/structural_reuse.rs), specialist suites |

Both structural-reuse suites additionally verify default/custom identity reuse in
both directions, stricter-key misses against representatives rather than old evidence,
coarser-key first-existing-edge selection, preserved representative context and fresh
evidence, unchanged old roots and mode-independent child lookup.

## Checks for the Agent entry-point refactor

- Python: **108 tests**, strict mypy on all four source modules, Ruff lint.
- Rust: **79 integration tests**, Clippy and rustdoc with warnings denied, formatting checks.
- All **ten provider-free examples**: `product_screening`, `agent_assets`,
  `history_replay`, `feedback`, and `specialized_subagents` in both languages.
- The new product examples assert **3 LLM calls for 3 initial products in 2
  classes**, **0 new LLM calls for a matching product**, and **1 new LLM call for
  changed ratings**. They count tool executions separately and verify old assets
  survive extension. Their assessment rules and model replies are toy fixtures.
- Relative Markdown links and heading anchors; `git diff --check`.
- The split-completion benchmark runs with Agent/AgentRun (100/400/1000 environments).
- No paid API calls or credentials. Existing diagrams/reports were not modified.

The file-inspection demos remain protocol examples, not the library's only
application. No legacy migration is required: obsolete alias/import fixtures were
removed. Current-format caller-owned storage roundtrips, lazy traversal, child
provenance, feedback, opaque evidence and collision coverage remain in the existing
suites and new entry-point tests. Transactional preparation, request routing,
arbitrary completion order and structural cost guards remain covered.
See [persistence values and API scope](agent-assets.md#persistence-values-and-api-scope).

## What fixture tests do not establish

The examples use deterministic model stand-ins and in-memory evidence. They do
not evaluate real LLM reasoning, perform genuine literature/compliance/legal
review, establish product suitability, or validate third-party tool purity.

Real deployment requires application-level evaluation and stable evidence. Tool
contracts, immutable payload ownership, and correct storage publication remain
caller responsibilities. No general cross-version wire promise, in-flight crash
recovery, automatic root merging, or dependency propagation is claimed.
