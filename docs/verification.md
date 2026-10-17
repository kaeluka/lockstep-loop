# Documentation verification

[Documentation](README.md) · [Visual guide](concepts.md) · [API mapping](agent-assets.md)

## What the documentation claims

lockstep-loop is a domain-neutral investigation state machine. Product ratings,
study documents, supplier packets, contracts, datasets, and repository snapshots
are caller-defined environments—not built-in integrations. The
[use-case guide](use-cases.md) distinguishes application designs from shipping APIs.

The documentation preserves these boundaries:

- An `Agent` contains its prompt, compatible execution contract, and recorded trie.
  An `Asset` is a read-only leaf view backed by its path, not merely its value.
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
| Naming aliases preserve old persisted records | [Pickle fixture](../python/tests/fixtures/README.md), [Serde fixture](../rust/tests/fixtures/README.md), both asset suites |

## Checks for the domain-neutral documentation rewrite

- Python: **96 tests**, strict mypy on the core and new product example, Ruff lint.
- Rust: **61 integration tests**, Clippy with warnings denied, formatting checks.
- All **ten provider-free examples**: `product_screening`, `agent_assets`,
  `history_replay`, `feedback`, and `specialized_subagents` in both languages.
- The new product examples assert **3 LLM calls for 3 initial products in 2
  classes**, **0 new LLM calls for a matching product**, and **1 new LLM call for
  changed ratings**. They count tool executions separately and verify old assets
  survive extension. Their assessment rules and model replies are toy fixtures.
- Relative Markdown links and heading anchors; `git diff --check`.
- All four editable diagrams were relabeled for varied domains, rendered to SVG
  and PNG from saved sources, and visually inspected for clipping, overlaps,
  arrow routing, and consistency with their surrounding text. No public upload
  or shared-canvas mutation was needed.

The existing file-inspection demos are retained as protocol examples, not treated
as the library's only application. No core behavior or persistence format changed
in this documentation rewrite. The earlier Agent/Asset naming migration likewise
preserves the record schema described in [the compatibility guide](agent-assets.md).

## What fixture tests do not establish

The examples use deterministic model stand-ins and in-memory evidence. They do
not evaluate real LLM reasoning, perform genuine literature/compliance/legal
review, establish product suitability, or validate third-party tool purity.

Real deployment requires application-level evaluation and stable evidence. Tool
contracts, immutable payload ownership, and correct storage publication remain
caller responsibilities. No general cross-version wire promise, in-flight crash
recovery, automatic root merging, or dependency propagation is claimed.
