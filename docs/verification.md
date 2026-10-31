# Documentation verification

[Documentation](README.md) · [Visual guide](concepts.md) · [API glossary](agent-assets.md)

## What the current API documentation claims

The [domain model](../AGENTS.md#domain-concepts-and-operations) defines the intended
contracts. API documentation distinguishes implemented behavior from open work,
including environment hygiene under custom equivalence.

lockstep-loop is a domain-neutral investigation state machine. Product ratings,
study documents, supplier packets, contracts, datasets, and repository snapshots
are contexts identified by caller-owned environment handles—not built-in
integrations. The [use-case guide](use-cases.md) distinguishes application designs
from shipping APIs.

The documentation preserves these boundaries:

- `Agent` is the mutable entry point. `AgentSnapshot` captures nonempty ordered
  initial envelopes, structural configuration, and one immutable history trie.
  Investigating environments is an operation; `AgentRun` holds its execution state.
- `Conversation` is a snapshot-anchored path view. An asset is the value extracted
  with `get_last_answer_asset()` or `get_final_tool_input_asset(name)`, not another
  wrapper or conversation identity. Missing values remain distinct from null.
- `MessageEnvelope` carries roles/call metadata around opaque message contents.
  System messages and multiple initial messages are supported; their full ordered
  structure participates in compatibility and does not consume turn budget.
- Successful completion publishes exactly once, including cached/budget starts.
  Overlapping runs do not merge: last completion wins. Terminal failure/abandonment
  does not publish, and inspecting an old run cannot replace a newer head.
  Python can roll back replay-time child compatibility mismatches and retry the
  pending completion; Rust recorded-turn incompatibility is terminal. Neither
  accepts incompatible history.
- Sharing follows tool-result equivalence, not topical similarity or equal final
  answers. Tools still run per environment; split groups do not remerge.
  Custom equivalence intentionally shares representative content while retaining
  actual outputs; it does not establish cross-environment isolation.
- There is no separate partition tree. Flat result classes describe the current
  execution, not every environment recorded in history.
- Replay does not guarantee coverage. Missing investigative steps need evaluation,
  explicit coverage requirements, or feedback.
- Feedback is inherited by every environment reaching that path in the new
  snapshot, not just the environment used to resolve it. Old snapshots stay valid.
- Specialized subagents can reuse independently selected child snapshots after
  parent initial-message changes. Recorded parent invocations remain pinned.
- Execution, evidence ingestion, provider adaptation, and persistence backends
  stay with the caller. The library owns history records, references, snapshots,
  and traversal; the caller owns codecs and backend I/O. Snapshotting live sources
  is not performed by the core.
- A final-tool-input accessor is read-only inspection, not a termination rule or
  publication mechanism. DAG push updates and designated result-tool publication
  remain future work.

## Regression evidence

These links identify executable regression coverage; they are not a claim that a
new test run occurred whenever this document was edited.

| Claim | Implementation / regression evidence |
| --- | --- |
| Equal product observations share work; new evidence replays or branches | Executable product-screening examples in [Python](../python/examples/product_screening.py) and [Rust](../rust/examples/product_screening.rs) |
| Snapshot-backed paths, distinct conversations with equal assets, old/open snapshots, null values | [Python conversations](../python/tests/test_agent_conversations.py), [Rust conversations](../rust/tests/agent_conversations.rs) |
| Ordered opaque initial messages, full structural compatibility, child initialization, final-tool asset accessors | [Python domain regressions](../python/tests/test_domain_model.py), [Rust domain regressions](../rust/tests/domain_model.rs) |
| Lazy inspection, exact evidence, immutable replay, arbitrary completion order | Conversation, history, history-foundation, and loop test suites in both languages |
| Feedback resolution and shared-path inheritance | [Python feedback](../python/tests/test_feedback.py), [Rust feedback](../rust/tests/feedback.rs) |
| Specialized registries, independent children, pinned revisions, concurrent updates | [Python specialists](../python/tests/test_specialized_subagents.py), [Rust specialists](../rust/tests/specialized_subagents.rs) |
| Completion recovery and serving costs | [Python transactions](../python/tests/test_transactional_transitions.py), [cost guards](../python/tests/test_transition_cost.py), [Rust transactions](../rust/tests/transactional_transitions.rs) |
| Mutable handles, independent snapshots, both overlapping completion orders, automatic publication, failure/budget semantics and lazy storage | [Python entry point](../python/tests/test_agent_entrypoint.py), [Rust entry point](../rust/tests/agent_entrypoint.rs) |
| Initial assistant context cannot make a budget fallback eligible for continuation | `test_initial_assistant_context_does_not_make_budget_fallback_a_completed_answer` in [Python domain regressions](../python/tests/test_domain_model.py), [Rust recording](../rust/tests/always_record.rs) |
| Explicit Python snapshot formatting: reachable records, opaque representations, lazy loads, deep sharing and lightweight repr | [Python formatting suite](../python/tests/test_snapshot_format.py) |
| Direct Python snapshot pickle: reachable graph, loader materialization, child bases/results, sharing, fresh writers and failures | [Python pickle suite](../python/tests/test_snapshot_pickle.py), [direct example](../python/examples/snapshot_pickle.py) |
| Rust local-write cursor identity, capture/export ordering, non-destructive reads and lazy-loaded exclusions | [Rust record-store changes](../rust/tests/record_store_changes.rs) |
| No keys needed; implementation changes reuse or branch; external policy is caller-owned | [Python structural reuse](../python/tests/test_structural_reuse.py), [Rust structural reuse](../rust/tests/structural_reuse.rs), specialist suites |

Structural-reuse suites verify default/custom identity reuse in both directions,
stricter-key misses against representatives rather than old evidence, coarser-key
first-existing-edge selection, preserved representative context and fresh evidence,
unchanged old roots, and mode-independent child lookup.

## Required regression coverage

Preserve request/completion behavior and verify the contracts in the domain
model. This checklist specifies required coverage, not a report of a new test run:

- Unchanged observations reuse recorded reasoning with zero new model requests,
  including changed tool implementations and changed parent initial messages for
  child lookup.
- External provider/grouping policy remains caller-owned, while structural
  incompatibilities are rejected.
- Nonempty ordered initial messages, including system messages and opaque content,
  survive requests, conversation inspection, child lookup, and storage round trips.
  Role, order, content, or call-metadata changes reject incompatible history.
- Empty child initialization leaves the parent completion correctable without
  leaking ordinary work or committing sibling preparation.
- Automatic exactly-once publication, sequential reuse, both overlapping
  completion orders without merging, immediate cached completion, and no
  publication on failure.
- Lazy snapshot roundtrips without repeating initial messages/specifications;
  independent execution snapshots and conversation views survive later agent
  publications.
- Changed results branch at the first divergence, including previously
  non-splitting turns.
- Old snapshots remain unchanged, and both old and new paths remain reusable.
- Default/custom identity replay works in both directions, including child
  lookup; stricter keys branch and coarser keys select the first existing match
  deterministically.
- Full evidence survives replay, including custom grouping and opaque messages.
- Parallel/multi-tool completions work in arbitrary order.
- Invalid completions leave pending work recoverable. Python callback exceptions
  and detected child compatibility mismatches during replay can roll back the
  advancing completion for correction/retry; cover the compatibility case in
  `test_last_tool_completion_prepares_replayed_children_before_consumption`.
  Rust recorded-turn incompatibility remains terminal. Stored terminal failures
  cancel work and prevent publication; rejection must never accept incompatible
  history.
- Storage round trips and lazy traversal require no core backend I/O.
- Direct Python pickle includes exactly the complete reachable graph, including
  child invocation bases/results, iteratively deduplicated by address. Preserve
  full pickleable values and sharing; exclude locks/loaders/runtime callbacks.
  Each load gets a fresh writer namespace; missing-record and unpickleable-value
  errors surface.
- Budgets count reused model responses and preserve real history checkpoints
  rather than synthetic model answers. Initial assistant context does not make
  a budget fallback eligible for continuation.
- Asset accessors distinguish absence from null, select the last matching call in
  conversation/call order, and include accepted pending calls and earlier values
  before open feedback without executing or terminating tools.
- Flat execution-result groups contain only the supplied environments; removing
  the second partition tree must not change refinement or split-depth statistics.

Document deliberate limitations, especially nested subagents, compatibility,
lazy loading, codecs, and cross-language/version persistence differences.
History persistence is not in-flight crash recovery.

## Running verification

From the repository root:

```sh
(cd python && uv run pytest && uv run mypy src && uv run ruff check .)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```

Run the provider-free examples: `product_screening`, `agent_conversations`,
`history_replay`, `feedback`, and `specialized_subagents` in both languages, plus
Python's `snapshot_pickle`. See [commands and sources](README.md#runnable-examples).
Check relative Markdown links/heading anchors and `git diff --check` after moving
APIs, tests, or examples. The Python split-completion benchmark exercises
Agent/AgentRun serving costs; run it when changing transition bookkeeping.

The product examples assert three initial model calls for three products in two
classes, zero new model calls for matching evidence, and one new call for changed
ratings. They count tool executions separately and preserve old conversation
views. Their assessment rules and model replies are toy fixtures.

No legacy migration is required. Current-format storage roundtrips, child
provenance, feedback, opaque evidence, collision handling, transactional
preparation, request routing, and structural-cost coverage must survive API
renaming. See [persistence values and API scope](agent-assets.md#persistence-values-and-api-scope).

## What fixture tests do not establish

The examples use deterministic model stand-ins and in-memory evidence. They do
not evaluate real LLM reasoning, perform genuine literature/compliance/legal
review, establish product suitability, or validate third-party tool purity.

Real deployment requires application-level evaluation and stable evidence. Tool
contracts, immutable message ownership, and correct durable publication remain
caller responsibilities. No general cross-version wire promise, in-flight crash
recovery, automatic root merging, cross-environment isolation under custom
equivalence, or dependency propagation is claimed.
