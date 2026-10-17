# Documentation audit

[Documentation](README.md) · [Visual guide](concepts.md)

Audited against **875a19b**, the implementation base of `docs/visual-concepts`.
The root, Python, and Rust READMEs were checked against both implementations,
relevant regression tests, and all four executable examples. Proposed specialized
subagents and naming changes were read only as design context, not shipping APIs.

## Corrections and reorganization

- Replaced the unqualified “without generating the same token twice” promise.
  Prefix sharing saves LLM work, but separate paths can repeat tokens/answers;
  tools still execute per environment. Equal answers do not remerge classes.
- Moved the root README's detailed history/storage/failure contracts into the
  [history reference](history-reference.md), retaining reference information and
  updating both language READMEs' links. The root is now a short entry point.
- Made the Rust README's opening sample genuinely standalone: `main`, no undefined
  wait function, no `todo!()` tool executor, and no rustdoc-only hidden line in
  plain Markdown. Compiled and ran the exact fenced snippet. Fixed its unresolved
  Markdown reference to `BatchLoop`.
- Labeled Python/provider/scheduler snippets and later Rust fragments as integration
  sketches; linked complete executable demos rather than pretending placeholders
  are runnable. Provider calls must be adapted to the loop's message/call types.
- Qualified Python-only committed-root/reentrant callback guarantees. Rust's
  `history()` can assemble child roots and allocate records; it does not implement
  Python's callback undo journal. The public snapshot remains immutable in both.
- Defined agent/asset without claiming an API rename, separate asset handles, or
  result-tool publication. Current terminal values are no-tool assistant replies.
- Documented the current single-subagent configuration and parent-linked child
  replay. Independent lookup after parent-prompt changes is design direction.
  DAG push updates remain future [issue #1](https://github.com/kaeluka/lockstep-loop/issues/1).

## Evidence checked

| Claim | Implementation | Regression evidence |
| --- | --- | --- |
| Ordered tuples, arbitrary completion order, recursive refinement | Python [`_plan_turn` / `_advance_turn`](../python/src/lockstep_loop/core.py); Rust [`advance_turn`](../rust/src/lib.rs) | [Python history](../python/tests/test_history.py), [Rust loop](../rust/tests/loop.rs), [Rust history](../rust/tests/history.rs) |
| Full actual evidence, representatives, immutable extension, lazy records | [Python history records](../python/src/lockstep_loop/history.py), [Rust history records](../rust/src/history.rs) | [Python foundation](../python/tests/test_history_foundation.py), [Rust foundation](../rust/tests/history_foundation.rs); 256-turn persistence and selected-path loading |
| Feedback resolves before reasoning, inherits by path, preserves old roots, allows same answer, fails on misses | Python [`_request_llm` / `_accept_assistant`](../python/src/lockstep_loop/core.py); Rust [`schedule_branch` / `accept_assistant`](../rust/src/lib.rs) | [Python feedback](../python/tests/test_feedback.py), [Rust feedback](../rust/tests/feedback.rs) |
| Nested child divergence can leave parent shared | Python [`_prepare_child` / `_rollup_children`](../python/src/lockstep_loop/core.py); Rust [`prepare_child` / `rollup_children`](../rust/src/lib.rs) | Both foundation suites above, including changed child observations with no new parent LLM work |
| Invalid completions preserve pending work; language-specific callback failures | Both `complete` implementations above | [Python transactions](../python/tests/test_transactional_transitions.py), [Python cost guards](../python/tests/test_transition_cost.py), [Rust transactions](../rust/tests/transactional_transitions.rs), request-routing suites |

## Checks run

- **Python:** 78 tests passed; mypy and Ruff passed. `uv` was unavailable in this
  environment, so these ran with Python 3.14 in a worktree-local virtualenv using
  the declared dev tools. The documented `uv` commands remain the project workflow.
- **Rust:** 44 integration tests passed; Clippy with warnings denied and formatting
  checks passed. Cargo's doctest phase contains **zero tests**; that alone does not
  validate README snippets, hence the separate snippet compilation above.
- Both `history_replay` demos and both `feedback` demos passed their assertions,
  with the documented output. They use deterministic caller-side stand-ins, not a
  real provider or a real repository scanner.
- A temporary caller-side smoke driver also exercised the guide's A/B/C/D repo
  walkthrough: three final classes, reverse-order multi-tool completions, zero LLM
  calls on a matching new commit, one on a changed `/foo` observation, reuse of all
  extended paths, and unchanged old records. It used fixture outputs, not live repo I/O.
- The Python split benchmark command ran for 100/400/1000 environments (three
  repeats). This validates the command/workload, not a general scalability promise.
- All relative Markdown links/heading anchors were checked; issue #1 resolved.
  All four editable sources were rendered locally to SVG and PNG. PNGs were
  inspected, then every SVG was screenshot in a local rendered Markdown guide at
  900 CSS-pixel content width. A child-heading/arrow overlap was fixed and checked
  again. Labels, boundaries, arrows, and alt text were readable; no public upload.

## Limits and remaining uncertainty

No new implementation defect was confirmed by this audit. One existing behavior
worth noting is that both serial drivers silently ignore additional subagent
entries after the first. The docs now explicitly discourage that unsupported
configuration; implementation changes belong to the separate subagent work.

These docs describe the checked-out base, not unmerged work. Specialized child
lookup and Agent/Asset API naming will need a documentation update when merged.
Real-provider adaptation, caller storage durability, payload immutability, and tool
purity cannot be established by these fixture-based tests. No stable cross-version
wire format, crash recovery, independent-root merge, or DAG propagation is claimed.
