# Naming compatibility fixture

`agent-naming-2c6d901.pickle` was generated with protocol 4 using the **unmodified
2c6d901** Python implementation, extracted from Git into a temporary directory in
this worktree. It uses the old exported `HistoryDescriptor`, `HistoryQuery`,
`HistoryNode`, `SubagentHistory`, and `ChildHistoryRun` classes. Only trusted test
fixtures should be unpickled.

It contains a descriptor/query and five individually addressed records: a lossy
multi-tool observation (A and B have distinct actual outputs), a prior answer,
user feedback, a new answer, and old/new child revisions with provenance. The
Agent/Asset tests load it, inspect contextual assets, replay it with the new API,
and assert stable legacy wire globals/field names and record values. Pickle memo
layout can differ with string interning; byte identity is not a format guarantee.
It is not a pickled executor, live store, or independently persisted asset. The Rust fixture uses the same record shape
with string payloads and the pre-rename Rust Serde codec.
