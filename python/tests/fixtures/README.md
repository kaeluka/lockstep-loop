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
and assert stable legacy class globals and evidence values. Contract removal tests
also import different legacy labels/modes and verify structural cache matching. Old
`contract`/`history_contract` labels and `custom_grouping` mode are discarded during
unpickling and omitted on new writes; the fixture binary is unchanged. Pickle memo layout can also differ
with string interning; byte/schema identity with old output is not promised.
It is not a pickled executor, live store, or independently persisted asset. The Rust fixture uses the same record shape
with string payloads and the pre-rename Rust Serde codec.
