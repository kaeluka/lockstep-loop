# Naming compatibility fixture

`agent-naming-2c6d901.json` was generated with the **unmodified 2c6d901** Rust
implementation, extracted from Git into a temporary directory in this worktree.
It contains a descriptor/query and five shallow records, encoded using the
pre-rename Serde implementations. The array of `(NodeRef, record)` pairs is a
caller-side fixture container, not a new core storage format.

The records cover lossy per-environment evidence, user feedback, an earlier and
later answer, and old/new child revisions with provenance. Tests deserialize via
the new aliases, assert identical JSON record/envelope data, inspect assets, and
replay normally. Nothing encodes an asset ID or a live resolver/execution.
