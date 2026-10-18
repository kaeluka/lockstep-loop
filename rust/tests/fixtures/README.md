# Naming compatibility fixture

`agent-naming-2c6d901.json` was generated with the **unmodified 2c6d901** Rust
implementation, extracted from Git into a temporary directory in this worktree.
It contains a descriptor/query and five shallow records, encoded using the
pre-rename Serde implementations. The array of `(NodeRef, record)` pairs is a
caller-side fixture container, not a new core storage format.

The records cover lossy per-environment evidence, user feedback, an earlier and
later answer, and old/new child revisions with provenance. Tests deserialize via
the aliases, inspect assets, and replay normally. Deprecated `contract` and
`history_contract` labels are now ignored on import and omitted on new writes;
all other JSON record/envelope data is preserved. Additional tests import different
labels and verify they cannot poison matching child caches. The fixture is unchanged.
Positional/binary codecs may need migration; JSON compatibility does not promise
general Serde wire compatibility. Nothing encodes an asset ID or live execution.
