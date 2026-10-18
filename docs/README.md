# Understand lockstep-loop

**[Take the visual tour →](concepts.md)**

See how product checks share reasoning, dataset releases reuse prior
investigations, supplier reviews improve with feedback, and research specialists
contain divergence. These are applications of the same domain-neutral loop.

- **Choose a workload:** [use cases and environment/tool designs](use-cases.md).
  Includes documents, studies, products, datasets, suppliers, and source code.
- **Name and inspect the model:** [Agent, run, snapshot and Asset glossary](agent-assets.md).
- **Serve requests:** [Python](../python/README.md) · [Rust](../rust/README.md).
- **Record, replay, and persist:** [technical reference](history-reference.md).

## Runnable examples

Run from the repository root. Each uses scripted model replies and local fixture
evidence; no provider credentials, real document analysis, or remote storage is
involved. They demonstrate the orchestration contract, not the quality of an LLM.

| Example | What it demonstrates | Python | Rust |
| --- | --- | --- | --- |
| `product_screening` | Product ratings, shared batch work, replay on new specifications | [source](../python/examples/product_screening.py) | [source](../rust/examples/product_screening.rs) |
| `snapshot_pickle` | Easy self-contained Python snapshot save/load; no manual record bundles | [source](../python/examples/snapshot_pickle.py) | Python-only pickle API |
| `history_replay` | Advanced shallow-record persistence, lazy loads, unchanged and changed paths | [source](../python/examples/history_replay.py) | [source](../rust/examples/history_replay.rs) |
| `feedback` | Shared-path continuation, unchanged answers, explicit resolution misses | [source](../python/examples/feedback.py) | [source](../rust/examples/feedback.rs) |
| `specialized_subagents` | Reuse a child's investigation after changing the parent's prompt | [source](../python/examples/specialized_subagents.py) | [source](../rust/examples/specialized_subagents.rs) |
| `agent_assets` | Inspect conversations and evidence; preserve old, open, and revised snapshots | [source](../python/examples/agent_assets.py) | [source](../rust/examples/agent_assets.rs) |

```sh
(cd python && PYTHONPATH=src python3 examples/product_screening.py)
cargo run -p lockstep-loop --example product_screening
# Substitute another example name from the table to explore its protocol.
```

The existing protocol demos use a small file-inspection fixture. That fixture is
not an API requirement: product ratings, document sections, and dataset statistics
use the same request/completion types. The [use-case guide](use-cases.md) explains
how to bind tools to other evidence sources.

## Reference and maintenance

- [Agent lifecycle and snapshot-backed Asset inspection](agent-assets.md)
- [Persistence, feedback, child caching, and limitations](history-reference.md)
- [Diagram sources and previews](diagrams/README.md)
- [Documentation verification](verification.md)

[Back to the project](../README.md)
