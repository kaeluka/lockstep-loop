# lockstep-loop

Investigate many similar repositories without repeating the reasoning they share.
Keep that experience, then replay it against the next commit.

lockstep-loop shares one conversation across read-only environments until their
observations differ. It then branches, keeping the common prefix. Recorded
investigations can be reused and continued with feedback, without rewriting old
snapshots.

**[Start with the visual guide →](docs/concepts.md)**

Agents, assets, conversations, replay, feedback, and subagents — through one
repository-analysis example.

## Try it

Two provider-neutral implementations; no LLM account needed for these examples:

| Implementation | API & setup | Replay demo | Feedback demo |
| --- | --- | --- | --- |
| Python ≥ 3.10 | [Python README](python/README.md) | [source](python/examples/history_replay.py) | [source](python/examples/feedback.py) |
| Rust | [Rust README](rust/README.md) | [source](rust/examples/history_replay.rs) | [source](rust/examples/feedback.rs) |

```sh
(cd python && PYTHONPATH=src python3 examples/history_replay.py)
cargo run -p lockstep-loop --example history_replay
# unchanged: 0 LLM calls; changed: 1; both paths reusable

(cd python && PYTHONPATH=src python3 examples/feedback.py)
cargo run -p lockstep-loop --example feedback

(cd python && PYTHONPATH=src python3 examples/specialized_subagents.py)
cargo run -p lockstep-loop --example specialized_subagents
# changed parent: 0 fresh child LLM calls; child tools still rerun

(cd python && PYTHONPATH=src python3 examples/agent_assets.py)
cargo run -p lockstep-loop --example agent_assets
```

## What the library owns

The core is a partition-refinement state machine, not a provider client, executor,
or storage backend. Your application pulls LLM/tool requests, executes them, and
returns completions. Independent requests may finish in any order; each turn
refines on the **complete, ordered tuple** of results. Classes split, never remerge
within an execution. Shared reasoning saves LLM work; tools still run per
environment, and unrelated paths may generate similar answers independently.

You supply pure/read-only tools over stable snapshots, identical tool specs across
environments, model-visible content that does not expose environment identities,
and immutable payloads. Recording/replay is opt-in on `BatchLoop`; you persist the
`Agent` snapshots. An `Asset` is a leaf investigation backed by its full conversation,
not just its value or environment class. Old `History` APIs remain compatibility aliases.

## Go deeper

- [Documentation map](docs/README.md)
- [Agent/Asset glossary, inspection API, and compatibility mapping](docs/agent-assets.md)
- [End-to-end agent/assets example](docs/agent-assets.md#end-to-end-examples)
- [History, replay, feedback, and persistence contracts](docs/history-reference.md)
- [Specialized subagents and independent child caches](docs/history-reference.md#specialized-tools-and-independently-reusable-child-histories)
- [README audit and verification](docs/verification.md)
- [Editable diagrams and local rendering](docs/diagrams/README.md)

## Development

```sh
# Python
cd python
uv sync
uv run pytest
uv run mypy src
uv run ruff check .

# Rust (from the repository root)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

MIT © 2026 Stephan Brandauer. See [LICENSE](LICENSE).
