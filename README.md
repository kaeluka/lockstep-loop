# lockstep-loop

**Run one investigation across many subjects. Reuse the reasoning they share.**

Compare supplier evidence packs, screen research papers, review contracts, check
product specifications, inspect datasets—or analyze source code. When the task
is the same and the observations match, why generate the same reasoning again?

lockstep-loop shares a conversation across read-only **environments**, splitting
only when tool results differ. Keep the recorded experience, replay it against
new subjects or updated snapshots, and continue an investigation with feedback.
Your application chooses the model, executes the tools, and stores the records.

## What is an environment?

A caller-owned handle identifying a stable subject and the evidence its tools can
read—for example, `owner/repo@commit`, not the repository itself. Other handles
can identify archived study documents, versioned product specifications, contract
bundles, or dataset releases. An environment is **not** necessarily a filesystem,
sandbox, or machine.

| Same investigation | Different environments |
| --- | --- |
| “Which studies meet these inclusion criteria?” | Study document bundles |
| “Does this product meet our operating requirements?” | Versioned product specifications |
| “Is this supplier's evidence complete?” | Supplier assessment packets |
| “Flag nonstandard renewal obligations.” | Contracts and their amendments |
| “Can this dataset enter our reporting pipeline?” | Frozen dataset releases |
| “Find unsafe path handling.” | Repository snapshots |

The gain depends on **shared observations**, not just similar-looking subjects.
Tools still execute per environment. See [use cases and tool design](docs/use-cases.md)
for what can be shared, what causes branching, and when this is a poor fit.

## The model

- An **Agent** is a mutable handle for starting investigations and accumulating experience.
- An **AgentSnapshot** captures ordered initial messages, structural configuration,
  and one immutable **history trie** sharing common conversation prefixes.
- An **AgentRun** is the execution handle: it captures the current snapshot, serves
  requests, and publishes on completion. Investigating is an operation on an agent.
- A **Conversation** is one path through the history trie, exposed as a
  snapshot-anchored view with messages and exact tool-result evidence.
- An **asset** is the value the caller extracts from a conversation: for example,
  its latest model answer or the input to a typed `submitFinalAnswer` tool.
- **Replay** re-executes tools and reuses recorded assistant turns until an
  observation differs. A fully matching completed path needs no new LLM calls.
- **Feedback** extends a resolved investigation in a new immutable snapshot.
- **Subagents** contain divergent investigations behind their returned findings;
  independently cached specialists can survive changes to parent initial messages.

**[Take the visual tour →](docs/concepts.md)**

## Try it

Both implementations are provider-neutral. These demos use scripted model replies
and in-memory evidence/storage; no LLM account or external service is needed.

| Implementation | API & setup | Product screening | Other runnable examples |
| --- | --- | --- | --- |
| Python ≥ 3.10 | [Python README](python/README.md) | [source](python/examples/product_screening.py) | [examples](docs/README.md#runnable-examples) |
| Rust | [Rust README](rust/README.md) | [source](rust/examples/product_screening.rs) | [examples](docs/README.md#runnable-examples) |

```sh
(cd python && PYTHONPATH=src python3 examples/product_screening.py)
cargo run -p lockstep-loop --example product_screening
# Initial products share work; a matching new product uses 0 new LLM calls.
```

## What the library owns

The core is a partition-refinement state machine—not a provider client, executor,
web crawler, or storage backend. Your application pulls LLM/tool requests,
executes them, and returns completions. Independent requests may finish in any
order; refinement waits for each turn's **complete, ordered tuple** of results.
Classes split, never remerge within an execution.

Initial input is a nonempty ordered sequence of `MessageEnvelope` values—for
example, a system message followed by several user messages. The library owns
role/call structure but treats the message contents as opaque, including nontext
or multimodal values.

Supply pure/read-only tools over stable snapshots, identical tool specs across
environments, and immutable message contents. Environment IDs are caller-side routing
metadata, not model-visible context. Snapshot live sources before using them;
recording does not make a changing web page or database query pure.

Recording is unconditional, including children. Start with `Agent`, drive
`agent.start(environments)`, and save `agent.snapshot()`. Restore with
`Agent.from_snapshot(snapshot, ...runtime callbacks...)`, without repeating initial
messages or tool specs. Successful completion automatically advances the handle exactly once.
Python supports `pickle.dump(agent.snapshot(), file)` and restoring the loaded
snapshot directly; only load trusted pickle data. Export eagerly includes all
reachable history and requires pickleable payloads, not runtime callbacks. The
lower-level descriptor/record API remains available for lazy/incremental storage
and Rust. The library owns immutable records, references, snapshots, and traversal;
the caller owns codecs, filesystem/blob/database backends, and durable publication.
The core performs no backend I/O. No manual version key is needed: pure tools are rerun, so implementation changes either
preserve observations (reuse) or change them (branch). The current grouping function
compares fresh outputs and recorded representatives; default/custom mode and
function identity are not cache keys. Coarser keys select the first matching stored
edge without merging history. External provider/grouping policy and cache isolation
remain caller responsibilities. This is observation-based reuse, not
semantic similarity search, model training, or a guarantee of exhaustive research.

**Warning — overlapping runs:** Each start captures the current root. Histories
are **not merged**; the last completion replaces the Agent's state, potentially
excluding other runs' experience. Retain each run's snapshot, or complete runs
sequentially to accumulate experience. Old snapshots remain usable; inspecting
an old run never republishes it. Failed/abandoned runs do not advance the handle.

## Go deeper

- [Documentation map and runnable examples](docs/README.md)
- [Use cases, environment boundaries, and sharing opportunities](docs/use-cases.md)
- [Agents, conversations, and asset accessors](docs/agent-assets.md)
- [Replay, feedback, persistence, and compatibility reference](docs/history-reference.md)
- [Specialized subagents and independent child caches](docs/history-reference.md#specialized-tools-and-independently-reusable-child-histories)
- [Verification](docs/verification.md) · [Editable diagrams](docs/diagrams/README.md)

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
