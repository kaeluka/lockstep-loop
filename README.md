# lockstep-loop

Run a single agent prompt across thousands of similar, read-only environments
without generating the same token twice.

This repository contains two implementations of the same provider-agnostic
partition-refinement loop:

| Implementation | Location | Package |
| --- | --- | --- |
| Python ≥ 3.10 | [`python/`](python/) | `lockstep-loop` |
| Rust | [`rust/`](rust/) | `lockstep-loop` |

Neither implementation calls an LLM or owns infrastructure. The caller pulls
provider-neutral LLM and tool requests, executes them with any scheduler or
provider, and returns completions in any order.

## The idea

- All environments begin in one equivalence class and share one conversation.
- A model tool call fans out over every environment still in that class.
- Environments with equal tool-result tuples stay together; differing results
  split the class into independent conversations.
- Multiple tool calls in one assistant turn are independent requests and may
  execute concurrently; refinement waits for the whole turn.
- A reserved subagent tool can spawn a nested lockstep loop and fold its final
  per-environment answers back into the parent turn.
- The result is a prefix-sharing split trie plus the final equivalence classes.

Cost therefore scales with the number of distinct tool-result traces rather
than directly with the number of environments.

The soundness requirements are the same in both implementations: tools must be
read-only/pure, tool specifications must be identical across environments, and
the model must not be shown environment identities. Requests do expose their
environment set to the caller for scheduling, accounting, and rate limiting.

## Repository layout

```text
python/   Python package, tests, and local examples
rust/     Rust crate and integration tests
```

The APIs follow the same vocabulary (`BatchLoop`, `LlmRequest`, `ToolRequest`,
completions, budgets, split trie, subagent), while retaining language-native
serving models:

- Python methods are internally synchronized.
- Rust methods use `&mut self`; callers can wrap a loop in their preferred
  mutex when cross-thread access is needed.
- Rust budgets take a caller-supplied exhaustion function because an opaque
  generic payload cannot safely be fabricated from a string sentinel.

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
