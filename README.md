# lockstep-loop

Run a single agent prompt across thousands of similar, read-only environments
without generating the same token twice.

## How it works

- All environments start in one equivalence class sharing **one** LLM
  conversation.
- When the model makes a tool call, the call is evaluated on **every**
  environment still in the class (SIMD-style fan-out).
- Results are canonicalized (`group_by`) and the class is refined — one
  sub-conversation per distinct result key. Branches then continue
  independently, DFS, so parent prefixes stay warm in provider prompt caches.
- The process repeats per branch until done, yielding a prefix-sharing trie
  whose leaves are **equivalence classes of environments**: cost scales with
  the number of distinct tool-call traces, not the number of environments.

Two invariants make this sound: tools must be **pure** (results depend only
on environment + arguments), and the model must never see which environment
it is in (tool specs identical across environments).

## Status

Skeleton. The planned entrypoint:

```python
from lockstep_loop import run_batch

result = run_batch(
    prompt="summarize what src/pkgA contains",
    env_ids=["fork-1", "fork-2", "fork-3"],
    tools=lambda env_id: make_tools(env_id),  # impls vary, specs must not
    group_by=None,        # default canonicalizer (planned)
    scheduler="dfs",
)
# result.tree, result.classes, result.results (env_id -> final message)
```

## Development

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
