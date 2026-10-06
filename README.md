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
  per-environment answers back into the parent turn. With `inherit_session`
  enabled, each child inherits the conversation before the calling assistant
  turn and appends its own user prompt. Multiple calls in one turn fork from
  the same prefix and join as separate tool results in call order.
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

## Usage: inherited subagent forks

These runnable examples use a scripted model reply so no API key is needed.
Replace `demo_llm` with your provider adapter for real model calls. The parent
requests two `fork` calls in one turn; each child inherits the initial user
message and appends either `user("do x")` or `user("do y")`. The calling
assistant turn is excluded from both children.

### Python

```python
from lockstep_loop import (
    LlmCompletion, LlmRequest, SubagentTool, ToolCall, run_to_completion,
)

fork = SubagentTool(
    name="fork",
    description="Investigate a task using the current conversation context.",
    parameters={
        "type": "object",
        "properties": {"task": {"type": "string"}},
        "required": ["task"],
    },
    build_prompt=lambda args: args["task"],
    sub_tool_specs=lambda env: [],  # Supply read-only tools for real investigations.
    inherit_session=True,
)


def demo_llm(request: LlmRequest[str]) -> LlmCompletion[str]:
    last = request.messages[-1]
    if len(request.messages) == 1:  # Parent asks for two independent forks.
        return LlmCompletion(
            request_id=request.request_id,
            content="",
            tool_calls=(
                ToolCall("x", "fork", {"task": "do x"}),
                ToolCall("y", "fork", {"task": "do y"}),
            ),
        )
    if last.role == "user":  # Child receives inherited history + its task.
        answer = f"Finished: {last.content}"
    else:  # Parent resumes with separate tool results, ordered X then Y.
        answer = "\n".join(m.content for m in request.messages if m.role == "tool")
    return LlmCompletion(request_id=request.request_id, content=answer)


result = run_to_completion(
    prompt="Investigate this repository without changing anything.",
    env_ids=["fork-a", "fork-b"],
    tools=lambda env: [fork],
    complete_llm=demo_llm,
)
print(result.results["fork-a"])
assert result.results["fork-a"] == result.results["fork-b"]
```

### Rust

```rust
use lockstep_loop::{
    run_to_completion, LlmCompletion, LlmRequest, Role, SubagentTool,
    ToolCall, ToolEntry, ToolSpec,
};
use serde_json::json;

fn demo_llm(request: &LlmRequest<String>) -> LlmCompletion<String> {
    let last = request.messages.last().unwrap();
    let (content, tool_calls) = if request.messages.len() == 1 {
        // Parent asks for two independent forks.
        (String::new(), vec![
            ToolCall::new("x", "fork", json!({"task": "do x"})),
            ToolCall::new("y", "fork", json!({"task": "do y"})),
        ])
    } else if last.role == Role::User {
        // Child receives inherited history + its task.
        (format!("Finished: {}", last.content), vec![])
    } else {
        // Parent resumes with separate tool results, ordered X then Y.
        let answers = request.messages.iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        (answers, vec![])
    };
    LlmCompletion {
        request_id: request.request_id.clone(),
        content,
        tool_calls,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fork = SubagentTool::new(
        ToolSpec::new("fork", "Investigate using the current context.", json!({
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
        })),
        |args| args["task"].as_str().unwrap().to_owned(),
        |_| vec![], // Supply read-only tools for real investigations.
        1, // Maximum subagent nesting depth.
    ).with_inherit_session(true);

    let result = run_to_completion(
        "Investigate this repository without changing anything.".to_owned(),
        vec!["fork-a".to_owned(), "fork-b".to_owned()],
        |_| vec![ToolEntry::Subagent(fork.clone())],
        demo_llm,
        None, // No step budget.
    )?;
    println!("{}", result.results["fork-a"]);
    assert_eq!(result.results["fork-a"], result.results["fork-b"]);
    Ok(())
}
```

Both examples print:

```text
Finished: do x
Finished: do y
```

Each child shares its model work across both environments. Only final answers
return to the parent, not child transcripts. The serial convenience driver
above serves requests one at a time; drive `BatchLoop` directly to execute
siblings concurrently or finish them out of order. The parent joins only after
all calls in its turn finish, grouping environments by their result tuple.

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

## License

MIT © 2026 Stephan Brandauer. See [`LICENSE`](LICENSE).
