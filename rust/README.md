# lockstep-loop for Rust

Provider-agnostic lockstep orchestration for running one LLM conversation over
many similar, read-only environments and splitting it only when tool results
diverge.

The crate never calls an LLM. Drive [`BatchLoop`] directly for arbitrary
scheduling, or use `run_to_completion` for a serial local-tool driver.

```rust
use lockstep_loop::{
    Completion, LlmCompletion, LoopConfig, BatchLoop, Request, RequestKind,
};

let mut loop_ = BatchLoop::new(LoopConfig::new(
    "inspect the repository".to_owned(),
    vec!["fork-a".to_owned(), "fork-b".to_owned()],
    vec![],
))?;

while !loop_.is_done() {
    match loop_.next_request(RequestKind::Any) {
        Some(Request::Llm(request)) => {
            // Adapt `request.messages` and `request.tools` to any provider.
            loop_.complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id,
                "done".to_owned(),
            )))?;
        }
        Some(Request::Tool(request)) => {
            // Execute request.tool_call once for every request.env_ids entry,
            // then return one result per environment.
            todo!()
        }
        None => wait_for_an_outstanding_completion(),
    }
}

let result = loop_.result()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Subagents and inherited sessions

`SubagentTool` starts a nested lockstep loop over the calling branch's environments.
Children start fresh by default. Opt into conversation inheritance with the
builder (or the public `inherit_session` field):

```rust
let fork = SubagentTool::new(
    fork_spec,
    |args| args["task"].as_str().unwrap().to_owned(),
    child_tool_specs,
    1, // maximum nesting depth
).with_inherit_session(true);
```

Each child inherits the parent history **before the calling assistant turn**,
then appends its generated prompt as a user message. Multiple calls in one turn
inherit the same prefix, excluding the calling turn and all sibling calls and
results. Earlier completed tool turns remain in the history. Child tool specs
still come from their configured factory; provider-side session state is not copied.

The existing driver serves children in any order. The parent waits for all calls
in the turn, including ordinary tools, then inserts separate tool results in call
order and refines its environment groups by the result tuple. Only final answers
return to the parent, not child transcripts. Descendants retain the inheritance
setting up to `max_depth`. Model routing and prefix caching are caller concerns.

## Payloads and grouping

Conversation content and tool results are generic in `P`. `BatchLoop::new`
groups results by value equality; `BatchLoop::with_group_by` accepts a custom
key function for deliberate lossy grouping. Tool arguments and JSON schemas
use `serde_json::Value`.

Rust budgets include an exhaustion callback (`Budget::new(max_steps, f)`) so
the caller can produce a valid value of the opaque payload type.

See `tests/loop.rs` for complete examples covering partition refinement,
multi-tool turns, custom grouping, bytes payloads, scheduling, and nested
subagents.
