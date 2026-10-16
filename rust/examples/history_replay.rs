//! Run from the repository root: cargo run -p lockstep-loop --example history_replay
use lockstep_loop::{
    BatchLoop, Completion, History, LlmCompletion, LoopConfig, Request, RequestKind, ToolCall,
    ToolCompletion, ToolSpec,
};
use serde_json::json;

fn investigate(
    env: &str,
    observation: &str,
    previous: Option<History<String>>,
) -> Result<(History<String>, usize, String), Box<dyn std::error::Error>> {
    let contract = "status-investigation/v1;exact-grouping;provider-policy-v1";
    let config = LoopConfig::new(
        "Read the revision's status.".to_owned(),
        vec![env.to_owned()],
        vec![ToolSpec::new(
            "status",
            "Read status without changing it.",
            json!({}),
        )],
    );
    let config = match previous {
        Some(history) => config.with_replay(history, contract),
        None => config.with_recording(contract),
    };
    let mut loop_ = BatchLoop::new(config)?;
    let mut llm_calls = 0;
    while !loop_.is_done() {
        // This example completes requests synchronously.
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(request) => {
                llm_calls += 1;
                // Replace this deterministic stand-in with a provider adapter.
                let reply = if request.messages.len() == 1 {
                    LlmCompletion {
                        request_id: request.request_id,
                        content: "Check status.".to_owned(),
                        tool_calls: vec![ToolCall::new("s", "status", json!({}))],
                    }
                } else {
                    LlmCompletion::final_answer(
                        request.request_id,
                        format!("Status: {}", request.messages.last().unwrap().content),
                    )
                };
                loop_.complete(Completion::Llm(reply))?;
            }
            Request::Tool(request) => {
                // Caller executes the pure tool in each requested environment.
                let results = request
                    .env_ids
                    .into_iter()
                    .map(|env| (env, observation.to_owned()))
                    .collect();
                loop_.complete(Completion::Tool(ToolCompletion {
                    request_id: request.request_id,
                    results,
                }))?;
            }
        }
    }
    let result = loop_.result()?;
    Ok((
        result.history.unwrap(),
        llm_calls,
        result.results[env].clone(),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (old, calls, _) = investigate("yesterday", "clean", None)?;
    assert_eq!(calls, 2);
    // Caller-side codec. Persist these bytes wherever you choose. No core I/O.
    let bytes = serde_json::to_vec(&old)?;
    let restored: History<String> = serde_json::from_slice(&bytes)?;
    let (_, calls, answer) = investigate("today", "clean", Some(restored.clone()))?;
    assert_eq!((calls, answer.as_str()), (0, "Status: clean"));
    let (extended, calls, _) = investigate("tomorrow", "dirty", Some(restored))?;
    assert_eq!(calls, 1);
    assert_eq!(old.root.observations.len(), 1);
    assert_eq!(extended.root.observations.len(), 2);
    for status in ["clean", "dirty"] {
        let (_, calls, _) = investigate("later", status, Some(extended.clone()))?;
        assert_eq!(calls, 0);
    }
    println!("unchanged: 0 LLM calls; changed: 1; both paths reusable");
    Ok(())
}
