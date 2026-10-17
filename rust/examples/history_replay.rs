//! Run from the repository root: cargo run -p lockstep-loop --example history_replay
use lockstep_loop::{
    Agent, BatchLoop, Completion, LlmCompletion, LoopConfig, NodeRef, RecordStore, Request,
    RequestKind, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

fn investigate(
    env: &str,
    observation: &str,
    previous: Option<Agent<String>>,
) -> Result<(Agent<String>, usize, String), Box<dyn std::error::Error>> {
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
        Some(agent) => config.with_agent(agent, contract),
        None => config
            .with_recording(contract)
            .with_record_store(RecordStore::new("writer-1")),
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
    let answer = result.results[env].clone();
    Ok((result.into_agent().unwrap(), llm_calls, answer))
}

type Backend = BTreeMap<NodeRef, Vec<u8>>;

fn persist(agent: &Agent<String>, backend: &mut Backend) -> Vec<u8> {
    // Persist shallow immutable records BEFORE atomically publishing descriptor.
    for (reference, node) in agent.store.records() {
        let bytes = serde_json::to_vec(&node).unwrap();
        if let Some(old) = backend.get(&reference) {
            assert_eq!(old, &bytes);
        } else {
            backend.insert(reference, bytes);
        }
    }
    serde_json::to_vec(&agent.descriptor).unwrap()
}

fn restore(descriptor: &[u8], backend: Backend, fresh_writer_namespace: &str) -> Agent<String> {
    // Reserve a globally unique writer namespace in real storage (e.g. a database
    // sequence or UUID). Never reuse writer-N across independent writable stores.
    Agent {
        descriptor: serde_json::from_slice(descriptor).unwrap(),
        store: RecordStore::with_loader(fresh_writer_namespace, move |reference| {
            // Caller-owned codec/I/O. No traversal or observation matching here.
            Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
        }),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (old, calls, _) = investigate("yesterday", "clean", None)?;
    assert_eq!(calls, 2);
    let mut backend = Backend::new();
    let old_descriptor = persist(&old, &mut backend);
    let restored = restore(&old_descriptor, backend.clone(), "writer-2");
    let (_, calls, answer) = investigate("today", "clean", Some(restored.clone()))?;
    assert_eq!((calls, answer.as_str()), (0, "Status: clean"));
    let (extended, calls, _) = investigate("tomorrow", "dirty", Some(restored))?;
    assert_eq!(calls, 1);
    assert_eq!(old.root().observations.len(), 1);
    assert_eq!(extended.root().observations.len(), 2);
    let extended_descriptor = persist(&extended, &mut backend);
    for (index, status) in ["clean", "dirty"].iter().enumerate() {
        let restored = restore(
            &extended_descriptor,
            backend.clone(),
            &format!("later-{index}"),
        );
        let (_, calls, _) = investigate("later", status, Some(restored))?;
        assert_eq!(calls, 0);
    }
    let restored = restore(&old_descriptor, backend, "old-snapshot-reader");
    let (_, calls, _) = investigate("old snapshot", "clean", Some(restored))?;
    assert_eq!(calls, 0);
    println!("unchanged: 0 LLM calls; changed: 1; both paths reusable");
    Ok(())
}
