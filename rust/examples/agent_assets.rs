//! cargo run -p lockstep-loop --example agent_assets
//! An Agent is prompt + structural configuration + accumulated trie. Assets are contextual leaf views.
use lockstep_loop::{
    Agent, BatchLoop, BatchResult, Completion, LlmCompletion, LoopConfig, NodeRef, RecordStore,
    Request, RequestKind, Role, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

const PROMPT: &str = "Inspect the repository";
const NOTE: &str = "Also inspect /foo";
fn config(envs: &[&str]) -> LoopConfig<String> {
    LoopConfig::new(
        PROMPT.into(),
        envs.iter().map(|s| (*s).into()).collect(),
        vec![
            ToolSpec::new("read", "Read entry", json!({})),
            ToolSpec::new("foo", "Inspect /foo", json!({})),
        ],
    )
}
fn serve(mut loop_: BatchLoop<String>) -> (BatchResult<String>, usize) {
    let mut llms = 0;
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                llms += 1;
                let calls = if r.messages.len() == 1 {
                    vec![ToolCall::new("read", "read", json!({}))]
                } else if r.messages.last().unwrap().role == Role::User {
                    vec![ToolCall::new("foo", "foo", json!({"path": "/foo"}))]
                } else {
                    vec![]
                };
                let value = if r.messages.last().unwrap().content == "vulnerability" {
                    "found"
                } else {
                    "no issues"
                };
                loop_
                    .complete(Completion::Llm(LlmCompletion {
                        request_id: r.request_id,
                        content: value.into(),
                        tool_calls: calls,
                    }))
                    .unwrap();
            }
            Request::Tool(r) => {
                let results = r
                    .env_ids
                    .iter()
                    .map(|env| {
                        let value = if r.tool_call.name == "read" {
                            if env == "D" { "unrelated" } else { "shared" }
                        } else if env == "B" {
                            "vulnerability"
                        } else {
                            "safe"
                        };
                        (env.clone(), value.into())
                    })
                    .collect();
                loop_
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results,
                    }))
                    .unwrap();
            }
        }
    }
    (loop_.result().unwrap(), llms)
}

type Backend = BTreeMap<NodeRef, Vec<u8>>;
fn persist(agent: &Agent<String>, backend: &mut Backend) -> Vec<u8> {
    // Caller persistence: shallow records first, descriptor publication last.
    for (reference, node) in agent.store.records() {
        let encoded = serde_json::to_vec(&node).unwrap();
        if let Some(old) = backend.get(&reference) {
            assert_eq!(old, &encoded);
        } else {
            backend.insert(reference, encoded);
        }
    }
    serde_json::to_vec(&agent.descriptor).unwrap()
}
fn restore(descriptor: &[u8], backend: Backend, fresh_writer: &str) -> Agent<String> {
    Agent {
        descriptor: serde_json::from_slice(descriptor).unwrap(),
        store: RecordStore::with_loader(fresh_writer, move |reference| {
            Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
        }),
    }
}
fn main() {
    let recorded = serve(
        BatchLoop::new(
            config(&["A", "B", "D"])
                .with_recording()
                .with_record_store(RecordStore::new("initial-writer")),
        )
        .unwrap(),
    )
    .0;
    let mut backend = Backend::new();
    let original_descriptor = persist(recorded.agent().unwrap(), &mut backend);
    let original = restore(&original_descriptor, backend.clone(), "replay-writer");
    let original_assets: Vec<_> = original.assets().collect();
    assert_eq!(original_assets.len(), 2);
    assert_eq!(original_assets[0].value(), original_assets[1].value());
    assert_ne!(original_assets[0].messages(), original_assets[1].messages());
    let (replayed, llms) =
        serve(BatchLoop::new(config(&["C"]).with_agent(original.clone())).unwrap());
    assert_eq!(llms, 0);
    let before = replayed.into_agent().unwrap();
    let old_asset = before.assets().next().unwrap();
    assert_eq!(old_asset.env_ids(), ["A", "B", "C"]);
    println!("Agent prompt: {}", before.prompt());
    println!("Asset known environments: {:?}", old_asset.env_ids());
    println!("Asset conversation: {:?}", old_asset.messages());

    // No Asset ID or endpoint is supplied: replay the agent + environment again.
    let mut loop_ = BatchLoop::new(
        config(&["C"])
            .with_agent(before.clone())
            .with_feedback(NOTE.into()),
    )
    .unwrap();
    let read = loop_.next_request(RequestKind::Any).unwrap();
    assert!(matches!(read, Request::Tool(_)));
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: read.request_id().into(),
            results: vec![("C".into(), "shared".into())],
        }))
        .unwrap();
    let opened = loop_.agent().unwrap();
    let pending = opened.assets().next().unwrap();
    assert!(!pending.is_complete() && pending.value().unwrap() == "no issues");
    let Request::Llm(request) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(request.messages, pending.messages());
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id,
            content: "Check /foo".into(),
            tool_calls: vec![ToolCall::new("foo", "foo", json!({"path": "/foo"}))],
        }))
        .unwrap();
    let tool = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id().into(),
            results: vec![("C".into(), "safe".into())],
        }))
        .unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "no issues".into(),
        )))
        .unwrap();
    let revised = loop_.result().unwrap().into_agent().unwrap();
    let revised_asset = revised.assets().next().unwrap();
    assert!(revised_asset.is_complete());
    assert_eq!(revised_asset.value(), old_asset.value());
    assert!(revised_asset.messages().len() > old_asset.messages().len());
    assert!(old_asset.node().user_continuations.is_empty());
    let nodes = revised_asset.nodes();
    assert!(!nodes[nodes.len() - 3].user_continuations.is_empty()); // old endpoint is an ancestor
    assert!(!pending.is_complete());

    let descriptor = persist(&revised, &mut backend);
    let loaded = restore(&descriptor, backend.clone(), "inherited-writer");
    let (inherited, _) =
        serve(BatchLoop::new(config(&["A", "B", "C", "D"]).with_agent(loaded)).unwrap());
    let agent = inherited.agent().unwrap();
    let assets: Vec<_> = agent.assets().collect();
    assert_eq!(assets.len(), 3);
    assert_eq!(inherited.results["B"], "found");
    persist(agent, &mut backend);
    let old_again = restore(&original_descriptor, backend, "old-reader");
    let (result, llms) = serve(BatchLoop::new(config(&["B"]).with_agent(old_again)).unwrap());
    assert_eq!(llms, 0);
    assert_eq!(result.results["B"], "no issues");
    println!(
        "Updated assets: {:?}",
        assets
            .iter()
            .map(|a| (a.env_ids(), a.value()))
            .collect::<Vec<_>>()
    );
    println!("Old, open, and updated agent snapshots remain independent; no asset IDs required");
}
