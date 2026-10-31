//! cargo run -p lockstep-loop --example agent_conversations
//! An AgentSnapshot contains initial messages, structural configuration, and one history trie. Conversations are contextual leaf views.
use lockstep_loop::{
    Agent, AgentConfig, AgentRun, AgentRuntime, AgentSnapshot, BatchResult, Completion,
    LlmCompletion, NodeRef, RecordStore, Request, RequestKind, Role, RunOptions, ToolCall,
    ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

const PROMPT: &str = "Inspect the repository";
const NOTE: &str = "Also inspect /foo";
fn config() -> AgentConfig<String> {
    AgentConfig::new(
        vec![lockstep_loop::MessageEnvelope::user(PROMPT.into())],
        vec![
            ToolSpec::new("read", "Read entry", json!({})),
            ToolSpec::new("foo", "Inspect /foo", json!({})),
        ],
    )
}
fn serve(mut loop_: AgentRun<String>) -> (BatchResult<String>, usize) {
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
fn persist(agent: &AgentSnapshot<String>, backend: &mut Backend) -> Vec<u8> {
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
fn restore(descriptor: &[u8], backend: Backend, fresh_writer: &str) -> AgentSnapshot<String> {
    AgentSnapshot {
        descriptor: serde_json::from_slice(descriptor).unwrap(),
        store: RecordStore::with_loader(fresh_writer, move |reference| {
            Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
        }),
    }
}
fn main() {
    let agent = Agent::new(config().with_record_store(RecordStore::new("initial-writer"))).unwrap();
    let recorded = serve(
        agent
            .start(
                vec!["A".into(), "B".into(), "D".into()],
                RunOptions::default(),
            )
            .unwrap(),
    )
    .0;
    let mut backend = Backend::new();
    let original_descriptor = persist(recorded.snapshot(), &mut backend);
    let original = restore(&original_descriptor, backend.clone(), "replay-writer");
    let original_conversations: Vec<_> = original.conversations().collect();
    assert_eq!(original_conversations.len(), 2);
    assert_eq!(
        original_conversations[0].get_last_answer_asset(),
        original_conversations[1].get_last_answer_asset()
    );
    assert_ne!(
        original_conversations[0].messages(),
        original_conversations[1].messages()
    );
    let agent = Agent::from_snapshot(original.clone(), AgentRuntime::default()).unwrap();
    let (replayed, llms) = serve(
        agent
            .start(vec!["C".into()], RunOptions::default())
            .unwrap(),
    );
    assert_eq!(llms, 0);
    let before = replayed.into_snapshot();
    let old_conversation = before.conversations().next().unwrap();
    assert_eq!(old_conversation.env_ids(), ["A", "B", "C"]);
    println!(
        "AgentSnapshot first initial message: {}",
        before.initial_messages()[0].content
    );
    println!(
        "Conversation known environments: {:?}",
        old_conversation.env_ids()
    );
    println!("Conversation messages: {:?}", old_conversation.messages());

    // No Conversation ID or endpoint is supplied: replay the agent + environment again.
    let mut loop_ = agent
        .start(
            vec!["C".into()],
            RunOptions {
                feedback: Some(NOTE.into()),
                budget: None,
            },
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
    let opened = loop_.snapshot();
    let pending = opened.conversations().next().unwrap();
    assert!(!pending.is_complete() && pending.get_last_answer_asset().unwrap() == "no issues");
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
    let revised = loop_.result().unwrap().into_snapshot();
    let revised_conversation = revised.conversations().next().unwrap();
    assert!(revised_conversation.is_complete());
    assert_eq!(
        revised_conversation.get_last_answer_asset(),
        old_conversation.get_last_answer_asset()
    );
    assert!(revised_conversation.messages().len() > old_conversation.messages().len());
    assert!(old_conversation.node().user_continuations.is_empty());
    let nodes = revised_conversation.nodes();
    assert!(!nodes[nodes.len() - 3].user_continuations.is_empty()); // old endpoint is an ancestor
    assert!(!pending.is_complete());

    let descriptor = persist(&revised, &mut backend);
    let loaded = restore(&descriptor, backend.clone(), "inherited-writer");
    let loaded = Agent::from_snapshot(loaded, AgentRuntime::default()).unwrap();
    let (inherited, _) = serve(
        loaded
            .start(
                vec!["A".into(), "B".into(), "C".into(), "D".into()],
                RunOptions::default(),
            )
            .unwrap(),
    );
    let agent = inherited.snapshot();
    let conversations: Vec<_> = agent.conversations().collect();
    assert_eq!(conversations.len(), 3);
    assert_eq!(inherited.results["B"], "found");
    persist(agent, &mut backend);
    let old_again = restore(&original_descriptor, backend, "old-reader");
    let old_again = Agent::from_snapshot(old_again, AgentRuntime::default()).unwrap();
    let (result, llms) = serve(
        old_again
            .start(vec!["B".into()], RunOptions::default())
            .unwrap(),
    );
    assert_eq!(llms, 0);
    assert_eq!(result.results["B"], "no issues");
    println!(
        "Updated conversations: {:?}",
        conversations
            .iter()
            .map(|a| (a.env_ids(), a.get_last_answer_asset()))
            .collect::<Vec<_>>()
    );
    println!(
        "Old, open, and updated agent snapshots remain independent; no conversation IDs required"
    );
}
