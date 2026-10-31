use lockstep_loop::{
    AgentDescriptor, AgentSnapshot, BatchLoop, Completion, EnvObservation, HistoryNode,
    LlmCompletion, LoopConfig, MessageEnvelope, Observation, OutputRecord, RecordStore, Request,
    RequestKind, Role, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::sync::{Arc, Mutex};

fn spec() -> ToolSpec {
    ToolSpec::new("read", "read-only", json!({}))
}
fn config(envs: &[&str]) -> LoopConfig<String> {
    LoopConfig::new(
        vec![lockstep_loop::MessageEnvelope::user("inspect".into())],
        envs.iter().map(|e| (*e).into()).collect(),
        vec![spec()],
    )
}
fn record() -> (
    AgentSnapshot<String>,
    AgentSnapshot<String>,
    AgentSnapshot<String>,
) {
    let mut loop_ = BatchLoop::new(config(&["A", "B", "D"])).unwrap();
    let empty = loop_.snapshot();
    let initial = empty.conversations().next().unwrap();
    assert!(!initial.is_complete() && !initial.has_answer());
    assert_eq!(initial.get_last_answer_asset(), None);
    assert!(initial.env_ids().is_empty());
    assert_eq!(initial.messages()[0].content, "inspect");
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "read".into(),
            tool_calls: vec![ToolCall::new("r", "read", json!({"path": "/entry"}))],
        }))
        .unwrap();
    let pending = loop_.snapshot();
    let conversation = pending.conversations().next().unwrap();
    assert!(!conversation.is_complete() && !conversation.has_answer());
    assert_eq!(conversation.env_ids(), ["A", "B", "D"]);
    assert_eq!(
        conversation.messages().last().unwrap().tool_calls[0].args,
        json!({"path": "/entry"})
    );
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![
                ("D".into(), "different".into()),
                ("B".into(), "shared".into()),
                ("A".into(), "shared".into()),
            ],
        }))
        .unwrap();
    let opened = loop_.snapshot();
    assert_eq!(
        opened
            .conversations()
            .map(|a| a.env_ids())
            .collect::<Vec<_>>(),
        [vec!["A", "B"], vec!["D"]]
    );
    assert!(
        opened
            .conversations()
            .all(|a| !a.is_complete() && !a.has_answer())
    );
    while !loop_.is_done() {
        let request = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id(),
                "no issues".into(),
            )))
            .unwrap();
    }
    let result = loop_.result().unwrap();
    assert_eq!(result.snapshot(), &result.snapshot);
    assert_eq!(loop_.snapshot(), result.snapshot);
    assert!(result.child_updates.is_empty());
    (result.snapshot, empty, opened)
}

#[test]
fn conversations_are_path_backed_leaves_not_asset_values_or_environment_classes() {
    let (agent, empty, opened) = record();
    assert_eq!(agent.initial_messages()[0].content, "inspect");
    let conversations: Vec<_> = agent.conversations().collect();
    assert_eq!(conversations.len(), 2);
    assert_eq!(conversations[0].env_ids(), ["A", "B"]);
    assert_eq!(conversations[1].env_ids(), ["D"]);
    assert!(
        conversations
            .iter()
            .all(|a| a.is_complete() && a.get_last_answer_asset().unwrap() == "no issues")
    );
    assert_ne!(conversations[0].messages(), conversations[1].messages()); // values do not identify conversations
    assert!(std::ptr::eq(conversations[0].snapshot(), &agent));
    assert_eq!(
        conversations[0]
            .messages()
            .iter()
            .map(|m| m.role)
            .collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
    assert_eq!(conversations[0].observations()[0].evidence[1].env_id, "B");
    assert!(std::ptr::eq(
        *conversations[0].nodes().last().unwrap(),
        conversations[0].node()
    ));
    assert!(!empty.conversations().next().unwrap().has_answer());
    assert!(opened.conversations().all(|a| !a.is_complete()));
}

#[test]
fn feedback_moves_the_leaf_without_mutating_old_conversations_or_overwriting_values() {
    let (old, _, _) = record();
    let old_conversation = old.conversations().next().unwrap();
    let mut loop_ = BatchLoop::new(
        config(&["A"])
            .with_snapshot(old.clone())
            .with_feedback("check /foo".into()),
    )
    .unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    assert!(matches!(request, Request::Tool(_))); // conversation inspection is not replay resolution
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("A".into(), "shared".into())],
        }))
        .unwrap();
    let pending_agent = loop_.snapshot();
    let pending = pending_agent.conversations().next().unwrap();
    assert_eq!(pending.env_ids(), ["A", "B"]);
    assert!(!pending.is_complete() && pending.has_answer());
    assert_eq!(pending.get_last_answer_asset().unwrap(), "no issues");
    assert_eq!(pending.messages().last().unwrap().content, "check /foo");
    assert!(pending_agent.conversations().nth(1).unwrap().is_complete());
    assert!(old_conversation.is_complete());
    assert!(old_conversation.node().user_continuations.is_empty());
    let Request::Llm(request) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(request.messages, pending.messages());
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            "no issues".into(),
        )))
        .unwrap();
    let current = loop_.result().unwrap().snapshot;
    let revised = current.conversations().next().unwrap();
    assert!(revised.is_complete());
    assert_eq!(
        revised.get_last_answer_asset(),
        old_conversation.get_last_answer_asset()
    );
    assert_eq!(
        revised.messages().len(),
        old_conversation.messages().len() + 2
    );
    let nodes = revised.nodes();
    assert_eq!(
        nodes[nodes.len() - 2].assistant,
        old_conversation.node().assistant
    );
    assert!(!nodes[nodes.len() - 2].user_continuations.is_empty());
    assert_eq!(current.conversations().count(), 2);
    assert!(!pending_agent.conversations().next().unwrap().is_complete());
    let mut replay = BatchLoop::new(config(&["C"]).with_snapshot(current)).unwrap();
    let request = replay.next_request(RequestKind::Any).unwrap();
    replay
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("C".into(), "shared".into())],
        }))
        .unwrap();
    assert!(replay.is_done());
    assert_eq!(replay.result().unwrap().results["C"], "no issues");
}

#[test]
fn conversation_views_retain_lossy_evidence_and_real_null_publications() {
    let config = LoopConfig::new(
        vec![lockstep_loop::MessageEnvelope::user(Some(
            "inspect".to_owned(),
        ))],
        vec!["A".into(), "B".into()],
        vec![spec()],
    );
    let mut loop_ = BatchLoop::with_comparison_key(config, |p: &Option<String>| {
        p.as_deref().unwrap_or_default().trim().to_owned()
    })
    .unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: Some("read".into()),
            tool_calls: vec![ToolCall::new("r", "read", json!({}))],
        }))
        .unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id().into(),
            results: vec![
                ("A".into(), Some("bytes ".into())),
                ("B".into(), Some("bytes".into())),
            ],
        }))
        .unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id(),
            None,
        )))
        .unwrap();
    let agent = loop_.snapshot();
    let conversation = agent.conversations().next().unwrap();
    assert!(conversation.is_complete() && conversation.has_answer());
    assert_eq!(conversation.get_last_answer_asset(), Some(&None));
    assert_eq!(conversation.last_answer().unwrap().content, None);
    let messages = conversation.messages();
    assert_eq!(
        messages[messages.len() - 2].content.as_deref(),
        Some("bytes ")
    );
    assert_eq!(
        conversation.observations()[0].evidence[1].outputs[0]
            .tool_output
            .as_deref(),
        Some("bytes")
    );
    assert_eq!(conversation.env_ids(), ["A", "B"]);
}

#[test]
fn conversation_enumeration_is_lazy_read_only_and_does_not_clone_payloads() {
    let (agent, _, _) = record();
    let records = agent.store.records();
    let loads = Arc::new(Mutex::new(vec![]));
    let loaded = loads.clone();
    let restored = AgentSnapshot {
        descriptor: agent.descriptor.clone(),
        store: RecordStore::with_loader("conversation-reader", move |r| {
            loaded.lock().unwrap().push(r.clone());
            records[r].clone()
        }),
    };
    let mut views = restored.conversations();
    assert!(loads.lock().unwrap().is_empty());
    let first = views.next().unwrap();
    assert_eq!(first.env_ids(), ["A", "B"]);
    assert_eq!(loads.lock().unwrap().len(), 2);
    assert!(
        !loads
            .lock()
            .unwrap()
            .contains(&agent.root().observations[1].continuation)
    );
    assert_eq!(
        first.messages(),
        agent.conversations().next().unwrap().messages()
    );
    assert_eq!(views.count(), 1);
    assert!(restored.store.records().is_empty());

    struct NonClone(&'static str);
    let store = RecordStore::new("non-clone");
    let root = store.put(HistoryNode {
        assistant: Some(MessageEnvelope::assistant(NonClone("answer"), vec![])),
        env_ids: vec!["e".into()],
        ..HistoryNode::default()
    });
    let opaque = AgentSnapshot {
        descriptor: AgentDescriptor {
            initial_messages: vec![MessageEnvelope::user(NonClone("prompt"))],
            tools: vec![],
            root,
            subagents: vec![],
        },
        store,
    };
    let conversation = opaque.conversations().next().unwrap();
    assert_eq!(conversation.get_last_answer_asset().unwrap().0, "answer"); // traversal/value access do not require P: Clone
    assert_eq!(
        conversation.snapshot().initial_messages()[0].content.0,
        "prompt"
    );
}

#[test]
fn deep_conversation_paths_are_iterative_and_inspection_never_writes_records() {
    let store = RecordStore::new("deep-conversations");
    let mut root = store.put(HistoryNode {
        assistant: Some(MessageEnvelope::assistant("done".to_owned(), vec![])),
        env_ids: vec!["A".into()],
        ..HistoryNode::default()
    });
    for i in (0..1500).rev() {
        let outputs = vec![OutputRecord {
            tool_output: i.to_string(),
        }];
        root = store.put(HistoryNode {
            assistant: Some(MessageEnvelope::assistant(
                "read".into(),
                vec![ToolCall::new("r", "read", json!({"step": i}))],
            )),
            env_ids: vec!["A".into()],
            observations: vec![Observation {
                representative: outputs.clone(),
                evidence: vec![EnvObservation {
                    env_id: "A".into(),
                    outputs,
                }],
                continuation: root,
            }],
            ..HistoryNode::default()
        });
    }
    let agent = AgentSnapshot {
        descriptor: AgentDescriptor {
            initial_messages: vec![MessageEnvelope::user("inspect".into())],
            tools: vec![spec()],
            root,
            subagents: vec![],
        },
        store,
    };
    let before = agent.store.records().len();
    let conversation = agent.conversations().next().unwrap();
    assert_eq!(conversation.observations().len(), 1500);
    assert_eq!(conversation.messages().len(), 3002);
    assert_eq!(conversation.get_last_answer_asset().unwrap(), "done");
    assert_eq!(conversation.env_ids(), ["A"]);
    assert_eq!(agent.store.records().len(), before);
}
