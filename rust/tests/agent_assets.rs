use lockstep_loop::{
    Agent, AgentDescriptor, AgentOptions, AgentQuery, AgentResolver, BatchLoop, ChildAgentRun,
    ChildAgentUpdate, ChildHistoryRun, ChildHistoryUpdate, Completion, EnvObservation, History,
    HistoryDescriptor, HistoryNode, HistoryOptions, HistoryQuery, HistoryResolver, LlmCompletion,
    LoopConfig, Message, NodeRef, Observation, OutputRecord, RecordStore, Request, RequestKind,
    Role, SubagentHistory, SubagentRecord, SubagentTool, ToolCall, ToolCompletion, ToolSpec,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn spec() -> ToolSpec {
    ToolSpec::new("read", "read-only", json!({}))
}
fn config(envs: &[&str]) -> LoopConfig<String> {
    LoopConfig::new(
        "inspect".into(),
        envs.iter().map(|e| (*e).into()).collect(),
        vec![spec()],
    )
}
fn record() -> (Agent<String>, Agent<String>, Agent<String>) {
    let mut loop_ = BatchLoop::new(config(&["A", "B", "D"]).with_recording()).unwrap();
    let empty = loop_.agent().unwrap();
    let initial = empty.assets().next().unwrap();
    assert!(!initial.is_complete() && !initial.has_value());
    assert_eq!(initial.value(), None);
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
    let pending = loop_.agent().unwrap();
    let asset = pending.assets().next().unwrap();
    assert!(!asset.is_complete() && !asset.has_value());
    assert_eq!(asset.env_ids(), ["A", "B", "D"]);
    assert_eq!(
        asset.messages().last().unwrap().tool_calls[0].args,
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
    let opened = loop_.agent().unwrap();
    assert_eq!(
        opened.assets().map(|a| a.env_ids()).collect::<Vec<_>>(),
        [vec!["A", "B"], vec!["D"]]
    );
    assert!(opened.assets().all(|a| !a.is_complete() && !a.has_value()));
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
    assert_eq!(result.agent(), result.history.as_ref());
    assert_eq!(loop_.agent(), loop_.history());
    assert_eq!(result.child_agents(), result.child_histories);
    (result.history.unwrap(), empty, opened)
}

#[test]
fn assets_are_path_backed_leaves_not_values_or_environment_classes() {
    let (agent, empty, opened) = record();
    assert_eq!(agent.prompt(), "inspect");
    let assets: Vec<_> = agent.assets().collect();
    assert_eq!(assets.len(), 2);
    assert_eq!(assets[0].env_ids(), ["A", "B"]);
    assert_eq!(assets[1].env_ids(), ["D"]);
    assert!(
        assets
            .iter()
            .all(|a| a.is_complete() && a.value().unwrap() == "no issues")
    );
    assert_ne!(assets[0].messages(), assets[1].messages()); // values do not identify assets
    assert!(std::ptr::eq(assets[0].agent(), &agent));
    assert_eq!(
        assets[0]
            .messages()
            .iter()
            .map(|m| m.role)
            .collect::<Vec<_>>(),
        [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
    assert_eq!(assets[0].observations()[0].evidence[1].env_id, "B");
    assert!(std::ptr::eq(
        *assets[0].nodes().last().unwrap(),
        assets[0].node()
    ));
    assert!(!empty.assets().next().unwrap().has_value());
    assert!(opened.assets().all(|a| !a.is_complete()));
}

#[test]
fn feedback_moves_the_leaf_without_mutating_old_assets_or_overwriting_values() {
    let (old, _, _) = record();
    let old_asset = old.assets().next().unwrap();
    let mut loop_ = BatchLoop::new(
        config(&["A"])
            .with_agent(old.clone())
            .with_feedback("check /foo".into()),
    )
    .unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    assert!(matches!(request, Request::Tool(_))); // asset inspection is not replay resolution
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("A".into(), "shared".into())],
        }))
        .unwrap();
    let pending_agent = loop_.agent().unwrap();
    let pending = pending_agent.assets().next().unwrap();
    assert_eq!(pending.env_ids(), ["A", "B"]);
    assert!(!pending.is_complete() && pending.has_value());
    assert_eq!(pending.value().unwrap(), "no issues");
    assert_eq!(pending.messages().last().unwrap().content, "check /foo");
    assert!(pending_agent.assets().nth(1).unwrap().is_complete());
    assert!(old_asset.is_complete());
    assert!(old_asset.node().user_continuations.is_empty());
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
    let current = loop_.result().unwrap().history.unwrap();
    let revised = current.assets().next().unwrap();
    assert!(revised.is_complete());
    assert_eq!(revised.value(), old_asset.value());
    assert_eq!(revised.messages().len(), old_asset.messages().len() + 2);
    let nodes = revised.nodes();
    assert_eq!(nodes[nodes.len() - 2].assistant, old_asset.node().assistant);
    assert!(!nodes[nodes.len() - 2].user_continuations.is_empty());
    assert_eq!(current.assets().count(), 2);
    assert!(!pending_agent.assets().next().unwrap().is_complete());
    let mut replay = BatchLoop::new(config(&["C"]).with_agent(current)).unwrap();
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
fn asset_views_retain_lossy_evidence_and_real_null_publications() {
    let config = LoopConfig::new(
        Some("inspect".to_owned()),
        vec!["A".into(), "B".into()],
        vec![spec()],
    )
    .with_recording();
    let mut loop_ = BatchLoop::with_group_by(config, |p: &Option<String>| {
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
    let agent = loop_.agent().unwrap();
    let asset = agent.assets().next().unwrap();
    assert!(asset.is_complete() && asset.has_value());
    assert_eq!(asset.value(), Some(&None));
    assert_eq!(asset.published_message().unwrap().content, None);
    let messages = asset.messages();
    assert_eq!(
        messages[messages.len() - 2].content.as_deref(),
        Some("bytes ")
    );
    assert_eq!(
        asset.observations()[0].evidence[1].outputs[0]
            .tool_output
            .as_deref(),
        Some("bytes")
    );
    assert_eq!(asset.env_ids(), ["A", "B"]);
}

#[test]
fn asset_enumeration_is_lazy_read_only_and_does_not_clone_payloads() {
    let (agent, _, _) = record();
    let records = agent.store.records();
    let loads = Arc::new(Mutex::new(vec![]));
    let loaded = loads.clone();
    let restored = Agent {
        descriptor: agent.descriptor.clone(),
        store: RecordStore::with_loader("asset-reader", move |r| {
            loaded.lock().unwrap().push(r.clone());
            records[r].clone()
        }),
    };
    let mut views = restored.assets();
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
    assert_eq!(first.messages(), agent.assets().next().unwrap().messages());
    assert_eq!(views.count(), 1);
    assert!(restored.store.records().is_empty());

    struct NonClone(&'static str);
    let store = RecordStore::new("non-clone");
    let root = store.put(HistoryNode {
        assistant: Some(Message::assistant(NonClone("answer"), vec![])),
        env_ids: vec!["e".into()],
        ..HistoryNode::default()
    });
    let opaque = Agent {
        descriptor: AgentDescriptor {
            prompt: NonClone("prompt"),
            tools: vec![],
            custom_grouping: false,
            root,
            subagents: vec![],
        },
        store,
    };
    let asset = opaque.assets().next().unwrap();
    assert_eq!(asset.value().unwrap().0, "answer"); // traversal/value access do not require P: Clone
    assert_eq!(asset.agent().prompt().0, "prompt");
}

#[test]
fn public_aliases_preserve_types_and_serde_records() {
    use std::any::type_name;
    assert_eq!(type_name::<Agent<String>>(), type_name::<History<String>>());
    assert_eq!(
        type_name::<AgentDescriptor<String>>(),
        type_name::<HistoryDescriptor<String>>()
    );
    assert_eq!(
        type_name::<AgentQuery<String>>(),
        type_name::<HistoryQuery<String>>()
    );
    assert_eq!(
        type_name::<AgentResolver<String>>(),
        type_name::<HistoryResolver<String>>()
    );
    assert_eq!(
        type_name::<AgentOptions<String>>(),
        type_name::<HistoryOptions<String>>()
    );
    assert_eq!(
        type_name::<SubagentRecord<String>>(),
        type_name::<SubagentHistory<String>>()
    );
    assert_eq!(
        type_name::<ChildAgentRun<String>>(),
        type_name::<ChildHistoryRun<String>>()
    );
    assert_eq!(
        type_name::<ChildAgentUpdate<String>>(),
        type_name::<ChildHistoryUpdate<String>>()
    );
    let (agent, _, _) = record();
    let serialized = serde_json::to_vec(&agent.descriptor).unwrap();
    let old: HistoryDescriptor<String> = serde_json::from_slice(&serialized).unwrap();
    let new: AgentDescriptor<String> = serde_json::from_slice(&serialized).unwrap();
    assert_eq!(old, new);
    assert_eq!(serde_json::to_vec(&new).unwrap(), serialized);
    for node in agent.store.records().values() {
        let restored: HistoryNode<String> =
            serde_json::from_slice(&serde_json::to_vec(node).unwrap()).unwrap();
        assert_eq!(&restored, node.as_ref());
    }
}

#[derive(Deserialize)]
struct Fixture {
    descriptor: AgentDescriptor<String>,
    query: AgentQuery<String>,
    records: Vec<(NodeRef, HistoryNode<String>)>,
}
#[test]
fn pre_rename_fixture_loads_inspects_replays_and_omits_legacy_labels() {
    let bytes = include_bytes!("fixtures/agent-naming-2c6d901.json");
    let mut raw: Value = serde_json::from_slice(bytes).unwrap();
    fn strip_labels(value: &mut Value) {
        match value {
            Value::Object(fields) => {
                fields.remove("contract");
                fields.remove("history_contract");
                fields.values_mut().for_each(strip_labels);
            }
            Value::Array(values) => values.iter_mut().for_each(strip_labels),
            _ => {}
        }
    }
    strip_labels(&mut raw);
    let fixture: Fixture = serde_json::from_slice(bytes).unwrap();
    assert_eq!(
        serde_json::to_value(&fixture.descriptor).unwrap(),
        raw["descriptor"]
    );
    assert_eq!(serde_json::to_value(&fixture.query).unwrap(), raw["query"]);
    assert_eq!(
        serde_json::to_value(&fixture.records).unwrap(),
        raw["records"]
    );
    let records: BTreeMap<_, _> = fixture
        .records
        .into_iter()
        .map(|(r, n)| (r, Arc::new(n)))
        .collect();
    let loaded = Arc::new(Mutex::new(vec![]));
    let loads = loaded.clone();
    let agent = Agent {
        descriptor: fixture.descriptor,
        store: RecordStore::with_loader("fixture-reader", move |r| {
            loads.lock().unwrap().push(r.clone());
            records[r].clone()
        }),
    };
    let asset = agent.assets().next().unwrap();
    assert_eq!(asset.value().unwrap(), "same answer");
    assert_eq!(asset.env_ids(), ["A", "B"]);
    assert_eq!(
        asset.messages().iter().map(|m| m.role).collect::<Vec<_>>(),
        [
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Tool,
            Role::Assistant,
            Role::User,
            Role::Assistant
        ]
    );
    assert_eq!(loaded.lock().unwrap().len(), 3); // not the child agents' roots
    assert_eq!(
        asset.observations()[0].evidence[1].outputs[0].tool_output,
        "same"
    );
    let root = agent.root();
    let entry = &root.subagents[0];
    assert!(std::ptr::eq(entry.agent_descriptor(), &entry.history));
    assert!(std::ptr::eq(
        entry.runs[0].agent_descriptor(),
        &entry.runs[0].history
    ));
    let child = agent.child(entry);
    assert_eq!(
        child.assets().next().unwrap().value().unwrap(),
        "child answer"
    );
    let sub = SubagentTool::new(
        ToolSpec::new("delegate", "child investigation", json!({})),
        |_| "child".to_owned(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let config = LoopConfig::new(
        agent.prompt().clone(),
        vec!["C".into()],
        agent.descriptor.tools.clone(),
    )
    .with_agent(agent.clone())
    .with_subagents(vec![sub.clone()]);
    let mut replay = BatchLoop::with_group_by(config, |s: &String| s.trim().to_owned()).unwrap();
    let request = replay.next_request(RequestKind::Any).unwrap();
    assert!(matches!(request, Request::Tool(_)));
    replay
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("C".into(), "same ".into())],
        }))
        .unwrap();
    assert!(replay.is_done());
    let result = replay.result().unwrap();
    assert_eq!(result.results["C"], "same answer");
    assert_eq!(result.child_agents().len(), 1);
    assert!(std::ptr::eq(
        result.child_agents()[0].agent(),
        &result.child_histories[0].history
    ));
    assert_eq!(
        serde_json::to_value(&agent.descriptor).unwrap(),
        raw["descriptor"]
    );
    let encoded_record = serde_json::to_value(agent.root()).unwrap();
    assert!(encoded_record["subagents"][0].get("history").is_some());
    assert!(
        encoded_record["subagents"][0]
            .get("agent_descriptor")
            .is_none()
    );
    let selected = entry.agent_descriptor().clone();
    let config = LoopConfig::new("new parent".to_owned(), vec!["new".into()], vec![])
        .with_subagents(vec![sub])
        .with_record_store(agent.store.clone())
        .with_agent_resolver(move |_| Ok(Some(selected.clone())));
    let mut new_parent =
        BatchLoop::with_group_by(config, |s: &String| s.trim().to_owned()).unwrap();
    let request = new_parent.next_request(RequestKind::Any).unwrap();
    new_parent
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "delegate".into(),
            tool_calls: vec![ToolCall::new("new", "delegate", json!({"q": "child"}))],
        }))
        .unwrap();
    let Request::Llm(request) = new_parent.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(request.messages[0].content, "new parent");
    assert_eq!(request.messages.last().unwrap().content, "child answer");
    new_parent
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            "parent answer".into(),
        )))
        .unwrap();
    assert!(new_parent.is_done());
    assert_eq!(new_parent.result().unwrap().child_agents().len(), 1);
}

#[test]
fn deep_asset_paths_are_iterative_and_inspection_never_writes_records() {
    let store = RecordStore::new("deep-assets");
    let mut root = store.put(HistoryNode {
        assistant: Some(Message::assistant("done".to_owned(), vec![])),
        env_ids: vec!["A".into()],
        ..HistoryNode::default()
    });
    for i in (0..1500).rev() {
        let outputs = vec![OutputRecord {
            tool_output: i.to_string(),
        }];
        root = store.put(HistoryNode {
            assistant: Some(Message::assistant(
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
    let agent = Agent {
        descriptor: AgentDescriptor {
            prompt: "inspect".into(),
            tools: vec![spec()],
            custom_grouping: false,
            root,
            subagents: vec![],
        },
        store,
    };
    let before = agent.store.records().len();
    let asset = agent.assets().next().unwrap();
    assert_eq!(asset.observations().len(), 1500);
    assert_eq!(asset.messages().len(), 3002);
    assert_eq!(asset.value().unwrap(), "done");
    assert_eq!(asset.env_ids(), ["A"]);
    assert_eq!(agent.store.records().len(), before);
}
