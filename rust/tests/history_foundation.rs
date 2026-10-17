use lockstep_loop::{
    BatchLoop, BatchResult, Completion, History, HistoryDescriptor, HistoryNode, LlmCompletion,
    LoopConfig, LoopError, Message, NodeRef, Observation, OutputRecord, RecordStore, Request,
    RequestKind, SubagentTool, ToolCall, ToolCompletion, ToolSpec, UserContinuation,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn specs() -> Vec<ToolSpec> {
    ["probe", "status", "investigate"]
        .iter()
        .map(|n| ToolSpec::new(*n, "", json!({})))
        .collect()
}
fn nested_config(
    envs: &[&str],
    previous: Option<History<String>>,
    depth: usize,
) -> LoopConfig<String> {
    let config = LoopConfig::new(
        "main".into(),
        envs.iter().map(|s| (*s).into()).collect(),
        specs(),
    )
    .with_subagent(SubagentTool::new(
        specs()[2].clone(),
        |a| a["question"].as_str().unwrap().into(),
        |_| specs(),
        depth,
    ));
    match previous {
        Some(h) => config.with_replay(h, "nested/v1"),
        None => config.with_recording("nested/v1"),
    }
}
struct Driven {
    result: BatchResult<String>,
    llms: Vec<String>,
    early: History<String>,
    tools: usize,
}
fn nested_drive(mut loop_: BatchLoop<String>, values: &[(&str, &str)]) -> Driven {
    let values: BTreeMap<_, _> = values.iter().copied().collect();
    let mut llms = Vec::new();
    let mut early = None;
    let mut tools = 0;
    while !loop_.is_done() {
        let mut pending = Vec::new();
        while let Some(r) = loop_.next_request(RequestKind::Any) {
            pending.push(r);
        }
        assert!(!pending.is_empty());
        for request in pending.into_iter().rev() {
            match request {
                Request::Llm(r) => {
                    let scope = r.messages[0].content.as_str();
                    llms.push(scope.to_owned());
                    let calls = if r.messages.len() == 1 {
                        match scope {
                            "main" => vec![
                                ToolCall::new("local", "status", json!({})),
                                ToolCall::new("child", "investigate", json!({"question": "child"})),
                                ToolCall::new(
                                    "sibling",
                                    "investigate",
                                    json!({"question": "sibling"}),
                                ),
                            ],
                            "child" => vec![
                                ToolCall::new(
                                    "nested",
                                    "investigate",
                                    json!({"question": "grand"}),
                                ),
                                ToolCall::new("probe", "probe", json!({"scope": scope})),
                            ],
                            _ => vec![ToolCall::new("probe", "probe", json!({"scope": scope}))],
                        }
                    } else {
                        vec![]
                    };
                    let content = if calls.is_empty() {
                        "same answer"
                    } else {
                        "thinking"
                    };
                    loop_
                        .complete(Completion::Llm(LlmCompletion {
                            request_id: r.request_id,
                            content: content.into(),
                            tool_calls: calls,
                        }))
                        .unwrap();
                }
                Request::Tool(r) => {
                    tools += 1;
                    assert_ne!(r.tool_call.name, "investigate");
                    let results = r
                        .env_ids
                        .iter()
                        .rev()
                        .map(|env| {
                            (
                                env.clone(),
                                if r.tool_call.name == "status" {
                                    "ok".into()
                                } else {
                                    format!(
                                        "{}:{}",
                                        values[env.as_str()],
                                        r.tool_call.args["scope"].as_str().unwrap()
                                    )
                                },
                            )
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
            if early.is_none() {
                early = loop_.history();
            }
        }
    }
    Driven {
        result: loop_.result().unwrap(),
        llms,
        early: early.unwrap(),
        tools,
    }
}
fn child(history: &History<String>, call: &str) -> History<String> {
    let root = history.root();
    let entry = root.subagents.iter().find(|s| s.call_id == call).unwrap();
    History {
        descriptor: entry.history.clone(),
        store: history.store.clone(),
    }
}

type Backend = BTreeMap<NodeRef, Vec<u8>>;
fn save(history: &History<String>, backend: &mut Backend) -> Vec<u8> {
    // Insert-only persistence: never silently overwrite a different record.
    for (reference, node) in history.store.records() {
        let bytes = serde_json::to_vec(&node).unwrap();
        if let Some(old) = backend.get(&reference) {
            assert_eq!(old, &bytes);
        } else {
            backend.insert(reference, bytes);
        }
    }
    serde_json::to_vec(&history.descriptor).unwrap()
}
fn restore(
    descriptor: &[u8],
    backend: Backend,
    namespace: &str,
    loads: Arc<Mutex<Vec<NodeRef>>>,
) -> History<String> {
    History {
        descriptor: serde_json::from_slice::<HistoryDescriptor<String>>(descriptor).unwrap(),
        store: RecordStore::with_loader(namespace, move |reference| {
            loads.lock().unwrap().push(reference.clone());
            Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
        }),
    }
}

#[test]
fn nested_histories_mixed_calls_depth_equal_parent_answers_and_snapshots() {
    let driven = nested_drive(
        BatchLoop::new(nested_config(&["a", "b"], None, 2)).unwrap(),
        &[("a", "A"), ("b", "B")],
    );
    assert_eq!(driven.result.classes.len(), 1);
    assert_eq!(driven.result.classes[0].env_ids, ["a", "b"]);
    assert_eq!(driven.llms.iter().filter(|s| *s == "main").count(), 2);
    let old = driven.result.history.unwrap();
    let grand = child(&child(&old, "child"), "nested");
    assert!(grand.descriptor.subagent.is_none());
    assert!(
        grand
            .descriptor
            .tools
            .iter()
            .all(|spec| spec.name != "investigate")
    );
    assert_eq!(grand.root().observations.len(), 2);
    let outputs: Vec<_> = grand
        .root()
        .observations
        .iter()
        .map(|e| e.evidence[0].outputs[0].tool_output.clone())
        .collect();
    assert_eq!(outputs, ["A:grand", "B:grand"]);
    assert_eq!(old.root().observations.len(), 1);
    assert_eq!(
        old.root().observations[0]
            .representative
            .iter()
            .map(|o| o.tool_output.as_str())
            .collect::<Vec<_>>(),
        ["ok", "same answer", "same answer"]
    );
    assert!(!driven.early.root().subagents.is_empty());
    assert!(
        driven.early.root().subagents.iter().all(|s| driven
            .early
            .load(&s.history.root)
            .assistant
            .is_none())
    );

    let mut backend = Backend::new();
    let descriptor = save(&old, &mut backend);
    let restored = restore(&descriptor, backend, "replay-writer", Arc::default());
    let replay = nested_drive(
        BatchLoop::new(nested_config(&["next"], Some(restored), 2)).unwrap(),
        &[("next", "A")],
    );
    assert!(replay.llms.is_empty());
    assert_eq!(replay.tools, 4);
    assert_eq!(replay.result.results["next"], "same answer");

    let changed = nested_drive(
        BatchLoop::new(nested_config(&["same", "new"], Some(old.clone()), 2)).unwrap(),
        &[("same", "A"), ("new", "C")],
    );
    assert!(!changed.llms.is_empty());
    assert!(changed.llms.iter().all(|scope| scope != "main"));
    assert_eq!(changed.result.classes.len(), 1);
    let extended = changed.result.history.unwrap();
    assert_eq!(
        child(&child(&extended, "child"), "nested")
            .root()
            .observations
            .len(),
        3
    );
    assert_eq!(grand.root().observations.len(), 2);
    assert_eq!(old.root().observations[0].evidence.len(), 2);
    for value in ["A", "B", "C"] {
        let replay = nested_drive(
            BatchLoop::new(nested_config(&["again"], Some(extended.clone()), 2)).unwrap(),
            &[("again", value)],
        );
        assert!(replay.llms.is_empty());
    }
    assert!(matches!(
        BatchLoop::new(nested_config(&["bad"], Some(old), 1)),
        Err(LoopError::IncompatibleHistory)
    ));
}

#[test]
fn repeated_subagent_call_ids_cannot_accept_stale_completions() {
    let mut loop_ = BatchLoop::new(nested_config(&["a"], None, 2)).unwrap();
    let mut old_child_id = None;
    for _ in 0..2 {
        let request = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion {
                request_id: request.request_id().into(),
                content: "delegate".into(),
                tool_calls: vec![ToolCall::new(
                    "same",
                    "investigate",
                    json!({"question": "child"}),
                )],
            }))
            .unwrap();
        let child_request = loop_.next_request(RequestKind::Any).unwrap();
        if let Some(id) = old_child_id {
            assert_ne!(child_request.request_id(), id);
            assert!(matches!(
                loop_.complete(Completion::Llm(LlmCompletion::final_answer(
                    id,
                    "stale".into()
                ))),
                Err(LoopError::UnknownRequest(_))
            ));
            assert_eq!(loop_.outstanding_llm(), 1);
        }
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                child_request.request_id(),
                "same answer".into(),
            )))
            .unwrap();
        old_child_id = Some(child_request.request_id().to_owned());
    }
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "done".into(),
        )))
        .unwrap();
    let old = loop_.history().unwrap();
    assert!(loop_.is_done());
    let mut replay = BatchLoop::new(nested_config(&["next"], Some(old.clone()), 2)).unwrap();
    assert!(replay.is_done());
    assert!(replay.next_request(RequestKind::Any).is_none());
    assert_eq!(replay.result().unwrap().results["next"], "done");
    assert_eq!(child(&old, "same").root().env_ids, ["a"]);
}

#[test]
fn invalid_nested_completion_preserves_routes_and_snapshot() {
    let mut loop_ = BatchLoop::new(nested_config(&["a", "b"], None, 2)).unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: "delegate".into(),
            tool_calls: vec![ToolCall::new(
                "child",
                "investigate",
                json!({"question": "child"}),
            )],
        }))
        .unwrap();
    let child_request = loop_.next_request(RequestKind::Any).unwrap();
    let before = loop_.history().unwrap();
    let duplicate = ToolCall::new("p", "probe", json!({}));
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: child_request.request_id().into(),
            content: "bad".into(),
            tool_calls: vec![duplicate.clone(), duplicate.clone()],
        })),
        Err(LoopError::DuplicateToolCallId(_))
    ));
    assert_eq!(loop_.history().unwrap(), before);
    assert_eq!(loop_.outstanding_llm(), 1);
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: child_request.request_id().into(),
            content: "probe".into(),
            tool_calls: vec![duplicate],
        }))
        .unwrap();
    let tool = loop_.next_request(RequestKind::Any).unwrap();
    let before = loop_.history().unwrap();
    assert!(matches!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id().into(),
            results: vec![("a".into(), "A".into())],
        })),
        Err(LoopError::InvalidToolCompletion(_))
    ));
    assert_eq!(loop_.outstanding_tool(), 1);
    assert_eq!(loop_.history().unwrap(), before);
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id().into(),
            results: vec![("a".into(), "A".into()), ("b".into(), "A".into())],
        }))
        .unwrap();
    for content in ["same", "done"] {
        let request = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id(),
                content.into(),
            )))
            .unwrap();
    }
    assert!(loop_.is_done());
    assert!(child(&before, "child").root().observations.is_empty());
    assert_eq!(
        child(&loop_.history().unwrap(), "child")
            .root()
            .observations[0]
            .evidence[1]
            .env_id,
        "b"
    );
}

fn simple(envs: &[&str], previous: Option<History<String>>) -> BatchLoop<String> {
    let config = LoopConfig::new(
        "inspect".into(),
        envs.iter().map(|s| (*s).into()).collect(),
        vec![specs()[0].clone()],
    );
    BatchLoop::new(match previous {
        Some(h) => config.with_replay(h, "simple/v1"),
        None => config
            .with_recording("simple/v1")
            .with_record_store(RecordStore::new("initial-writer")),
    })
    .unwrap()
}
fn sequential<K: Eq + std::hash::Hash + std::fmt::Debug + 'static>(
    loop_: &mut BatchLoop<String, K>,
    values: &[(&str, &str)],
    turns: usize,
) -> usize {
    let values: BTreeMap<_, _> = values.iter().copied().collect();
    let mut llms = 0;
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                llms += 1;
                let calls = if r.stats.depth < turns {
                    vec![ToolCall::new("p", "probe", json!({"turn": r.stats.depth}))]
                } else {
                    vec![]
                };
                loop_
                    .complete(Completion::Llm(LlmCompletion {
                        request_id: r.request_id,
                        content: "same final".into(),
                        tool_calls: calls,
                    }))
                    .unwrap();
            }
            Request::Tool(r) => {
                let results = r
                    .env_ids
                    .iter()
                    .map(|env| (env.clone(), values[env.as_str()].to_owned()))
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
    llms
}

#[test]
fn deep_individual_records_roundtrip_and_lazy_traversal_then_publish_new_path() {
    let mut original = simple(&["left", "right"], None);
    sequential(&mut original, &[("left", "L"), ("right", "R")], 256);
    let old = original.history().unwrap();
    let mut backend = Backend::new();
    // Each record AND envelope deserialize using ordinary serde_json defaults.
    // No recursion-limit increase, recursive export tree, or caller replay logic.
    let descriptor = save(&old, &mut backend);
    let loads = Arc::new(Mutex::new(Vec::new()));
    let restored = restore(&descriptor, backend.clone(), "second-writer", loads.clone());
    assert!(loads.lock().unwrap().is_empty());
    let right = old.root().observations[1].continuation.clone();
    let mut replay = simple(&["today"], Some(restored));
    assert_eq!(sequential(&mut replay, &[("today", "L")], 256), 0);
    assert_eq!(loads.lock().unwrap().len(), 257);
    assert!(!loads.lock().unwrap().contains(&right));
    assert_eq!(old.root().observations[0].evidence.len(), 1);
    let new_descriptor = save(&replay.history().unwrap(), &mut backend);
    let restored = restore(
        &new_descriptor,
        backend.clone(),
        "third-writer",
        Arc::default(),
    );
    let mut changed = simple(&["new"], Some(restored));
    assert_eq!(sequential(&mut changed, &[("new", "NEW")], 1), 1);
    let extended_descriptor = save(&changed.history().unwrap(), &mut backend);
    for (index, value) in ["L", "R", "NEW"].iter().enumerate() {
        let restored = restore(
            &extended_descriptor,
            backend.clone(),
            &format!("later-{index}"),
            Arc::default(),
        );
        assert_eq!(
            sequential(
                &mut simple(&["again"], Some(restored)),
                &[("again", value)],
                256
            ),
            0
        );
    }
    // Original published descriptor still resolves both old paths after additions.
    let restored = restore(&descriptor, backend, "old-snapshot-reader", Arc::default());
    assert_eq!(
        sequential(&mut simple(&["old"], Some(restored)), &[("old", "R")], 256),
        0
    );
}

#[test]
fn user_continuations_belong_to_resolved_investigation_not_answer_value() {
    let mut original = simple(&["a", "b"], None);
    sequential(&mut original, &[("a", "A"), ("b", "B")], 1);
    let old = original.history().unwrap();
    let mut resolved = simple(&["a"], Some(old.clone()));
    assert_eq!(sequential(&mut resolved, &[("a", "A")], 1), 0);
    let extended = resolved
        .append_user_message("Please check again".into())
        .unwrap();
    let answered = extended.load(&extended.root().observations[0].continuation);
    let untouched = extended.load(&extended.root().observations[1].continuation);
    assert_eq!(answered.assistant, untouched.assistant);
    assert!(untouched.user_continuations.is_empty());
    let edge = &answered.user_continuations[0];
    assert_eq!(edge.resolved_env_id.as_deref(), Some("a"));
    assert_eq!(edge.message, Message::user("Please check again".to_owned()));
    assert!(extended.load(&edge.continuation).assistant.is_none());
    assert!(
        old.load(&old.root().observations[0].continuation)
            .user_continuations
            .is_empty()
    );
    assert!(!resolved.history().unwrap().descriptor.requires_user_policy);
    let config = LoopConfig::new("inspect".into(), vec!["a".into()], vec![specs()[0].clone()])
        .with_replay(extended.clone(), "simple/v1");
    assert!(matches!(
        BatchLoop::new(config),
        Err(LoopError::UserContinuationPolicyRequired)
    ));

    // Representation of more tools, the SAME answer, and another user message.
    // This is storage-level construction, NOT a session ID or inheritance policy.
    let store = &extended.store;
    let open = store.put(HistoryNode::default());
    let same = store.put(HistoryNode {
        assistant: Some(Message::assistant("same final".into(), vec![])),
        user_continuations: vec![UserContinuation {
            message: Message::user("One more check".into()),
            continuation: open,
            resolved_env_id: None,
        }],
        ..HistoryNode::default()
    });
    let investigate = store.put(HistoryNode {
        assistant: Some(Message::assistant(
            "checking".into(),
            vec![ToolCall::new("p", "probe", json!({}))],
        )),
        observations: vec![Observation {
            representative: vec![OutputRecord {
                tool_output: "A".into(),
            }],
            evidence: vec![],
            continuation: same,
        }],
        ..HistoryNode::default()
    });
    let mut populated_answer = answered.as_ref().clone();
    populated_answer.user_continuations[0].continuation = investigate;
    let mut root = extended.root().as_ref().clone();
    root.observations[0].continuation = store.put(populated_answer);
    let mut populated = extended.clone();
    populated.descriptor.root = store.put(root);
    let mut backend = Backend::new();
    let descriptor = save(&populated, &mut backend);
    let restored = restore(&descriptor, backend, "inspection", Arc::default());
    let answer = restored.load(&restored.root().observations[0].continuation);
    let checked = restored.load(&answer.user_continuations[0].continuation);
    let final_ = restored.load(&checked.observations[0].continuation);
    assert_eq!(final_.assistant, answered.assistant);
    assert_eq!(
        final_.user_continuations[0].message.content,
        "One more check"
    );
    assert!(extended.load(&edge.continuation).assistant.is_none());
    assert_eq!(
        sequential(&mut simple(&["a"], Some(old)), &[("a", "A")], 1),
        0
    );
}

#[test]
fn storage_addresses_are_not_content_hashes_and_writer_reuse_is_rejected() {
    let store = RecordStore::<String>::new("reserved-writer");
    let first = store.put(HistoryNode::default());
    let second = store.put(HistoryNode::default());
    assert_ne!(first, second);
    assert_eq!(store.load(&first), store.load(&second));
    let invalid_writer =
        RecordStore::with_loader("reserved-writer", move |reference| store.load(reference));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invalid_writer.load(&second)))
            .is_err()
    );
}

#[derive(PartialEq, Eq)]
struct CollidingKey(bool);
impl std::fmt::Debug for CollidingKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("same")
    }
}
impl std::hash::Hash for CollidingKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&0, state);
    }
}

#[test]
fn non_recording_colliding_labels_do_not_identify_or_drop_groups() {
    let config = LoopConfig::new(
        "inspect".into(),
        vec!["a".into(), "b".into()],
        vec![specs()[0].clone()],
    );
    let mut loop_ =
        BatchLoop::with_group_by(config, |value: &String| CollidingKey(value == "A")).unwrap();
    sequential(&mut loop_, &[("a", "A"), ("b", "B")], 1);
    let result = loop_.result().unwrap();
    assert_eq!(result.tree.children.len(), 2);
    assert_eq!(result.tree.children[0].group_key, "[same]");
    assert_eq!(result.tree.children[1].group_key, "[same]");
    assert_ne!(result.classes[0].id, result.classes[1].id);
}

#[test]
fn non_recording_legacy_labels_retain_meaning() {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "inspect".into(),
        vec!["a".into(), "b".into()],
        vec![specs()[0].clone()],
    ))
    .unwrap();
    sequential(&mut loop_, &[("a", "A"), ("b", "B")], 1);
    let result = loop_.result().unwrap();
    assert!(result.history.is_none());
    assert_eq!(
        result
            .tree
            .children
            .iter()
            .map(|e| e.group_key.as_str())
            .collect::<Vec<_>>(),
        ["[\"A\"]", "[\"B\"]"]
    );
    assert_ne!(result.classes[0].id, result.classes[1].id);
}
