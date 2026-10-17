use lockstep_loop::{
    BatchLoop, BatchResult, Budget, Completion, History, HistoryNode, LlmCompletion, LlmRequest,
    LoopConfig, LoopError, NodeRef, RecordStore, Request, RequestKind, Role, SubagentTool,
    ToolCall, ToolCompletion, ToolRequest, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

const PROMPT: &str = "Inspect the repository";
const NOTE: &str = "Also inspect /foo";
const OLD: &str = "no issues";
fn specs() -> Vec<ToolSpec> {
    ["read", "detail", "extra"]
        .iter()
        .map(|s| ToolSpec::new(*s, "read-only", json!({})))
        .collect()
}
fn config(
    envs: &[&str],
    history: Option<History<String>>,
    feedback: Option<&str>,
) -> LoopConfig<String> {
    let c = LoopConfig::new(
        PROMPT.into(),
        envs.iter().map(|e| (*e).into()).collect(),
        specs(),
    );
    let mut c = match history {
        Some(h) => c.with_replay(h, "feedback-v1"),
        None => c.with_recording("feedback-v1"),
    };
    if let Some(content) = feedback {
        c = c.with_feedback(content.into());
    }
    c
}
fn loop_for(
    envs: &[&str],
    history: Option<History<String>>,
    feedback: Option<&str>,
) -> BatchLoop<String> {
    BatchLoop::new(config(envs, history, feedback)).unwrap()
}
fn next_llm(loop_: &mut BatchLoop<String>) -> LlmRequest<String> {
    let Request::Llm(r) = loop_.next_request(RequestKind::Llm).unwrap() else {
        panic!()
    };
    r
}
fn next_tool(loop_: &mut BatchLoop<String>) -> ToolRequest {
    let Request::Tool(r) = loop_.next_request(RequestKind::Tool).unwrap() else {
        panic!()
    };
    r
}
fn output(request: &ToolRequest) -> ToolCompletion<String> {
    let results = request
        .env_ids
        .iter()
        .rev()
        .map(|env| {
            let value = match request.tool_call.name.as_str() {
                "read" => {
                    if env == "D" {
                        "unrelated"
                    } else {
                        "shared"
                    }
                }
                "detail" => {
                    if env == "B" || env == "NEW" {
                        "vulnerability"
                    } else {
                        "safe"
                    }
                }
                "extra" => "metadata",
                _ => panic!("unexpected tool"),
            };
            (env.clone(), value.into())
        })
        .collect();
    ToolCompletion {
        request_id: request.request_id.clone(),
        results,
    }
}
fn model(request: &LlmRequest<String>) -> LlmCompletion<String> {
    let calls = if request.messages.len() == 1 {
        vec![ToolCall::new("r", "read", json!({"path": "/entry"}))]
    } else if request.messages.last().unwrap().role == Role::User {
        vec![
            ToolCall::new("d", "detail", json!({"path": "/foo"})),
            ToolCall::new("e", "extra", json!({"path": "/metadata"})),
        ]
    } else {
        vec![]
    };
    let last_detail = request
        .messages
        .iter()
        .rev()
        .find(|m| m.tool_call_id.as_deref() == Some("d"));
    let answer = if last_detail.is_some_and(|m| m.content == "vulnerability") {
        "found"
    } else {
        OLD
    };
    LlmCompletion {
        request_id: request.request_id.clone(),
        content: if calls.is_empty() {
            answer
        } else {
            "investigating"
        }
        .into(),
        tool_calls: calls,
    }
}
struct Driven {
    result: BatchResult<String>,
    llms: Vec<LlmRequest<String>>,
    tools: Vec<ToolRequest>,
}
fn drive(loop_: &mut BatchLoop<String>, allow_llm: bool) -> Driven {
    let (mut llms, mut tools) = (Vec::new(), Vec::new());
    while !loop_.is_done() {
        let mut pending = Vec::new();
        while let Some(r) = loop_.next_request(RequestKind::Any) {
            pending.push(r);
        }
        assert!(!pending.is_empty(), "stranded work");
        for r in pending.into_iter().rev() {
            match r {
                Request::Llm(r) => {
                    assert!(allow_llm, "unexpected fresh reasoning during replay");
                    loop_.complete(Completion::Llm(model(&r))).unwrap();
                    llms.push(r);
                }
                Request::Tool(r) => {
                    loop_.complete(Completion::Tool(output(&r))).unwrap();
                    tools.push(r);
                }
            }
        }
    }
    assert!(loop_.error().is_none());
    assert_eq!(loop_.outstanding_count(), 0);
    Driven {
        result: loop_.result().unwrap(),
        llms,
        tools,
    }
}
fn initial() -> History<String> {
    drive(&mut loop_for(&["A", "B", "D"], None, None), true)
        .result
        .history
        .unwrap()
}
fn shared_node(history: &History<String>) -> Arc<HistoryNode<String>> {
    let root = history.root();
    let edge = root
        .observations
        .iter()
        .find(|e| e.representative[0].tool_output.trim() == "shared")
        .unwrap();
    history.load(&edge.continuation)
}
type Backend = BTreeMap<NodeRef, Vec<u8>>;
fn publish(history: &History<String>, backend: &mut Backend) -> Vec<u8> {
    for (r, n) in history.store.records() {
        let bytes = serde_json::to_vec(&n).unwrap();
        if let Some(old) = backend.get(&r) {
            assert_eq!(old, &bytes);
        } else {
            backend.insert(r, bytes);
        }
    }
    serde_json::to_vec(&history.descriptor).unwrap()
}
fn restore(descriptor: &[u8], backend: Backend, namespace: &str) -> History<String> {
    History {
        descriptor: serde_json::from_slice(descriptor).unwrap(),
        store: RecordStore::with_loader(namespace, move |r| {
            Arc::new(serde_json::from_slice(&backend[r]).unwrap())
        }),
    }
}

#[test]
fn resume_once_shared_path_inheritance_splits_and_immutable_snapshots() {
    let old = initial();
    let old_shared = shared_node(&old);
    let old_unrelated = old.root().observations[1].continuation.clone();
    let mut loop_ = loop_for(&["A"], Some(old.clone()), Some(NOTE));
    assert!(loop_.next_request(RequestKind::Llm).is_none());
    let read = next_tool(&mut loop_);
    assert_eq!(read.tool_call.name, "read");
    loop_.complete(Completion::Tool(output(&read))).unwrap();
    let request = next_llm(&mut loop_);
    assert_eq!(
        request.messages.iter().map(|m| m.role).collect::<Vec<_>>(),
        [
            Role::User,
            Role::Assistant,
            Role::Tool,
            Role::Assistant,
            Role::User
        ]
    );
    assert_eq!(request.messages[0].content, PROMPT);
    assert_eq!(request.messages[3].content, OLD);
    assert_eq!(request.messages[4].content, NOTE);
    assert_eq!(
        request.messages[1].tool_calls[0].args,
        json!({"path": "/entry"})
    );
    let opened = loop_.history().unwrap();
    let edge = shared_node(&opened).user_continuations[0].clone();
    assert_eq!(edge.message.content, NOTE);
    assert!(opened.load(&edge.continuation).assistant.is_none());
    assert!(old_shared.user_continuations.is_empty());
    assert_eq!(opened.root().observations[1].continuation, old_unrelated);

    let before = loop_.history().unwrap();
    assert!(matches!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id.clone(),
            results: vec![("A".into(), "bad".into())],
        })),
        Err(LoopError::WrongCompletionKind { .. })
    ));
    let duplicate = ToolCall::new("dup", "detail", json!({}));
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id.clone(),
            content: "bad".into(),
            tool_calls: vec![duplicate.clone(), duplicate],
        })),
        Err(LoopError::DuplicateToolCallId(_))
    ));
    assert_eq!(loop_.history().unwrap(), before);
    assert_eq!(loop_.outstanding_llm(), 1);
    loop_.complete(Completion::Llm(model(&request))).unwrap();
    let detail = next_tool(&mut loop_);
    let extra = next_tool(&mut loop_);
    let before = loop_.history().unwrap();
    assert!(matches!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: extra.request_id.clone(),
            results: vec![("wrong".into(), "bad".into())],
        })),
        Err(LoopError::InvalidToolCompletion(_))
    ));
    assert_eq!(loop_.history().unwrap(), before);
    assert_eq!(loop_.outstanding_tool(), 2);
    loop_.complete(Completion::Tool(output(&extra))).unwrap();
    assert!(loop_.next_request(RequestKind::Llm).is_none());
    loop_.complete(Completion::Tool(output(&detail))).unwrap();
    assert!(matches!(
        loop_.complete(Completion::Tool(output(&detail))),
        Err(LoopError::UnknownRequest(_))
    ));
    let final_ = next_llm(&mut loop_);
    assert_eq!(
        final_
            .messages
            .iter()
            .rev()
            .take(2)
            .map(|m| m.tool_call_id.as_deref().unwrap())
            .collect::<Vec<_>>(),
        ["e", "d"]
    );
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            final_.request_id,
            OLD.into(),
        )))
        .unwrap();
    assert!(loop_.is_done());
    assert!(loop_.next_request(RequestKind::Any).is_none());
    let revised = loop_.result().unwrap().history.unwrap();
    let revised_shared = shared_node(&revised);
    assert_eq!(revised_shared.assistant.as_ref().unwrap().content, OLD);
    let after_user = revised.load(&revised_shared.user_continuations[0].continuation);
    assert_eq!(
        after_user.assistant.as_ref().unwrap().tool_calls[0].args,
        json!({"path": "/foo"})
    );
    assert_eq!(
        after_user.observations[0].evidence[0]
            .outputs
            .iter()
            .map(|o| o.tool_output.as_str())
            .collect::<Vec<_>>(),
        ["safe", "metadata"]
    );
    assert_eq!(
        revised
            .load(&after_user.observations[0].continuation)
            .assistant
            .as_ref()
            .unwrap()
            .content,
        OLD
    );
    assert!(opened.load(&edge.continuation).assistant.is_none());

    let driven = drive(
        &mut loop_for(&["A", "B", "C", "D"], Some(revised.clone()), None),
        true,
    );
    assert_eq!(
        driven.result.results,
        BTreeMap::from([
            ("A".into(), OLD.into()),
            ("B".into(), "found".into()),
            ("C".into(), OLD.into()),
            ("D".into(), OLD.into())
        ])
    );
    assert_eq!(driven.llms.len(), 1);
    assert_eq!(driven.llms[0].env_ids, ["B"]);
    assert_eq!(
        driven.llms[0]
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        [PROMPT, NOTE]
    );
    assert!(
        driven
            .tools
            .iter()
            .filter(|r| r.tool_call.name != "read")
            .all(|r| r.env_ids == ["A", "B", "C"])
    );
    let mut classes: Vec<_> = driven
        .result
        .classes
        .iter()
        .map(|c| c.env_ids.clone())
        .collect();
    classes.sort();
    assert_eq!(classes, [vec!["A", "C"], vec!["B"], vec!["D"]]);
    assert!(old_shared.user_continuations.is_empty());
    assert_eq!(old.root().env_ids, ["A", "B", "D"]);
    assert_eq!(after_user.observations.len(), 1);
    let extended = driven.result.history.unwrap();
    assert!(
        extended
            .load(&extended.root().observations[1].continuation)
            .user_continuations
            .is_empty()
    );
    let prior = drive(&mut loop_for(&["B"], Some(old), None), false);
    assert_eq!(prior.tools.len(), 1);
    assert_eq!(prior.tools[0].tool_call.name, "read");
    for env in ["A", "B", "C", "D"] {
        drive(&mut loop_for(&[env], Some(extended.clone()), None), false);
    }
}

#[test]
fn interleaved_feedback_paths_preserve_each_others_updates() {
    let first = drive(&mut loop_for(&["A"], Some(initial()), Some(NOTE)), true)
        .result
        .history
        .unwrap();
    let second = drive(
        &mut loop_for(&["D"], Some(first.clone()), Some("Check another path")),
        true,
    )
    .result
    .history
    .unwrap();
    assert!(
        first
            .load(&first.root().observations[1].continuation)
            .user_continuations
            .is_empty()
    );
    for seed in 0..8_u64 {
        let mut state = seed;
        let mut choose = |n: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 32) as usize % n
        };
        let mut loop_ = loop_for(&["A", "B", "C", "D"], Some(second.clone()), None);
        let (mut pending, mut llms) = (vec![], vec![]);
        while !loop_.is_done() {
            for _ in 0..(choose(4) + 1) {
                let kind = [RequestKind::Any, RequestKind::Llm, RequestKind::Tool][choose(3)];
                let request = loop_
                    .next_request(kind)
                    .or_else(|| loop_.next_request(RequestKind::Any));
                if let Some(r) = request {
                    pending.push(r);
                } else {
                    break;
                }
            }
            assert!(!pending.is_empty());
            let request = pending.remove(choose(pending.len()));
            match request {
                Request::Llm(r) => {
                    assert_eq!(
                        r.messages
                            .iter()
                            .filter(|m| m.role == Role::User)
                            .map(|m| m.content.as_str())
                            .collect::<Vec<_>>(),
                        [PROMPT, NOTE]
                    );
                    loop_.complete(Completion::Llm(model(&r))).unwrap();
                    llms.push(r);
                }
                Request::Tool(r) => loop_.complete(Completion::Tool(output(&r))).unwrap(),
            }
            assert_eq!(loop_.outstanding_count(), pending.len());
        }
        assert!(pending.is_empty());
        assert_eq!(llms.len(), 1);
        assert_eq!(llms[0].env_ids, ["B"]);
        let result = loop_.result().unwrap();
        assert_eq!(
            result.results,
            BTreeMap::from([
                ("A".into(), OLD.into()),
                ("B".into(), "found".into()),
                ("C".into(), OLD.into()),
                ("D".into(), OLD.into())
            ])
        );
        assert_eq!(result.classes.len(), 3);
        assert_eq!(
            shared_node(&result.history.unwrap())
                .user_continuations
                .len(),
            1
        );
    }
}

#[test]
fn repeated_identical_feedback_after_storage_roundtrip_targets_revised_endpoint() {
    let first = drive(&mut loop_for(&["B"], Some(initial()), Some(NOTE)), true).result;
    assert_eq!(first.results["B"], "found");
    let first_history = first.history.unwrap();
    let mut backend = Backend::new();
    let descriptor = publish(&first_history, &mut backend);
    let restored = restore(&descriptor, backend.clone(), "second-intervention");
    let mut loop_ = loop_for(&["B"], Some(restored.clone()), Some(NOTE));
    let mut tool_count = 0;
    let request = loop {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => break r,
            Request::Tool(r) => {
                tool_count += 1;
                loop_.complete(Completion::Tool(output(&r))).unwrap();
            }
        }
    };
    assert_eq!(tool_count, 3);
    assert_eq!(
        request
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        [PROMPT, NOTE, NOTE]
    );
    assert_eq!(
        request.messages[request.messages.len() - 2].content,
        "found"
    );
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            "found".into(),
        )))
        .unwrap();
    assert!(loop_.is_done());
    let second = loop_.result().unwrap().history.unwrap();
    let descriptor = publish(&second, &mut backend);
    let reopened = restore(&descriptor, backend, "replay-second");
    let driven = drive(&mut loop_for(&["NEW"], Some(reopened), None), false);
    assert_eq!(driven.result.results["NEW"], "found");
    assert_eq!(driven.tools.len(), 3);
    let tools_node = restored.load(&shared_node(&restored).user_continuations[0].continuation);
    let first_end = restored.load(&tools_node.observations[0].continuation);
    assert_eq!(first_end.assistant.as_ref().unwrap().content, "found");
    assert!(first_end.user_continuations.is_empty());
}

fn assert_failed(loop_: &mut BatchLoop<String>, old: &History<String>) {
    assert!(loop_.is_done());
    assert_eq!(loop_.error(), Some(&LoopError::FeedbackResolutionMiss));
    assert!(loop_.next_request(RequestKind::Any).is_none());
    assert_eq!(loop_.outstanding_count(), 0);
    assert_eq!(loop_.history().as_ref(), Some(old));
    assert_eq!(loop_.result(), Err(LoopError::FeedbackResolutionMiss));
}
#[test]
fn feedback_miss_is_terminal_not_correctable_validation_or_fresh_reasoning() {
    let old = initial();
    let mut loop_ = loop_for(&["A"], Some(old.clone()), Some(NOTE));
    let request = next_tool(&mut loop_);
    assert!(matches!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id.clone(),
            results: vec![]
        })),
        Err(LoopError::InvalidToolCompletion(_))
    ));
    assert!(loop_.error().is_none());
    assert_eq!(loop_.outstanding_count(), 1);
    assert_eq!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id.clone(),
            results: vec![("A".into(), "unseen".into())]
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert_failed(&mut loop_, &old);
    assert_eq!(
        loop_.complete(Completion::Tool(output(&request))),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert!(shared_node(&old).user_continuations.is_empty());
}

#[test]
fn partial_paths_and_unfinished_feedback_are_not_targets() {
    let mut pending = loop_for(&["A"], None, None);
    let empty = pending.history().unwrap();
    assert!(matches!(
        BatchLoop::new(config(&["A"], Some(empty), Some(NOTE))),
        Err(LoopError::FeedbackResolutionMiss)
    ));
    let request = next_llm(&mut pending);
    pending.complete(Completion::Llm(model(&request))).unwrap();
    let request = next_tool(&mut pending);
    pending
        .complete(Completion::Tool(output(&request)))
        .unwrap();
    let partial = pending.history().unwrap();
    let mut loop_ = loop_for(&["A"], Some(partial.clone()), Some(NOTE));
    let tool = next_tool(&mut loop_);
    assert_eq!(
        loop_.complete(Completion::Tool(output(&tool))),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert_failed(&mut loop_, &partial);
    let mut first = loop_for(&["A"], Some(initial()), Some(NOTE));
    let tool = next_tool(&mut first);
    first.complete(Completion::Tool(output(&tool))).unwrap();
    let open = first.history().unwrap();
    let mut second = loop_for(&["A"], Some(open.clone()), Some("another intervention"));
    let tool = next_tool(&mut second);
    assert_eq!(
        second.complete(Completion::Tool(output(&tool))),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert_failed(&mut second, &open);
}

#[test]
fn opaque_null_feedback_presence_and_constructor_limits() {
    let base = || LoopConfig::new(Some(vec![0_u8, 255]), vec!["A".into()], vec![]);
    let mut initial_loop = BatchLoop::new(base().with_recording("bytes")).unwrap();
    let request = initial_loop.next_request(RequestKind::Any).unwrap();
    initial_loop
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            Some(vec![1, 254]),
        )))
        .unwrap();
    let old = initial_loop.history().unwrap();
    let mut updated = old.clone();
    for _ in 0..3 {
        let mut loop_ =
            BatchLoop::new(base().with_replay(updated, "bytes").with_feedback(None)).unwrap();
        let Request::Llm(request) = loop_.next_request(RequestKind::Any).unwrap() else {
            panic!()
        };
        assert_eq!(request.messages.last().unwrap().role, Role::User);
        assert_eq!(request.messages.last().unwrap().content, None);
        assert_eq!(
            request.messages[request.messages.len() - 2].content,
            Some(vec![1, 254])
        );
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id,
                Some(vec![1, 254]),
            )))
            .unwrap();
        assert!(loop_.is_done());
        updated = loop_.history().unwrap();
    }
    let mut replay_config = base().with_replay(updated, "bytes");
    replay_config.env_ids = vec!["C".into()];
    let mut replay = BatchLoop::new(replay_config).unwrap();
    assert!(replay.is_done());
    assert_eq!(replay.result().unwrap().results["C"], Some(vec![1, 254]));
    assert!(old.root().user_continuations.is_empty());
    assert!(matches!(
        BatchLoop::new(base().with_feedback(None)),
        Err(LoopError::InvalidFeedback)
    ));
    let mut multiple = base().with_replay(old.clone(), "bytes").with_feedback(None);
    multiple.env_ids.push("B".into());
    assert!(matches!(
        BatchLoop::new(multiple),
        Err(LoopError::InvalidFeedback)
    ));
    assert!(matches!(
        BatchLoop::new(
            base()
                .with_replay(old.clone(), "changed")
                .with_feedback(None)
        ),
        Err(LoopError::IncompatibleHistory)
    ));
    assert!(matches!(
        BatchLoop::new(
            base()
                .with_replay(old, "bytes")
                .with_feedback(None)
                .with_budget(Budget::new(100, |_| None))
        ),
        Err(LoopError::UnsupportedHistory)
    ));
}

#[test]
fn lossy_grouping_keeps_actual_evidence_and_shared_feedback_context() {
    let mut loop_ = BatchLoop::with_group_by(config(&["A", "B"], None, None), |s: &String| {
        s.trim().to_owned()
    })
    .unwrap();
    let request = next_llm(&mut loop_);
    loop_.complete(Completion::Llm(model(&request))).unwrap();
    let request = next_tool(&mut loop_);
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id,
            results: vec![
                ("A".into(), "shared ".into()),
                ("B".into(), "shared".into()),
            ],
        }))
        .unwrap();
    let request = next_llm(&mut loop_);
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            OLD.into(),
        )))
        .unwrap();
    let mut feedback =
        BatchLoop::with_group_by(config(&["B"], loop_.history(), Some(NOTE)), |s: &String| {
            s.trim().to_owned()
        })
        .unwrap();
    let tool = next_tool(&mut feedback);
    feedback
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![("B".into(), "shared\n".into())],
        }))
        .unwrap();
    let request = next_llm(&mut feedback);
    assert_eq!(request.messages[2].content, "shared ");
    feedback
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            OLD.into(),
        )))
        .unwrap();
    let history = feedback.history().unwrap();
    assert_eq!(
        history.root().observations[0]
            .evidence
            .last()
            .unwrap()
            .outputs[0]
            .tool_output,
        "shared\n"
    );
    let mut replay =
        BatchLoop::with_group_by(config(&["C"], Some(history.clone()), None), |s: &String| {
            s.trim().to_owned()
        })
        .unwrap();
    let tool = next_tool(&mut replay);
    replay
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![("C".into(), "shared\t".into())],
        }))
        .unwrap();
    assert!(replay.is_done());
    assert_eq!(replay.result().unwrap().results["C"], OLD);
    assert_eq!(
        history.root().observations[0]
            .evidence
            .last()
            .unwrap()
            .outputs[0]
            .tool_output,
        "shared\n"
    );
}

#[test]
fn ambiguous_user_edges_are_rejected_not_implicitly_merged() {
    let mut pending = loop_for(&["A"], Some(initial()), Some(NOTE));
    let tool = next_tool(&mut pending);
    pending.complete(Completion::Tool(output(&tool))).unwrap();
    let mut malformed = pending.history().unwrap();
    let mut node = shared_node(&malformed).as_ref().clone();
    node.user_continuations
        .push(node.user_continuations[0].clone());
    let bad_node = malformed.store.put(node);
    let mut root = malformed.root().as_ref().clone();
    root.observations[0].continuation = bad_node;
    malformed.descriptor.root = malformed.store.put(root);
    let mut loop_ = loop_for(&["A"], Some(malformed.clone()), None);
    let tool = next_tool(&mut loop_);
    assert_eq!(
        loop_.complete(Completion::Tool(output(&tool))),
        Err(LoopError::InvalidUserContinuation)
    );
    assert!(loop_.is_done());
    assert_eq!(loop_.error(), Some(&LoopError::InvalidUserContinuation));
    drive(&mut loop_for(&["D"], Some(malformed), None), false);
}

fn nested_config(previous: Option<History<String>>, feedback: Option<&str>) -> LoopConfig<String> {
    let specs = || {
        ["investigate", "probe"]
            .iter()
            .map(|s| ToolSpec::new(*s, "", json!({})))
            .collect()
    };
    let config = LoopConfig::new("parent".into(), vec!["A".into()], specs()).with_subagent(
        SubagentTool::new(
            ToolSpec::new("investigate", "", json!({})),
            |a| a["prompt"].as_str().unwrap().into(),
            move |_| specs(),
            1,
        ),
    );
    let mut config = match previous {
        Some(h) => config.with_replay(h, "nested-feedback"),
        None => config.with_recording("nested-feedback"),
    };
    if let Some(f) = feedback {
        config = config.with_feedback(f.into());
    }
    config
}
fn nested_loop(previous: Option<History<String>>, feedback: Option<&str>) -> BatchLoop<String> {
    BatchLoop::new(nested_config(previous, feedback)).unwrap()
}
fn serve_nested(r: &LlmRequest<String>) -> LlmCompletion<String> {
    let scope = r.messages[0].content.as_str();
    let calls = if scope == "parent"
        && (r.messages.len() == 1 || r.messages.last().unwrap().role == Role::User)
    {
        let child = if r.messages.len() == 1 {
            "old-child"
        } else {
            "new-child"
        };
        vec![
            ToolCall::new("child", "investigate", json!({"prompt": child})),
            ToolCall::new("local", "probe", json!({"scope": "local"})),
        ]
    } else if scope != "parent" && r.messages.len() == 1 {
        vec![ToolCall::new("p", "probe", json!({"scope": scope}))]
    } else {
        vec![]
    };
    LlmCompletion {
        request_id: r.request_id.clone(),
        content: if calls.is_empty() { "same" } else { "checking" }.into(),
        tool_calls: calls,
    }
}
fn nested_drive(loop_: &mut BatchLoop<String>, replay: bool) -> Vec<LlmRequest<String>> {
    let mut llms = vec![];
    while !loop_.is_done() {
        let mut pending = vec![];
        while let Some(r) = loop_.next_request(RequestKind::Any) {
            pending.push(r);
        }
        assert!(!pending.is_empty());
        for r in pending.into_iter().rev() {
            match r {
                Request::Llm(r) => {
                    assert!(!replay);
                    loop_.complete(Completion::Llm(serve_nested(&r))).unwrap();
                    llms.push(r);
                }
                Request::Tool(r) => loop_
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results: vec![("A".into(), "known".into())],
                    }))
                    .unwrap(),
            }
        }
    }
    llms
}
#[test]
fn resolution_miss_in_child_cancels_other_work_and_new_children_can_resume() {
    let mut pending = nested_loop(None, None);
    let request = next_llm(&mut pending);
    pending
        .complete(Completion::Llm(serve_nested(&request)))
        .unwrap();
    assert!(matches!(
        BatchLoop::new(nested_config(pending.history(), Some(NOTE))),
        Err(LoopError::FeedbackResolutionMiss)
    ));
    let mut original = nested_loop(None, None);
    nested_drive(&mut original, false);
    let old = original.history().unwrap();
    let mut loop_ = nested_loop(Some(old.clone()), Some(NOTE));
    let child = next_tool(&mut loop_);
    let local = next_tool(&mut loop_);
    assert_eq!(child.tool_call.args["scope"], "old-child");
    assert_eq!(local.tool_call.args["scope"], "local");
    assert_eq!(loop_.outstanding_count(), 2);
    assert_eq!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: child.request_id,
            results: vec![("A".into(), "unseen child evidence".into())]
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert_failed(&mut loop_, &old);
    assert_eq!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: local.request_id,
            results: vec![("A".into(), "known".into())]
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    let mut feedback = nested_loop(Some(old.clone()), Some(NOTE));
    let requests = nested_drive(&mut feedback, false);
    assert_eq!(requests[0].messages.last().unwrap().content, NOTE);
    assert_eq!(
        requests
            .iter()
            .map(|r| r.messages[0].content.as_str())
            .collect::<Vec<_>>(),
        ["parent", "new-child", "new-child", "parent"]
    );
    assert!(nested_drive(&mut nested_loop(feedback.history(), None), true).is_empty());
    assert!(
        old.load(&old.root().observations[0].continuation)
            .user_continuations
            .is_empty()
    );
}
