//! Library-detected nested incompatibility must be a structured failure, not a panic.
use lockstep_loop::{
    AgentSnapshot, BatchLoop, Completion, LlmCompletion, LlmRequest, LoopConfig, LoopError,
    NodeRef, RecordStore, Request, RequestKind, Role, SubagentTool, ToolCall, ToolCompletion,
    ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy)]
enum Mismatch {
    Prompt,
    Specs,
}
fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new("probe", "", json!({})),
        ToolSpec::new("investigate", "", json!({})),
    ]
}
fn subagent(
    fault: &Arc<AtomicBool>,
    calls: &Arc<Mutex<Vec<String>>>,
    mismatch: Mismatch,
    depth: usize,
) -> SubagentTool<String> {
    let prompt_fault = fault.clone();
    let prompt_calls = calls.clone();
    let spec_fault = fault.clone();
    let spec_calls = calls.clone();
    SubagentTool::new(
        specs()[1].clone(),
        move |args| {
            let name = args["required"].as_str().unwrap();
            prompt_calls.lock().unwrap().push(format!("prompt:{name}"));
            if prompt_fault.load(Ordering::Relaxed)
                && matches!(mismatch, Mismatch::Prompt)
                && name == "bad"
            {
                "incompatible prompt".into()
            } else {
                name.into()
            }
        },
        move |_| {
            spec_calls.lock().unwrap().push("specs".into());
            let mut result = specs();
            if spec_fault.load(Ordering::Relaxed) && matches!(mismatch, Mismatch::Specs) {
                result[0].description = "incompatible spec".into();
            }
            result
        },
        depth,
    )
}
fn config(
    sub: SubagentTool<String>,
    snapshot: Option<AgentSnapshot<String>>,
    env: &str,
) -> LoopConfig<String> {
    let config = LoopConfig::new("parent".into(), vec![env.into()], specs()).with_subagent(sub);
    if let Some(snapshot) = snapshot {
        config.with_snapshot(snapshot)
    } else {
        config
    }
}
fn llm(loop_: &mut BatchLoop<String>) -> LlmRequest<String> {
    let Request::Llm(r) = loop_.next_request(RequestKind::Llm).unwrap() else {
        panic!()
    };
    r
}
fn mixed(request_id: &str) -> Completion<String> {
    Completion::Llm(LlmCompletion {
        request_id: request_id.into(),
        content: "delegate".into(),
        tool_calls: vec![
            ToolCall::new("local", "probe", json!({})),
            ToolCall::new("good", "investigate", json!({"required": "good"})),
            ToolCall::new("bad", "investigate", json!({"required": "bad"})),
        ],
    })
}
fn finish(loop_: &mut BatchLoop<String>, replay: bool) -> BTreeMap<String, String> {
    while !loop_.is_done() {
        let mut pending = vec![];
        while let Some(r) = loop_.next_request(RequestKind::Any) {
            pending.push(r);
        }
        assert!(!pending.is_empty(), "stranded work");
        for r in pending.into_iter().rev() {
            match r {
                Request::Tool(r) => loop_
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results: r.env_ids.into_iter().map(|e| (e, "local".into())).collect(),
                    }))
                    .unwrap(),
                Request::Llm(r) => {
                    assert!(!replay);
                    let answer = if r.messages[0].content == "parent" {
                        r.messages
                            .iter()
                            .filter(|m| m.role == Role::Tool)
                            .map(|m| m.content.as_str())
                            .collect::<Vec<_>>()
                            .join("|")
                    } else {
                        format!("answer:{}", r.messages[0].content)
                    };
                    loop_
                        .complete(Completion::Llm(LlmCompletion::final_answer(
                            r.request_id,
                            answer,
                        )))
                        .unwrap();
                }
            }
        }
    }
    assert_eq!(loop_.outstanding_count(), 0);
    loop_.result().unwrap().results
}
fn persisted(snapshot: &AgentSnapshot<String>) -> AgentSnapshot<String> {
    let records: BTreeMap<NodeRef, Vec<u8>> = snapshot
        .store
        .records()
        .into_iter()
        .map(|(r, n)| (r, serde_json::to_vec(&n).unwrap()))
        .collect();
    AgentSnapshot {
        descriptor: serde_json::from_slice(&serde_json::to_vec(&snapshot.descriptor).unwrap())
            .unwrap(),
        store: RecordStore::with_loader("restored", move |r| {
            Arc::new(serde_json::from_slice(&records[r]).unwrap())
        }),
    }
}

#[test]
fn children_are_prepared_once_and_corrected_completions_replay_consistently() {
    let fault = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(Mutex::new(vec![]));
    let sub = subagent(&fault, &calls, Mismatch::Prompt, 1);
    let mut loop_ = BatchLoop::new(config(sub.clone(), None, "old")).unwrap();
    let request = llm(&mut loop_);
    let before = loop_.snapshot();
    let duplicate = ToolCall::new("same", "investigate", json!({"required": "bad"}));
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id.clone(),
            content: "rejected".into(),
            tool_calls: vec![duplicate.clone(), duplicate],
        })),
        Err(LoopError::DuplicateToolCallId(_))
    ));
    assert!(calls.lock().unwrap().is_empty());
    assert_eq!(loop_.snapshot(), before);
    assert_eq!(loop_.outstanding_llm(), 1);
    loop_.complete(mixed(&request.request_id)).unwrap();
    let expected_calls = vec!["specs", "prompt:good", "specs", "prompt:bad"];
    assert_eq!(*calls.lock().unwrap(), expected_calls);
    let expected = "local|answer:good|answer:bad";
    assert_eq!(finish(&mut loop_, false)["old"], expected);
    assert_eq!(*calls.lock().unwrap(), expected_calls); // no factory rerun after commit
    let snapshot = loop_.snapshot();
    assert_eq!(
        snapshot.root().assistant.as_ref().unwrap().content,
        "delegate"
    );
    let mut replay = BatchLoop::new(config(sub, Some(persisted(&snapshot)), "new")).unwrap();
    assert_eq!(finish(&mut replay, true)["new"], expected);
}

#[test]
fn incompatible_nested_history_is_rejected_without_panic_at_construction_or_after_tools() {
    for delayed in [false, true] {
        for mismatch in [Mismatch::Prompt, Mismatch::Specs] {
            let fault = Arc::new(AtomicBool::new(false));
            let calls = Arc::new(Mutex::new(vec![]));
            let sub = subagent(&fault, &calls, mismatch, 1);
            let mut original = BatchLoop::new(config(sub.clone(), None, "old")).unwrap();
            if delayed {
                let request = llm(&mut original);
                original
                    .complete(Completion::Llm(LlmCompletion {
                        request_id: request.request_id,
                        content: "read first".into(),
                        tool_calls: vec![ToolCall::new("first", "probe", json!({}))],
                    }))
                    .unwrap();
                let Request::Tool(tool) = original.next_request(RequestKind::Tool).unwrap() else {
                    panic!()
                };
                original
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: tool.request_id,
                        results: vec![("old".into(), "local".into())],
                    }))
                    .unwrap();
            }
            let request = llm(&mut original);
            original.complete(mixed(&request.request_id)).unwrap();
            let expected = finish(&mut original, false)["old"].clone();
            let snapshot = original.snapshot();
            let old_turn = if delayed {
                snapshot.load(&snapshot.root().observations[0].continuation)
            } else {
                snapshot.root()
            };
            fault.store(true, Ordering::Relaxed);
            if delayed {
                let mut replay =
                    BatchLoop::new(config(sub.clone(), Some(snapshot.clone()), "new")).unwrap();
                let Request::Tool(tool) = replay.next_request(RequestKind::Tool).unwrap() else {
                    panic!()
                };
                let completion = Completion::Tool(ToolCompletion {
                    request_id: tool.request_id,
                    results: vec![("new".into(), "local".into())],
                });
                assert_eq!(
                    replay.complete(completion.clone()),
                    Err(LoopError::IncompatibleHistory)
                );
                assert_eq!(replay.error(), Some(&LoopError::IncompatibleHistory));
                assert!(replay.is_done());
                assert_eq!(replay.outstanding_count(), 0);
                assert!(replay.next_request(RequestKind::Any).is_none());
                assert_eq!(replay.result(), Err(LoopError::IncompatibleHistory));
                assert_eq!(
                    replay.complete(completion),
                    Err(LoopError::IncompatibleHistory)
                );
                let rejected = replay.snapshot();
                let turn = rejected.load(&rejected.root().observations[0].continuation);
                // A prior ordinary observation was accepted, but the incompatible
                // assistant turn has NOT been fulfilled/visited for the new env.
                assert_eq!(turn.env_ids, ["old"]);
                assert_eq!(turn.as_ref(), old_turn.as_ref());
            } else {
                assert!(matches!(
                    BatchLoop::new(config(sub.clone(), Some(snapshot.clone()), "new")),
                    Err(LoopError::IncompatibleHistory)
                ));
            }
            assert_eq!(old_turn.env_ids, ["old"]);
            assert_eq!(snapshot.root().env_ids, ["old"]);
            fault.store(false, Ordering::Relaxed);
            let mut retry =
                BatchLoop::new(config(sub, Some(persisted(&snapshot)), "again")).unwrap();
            assert_eq!(finish(&mut retry, true)["again"], expected);
        }
    }
}

#[test]
fn late_incompatibility_in_a_grandchild_cancels_root_routes_coherently() {
    let fault = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(Mutex::new(vec![]));
    let sub = subagent(&fault, &calls, Mismatch::Prompt, 2);
    let mut original = BatchLoop::new(config(sub.clone(), None, "old")).unwrap();
    let request = llm(&mut original);
    original
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id,
            content: "outer".into(),
            tool_calls: vec![
                ToolCall::new("outer", "investigate", json!({"required": "outer"})),
                ToolCall::new("local", "probe", json!({})),
            ],
        }))
        .unwrap();
    let request = llm(&mut original);
    original
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id,
            content: "probe".into(),
            tool_calls: vec![ToolCall::new("p", "probe", json!({}))],
        }))
        .unwrap();
    let Request::Tool(tool) = original.next_request(RequestKind::Tool).unwrap() else {
        panic!()
    };
    original
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![("old".into(), "local".into())],
        }))
        .unwrap();
    let request = llm(&mut original);
    original
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id,
            content: "inner".into(),
            tool_calls: vec![ToolCall::new(
                "inner",
                "investigate",
                json!({"required": "bad"}),
            )],
        }))
        .unwrap();
    finish(&mut original, false);
    let snapshot = original.snapshot();
    fault.store(true, Ordering::Relaxed);
    let mut replay = BatchLoop::new(config(sub, Some(snapshot.clone()), "new")).unwrap();
    let Request::Tool(child_tool) = replay.next_request(RequestKind::Tool).unwrap() else {
        panic!()
    };
    let other = replay.next_request(RequestKind::Tool).unwrap();
    assert_eq!(replay.outstanding_count(), 2);
    assert_eq!(
        replay.complete(Completion::Tool(ToolCompletion {
            request_id: child_tool.request_id,
            results: vec![("new".into(), "local".into())]
        })),
        Err(LoopError::IncompatibleHistory)
    );
    assert!(replay.is_done());
    assert_eq!(replay.outstanding_count(), 0);
    assert_eq!(replay.result(), Err(LoopError::IncompatibleHistory));
    assert_eq!(
        replay.complete(Completion::Tool(ToolCompletion {
            request_id: other.request_id().into(),
            results: vec![("new".into(), "local".into())]
        })),
        Err(LoopError::IncompatibleHistory)
    );
    assert_eq!(snapshot.root().env_ids, ["old"]);
}
