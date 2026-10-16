use lockstep_loop::{
    BatchLoop, Completion, History, LlmCompletion, LlmRequest, LoopConfig, LoopError, Request,
    RequestKind, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

fn specs() -> Vec<ToolSpec> {
    vec![ToolSpec::new(
        "read",
        "read-only",
        json!({"type": "object"}),
    )]
}

fn config(envs: &[&str], history: Option<History<String>>) -> LoopConfig<String> {
    let config = LoopConfig::new(
        "inspect".into(),
        envs.iter().map(|s| (*s).into()).collect(),
        specs(),
    );
    match history {
        Some(h) => config.with_replay(h, "inspect/read/exact/v1"),
        None => config.with_recording("inspect/read/exact/v1"),
    }
}

fn drive<K: Eq + Hash + fmt::Debug + 'static>(
    loop_: &mut BatchLoop<String, K>,
    values: &[(&str, &[(&str, &str)])],
    turns: usize,
) -> Vec<LlmRequest<String>> {
    let values: BTreeMap<_, _> = values.iter().copied().collect();
    let mut llms = Vec::new();
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                let (content, tool_calls) = if r.stats.depth < turns {
                    (
                        format!("think {}", r.stats.depth),
                        vec![
                            ToolCall::new("a", "read", json!({"step": r.stats.depth, "slot": 0})),
                            ToolCall::new("b", "read", json!({"step": r.stats.depth, "slot": 1})),
                        ],
                    )
                } else {
                    (r.messages.last().unwrap().content.clone(), vec![])
                };
                loop_
                    .complete(Completion::Llm(LlmCompletion {
                        request_id: r.request_id.clone(),
                        content,
                        tool_calls,
                    }))
                    .unwrap();
                llms.push(r);
            }
            Request::Tool(first) => {
                let Request::Tool(second) = loop_.next_request(RequestKind::Tool).unwrap() else {
                    panic!("expected second parallel tool")
                };
                for tool in [second, first] {
                    let step = tool.tool_call.args["step"].as_u64().unwrap() as usize;
                    let slot = tool.tool_call.args["slot"].as_u64().unwrap();
                    let results = tool
                        .env_ids
                        .iter()
                        .rev()
                        .map(|env| {
                            let pair = values[env.as_str()][step];
                            (
                                env.clone(),
                                if slot == 0 { pair.0 } else { pair.1 }.to_owned(),
                            )
                        })
                        .collect();
                    loop_
                        .complete(Completion::Tool(ToolCompletion {
                            request_id: tool.request_id,
                            results,
                        }))
                        .unwrap();
                }
            }
        }
    }
    llms
}

#[test]
fn non_splitting_checkpoints_replay_branch_and_share_old_paths() {
    let mut loop_ = BatchLoop::new(config(&["old"], None)).unwrap();
    let empty = loop_.history().unwrap();
    let values = [("same", "one"), ("same", "two")];
    assert_eq!(drive(&mut loop_, &[("old", &values)], 2).len(), 3);
    let result = loop_.result().unwrap();
    assert!(result.tree.children.is_empty());
    assert!(empty.root.assistant.is_none());
    let old = result.history.unwrap();
    let first = &old.root.observations[0];
    let second = &first.continuation.observations[0];
    assert_eq!(
        second.continuation.assistant.as_ref().unwrap().content,
        "two"
    );

    let mut replay = BatchLoop::new(config(&["new"], Some(old.clone()))).unwrap();
    assert!(drive(&mut replay, &[("new", &values)], 2).is_empty());
    assert_eq!(replay.result().unwrap().results["new"], "two");
    assert_eq!(first.evidence.len(), 1);
    assert_eq!(
        replay.history().unwrap().root.observations[0]
            .evidence
            .len(),
        2
    );

    let changed_values = [("same", "one"), ("same", "CHANGED")];
    let mut changed = BatchLoop::new(config(&["tomorrow"], Some(old.clone()))).unwrap();
    let requests = drive(&mut changed, &[("tomorrow", &changed_values)], 2);
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].stats.depth, 2);
    assert_eq!(requests[0].messages.last().unwrap().content, "CHANGED");
    let extended = changed.history().unwrap();
    let edges = &extended.root.observations[0].continuation.observations;
    assert_eq!(edges.len(), 2);
    assert!(Arc::ptr_eq(&edges[0].continuation, &second.continuation));
    assert_eq!(first.continuation.observations.len(), 1);
    for observations in [&values, &changed_values] {
        let mut again = BatchLoop::new(config(&["again"], Some(extended.clone()))).unwrap();
        assert!(drive(&mut again, &[("again", observations)], 2).is_empty());
    }
    let mut early = BatchLoop::new(config(&["early"], Some(old))).unwrap();
    let requests = drive(
        &mut early,
        &[("early", &[("different", "first"), ("later", "second")])],
        2,
    );
    assert_eq!(
        requests.iter().map(|r| r.stats.depth).collect::<Vec<_>>(),
        [1, 2]
    );
}

#[test]
fn concurrent_branches_and_round_trip_keep_all_paths() {
    let mut old_loop = BatchLoop::new(config(&["old"], None)).unwrap();
    drive(&mut old_loop, &[("old", &[("x", "y")])], 1);
    let old = old_loop.history().unwrap();
    let mut extended = BatchLoop::new(config(&["same", "a", "b"], Some(old.clone()))).unwrap();
    assert_eq!(
        drive(
            &mut extended,
            &[
                ("same", &[("x", "y")]),
                ("a", &[("x", "a")]),
                ("b", &[("x", "b")]),
            ],
            1
        )
        .len(),
        2
    );
    let result = extended.result().unwrap();
    assert_eq!(result.classes.len(), 3);
    let history = result.history.unwrap();
    assert_eq!(history.root.observations.len(), 3);
    assert_eq!(old.root.observations.len(), 1);
    // Codec and storage belong entirely to the caller. No core I/O.
    let encoded = serde_json::to_vec(&history).unwrap();
    let restored: History<String> = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored, history);
    for value in ["y", "a", "b"] {
        let mut replay = BatchLoop::new(config(&["next"], Some(restored.clone()))).unwrap();
        assert!(drive(&mut replay, &[("next", &[("x", value)])], 1).is_empty());
    }
}

#[test]
fn lossy_grouping_preserves_actual_evidence_and_original_conversation() {
    let mut loop_ =
        BatchLoop::with_group_by(config(&["a", "b"], None), |s: &String| s.trim().to_owned())
            .unwrap();
    drive(
        &mut loop_,
        &[("a", &[("one ", "two ")]), ("b", &[("one", "two")])],
        1,
    );
    let history = loop_.history().unwrap();
    let edge = &history.root.observations[0];
    assert_eq!(edge.representative[1].tool_output, "two ");
    assert_eq!(edge.evidence[1].outputs[1].tool_output, "two");
    let mut replay =
        BatchLoop::with_group_by(config(&["c"], Some(history.clone())), |s: &String| {
            s.trim().to_owned()
        })
        .unwrap();
    assert!(drive(&mut replay, &[("c", &[("one\n", "two\n")])], 1).is_empty());
    assert_eq!(replay.result().unwrap().results["c"], "two ");
    assert_eq!(
        replay.history().unwrap().root.observations[0].evidence[2].outputs[1].tool_output,
        "two\n"
    );
    assert_eq!(edge.evidence.len(), 2);
}

#[derive(Clone, Eq, PartialEq)]
struct Collision(&'static str);
impl fmt::Debug for Collision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("identical-debug-string")
    }
}
impl Hash for Collision {
    fn hash<H: Hasher>(&self, state: &mut H) {
        0.hash(state);
    }
}

#[test]
fn hash_and_debug_collisions_cannot_conflate_records() {
    let mut loop_ = BatchLoop::with_group_by(config(&["a", "b"], None), |s: &String| {
        if s == "a" {
            Collision("a")
        } else {
            Collision("b")
        }
    })
    .unwrap();
    drive(&mut loop_, &[("a", &[("a", "a")]), ("b", &[("b", "b")])], 1);
    let result = loop_.result().unwrap();
    assert_eq!(result.results["a"], "a");
    assert_eq!(result.results["b"], "b");
    assert_ne!(result.classes[0].id, result.classes[1].id);
    assert_ne!(
        result.tree.children[0].group_key,
        result.tree.children[1].group_key
    );
    let history = result.history.unwrap();
    assert_eq!(history.root.observations.len(), 2);
    for value in ["a", "b"] {
        let mut replay =
            BatchLoop::with_group_by(config(&["next"], Some(history.clone())), |s: &String| {
                if s == "a" {
                    Collision("a")
                } else {
                    Collision("b")
                }
            })
            .unwrap();
        assert!(drive(&mut replay, &[("next", &[(value, value)])], 1).is_empty());
        assert_eq!(replay.result().unwrap().results["next"], value);
    }
}

#[test]
fn invalid_completions_do_not_fulfill_history_and_can_be_corrected() {
    let mut loop_ = BatchLoop::new(config(&["a", "b"], None)).unwrap();
    let Request::Llm(llm) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    let empty = loop_.history().unwrap();
    let call = ToolCall::new("dup", "read", json!({}));
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: llm.request_id.clone(),
            content: "bad".into(),
            tool_calls: vec![call.clone(), call.clone()],
        })),
        Err(LoopError::DuplicateToolCallId(_))
    ));
    assert!(Arc::ptr_eq(&empty.root, &loop_.history().unwrap().root));
    assert_eq!(loop_.outstanding_llm(), 1);
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: llm.request_id,
            content: "ok".into(),
            tool_calls: vec![call],
        }))
        .unwrap();
    let Request::Tool(tool) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    let snapshot = loop_.history().unwrap();
    for results in [
        vec![("a".into(), "x".into())],
        vec![("a".into(), "x".into()), ("a".into(), "x".into())],
    ] {
        assert!(matches!(
            loop_.complete(Completion::Tool(ToolCompletion {
                request_id: tool.request_id.clone(),
                results,
            })),
            Err(LoopError::InvalidToolCompletion(_))
        ));
        assert_eq!(loop_.outstanding_tool(), 1);
        assert!(Arc::ptr_eq(&snapshot.root, &loop_.history().unwrap().root));
    }
    let valid = Completion::Tool(ToolCompletion {
        request_id: tool.request_id,
        results: vec![("a".into(), "x".into()), ("b".into(), "x".into())],
    });
    loop_.complete(valid.clone()).unwrap();
    assert!(matches!(
        loop_.complete(valid),
        Err(LoopError::UnknownRequest(_))
    ));
    let Request::Llm(llm) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            llm.request_id,
            "done".into(),
        )))
        .unwrap();
    assert!(loop_.is_done());
    assert!(snapshot.root.observations.is_empty());
}

#[test]
fn explicit_contract_checks_final_only_replay_and_unsupported_budget() {
    let mut loop_ = BatchLoop::new(config(&["old"], None)).unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "done".into(),
        )))
        .unwrap();
    let history = loop_.history().unwrap();
    let mut replay = BatchLoop::new(config(&["new"], Some(history.clone()))).unwrap();
    assert!(replay.is_done());
    assert!(replay.next_request(RequestKind::Any).is_none());
    assert_eq!(replay.result().unwrap().results["new"], "done");
    let changed = config(&["new"], None).with_replay(history.clone(), "changed");
    assert!(matches!(
        BatchLoop::new(changed),
        Err(LoopError::IncompatibleHistory)
    ));
    let mut changed = config(&["new"], Some(history.clone()));
    changed.prompt = "changed prompt".into();
    assert!(matches!(
        BatchLoop::new(changed),
        Err(LoopError::IncompatibleHistory)
    ));
    let mut changed = config(&["new"], Some(history.clone()));
    changed.tool_specs.clear();
    assert!(matches!(
        BatchLoop::new(changed),
        Err(LoopError::IncompatibleHistory)
    ));
    assert!(matches!(
        BatchLoop::with_group_by(config(&["new"], Some(history)), Clone::clone),
        Err(LoopError::IncompatibleHistory)
    ));
    assert!(matches!(
        BatchLoop::new(
            config(&["new"], None)
                .with_budget(lockstep_loop::Budget::new(1, |_| "exhausted".into()))
        ),
        Err(LoopError::UnsupportedHistory)
    ));
}

#[test]
fn reverse_branch_completions_keep_siblings_and_never_merge() {
    let mut loop_ = BatchLoop::new(config(&["a", "b"], None)).unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "look".into(),
            tool_calls: vec![ToolCall::new("r", "read", json!({}))],
        }))
        .unwrap();
    let tool = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id().into(),
            results: vec![("a".into(), "one".into()), ("b".into(), "two".into())],
        }))
        .unwrap();
    let before = loop_.history().unwrap();
    let first = loop_.next_request(RequestKind::Any).unwrap();
    let second = loop_.next_request(RequestKind::Any).unwrap();
    for request in [second, first] {
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id(),
                "same final".into(),
            )))
            .unwrap();
    }
    let result = loop_.result().unwrap();
    assert_eq!(result.classes.len(), 2);
    assert!(
        before
            .root
            .observations
            .iter()
            .all(|e| e.continuation.assistant.is_none())
    );
    assert!(
        result
            .history
            .unwrap()
            .root
            .observations
            .iter()
            .all(|e| { e.continuation.assistant.as_ref().unwrap().content == "same final" })
    );
}

#[test]
fn opaque_bytes_and_fulfilled_assistant_snapshot_restart_tools() {
    let make_config = |env: &str| LoopConfig::new(vec![0, 255], vec![env.into()], specs());
    let mut loop_ = BatchLoop::new(make_config("old").with_recording("bytes-v1")).unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: vec![1, 254],
            tool_calls: vec![ToolCall::new("read", "read", json!({}))],
        }))
        .unwrap();
    let snapshot = loop_.history().unwrap();
    let mut replay =
        BatchLoop::new(make_config("new").with_replay(snapshot.clone(), "bytes-v1")).unwrap();
    let Request::Tool(tool) = replay.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    replay
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![("new".into(), vec![2, 253])],
        }))
        .unwrap();
    let request = replay.next_request(RequestKind::Any).unwrap();
    replay
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            vec![3, 252],
        )))
        .unwrap();
    let history = replay.history().unwrap();
    assert_eq!(
        history.root.observations[0].evidence[0].outputs[0].tool_output,
        [2, 253]
    );
    let restored: History<Vec<u8>> =
        serde_json::from_slice(&serde_json::to_vec(&history).unwrap()).unwrap();
    assert_eq!(restored, history);
    let mut again = BatchLoop::new(make_config("again").with_replay(restored, "bytes-v1")).unwrap();
    let Request::Tool(tool) = again.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    again
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![("again".into(), vec![2, 253])],
        }))
        .unwrap();
    assert!(again.is_done());
    assert_eq!(again.result().unwrap().results["again"], [3, 252]);
    assert!(snapshot.root.observations.is_empty());
}
