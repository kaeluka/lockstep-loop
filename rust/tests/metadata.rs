use lockstep_loop::*;
use serde_json::json;
use std::sync::Arc;

fn metadata(model: &str) -> Metadata {
    Arc::new(
        json!({"model": model, "tags": ["runtime-only"]})
            .as_object()
            .unwrap()
            .clone(),
    )
}

#[test]
fn root_metadata_survives_splits_replay_and_replacement_without_deep_cloning() {
    let original = metadata("original");
    let agent = Agent::new(
        AgentConfig::new(
            vec![MessageEnvelope::user("question".to_owned())],
            vec![ToolSpec::new("read", "", json!({}))],
        )
        .with_metadata(original.clone()),
    )
    .unwrap();
    let mut run = agent
        .start(vec!["a".into(), "b".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Any).unwrap();
    assert!(Arc::ptr_eq(request.metadata(), &original));
    run.complete(Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: "inspect".into(),
        tool_calls: vec![ToolCall::new("r", "read", json!({}))],
    }))
    .unwrap();
    let Request::Tool(request) = run.next_request(RequestKind::Any).unwrap() else {
        panic!("expected tool")
    };
    assert!(Arc::ptr_eq(request.metadata(), &original));
    run.complete(Completion::Tool(ToolCompletion {
        request_id: request.request_id,
        results: vec![("a".into(), "one".into()), ("b".into(), "two".into())],
    }))
    .unwrap();
    for _ in 0..2 {
        let Request::Llm(request) = run.next_request(RequestKind::Any).unwrap() else {
            panic!("expected split model request")
        };
        assert!(Arc::ptr_eq(request.metadata(), &original));
        run.complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            "answer".into(),
        )))
        .unwrap();
    }
    assert!(run.is_done());

    let snapshot = run.snapshot();
    assert!(
        !serde_json::to_string(&snapshot.descriptor)
            .unwrap()
            .contains("runtime-only")
    );
    let current = metadata("current");
    let restored = Agent::from_snapshot(
        snapshot.clone(),
        AgentRuntime::default().with_metadata(current.clone()),
    )
    .unwrap();
    let mut replay = restored
        .start(vec!["c".into()], RunOptions::default())
        .unwrap();
    let request = replay.next_request(RequestKind::Any).unwrap();
    assert!(Arc::ptr_eq(request.metadata(), &current));
    replay
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("c".into(), "one".into())],
        }))
        .unwrap();
    assert!(replay.is_done());
    let mut changed = restored
        .start(vec!["d".into()], RunOptions::default())
        .unwrap();
    let request = changed.next_request(RequestKind::Any).unwrap();
    changed
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("d".into(), "different".into())],
        }))
        .unwrap();
    let Request::Llm(request) = changed.next_request(RequestKind::Any).unwrap() else {
        panic!("expected fresh reasoning")
    };
    assert!(Arc::ptr_eq(request.metadata(), &current));
    let default = Agent::from_snapshot(snapshot, AgentRuntime::default()).unwrap();
    let mut run = default
        .start(vec!["e".into()], RunOptions::default())
        .unwrap();
    assert!(
        run.next_request(RequestKind::Any)
            .unwrap()
            .metadata()
            .is_empty()
    );
}

#[test]
fn nested_children_keep_own_metadata_and_unconfigured_sibling_does_not_inherit() {
    let leaf = SubagentTool::new(
        ToolSpec::new("leaf", "", json!({})),
        |_| vec![MessageEnvelope::user("leaf".to_owned())],
        |_| vec![],
        8,
    )
    .with_subagents(vec![])
    .with_metadata(metadata("leaf"));
    let child = SubagentTool::new(
        ToolSpec::new("child", "", json!({})),
        |_| vec![MessageEnvelope::user("child".to_owned())],
        |_| vec![],
        2,
    )
    .with_subagents(vec![leaf])
    .with_metadata(metadata("child"));
    let sibling = SubagentTool::new(
        ToolSpec::new("sibling", "", json!({})),
        |_| vec![MessageEnvelope::user("sibling".to_owned())],
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    // Exercise the lower-level config as well as AgentConfig.
    let mut run = BatchLoop::new(
        LoopConfig::new(
            vec![MessageEnvelope::user("parent".to_owned())],
            vec!["e".into()],
            vec![],
        )
        .with_metadata(metadata("parent"))
        .with_subagents(vec![child, sibling]),
    )
    .unwrap();
    let req = run.next_request(RequestKind::Any).unwrap();
    assert_eq!(req.metadata()["model"], "parent");
    run.complete(Completion::Llm(LlmCompletion {
        request_id: req.request_id().into(),
        content: "delegate".into(),
        tool_calls: vec![
            ToolCall::new("c", "child", json!({})),
            ToolCall::new("s", "sibling", json!({})),
        ],
    }))
    .unwrap();
    let req = run.next_request(RequestKind::Any).unwrap();
    assert_eq!(req.metadata()["model"], "child");
    run.complete(Completion::Llm(LlmCompletion {
        request_id: req.request_id().into(),
        content: "delegate".into(),
        tool_calls: vec![ToolCall::new("l", "leaf", json!({}))],
    }))
    .unwrap();
    for expected in [Some("leaf"), Some("child"), None, Some("parent")] {
        let req = run.next_request(RequestKind::Any).unwrap();
        assert_eq!(
            req.metadata().get("model").and_then(|v| v.as_str()),
            expected
        );
        run.complete(Completion::Llm(LlmCompletion::final_answer(
            req.request_id(),
            "answer".into(),
        )))
        .unwrap();
    }
    assert!(run.is_done());
}
