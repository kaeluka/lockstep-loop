use lockstep_loop::{
    Agent, BatchLoop, Budget, Completion, LlmCompletion, LoopConfig, LoopError, Request,
    RequestKind, SubagentTool, ToolCall, ToolCompletion, ToolSpec, run_to_completion,
    run_to_completion_by,
};
use serde_json::json;

fn read() -> ToolSpec {
    ToolSpec::new("read", "", json!({}))
}
fn config(agent: Option<Agent<String>>, budget: Option<usize>, env: &str) -> LoopConfig<String> {
    let mut c = LoopConfig::new("p".into(), vec![env.into()], vec![read()]);
    if let Some(agent) = agent {
        c = c.with_agent(agent);
    }
    if let Some(limit) = budget {
        c = c.with_budget(Budget::new(limit, |steps| format!("exhausted:{steps}")));
    }
    c
}
fn answer(loop_: &mut BatchLoop<String>, content: &str, calls: Vec<ToolCall>) {
    let Request::Llm(r) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id,
            content: content.into(),
            tool_calls: calls,
        }))
        .unwrap();
}
fn observe(loop_: &mut BatchLoop<String>) {
    let Request::Tool(r) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id,
            results: r
                .env_ids
                .into_iter()
                .map(|e| (e, "observed".into()))
                .collect(),
        }))
        .unwrap();
}
fn observations(loop_: &mut BatchLoop<String>) {
    let requests = [
        loop_.next_request(RequestKind::Any).unwrap(),
        loop_.next_request(RequestKind::Any).unwrap(),
    ];
    for r in requests.into_iter().rev() {
        let Request::Tool(r) = r else { panic!() };
        loop_
            .complete(Completion::Tool(ToolCompletion {
                request_id: r.request_id,
                results: r
                    .env_ids
                    .into_iter()
                    .map(|e| (e, r.tool_call.id.clone()))
                    .collect(),
            }))
            .unwrap();
    }
}

#[test]
fn constructor_and_sync_driver_publish_immutable_agents() {
    let mut loop_ = BatchLoop::new(config(None, None, "e")).unwrap();
    let empty = loop_.agent();
    answer(&mut loop_, "answer", vec![]);
    let result = loop_.result().unwrap();
    assert_eq!(
        result.agent().root().assistant.as_ref().unwrap().content,
        "answer"
    );
    assert!(empty.root().assistant.is_none());
    let mut lower = BatchLoop::new(config(Some(result.into_agent()), Some(0), "new")).unwrap();
    assert!(lower.is_done());
    assert_eq!(lower.agent().root().env_ids, ["e"]);
    assert_eq!(lower.result().unwrap().results["new"], "exhausted:0");
    assert_eq!(
        lower.append_user_message("feedback".into()),
        Err(LoopError::UnresolvedConversation)
    );
    let result = run_to_completion(
        "p".to_owned(),
        vec!["e".into()],
        |_| vec![],
        |r| LlmCompletion::final_answer(&r.request_id, "answer".into()),
        None,
    )
    .unwrap();
    assert_eq!(
        result.agent().root().assistant.as_ref().unwrap().content,
        "answer"
    );
}

#[test]
fn zero_budget_is_an_open_agent_and_generic_fallback_needs_only_clone() {
    let mut loop_ = BatchLoop::new(config(None, Some(0), "e")).unwrap();
    assert!(loop_.is_done());
    assert!(loop_.next_request(RequestKind::Any).is_none());
    assert_eq!(
        loop_.append_user_message("feedback".into()),
        Err(LoopError::UnresolvedConversation)
    );
    let result = loop_.result().unwrap();
    assert_eq!(result.results["e"], "exhausted:0");
    assert!(result.agent().root().assistant.is_none());
    assert!(result.agent().root().env_ids.is_empty());
    let mut resumed = BatchLoop::new(config(Some(result.agent().clone()), None, "new")).unwrap();
    answer(&mut resumed, "real answer", vec![]);
    assert_eq!(resumed.result().unwrap().results["new"], "real answer");
    assert!(result.agent().root().assistant.is_none());

    // No PartialEq, Eq, Hash, Debug, Serialize, Send or Sync bound on payloads.
    #[derive(Clone)]
    struct Opaque(std::rc::Rc<usize>);
    let result = run_to_completion_by(
        Opaque(std::rc::Rc::new(10)),
        vec!["e".into()],
        |_| vec![],
        |_| panic!("zero budget must not call model"),
        |p| *p.0,
        Some(Budget::new(0, |steps| Opaque(std::rc::Rc::new(steps)))),
    )
    .unwrap();
    assert_eq!(*result.results["e"].0, 0);
    assert!(result.agent().root().assistant.is_none());
    assert_eq!(*result.agent().descriptor.prompt.0, 10);
}

#[test]
fn budget_checkpoint_resumes_and_replay_obeys_lower_limit() {
    let mut loop_ = BatchLoop::new(config(None, Some(1), "e")).unwrap();
    let empty = loop_.agent();
    answer(
        &mut loop_,
        "inspect",
        vec![
            ToolCall::new("one", "read", json!({})),
            ToolCall::new("two", "read", json!({})),
        ],
    );
    let pending = loop_.agent();
    observations(&mut loop_);
    let result = loop_.result().unwrap();
    assert_eq!(result.results["e"], "exhausted:1");
    let old = result.into_agent();
    let edge = old.root().observations[0].clone();
    assert!(old.load(&edge.continuation).assistant.is_none());
    assert_eq!(
        edge.evidence[0]
            .outputs
            .iter()
            .map(|o| o.tool_output.as_str())
            .collect::<Vec<_>>(),
        ["one", "two"]
    );
    assert!(empty.root().assistant.is_none());
    assert!(pending.root().observations.is_empty());

    let mut resumed = BatchLoop::new(config(Some(old.clone()), Some(2), "new")).unwrap();
    observations(&mut resumed);
    let Request::Llm(r) = resumed.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(r.stats.depth, 1);
    assert_eq!(
        r.messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        ["p", "inspect", "one", "two"]
    );
    resumed
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            "real answer".into(),
        )))
        .unwrap();
    let full = resumed.result().unwrap().into_agent();
    assert!(old.load(&edge.continuation).assistant.is_none());

    let mut lower = BatchLoop::new(config(Some(full.clone()), Some(1), "lower")).unwrap();
    observations(&mut lower);
    assert!(lower.is_done());
    assert!(lower.next_request(RequestKind::Any).is_none());
    assert_eq!(lower.result().unwrap().results["lower"], "exhausted:1");
    let snapshot = lower.agent();
    let endpoint = snapshot.load(&snapshot.root().observations[0].continuation);
    assert_eq!(endpoint.assistant.as_ref().unwrap().content, "real answer");
    assert_eq!(endpoint.env_ids, ["new"]); // not accepted by the lower-budget run
    for limit in [None, Some(2)] {
        let mut replay = BatchLoop::new(config(Some(snapshot.clone()), limit, "again")).unwrap();
        observations(&mut replay);
        assert!(replay.is_done());
        assert_eq!(replay.result().unwrap().results["again"], "real answer");
    }
    assert_eq!(
        full.load(&full.root().observations[0].continuation).env_ids,
        ["new"]
    );

    let mut feedback = BatchLoop::new(
        config(Some(full.clone()), Some(1), "feedback").with_feedback("revise".into()),
    )
    .unwrap();
    let requests = [
        feedback.next_request(RequestKind::Any).unwrap(),
        feedback.next_request(RequestKind::Any).unwrap(),
    ];
    for (i, r) in requests.into_iter().enumerate() {
        let Request::Tool(r) = r else { panic!() };
        let result = feedback.complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id,
            results: vec![("feedback".into(), r.tool_call.id)],
        }));
        assert_eq!(
            result,
            if i == 0 {
                Ok(())
            } else {
                Err(LoopError::FeedbackResolutionMiss)
            }
        );
    }
    assert_eq!(feedback.agent(), full);
}

#[test]
fn budget_after_feedback_preserves_open_continuation_and_counts_replayed_answers() {
    let mut original = BatchLoop::new(config(None, None, "e")).unwrap();
    answer(&mut original, "old answer", vec![]);
    let mut bounded =
        BatchLoop::new(config(Some(original.agent()), Some(1), "e").with_feedback("revise".into()))
            .unwrap();
    assert!(bounded.is_done());
    assert_eq!(bounded.result().unwrap().results["e"], "exhausted:1");
    let opened = bounded.agent();
    let continuation = opened.root().user_continuations[0].clone();
    assert_eq!(continuation.message.content, "revise");
    assert!(opened.load(&continuation.continuation).assistant.is_none());
    assert!(original.agent().root().user_continuations.is_empty());
    let mut resumed = BatchLoop::new(config(Some(opened.clone()), None, "e")).unwrap();
    let Request::Llm(r) = resumed.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(
        r.messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        ["p", "old answer", "revise"]
    );
    assert_eq!(r.stats.depth, 1);
    resumed
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            "new answer".into(),
        )))
        .unwrap();
    let full = resumed.agent();
    let mut lower = BatchLoop::new(config(Some(full.clone()), Some(1), "lower")).unwrap();
    assert!(lower.is_done());
    assert_eq!(lower.result().unwrap().results["lower"], "exhausted:1");
    let snapshot = lower.agent();
    let endpoint = snapshot.load(&snapshot.root().user_continuations[0].continuation);
    assert_eq!(endpoint.assistant.as_ref().unwrap().content, "new answer");
    assert_eq!(endpoint.env_ids, ["e"]);
    let mut replay = BatchLoop::new(config(Some(full), Some(2), "e")).unwrap();
    assert_eq!(replay.result().unwrap().results["e"], "new answer");
    assert!(opened.load(&continuation.continuation).assistant.is_none());
}

#[test]
fn child_budget_keeps_child_open_and_parent_records_actual_tool_result() {
    let sub = SubagentTool::new(
        ToolSpec::new("child", "", json!({})),
        |_| "child".to_owned(),
        |_| vec![read()],
        1,
    )
    .with_subagents(vec![]);
    let mut loop_ = BatchLoop::new(config(None, Some(1), "e").with_subagent(sub.clone())).unwrap();
    answer(
        &mut loop_,
        "delegate",
        vec![ToolCall::new("c", "child", json!({}))],
    );
    answer(
        &mut loop_,
        "inspect",
        vec![ToolCall::new("r", "read", json!({}))],
    );
    observe(&mut loop_);
    let result = loop_.result().unwrap();
    let child = result.child_agents()[0].agent();
    assert!(
        child
            .load(&child.root().observations[0].continuation)
            .assistant
            .is_none()
    );
    assert_eq!(
        result.agent().root().observations[0].representative[0].tool_output,
        result.results["e"]
    );
    assert_eq!(
        &result.agent().child(&result.agent().root().subagents[0]),
        child
    );

    let mut resumed =
        BatchLoop::new(config(Some(result.agent().clone()), Some(2), "new").with_subagent(sub))
            .unwrap();
    observe(&mut resumed);
    answer(&mut resumed, "child answer", vec![]);
    let Request::Llm(r) = resumed.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(r.messages.last().unwrap().content, "child answer");
    resumed
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            "parent answer".into(),
        )))
        .unwrap();
    assert_eq!(resumed.result().unwrap().results["new"], "parent answer");
    assert_eq!(resumed.agent().root().observations.len(), 2);
    assert_eq!(result.agent().root().observations.len(), 1);
    assert!(
        child
            .load(&child.root().observations[0].continuation)
            .assistant
            .is_none()
    );
}
