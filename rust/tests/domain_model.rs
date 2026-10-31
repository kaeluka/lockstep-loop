use lockstep_loop::{
    Agent, AgentConfig, AgentDescriptor, AgentResolver, AgentRuntime, BatchLoop, Completion,
    LlmCompletion, LoopConfig, LoopError, MessageEnvelope, Request, RequestKind, Role, RunOptions,
    SubagentTool, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

fn spec(name: &str) -> ToolSpec {
    ToolSpec::new(name, "", json!({}))
}

#[test]
fn initial_messages_are_nonempty_ordered_opaque_envelopes_and_roundtrip() {
    let initial = vec![
        MessageEnvelope::system(vec![0, 255, 3]),
        MessageEnvelope::user(vec![12, 0]),
        MessageEnvelope::user(vec![128, 255]),
    ];
    let agent = Agent::new(AgentConfig::new(initial.clone(), vec![])).unwrap();
    assert_eq!(agent.snapshot().initial_messages(), initial);
    let mut run = agent
        .start(vec!["a".into()], RunOptions::default())
        .unwrap();
    let Request::Llm(request) = run.next_request(RequestKind::Llm).unwrap() else {
        panic!("expected model request");
    };
    assert_eq!(request.messages, initial);
    let answer = vec![0, 128, 64];
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        request.request_id,
        answer.clone(),
    )))
    .unwrap();
    let original = run.snapshot();
    let conversation = original.conversations().next().unwrap();
    assert_eq!(&conversation.messages()[..3], initial);
    assert_eq!(conversation.get_last_answer_asset(), Some(&answer));
    let descriptor: AgentDescriptor<Vec<u8>> =
        serde_json::from_slice(&serde_json::to_vec(&original.descriptor).unwrap()).unwrap();
    let restored = Agent::from_snapshot(
        lockstep_loop::AgentSnapshot {
            descriptor,
            store: original.store.clone(),
        },
        AgentRuntime::default(),
    )
    .unwrap();
    let mut replay = restored
        .start(vec!["b".into()], RunOptions::default())
        .unwrap();
    assert!(replay.is_done());
    assert!(replay.next_request(RequestKind::Any).is_none());
    assert_eq!(replay.result().unwrap().results["b"], answer);
    assert_eq!(original.root().env_ids, ["a"]);

    assert!(matches!(
        Agent::<String>::new(AgentConfig::new(vec![], vec![])),
        Err(LoopError::EmptyInitialMessages)
    ));
    assert!(matches!(
        BatchLoop::<String>::new(LoopConfig::new(vec![], vec!["e".into()], vec![])),
        Err(LoopError::EmptyInitialMessages)
    ));
    let mut empty_snapshot = original;
    empty_snapshot.descriptor.initial_messages.clear();
    assert!(matches!(
        Agent::from_snapshot(empty_snapshot, AgentRuntime::default()),
        Err(LoopError::EmptyInitialMessages)
    ));
}

#[test]
fn compatibility_checks_every_initial_envelope_and_its_structure() {
    let initial = vec![
        MessageEnvelope::system("system".to_owned()),
        MessageEnvelope::user("one".to_owned()),
        MessageEnvelope::user("two".to_owned()),
    ];
    let agent = Agent::new(AgentConfig::new(initial.clone(), vec![])).unwrap();
    let snapshot = agent.snapshot();
    let mut variants = Vec::new();
    let mut changed = initial.clone();
    changed[2].content = "changed".into();
    variants.push(changed);
    let mut changed = initial.clone();
    changed[0].role = Role::User;
    variants.push(changed);
    let mut changed = initial.clone();
    changed.swap(1, 2);
    variants.push(changed);
    let mut changed = initial.clone();
    changed.pop();
    variants.push(changed);
    let mut changed = initial.clone();
    changed[1]
        .tool_calls
        .push(ToolCall::new("id", "tool", json!({})));
    variants.push(changed);
    let mut changed = initial.clone();
    changed[2].tool_call_id = Some("other".into());
    variants.push(changed);
    for variant in variants {
        assert!(matches!(
            BatchLoop::new(
                LoopConfig::new(variant, vec!["e".into()], vec![]).with_snapshot(snapshot.clone())
            ),
            Err(LoopError::IncompatibleHistory)
        ));
    }
    assert!(
        BatchLoop::new(LoopConfig::new(initial, vec!["e".into()], vec![]).with_snapshot(snapshot))
            .is_ok()
    );
}

#[test]
fn child_lookup_and_replay_use_the_complete_initial_sequence() {
    let child_initial = vec![
        MessageEnvelope::system("child system".to_owned()),
        MessageEnvelope::user("child input".to_owned()),
        MessageEnvelope::user("child question".to_owned()),
    ];
    let child = Agent::new(AgentConfig::new(child_initial.clone(), vec![])).unwrap();
    let mut child_run = child
        .start(vec!["old".into()], RunOptions::default())
        .unwrap();
    let request = child_run.next_request(RequestKind::Llm).unwrap();
    child_run
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "child answer".to_owned(),
        )))
        .unwrap();
    let child_snapshot = child.snapshot();
    let candidate = child_snapshot.descriptor.clone();
    let queries = Arc::new(Mutex::new(Vec::new()));
    let seen = queries.clone();
    let resolver = AgentResolver::new(move |query| {
        seen.lock().unwrap().push(query.clone());
        Ok(Some(candidate.clone()))
    });
    let factory_messages = child_initial.clone();
    let tool = SubagentTool::new(
        spec("child"),
        move |_| factory_messages.clone(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let parent = Agent::new(
        AgentConfig::new(
            vec![MessageEnvelope::user("unrelated parent".to_owned())],
            vec![],
        )
        .with_subagents(vec![tool.clone()])
        .with_record_store(child_snapshot.store.clone())
        .with_agent_resolver(resolver.clone()),
    )
    .unwrap();
    let mut run = parent
        .start(vec!["new".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    run.complete(Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: "delegate".into(),
        tool_calls: vec![ToolCall::new("c", "child", json!({}))],
    }))
    .unwrap();
    let Request::Llm(request) = run.next_request(RequestKind::Any).unwrap() else {
        panic!("cached child should return to parent");
    };
    assert_eq!(request.messages[0].content, "unrelated parent");
    assert_eq!(request.stats.depth, 1);
    assert_eq!(request.messages.last().unwrap().content, "child answer");
    assert_eq!(queries.lock().unwrap()[0].initial_messages, child_initial);
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        request.request_id,
        "done".into(),
    )))
    .unwrap();
    assert_eq!(run.result().unwrap().child_updates.len(), 1);

    // A resolver hit with only the first child message equal is incompatible.
    let mut changed = child_initial;
    changed[2].content = "different question".into();
    let changed_tool = SubagentTool::new(spec("child"), move |_| changed.clone(), |_| vec![], 1)
        .with_subagents(vec![]);
    let other_parent = Agent::new(
        AgentConfig::new(
            vec![MessageEnvelope::user("other parent".to_owned())],
            vec![],
        )
        .with_subagents(vec![changed_tool])
        .with_record_store(child_snapshot.store)
        .with_agent_resolver(resolver),
    )
    .unwrap();
    let mut run = other_parent
        .start(vec!["new".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    let completion = Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: "delegate".into(),
        tool_calls: vec![ToolCall::new("c", "child", json!({}))],
    });
    assert_eq!(
        run.complete(completion),
        Err(LoopError::IncompatibleHistory)
    );
    assert_eq!(run.outstanding_llm(), 1);
    assert!(other_parent.snapshot().root().assistant.is_none());
}

#[test]
fn empty_child_initial_sequence_does_not_consume_parent_request() {
    let tool = SubagentTool::new(spec("child"), |_| vec![], |_| vec![], 1).with_subagents(vec![]);
    let parent = Agent::new(
        AgentConfig::new(vec![MessageEnvelope::user("parent".to_owned())], vec![])
            .with_subagents(vec![tool]),
    )
    .unwrap();
    let mut run = parent
        .start(vec!["e".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    assert_eq!(
        run.complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "delegate".into(),
            tool_calls: vec![ToolCall::new("c", "child", json!({}))],
        })),
        Err(LoopError::EmptyInitialMessages)
    );
    assert_eq!(run.outstanding_llm(), 1);
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        request.request_id(),
        "corrected".into(),
    )))
    .unwrap();
    assert!(run.is_done());
}

#[test]
fn asset_accessors_preserve_missing_null_and_latest_named_call_semantics() {
    let agent = Agent::new(AgentConfig::new(
        vec![MessageEnvelope::user(Some("start".to_owned()))],
        vec![spec("submitFinalAnswer"), spec("other")],
    ))
    .unwrap();
    let empty = agent.snapshot();
    let empty_conversation = empty.conversations().next().unwrap();
    assert_eq!(empty_conversation.get_last_answer_asset(), None);
    assert_eq!(
        empty_conversation.get_final_tool_input_asset("submitFinalAnswer"),
        None
    );
    let mut run = agent
        .start(vec!["a".into(), "b".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    run.complete(Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: Some("submit".into()),
        tool_calls: vec![
            ToolCall::new("first", "submitFinalAnswer", json!({"answer": 1})),
            ToolCall::new("second", "submitFinalAnswer", json!({"answer": 2})),
            ToolCall::new("third", "other", json!(3)),
        ],
    }))
    .unwrap();
    let pending = run.snapshot();
    let conversation = pending.conversations().next().unwrap();
    assert_eq!(
        conversation.get_final_tool_input_asset("submitFinalAnswer"),
        Some(&json!({"answer": 2}))
    );
    assert_eq!(conversation.get_last_answer_asset(), None);
    let mut requests = Vec::new();
    while let Some(Request::Tool(request)) = run.next_request(RequestKind::Tool) {
        requests.push(request);
    }
    for request in requests.into_iter().rev() {
        run.complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id,
            results: request
                .env_ids
                .into_iter()
                .map(|env| (env, Some("ack, not asset".into())))
                .collect(),
        }))
        .unwrap();
    }
    let request = run.next_request(RequestKind::Llm).unwrap();
    run.complete(Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: None,
        tool_calls: vec![ToolCall::new("fourth", "submitFinalAnswer", Value::Null)],
    }))
    .unwrap();
    let Request::Tool(request) = run.next_request(RequestKind::Tool).unwrap() else {
        panic!()
    };
    run.complete(Completion::Tool(ToolCompletion {
        request_id: request.request_id,
        results: request
            .env_ids
            .into_iter()
            .map(|env| (env, Some("ack".into())))
            .collect(),
    }))
    .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        request.request_id(),
        None,
    )))
    .unwrap();
    let result = run.result().unwrap();
    assert_eq!(result.classes.len(), 1);
    assert_eq!(result.classes[0].env_ids, ["a", "b"]);
    let conversation = result.snapshot.conversations().next().unwrap();
    assert_eq!(conversation.observations().len(), 2); // Both unsplit checkpoints persist.
    assert_eq!(conversation.get_last_answer_asset(), Some(&None));
    assert_eq!(
        conversation.get_final_tool_input_asset("submitFinalAnswer"),
        Some(&Value::Null)
    );
    assert_eq!(conversation.get_final_tool_input_asset("absent"), None);
    assert_eq!(
        pending
            .conversations()
            .next()
            .unwrap()
            .get_final_tool_input_asset("submitFinalAnswer"),
        Some(&json!({"answer": 2}))
    );

    let mut replay = agent
        .start(vec!["c".into()], RunOptions::default())
        .unwrap();
    while let Some(request) = replay.next_request(RequestKind::Any) {
        let Request::Tool(request) = request else {
            panic!("unchanged results must reuse reasoning")
        };
        let output = if request.tool_call.id == "fourth" {
            "ack"
        } else {
            "ack, not asset"
        };
        replay
            .complete(Completion::Tool(ToolCompletion {
                request_id: request.request_id,
                results: vec![("c".into(), Some(output.into()))],
            }))
            .unwrap();
    }
    let replay_result = replay.result().unwrap();
    assert_eq!(replay_result.classes.len(), 1);
    assert_eq!(replay_result.classes[0].env_ids, ["c"]);
    assert_eq!(result.classes[0].env_ids, ["a", "b"]);
}

#[test]
fn conversation_accessors_include_initial_model_messages_without_cloning_content() {
    struct Opaque;
    let initial = vec![
        MessageEnvelope::system(Opaque),
        MessageEnvelope::assistant(Opaque, vec![]),
        MessageEnvelope::assistant(
            Opaque,
            vec![ToolCall::new(
                "f",
                "submitFinalAnswer",
                json!({"typed": [1, 2]}),
            )],
        ),
        MessageEnvelope::user(Opaque),
    ];
    let store = lockstep_loop::RecordStore::new("opaque-initial");
    let snapshot = lockstep_loop::AgentSnapshot {
        descriptor: AgentDescriptor {
            initial_messages: initial,
            tools: vec![],
            subagents: vec![],
            root: store.put(lockstep_loop::HistoryNode::default()),
        },
        store,
    };
    let conversation = snapshot.conversations().next().unwrap();
    assert!(conversation.get_last_answer_asset().is_some());
    assert_eq!(
        conversation.get_final_tool_input_asset("submitFinalAnswer"),
        Some(&json!({"typed": [1, 2]}))
    );
    assert!(!conversation.is_complete());
}
