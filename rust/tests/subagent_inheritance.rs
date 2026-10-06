use lockstep_loop::{
    BatchLoop, Completion, LlmCompletion, LlmRequest, LoopConfig, Message, Request, RequestKind,
    SubagentTool, ToolCall, ToolCompletion, ToolRequest, ToolSpec,
};
use serde_json::json;

fn spec(name: &str) -> ToolSpec {
    ToolSpec::new(name, name, json!({}))
}

fn llm(loop_: &mut BatchLoop<String>) -> LlmRequest<String> {
    match loop_.next_request(RequestKind::Llm) {
        Some(Request::Llm(request)) => request,
        _ => panic!("expected an LLM request"),
    }
}

fn tool(loop_: &mut BatchLoop<String>) -> ToolRequest {
    match loop_.next_request(RequestKind::Tool) {
        Some(Request::Tool(request)) => request,
        _ => panic!("expected a tool request"),
    }
}

fn reply(loop_: &mut BatchLoop<String>, request: &LlmRequest<String>, content: &str) {
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            &request.request_id,
            content.to_owned(),
        )))
        .unwrap();
}

fn call(
    loop_: &mut BatchLoop<String>,
    request: &LlmRequest<String>,
    content: &str,
    calls: Vec<ToolCall>,
) {
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id.clone(),
            content: content.to_owned(),
            tool_calls: calls,
        }))
        .unwrap();
}

fn results(loop_: &mut BatchLoop<String>, request: &ToolRequest, values: &[(&str, &str)]) {
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id.clone(),
            results: values
                .iter()
                .map(|(env, value)| ((*env).to_owned(), (*value).to_owned()))
                .collect(),
        }))
        .unwrap();
}

#[test]
fn multiple_subagents_inherit_prefix_and_join_out_of_order() {
    for (inherit_session, ordinary_last) in
        [(false, false), (false, true), (true, false), (true, true)]
    {
        let envs = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let subagent = SubagentTool::new(
            spec("fork"),
            |args| args["task"].as_str().unwrap().to_owned(),
            |_| vec![spec("probe")],
            1,
        )
        .with_inherit_session(inherit_session);
        let mut loop_ = BatchLoop::new(
            LoopConfig::new(
                "main".to_owned(),
                envs.clone(),
                vec![spec("read"), spec("fork")],
            )
            .with_subagent(subagent),
        )
        .unwrap();

        // Include a fully answered tool turn in the inherited history.
        let first = llm(&mut loop_);
        let earlier_call = ToolCall::new("earlier", "read", json!({}));
        call(&mut loop_, &first, "look first", vec![earlier_call.clone()]);
        let earlier = tool(&mut loop_);
        results(
            &mut loop_,
            &earlier,
            &[("a", "context"), ("b", "context"), ("c", "context")],
        );
        let parent = llm(&mut loop_);
        let prefix = parent.messages.clone();
        assert_eq!(
            prefix,
            vec![
                Message::user("main".to_owned()),
                Message::assistant("look first".to_owned(), vec![earlier_call]),
                Message::tool("context".to_owned(), "earlier"),
            ]
        );

        let calls = vec![
            ToolCall::new("x", "fork", json!({"task": "do x"})),
            ToolCall::new("y", "fork", json!({"task": "do y"})),
            ToolCall::new("regular", "read", json!({})),
        ];
        call(&mut loop_, &parent, "fork now", calls.clone());
        let x = llm(&mut loop_);
        let y = llm(&mut loop_);
        let regular = tool(&mut loop_);
        assert_eq!(x.env_ids, envs);
        assert_eq!(y.env_ids, envs);
        assert_eq!(regular.env_ids, envs);
        for (request, task) in [(&x, "do x"), (&y, "do y")] {
            let mut expected = if inherit_session {
                prefix.clone()
            } else {
                vec![]
            };
            expected.push(Message::user(task.to_owned()));
            assert_eq!(request.messages, expected);
            assert_eq!(request.tools, vec![spec("probe")]);
        }
        assert_ne!(x.request_id, y.request_id);
        assert_ne!(x.request_id, regular.request_id);
        assert_ne!(y.request_id, regular.request_id);

        // Y splits and finishes first; X and the ordinary parent tool wait.
        let probe_call = ToolCall::new("p", "probe", json!({}));
        call(&mut loop_, &y, "", vec![probe_call.clone()]);
        let probe = tool(&mut loop_);
        assert_eq!(probe.tool_call, probe_call);
        results(
            &mut loop_,
            &probe,
            &[("a", "same"), ("b", "same"), ("c", "other")],
        );
        let y_same = llm(&mut loop_);
        let y_other = llm(&mut loop_);
        assert_eq!(y_same.env_ids, vec!["a", "b"]);
        assert_eq!(y_other.env_ids, vec!["c"]);
        reply(&mut loop_, &y_other, "Z");
        reply(&mut loop_, &y_same, "Y");
        assert!(loop_.next_request(RequestKind::Any).is_none());
        assert!(!loop_.is_done());
        assert_eq!(loop_.outstanding_count(), 2);

        let mut final_completions = vec![
            Completion::Tool(ToolCompletion {
                request_id: regular.request_id,
                results: envs
                    .iter()
                    .map(|env| (env.clone(), "ordinary".to_owned()))
                    .collect(),
            }),
            Completion::Llm(LlmCompletion::final_answer(&x.request_id, "X".to_owned())),
        ];
        if ordinary_last {
            final_completions.reverse();
        }
        let last = final_completions.pop().unwrap();
        loop_.complete(final_completions.pop().unwrap()).unwrap();
        assert!(loop_.next_request(RequestKind::Any).is_none());
        loop_.complete(last).unwrap();
        let same = llm(&mut loop_);
        let other = llm(&mut loop_);
        assert_eq!(same.env_ids, vec!["a", "b"]);
        assert_eq!(other.env_ids, vec!["c"]);
        for (request, answer_y) in [(&other, "Z"), (&same, "Y")] {
            let mut expected = prefix.clone();
            expected.extend([
                Message::assistant("fork now".to_owned(), calls.clone()),
                Message::tool("X".to_owned(), "x"),
                Message::tool(answer_y.to_owned(), "y"),
                Message::tool("ordinary".to_owned(), "regular"),
            ]);
            assert_eq!(request.messages, expected);
            reply(&mut loop_, request, &format!("X/{answer_y}"));
        }
        assert!(loop_.is_done());
        let result = loop_.result().unwrap();
        assert_eq!(result.results["a"], "X/Y");
        assert_eq!(result.results["b"], "X/Y");
        assert_eq!(result.results["c"], "X/Z");
    }
}

#[test]
fn inheritance_uses_each_parent_branch_history() {
    let subagent = SubagentTool::new(spec("fork"), |_| "child".to_owned(), |_| vec![], 1)
        .with_inherit_session(true);
    let mut loop_ = BatchLoop::new(
        LoopConfig::new(
            "main".to_owned(),
            vec!["a".to_owned(), "b".to_owned()],
            vec![spec("read"), spec("fork")],
        )
        .with_subagent(subagent),
    )
    .unwrap();
    let root = llm(&mut loop_);
    call(
        &mut loop_,
        &root,
        "",
        vec![ToolCall::new("read", "read", json!({}))],
    );
    let read = tool(&mut loop_);
    results(&mut loop_, &read, &[("a", "left"), ("b", "right")]);
    let parents = [llm(&mut loop_), llm(&mut loop_)];
    let mut prefixes = std::collections::HashMap::new();
    for parent in &parents {
        prefixes.insert(parent.env_ids.clone(), parent.messages.clone());
        call(
            &mut loop_,
            parent,
            "",
            vec![ToolCall::new("fork", "fork", json!({}))],
        );
    }
    let children = [llm(&mut loop_), llm(&mut loop_)];
    for child in children.iter().rev() {
        let mut expected = prefixes[&child.env_ids].clone();
        expected.push(Message::user("child".to_owned()));
        assert_eq!(child.messages, expected);
        reply(&mut loop_, child, "child answer");
    }
    for _ in &parents {
        let resumed = llm(&mut loop_);
        assert_eq!(
            resumed.messages[..resumed.messages.len() - 2],
            prefixes[&resumed.env_ids]
        );
        assert_eq!(
            resumed.messages.last(),
            Some(&Message::tool("child answer".to_owned(), "fork"))
        );
        reply(&mut loop_, &resumed, "done");
    }
    let result = loop_.result().unwrap();
    assert_eq!(result.results["a"], "done");
    assert_eq!(result.results["b"], "done");
}

#[test]
fn nested_subagents_preserve_inheritance_setting() {
    let subagent = SubagentTool::new(
        spec("fork"),
        |args| args["task"].as_str().unwrap().to_owned(),
        |_| vec![spec("fork")],
        2,
    )
    .with_inherit_session(true);
    assert!(subagent.clone().inherit_session);
    let mut loop_ = BatchLoop::new(
        LoopConfig::new("main".to_owned(), vec!["a".to_owned()], vec![spec("fork")])
            .with_subagent(subagent),
    )
    .unwrap();
    let parent = llm(&mut loop_);
    call(
        &mut loop_,
        &parent,
        "",
        vec![ToolCall::new("x", "fork", json!({"task": "child"}))],
    );
    let child = llm(&mut loop_);
    assert_eq!(
        child.messages,
        vec![
            Message::user("main".to_owned()),
            Message::user("child".to_owned())
        ]
    );
    call(
        &mut loop_,
        &child,
        "",
        vec![ToolCall::new("y", "fork", json!({"task": "grandchild"}))],
    );
    let grandchild = llm(&mut loop_);
    let mut expected = child.messages.clone();
    expected.push(Message::user("grandchild".to_owned()));
    assert_eq!(grandchild.messages, expected);
    reply(&mut loop_, &grandchild, "deep answer");
    let resumed_child = llm(&mut loop_);
    assert_eq!(
        resumed_child.messages.last(),
        Some(&Message::tool("deep answer".to_owned(), "y"))
    );
    reply(&mut loop_, &resumed_child, "child answer");
    let resumed_parent = llm(&mut loop_);
    assert_eq!(
        resumed_parent.messages.last(),
        Some(&Message::tool("child answer".to_owned(), "x"))
    );
    reply(&mut loop_, &resumed_parent, "done");
    assert_eq!(loop_.result().unwrap().results["a"], "done");
}
