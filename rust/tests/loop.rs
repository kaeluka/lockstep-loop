use lockstep_loop::{
    BatchLoop, Budget, Completion, FunctionTool, LlmCompletion, LoopConfig, LoopError, Message,
    Request, RequestKind, SubagentTool, ToolCall, ToolCompletion, ToolEntry, ToolSpec,
    run_to_completion, run_to_completion_by,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

fn spec(name: &str) -> ToolSpec {
    ToolSpec::new(name, "", json!({"type": "object"}))
}

fn constant_tool<P>(name: &str, value: P) -> ToolEntry<P>
where
    P: Clone + Send + Sync + 'static,
{
    ToolEntry::Callable(Arc::new(FunctionTool::new(spec(name), move |_| {
        value.clone()
    })))
}

#[test]
fn single_env_no_tool_calls() {
    let result = run_to_completion(
        "hi".to_owned(),
        vec!["a".to_owned()],
        |_| Vec::new(),
        |request| LlmCompletion::final_answer(&request.request_id, "done".to_owned()),
        None,
    )
    .unwrap();

    assert_eq!(result.results["a"], "done");
    assert_eq!(result.tree.env_ids, ["a"]);
    assert!(result.tree.children.is_empty());
    assert_eq!(result.classes[0].env_ids, ["a"]);
}

#[test]
fn splits_on_divergent_results_and_llm_requests_carry_env_sets() {
    let data = BTreeMap::from([("a", "hello"), ("b", "hello"), ("c", "goodbye")]);
    let mut request_envs = Vec::new();
    let result = run_to_completion(
        "read it".to_owned(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        |env| vec![constant_tool("read_file", data[env].to_owned())],
        |request| {
            request_envs.push(request.env_ids.clone());
            if request.messages.len() == 1 {
                LlmCompletion {
                    request_id: request.request_id.clone(),
                    content: String::new(),
                    tool_calls: vec![ToolCall::new("t1", "read_file", json!({}))],
                }
            } else {
                LlmCompletion::final_answer(
                    &request.request_id,
                    format!("saw:{}", request.messages.last().unwrap().content),
                )
            }
        },
        None,
    )
    .unwrap();

    assert_eq!(result.results["a"], "saw:hello");
    assert_eq!(result.results["b"], "saw:hello");
    assert_eq!(result.results["c"], "saw:goodbye");
    assert_eq!(request_envs[0], ["a", "b", "c"]);
    let mut later_sizes = request_envs[1..].iter().map(Vec::len).collect::<Vec<_>>();
    later_sizes.sort_unstable();
    assert_eq!(later_sizes, [1, 2]);
    assert_eq!(result.tree.children.len(), 2);
}

#[test]
fn multi_tool_turn_groups_by_result_tuple() {
    let data = BTreeMap::from([("a", ("1", "x")), ("b", ("1", "x")), ("c", ("1", "y"))]);
    let result = run_to_completion(
        "x".to_owned(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        |env| {
            vec![
                constant_tool("t1", data[env].0.to_owned()),
                constant_tool("t2", data[env].1.to_owned()),
            ]
        },
        |request| {
            if request.messages.len() == 1 {
                LlmCompletion {
                    request_id: request.request_id.clone(),
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall::new("c1", "t1", json!({})),
                        ToolCall::new("c2", "t2", json!({})),
                    ],
                }
            } else {
                LlmCompletion::final_answer(
                    &request.request_id,
                    request.messages.last().unwrap().content.clone(),
                )
            }
        },
        None,
    )
    .unwrap();

    assert_eq!(result.results["a"], "x");
    assert_eq!(result.results["b"], "x");
    assert_eq!(result.results["c"], "y");
    let mut classes = result
        .classes
        .iter()
        .map(|class| class.env_ids.clone())
        .collect::<Vec<_>>();
    classes.sort();
    assert_eq!(
        classes,
        [vec!["a".to_owned(), "b".to_owned()], vec!["c".to_owned()]]
    );
}

#[test]
fn opaque_byte_payloads_group_by_value() {
    let data = BTreeMap::from([
        ("a", b"PNG-a".to_vec()),
        ("b", b"PNG-a".to_vec()),
        ("c", b"PNG-b".to_vec()),
    ]);
    let result = run_to_completion(
        b"go".to_vec(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        |env| vec![constant_tool("snap", data[env].clone())],
        |request| {
            if request.messages.len() == 1 {
                LlmCompletion {
                    request_id: request.request_id.clone(),
                    content: Vec::new(),
                    tool_calls: vec![ToolCall::new("s", "snap", json!({}))],
                }
            } else {
                LlmCompletion::final_answer(
                    &request.request_id,
                    request.messages.last().unwrap().content.clone(),
                )
            }
        },
        None,
    )
    .unwrap();

    assert_eq!(result.results["a"], b"PNG-a");
    assert_eq!(result.results["c"], b"PNG-b");
}

#[test]
fn custom_grouping_can_merge_unequal_values() {
    let data = BTreeMap::from([("a", "result v1\n"), ("b", "result v1")]);
    let result = run_to_completion_by(
        "x".to_owned(),
        vec!["a".to_owned(), "b".to_owned()],
        |env| vec![constant_tool("t", data[env].to_owned())],
        |request| {
            if request.messages.len() == 1 {
                LlmCompletion {
                    request_id: request.request_id.clone(),
                    content: String::new(),
                    tool_calls: vec![ToolCall::new("t1", "t", json!({}))],
                }
            } else {
                LlmCompletion::final_answer(&request.request_id, "fin".to_owned())
            }
        },
        |value: &String| value.trim().to_owned(),
        None,
    )
    .unwrap();

    assert_eq!(result.classes.len(), 1);
    assert_eq!(result.classes[0].env_ids, ["a", "b"]);
}

#[test]
fn subagent_answers_refine_the_parent_branch() {
    let answers = Arc::new(BTreeMap::from([("a", "yes"), ("b", "yes"), ("c", "no")]));
    let subagent = SubagentTool::new(
        spec("investigate"),
        |args| format!("Answer: {}", args["question"].as_str().unwrap()),
        |_| vec![spec("probe")],
        1,
    );
    let factory_subagent = subagent.clone();
    let factory_answers = Arc::clone(&answers);
    let mut request_ids = Vec::new();
    let result = run_to_completion(
        "main".to_owned(),
        vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
        move |env| {
            vec![
                constant_tool("probe", factory_answers[env].to_owned()),
                ToolEntry::Subagent(factory_subagent.clone()),
            ]
        },
        |request| {
            request_ids.push(request.request_id.clone());
            if request.messages[0].content == "main" {
                if request.messages.len() == 1 {
                    LlmCompletion {
                        request_id: request.request_id.clone(),
                        content: String::new(),
                        tool_calls: vec![ToolCall::new(
                            "inv",
                            "investigate",
                            json!({"question": "q?"}),
                        )],
                    }
                } else {
                    LlmCompletion::final_answer(
                        &request.request_id,
                        format!("done:{}", request.messages.last().unwrap().content),
                    )
                }
            } else if request.messages.len() == 1 {
                LlmCompletion {
                    request_id: request.request_id.clone(),
                    content: String::new(),
                    tool_calls: vec![ToolCall::new("pr", "probe", json!({}))],
                }
            } else {
                LlmCompletion::final_answer(
                    &request.request_id,
                    request.messages.last().unwrap().content.clone(),
                )
            }
        },
        None,
    )
    .unwrap();

    assert_eq!(result.results["a"], "done:yes");
    assert_eq!(result.results["b"], "done:yes");
    assert_eq!(result.results["c"], "done:no");
    assert!(request_ids.iter().any(|id| id.starts_with("sub:root/inv/")));
    assert_eq!(
        request_ids
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        request_ids.len()
    );
}

#[test]
fn zero_subagent_depth_is_rejected() {
    let config = LoopConfig::new("x".to_owned(), vec!["e".to_owned()], Vec::new()).with_subagent(
        SubagentTool::new(spec("s"), |_| "p".to_owned(), |_| Vec::new(), 0),
    );
    assert!(matches!(
        BatchLoop::new(config),
        Err(LoopError::InvalidSubagentDepth)
    ));
}

#[test]
fn budget_uses_caller_supplied_generic_payload() {
    let result = run_to_completion(
        "x".to_owned(),
        vec!["a".to_owned()],
        |_| vec![constant_tool("noop", "ok".to_owned())],
        |request| LlmCompletion {
            request_id: request.request_id.clone(),
            content: String::new(),
            tool_calls: vec![ToolCall::new(
                format!("c{}", request.messages.len()),
                "noop",
                json!({}),
            )],
        },
        Some(Budget::new(3, |steps| format!("exhausted after {steps}"))),
    )
    .unwrap();

    assert_eq!(result.results["a"], "exhausted after 3");
}

#[test]
fn multi_tool_completions_may_arrive_out_of_order() {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "x".to_owned(),
        vec!["a".to_owned(), "b".to_owned()],
        vec![spec("t1"), spec("t2")],
    ))
    .unwrap();
    let llm = match loop_.next_request(RequestKind::Any).unwrap() {
        Request::Llm(request) => request,
        _ => panic!(),
    };
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: llm.request_id,
            content: String::new(),
            tool_calls: vec![
                ToolCall::new("c1", "t1", json!({})),
                ToolCall::new("c2", "t2", json!({})),
            ],
        }))
        .unwrap();
    let first = match loop_.next_request(RequestKind::Tool).unwrap() {
        Request::Tool(request) => request,
        _ => panic!(),
    };
    let second = match loop_.next_request(RequestKind::Tool).unwrap() {
        Request::Tool(request) => request,
        _ => panic!(),
    };

    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: second.request_id,
            results: vec![
                ("a".to_owned(), "x".to_owned()),
                ("b".to_owned(), "y".to_owned()),
            ],
        }))
        .unwrap();
    assert!(loop_.next_request(RequestKind::Llm).is_none());
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: first.request_id,
            results: vec![
                ("a".to_owned(), "same".to_owned()),
                ("b".to_owned(), "same".to_owned()),
            ],
        }))
        .unwrap();

    assert!(matches!(
        loop_.next_request(RequestKind::Llm),
        Some(Request::Llm(_))
    ));
    assert!(matches!(
        loop_.next_request(RequestKind::Llm),
        Some(Request::Llm(_))
    ));
}

#[test]
fn filtered_pulls_and_any_use_issue_order() {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "x".to_owned(),
        vec!["a".to_owned(), "b".to_owned()],
        vec![spec("t")],
    ))
    .unwrap();

    let first = match loop_.next_request(RequestKind::Llm).unwrap() {
        Request::Llm(request) => request,
        _ => panic!("expected LLM request"),
    };
    assert!(loop_.next_request(RequestKind::Tool).is_none());
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: first.request_id,
            content: String::new(),
            tool_calls: vec![ToolCall::new("c1", "t", json!({}))],
        }))
        .unwrap();

    let tool = match loop_.next_request(RequestKind::Tool).unwrap() {
        Request::Tool(request) => request,
        _ => panic!("expected tool request"),
    };
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id,
            results: vec![
                ("a".to_owned(), "1".to_owned()),
                ("b".to_owned(), "2".to_owned()),
            ],
        }))
        .unwrap();

    let llm_a = match loop_.next_request(RequestKind::Llm).unwrap() {
        Request::Llm(request) => request,
        _ => panic!("expected LLM request"),
    };
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: llm_a.request_id,
            content: String::new(),
            tool_calls: vec![ToolCall::new("c2", "t", json!({}))],
        }))
        .unwrap();

    assert!(matches!(
        loop_.next_request(RequestKind::Any),
        Some(Request::Llm(_))
    ));
    assert!(matches!(
        loop_.next_request(RequestKind::Any),
        Some(Request::Tool(_))
    ));
}

#[test]
fn completion_errors_leave_the_request_outstanding() {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "x".to_owned(),
        vec!["a".to_owned()],
        Vec::new(),
    ))
    .unwrap();
    assert_eq!(loop_.result(), Err(LoopError::NotDone));
    let request = match loop_.next_request(RequestKind::Any).unwrap() {
        Request::Llm(request) => request,
        _ => panic!("expected LLM request"),
    };
    let error = loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id.clone(),
            results: vec![("a".to_owned(), "x".to_owned())],
        }))
        .unwrap_err();
    assert!(matches!(error, LoopError::WrongCompletionKind { .. }));
    assert_eq!(loop_.outstanding_count(), 1);
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id,
            "done".to_owned(),
        )))
        .unwrap();
    assert!(loop_.is_done());
}

#[test]
fn validates_environment_and_completion_sets() {
    assert!(matches!(
        BatchLoop::<String>::new(LoopConfig::new("x".to_owned(), Vec::new(), Vec::new())),
        Err(LoopError::EmptyEnvIds)
    ));
    assert!(matches!(
        BatchLoop::<String>::new(LoopConfig::new(
            "x".to_owned(),
            vec!["a".to_owned(), "a".to_owned()],
            Vec::new()
        )),
        Err(LoopError::DuplicateEnvId(_))
    ));

    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "x".to_owned(),
        vec!["a".to_owned(), "b".to_owned()],
        vec![spec("t")],
    ))
    .unwrap();
    let llm = match loop_.next_request(RequestKind::Any).unwrap() {
        Request::Llm(request) => request,
        _ => panic!(),
    };
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: llm.request_id,
            content: String::new(),
            tool_calls: vec![ToolCall::new("t1", "t", json!({}))],
        }))
        .unwrap();
    let tool = match loop_.next_request(RequestKind::Any).unwrap() {
        Request::Tool(request) => request,
        _ => panic!(),
    };
    assert_eq!(
        loop_.complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id.clone(),
            results: vec![("a".to_owned(), "1".to_owned())],
        })),
        Err(LoopError::InvalidToolCompletion(tool.request_id))
    );
    assert_eq!(loop_.outstanding_tool(), 1);
}

#[test]
fn duplicate_tool_call_ids_are_rejected_without_consuming_request() {
    let mut loop_ = BatchLoop::new(LoopConfig::new(
        "x".to_owned(),
        vec!["a".to_owned()],
        vec![spec("t")],
    ))
    .unwrap();
    let request = match loop_.next_request(RequestKind::Any).unwrap() {
        Request::Llm(request) => request,
        _ => panic!(),
    };
    let error = loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id,
            content: String::new(),
            tool_calls: vec![
                ToolCall::new("same", "t", json!({})),
                ToolCall::new("same", "t", json!({})),
            ],
        }))
        .unwrap_err();
    assert_eq!(error, LoopError::DuplicateToolCallId("same".to_owned()));
    assert_eq!(loop_.outstanding_llm(), 1);
}

#[test]
fn message_helpers_match_provider_neutral_shape() {
    assert_eq!(Message::user("x").tool_calls, Vec::<ToolCall>::new());
}
