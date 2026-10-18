//! Routing must never interpret an opaque provider tool-call ID as an address.
use lockstep_loop::{
    BatchLoop, Completion, History, LlmCompletion, LlmRequest, LoopConfig, LoopError, Request,
    RequestKind, Role, SubagentTool, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::HashSet;

const OPAQUE_IDS: [&str; 2] = ["a/turn-1/sub:root/b", "\0/sub:1/turn-1/llm-1/☃:b"];
const PROBE_ID: &str = "probe/turn-1/sub:root/b/llm-1:\0☃";
const EXPECTED: &str = "child1:grandchild:observed:grandchild|child2:observed:child2";

fn specs() -> Vec<ToolSpec> {
    ["investigate", "probe"]
        .iter()
        .map(|name| ToolSpec::new(*name, "", json!({})))
        .collect()
}
fn make_loop(recording: bool, env: &str, previous: Option<History<String>>) -> BatchLoop<String> {
    let config = LoopConfig::new("parent".into(), vec![env.into()], specs()).with_subagent(
        SubagentTool::new(
            specs()[0].clone(),
            |args| args["prompt"].as_str().unwrap().into(),
            |_| specs(),
            2,
        ),
    );
    let config = if let Some(history) = previous {
        config.with_replay(history)
    } else if recording {
        config.with_recording()
    } else {
        config
    };
    BatchLoop::new(config).unwrap()
}
fn answer(request: &LlmRequest<String>, opaque_id: &str) -> LlmCompletion<String> {
    let scope = request.messages[0].content.as_str();
    let observed: Vec<_> = request
        .messages
        .iter()
        .filter(|m| m.role == Role::Tool)
        .collect();
    let calls = if scope == "parent" {
        // Reuse both opaque call IDs on a second parent turn as well.
        assert!([0, 2, 4].contains(&observed.len()));
        for pair in observed.chunks_exact(2) {
            assert_eq!(pair[0].tool_call_id.as_deref(), Some("a"));
            assert_eq!(pair[0].content, "child1:grandchild:observed:grandchild");
            assert_eq!(pair[1].tool_call_id.as_deref(), Some(opaque_id));
            assert_eq!(pair[1].content, "child2:observed:child2");
        }
        if observed.len() == 4 {
            return LlmCompletion::final_answer(&request.request_id, EXPECTED.into());
        }
        vec![
            ToolCall::new("a", "investigate", json!({"prompt": "child1"})),
            ToolCall::new(opaque_id, "investigate", json!({"prompt": "child2"})),
        ]
    } else if observed.is_empty() {
        if scope == "child1" {
            vec![ToolCall::new(
                "b",
                "investigate",
                json!({"prompt": "grandchild"}),
            )]
        } else {
            assert!(["grandchild", "child2"].contains(&scope));
            vec![ToolCall::new(PROBE_ID, "probe", json!({"scope": scope}))]
        }
    } else {
        assert_eq!(observed.len(), 1);
        if scope == "child1" {
            assert_eq!(observed[0].tool_call_id.as_deref(), Some("b"));
            assert_eq!(observed[0].content, "grandchild:observed:grandchild");
        } else {
            assert_eq!(observed[0].tool_call_id.as_deref(), Some(PROBE_ID));
            assert_eq!(observed[0].content, format!("observed:{scope}"));
        }
        return LlmCompletion::final_answer(
            &request.request_id,
            format!("{scope}:{}", observed[0].content),
        );
    };
    LlmCompletion {
        request_id: request.request_id.clone(),
        content: "delegate".into(),
        tool_calls: calls,
    }
}
fn pull(
    loop_: &mut BatchLoop<String>,
    seen: &mut HashSet<String>,
    kind: RequestKind,
) -> Option<Request<String>> {
    let request = loop_.next_request(kind);
    if let Some(r) = &request {
        assert!(
            seen.insert(r.request_id().to_owned()),
            "request ID reused: {}",
            r.request_id()
        );
    }
    request
}
fn pull_llm(loop_: &mut BatchLoop<String>, seen: &mut HashSet<String>) -> LlmRequest<String> {
    let Request::Llm(request) = pull(loop_, seen, RequestKind::Llm).unwrap() else {
        panic!("expected LLM request")
    };
    request
}
fn finish(
    loop_: &mut BatchLoop<String>,
    seen: &mut HashSet<String>,
    opaque_id: &str,
    reverse: bool,
    replay: bool,
) -> (usize, usize) {
    let (mut llms, mut tools) = (0, 0);
    while !loop_.is_done() {
        let mut pending = Vec::new();
        while let Some(request) = pull(loop_, seen, RequestKind::Any) {
            pending.push(request);
        }
        assert!(!pending.is_empty(), "outstanding work was stranded");
        assert_eq!(loop_.outstanding_count(), pending.len());
        if reverse {
            pending.reverse();
        }
        for request in pending {
            match request {
                Request::Llm(request) => {
                    assert!(!replay, "unchanged nested replay requested fresh reasoning");
                    llms += 1;
                    loop_
                        .complete(Completion::Llm(answer(&request, opaque_id)))
                        .unwrap();
                }
                Request::Tool(request) => {
                    tools += 1;
                    assert_eq!(request.tool_call.id, PROBE_ID);
                    let output = format!(
                        "observed:{}",
                        request.tool_call.args["scope"].as_str().unwrap()
                    );
                    let results = request
                        .env_ids
                        .into_iter()
                        .map(|env| (env, output.clone()))
                        .collect();
                    loop_
                        .complete(Completion::Tool(ToolCompletion {
                            request_id: request.request_id,
                            results,
                        }))
                        .unwrap();
                }
            }
        }
        assert_eq!(loop_.outstanding_count(), 0);
    }
    assert_eq!(loop_.outstanding_llm(), 0);
    assert_eq!(loop_.outstanding_tool(), 0);
    assert_eq!(loop_.outstanding_count(), 0);
    assert!(loop_.next_request(RequestKind::Any).is_none());
    (llms, tools)
}
fn exercise(recording: bool) {
    for opaque_id in OPAQUE_IDS {
        for reverse in [false, true] {
            let mut loop_ = make_loop(recording, "old", None);
            let mut seen = HashSet::new();
            let parent = pull_llm(&mut loop_, &mut seen);
            loop_
                .complete(Completion::Llm(answer(&parent, opaque_id)))
                .unwrap();
            let child1 = pull_llm(&mut loop_, &mut seen);
            assert_eq!(child1.messages[0].content, "child1");
            loop_
                .complete(Completion::Llm(answer(&child1, opaque_id)))
                .unwrap();
            let grandchild = pull_llm(&mut loop_, &mut seen);
            let child2 = pull_llm(&mut loop_, &mut seen);
            // Exact reproduction: both previously had the ID
            // sub:root/a/turn-1/sub:root/b/turn-1/llm-1, overwriting one route.
            assert_eq!(grandchild.messages[0].content, "grandchild");
            assert_eq!(child2.messages[0].content, "child2");
            assert_ne!(grandchild.request_id, child2.request_id);
            assert_eq!(loop_.outstanding_llm(), 2);
            let stale_id = grandchild.request_id.clone();
            let mut pending = vec![grandchild, child2];
            if reverse {
                pending.reverse();
            }
            for request in pending {
                loop_
                    .complete(Completion::Llm(answer(&request, opaque_id)))
                    .unwrap();
            }
            assert_eq!(loop_.outstanding_count(), 0);
            assert_eq!(
                finish(&mut loop_, &mut seen, opaque_id, reverse, false).1,
                4
            );
            assert_eq!(loop_.result().unwrap().results["old"], EXPECTED);
            assert!(matches!(
                loop_.complete(Completion::Llm(LlmCompletion::final_answer(
                    stale_id,
                    "stale".into()
                ))),
                Err(LoopError::UnknownRequest(_))
            ));
            assert_eq!(loop_.outstanding_count(), 0);
            if recording {
                let history = loop_.history().unwrap();
                assert_eq!(
                    history.root().assistant.as_ref().unwrap().tool_calls[1].id,
                    opaque_id
                );
                let ids: HashSet<_> = history
                    .root()
                    .subagents
                    .iter()
                    .map(|s| s.call_id.clone())
                    .collect();
                assert_eq!(ids, HashSet::from(["a".to_owned(), opaque_id.to_owned()]));
                for replay_reverse in [false, true] {
                    let mut replay = make_loop(true, "new", Some(history.clone()));
                    assert_eq!(
                        finish(
                            &mut replay,
                            &mut HashSet::new(),
                            opaque_id,
                            replay_reverse,
                            true
                        ),
                        (0, 4)
                    );
                    assert_eq!(replay.result().unwrap().results["new"], EXPECTED);
                }
                assert_eq!(history.root().env_ids, ["old"]);
            } else {
                assert!(loop_.history().is_none());
            }
        }
    }
}

#[test]
fn opaque_nested_ids_route_correctly_without_recording() {
    exercise(false);
}

#[test]
fn opaque_nested_ids_route_correctly_with_recording_and_replay() {
    exercise(true);
}
