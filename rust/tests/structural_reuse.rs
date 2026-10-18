use lockstep_loop::{
    AgentDescriptor, BatchLoop, BatchResult, Budget, Completion, HistoryNode, LlmCompletion,
    LoopConfig, LoopError, NodeRef, RecordStore, Request, RequestKind, SubagentTool, ToolCall,
    ToolCompletion, ToolSpec,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new("read", "read-only", json!({})),
        ToolSpec::new("other", "read-only", json!({})),
    ]
}
fn config() -> LoopConfig<String> {
    LoopConfig::new("inspect".into(), vec!["e".into()], specs())
}
fn drive(
    mut loop_: BatchLoop<String>,
    implementation: impl Fn(&Value) -> String,
    provider: impl Fn() -> String,
) -> (BatchResult<String>, usize) {
    let mut llms = 0;
    while !loop_.is_done() {
        let reply = match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                llms += 1;
                Completion::Llm(if r.messages.len() == 1 {
                    LlmCompletion {
                        request_id: r.request_id,
                        content: "read".into(),
                        tool_calls: vec![ToolCall::new("r", "read", json!({}))],
                    }
                } else {
                    LlmCompletion::final_answer(r.request_id, provider())
                })
            }
            Request::Tool(r) => Completion::Tool(ToolCompletion {
                request_id: r.request_id,
                results: r
                    .env_ids
                    .into_iter()
                    .map(|e| (e, implementation(&r.tool_call.args)))
                    .collect(),
            }),
        };
        loop_.complete(reply).unwrap();
    }
    (loop_.result().unwrap(), llms)
}
fn answer() -> String {
    "answer".into()
}

#[test]
fn changed_implementation_reuses_equal_outputs_and_branches_on_different_outputs() {
    let (original, count) = drive(BatchLoop::new(config()).unwrap(), |_| "same".into(), answer);
    assert_eq!(count, 2);
    let old = original.agent().clone();
    let (same, count) = drive(
        BatchLoop::new(config().with_agent(old.clone())).unwrap(),
        |_| ["sa", "me"].concat(),
        answer,
    );
    assert_eq!(count, 0);
    assert_eq!(same.results, original.results);
    let (changed, count) = drive(
        BatchLoop::new(config().with_agent(same.into_agent())).unwrap(),
        |_| "different".into(),
        answer,
    );
    assert_eq!(count, 1);
    assert_eq!(changed.agent().root().observations.len(), 2);
    assert_eq!(old.root().observations.len(), 1);
    for value in ["same", "different"] {
        let (_, count) = drive(
            BatchLoop::new(config().with_agent(changed.agent().clone())).unwrap(),
            |_| value.into(),
            answer,
        );
        assert_eq!(count, 0);
    }
    let mut reordered = config().with_agent(old);
    reordered.tool_specs.reverse();
    assert!(matches!(
        BatchLoop::new(reordered),
        Err(LoopError::IncompatibleHistory)
    ));
}

#[test]
fn provider_and_custom_grouping_policy_are_caller_owned_not_inferred() {
    let (old, _) = drive(
        BatchLoop::with_comparison_key(config(), |s: &String| s.trim().to_owned()).unwrap(),
        |_| "same ".into(),
        answer,
    );
    let (replay, count) = drive(
        BatchLoop::with_comparison_key(config().with_agent(old.agent().clone()), |s: &String| {
            s.trim().to_lowercase()
        })
        .unwrap(),
        |_| "SAME".into(),
        || panic!("external provider change does not invalidate recorded turns"),
    );
    assert_eq!(count, 0);
    assert_eq!(replay.results, old.results);
    let root = replay.agent().root();
    let evidence: Vec<_> = root.observations[0]
        .evidence
        .iter()
        .map(|e| e.outputs[0].tool_output.as_str())
        .collect();
    assert_eq!(evidence, ["same ", "SAME"]);
    // Mode and function identity are not checked. Today's function compares raw
    // outputs to historical representatives; provider policy remains external.
    let (exact, count) = drive(
        BatchLoop::new(config().with_agent(old.agent().clone())).unwrap(),
        |_| "same ".into(),
        || panic!("recorded provider answer must be reused"),
    );
    assert_eq!(count, 0);
    assert_eq!(exact.results, old.results);
}

#[derive(Deserialize)]
struct Fixture {
    descriptor: AgentDescriptor<String>,
    records: Vec<(NodeRef, HistoryNode<String>)>,
}
#[test]
fn imported_legacy_labels_do_not_poison_structural_child_cache_matching() {
    let raw: Value =
        serde_json::from_slice(include_bytes!("fixtures/agent-naming-2c6d901.json")).unwrap();
    let fixture: Fixture = serde_json::from_value(raw.clone()).unwrap();
    let mut different = raw;
    fn relabel(value: &mut Value) {
        match value {
            Value::Object(fields) => {
                for (name, value) in fields {
                    if name == "contract" || name == "history_contract" {
                        *value = json!("unrelated legacy label");
                    } else if name == "custom_grouping" {
                        *value = json!(false);
                    } else {
                        relabel(value);
                    }
                }
            }
            Value::Array(values) => values.iter_mut().for_each(relabel),
            _ => {}
        }
    }
    relabel(&mut different);
    let imported: Fixture = serde_json::from_value(different).unwrap();
    assert_eq!(fixture.descriptor, imported.descriptor);
    assert_eq!(fixture.records, imported.records);
    let records: BTreeMap<_, _> = imported
        .records
        .into_iter()
        .map(|(r, n)| (r, Arc::new(n)))
        .collect();
    let selected = records[&imported.descriptor.root].subagents[0]
        .history
        .clone();
    assert!(
        serde_json::to_value(&selected)
            .unwrap()
            .get("contract")
            .is_none()
    );
    let store = RecordStore::with_loader("legacy-labels", move |r| records[r].clone());
    let queries = Arc::new(Mutex::new(vec![]));
    let tool = SubagentTool::new(
        ToolSpec::new("delegate", "child investigation", json!({})),
        |_| "child".to_owned(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    for (parent, custom) in [("parent one", true), ("changed parent", false)] {
        let seen = queries.clone();
        let candidate = selected.clone();
        let config = LoopConfig::new(parent.to_owned(), vec!["new".into()], vec![])
            .with_subagents(vec![tool.clone()])
            .with_record_store(store.clone())
            .with_agent_resolver(move |q| {
                seen.lock().unwrap().push(q.clone());
                Ok((*q == candidate.query()).then(|| candidate.clone()))
            });
        let mut loop_ = if custom {
            BatchLoop::with_comparison_key(config, |s: &String| s.trim().to_owned())
        } else {
            BatchLoop::new(config)
        }
        .unwrap();
        let r = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion {
                request_id: r.request_id().into(),
                content: "delegate".into(),
                tool_calls: vec![ToolCall::new("c", "delegate", json!({}))],
            }))
            .unwrap();
        let Request::Llm(r) = loop_.next_request(RequestKind::Any).unwrap() else {
            panic!()
        };
        assert_eq!(r.messages[0].content, parent); // no child LLM request
        assert_eq!(r.messages.last().unwrap().content, "child answer");
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                r.request_id,
                "done".into(),
            )))
            .unwrap();
        assert_eq!(loop_.agent().root().subagents.len(), 1);
        assert_eq!(
            loop_.result().unwrap().child_agents()[0].base.as_ref(),
            Some(&selected)
        );
    }
    let queries = queries.lock().unwrap();
    assert_eq!(queries[0], queries[1]);
}

#[test]
fn default_and_custom_identity_reuse_each_other() {
    for initial_custom in [false, true] {
        let c = config();
        let initial = if initial_custom {
            BatchLoop::with_comparison_key(c, Clone::clone)
        } else {
            BatchLoop::new(c)
        }
        .unwrap();
        let (old, _) = drive(initial, |_| "same".into(), answer);
        let agent = old.agent();
        let before = agent.root();
        let c = config().with_agent(agent.clone());
        let replay = if initial_custom {
            BatchLoop::new(c)
        } else {
            BatchLoop::with_comparison_key(c, Clone::clone)
        }
        .unwrap();
        let (replay, count) = drive(replay, |_| "same".into(), answer);
        assert_eq!(count, 0);
        assert_eq!(replay.results, old.results);
        assert_eq!(agent.root(), before);
        assert_eq!(before.observations[0].evidence.len(), 1);
        assert_eq!(replay.agent().root().observations[0].evidence.len(), 2);
        assert!(
            serde_json::to_value(agent.descriptor.query())
                .unwrap()
                .get("custom_grouping")
                .is_none()
        );
    }
}

#[test]
fn stricter_keys_match_representatives_not_old_evidence() {
    let (old, _) = drive(
        BatchLoop::with_comparison_key(config(), |v: &String| v.trim().to_owned()).unwrap(),
        |_| "same ".into(),
        answer,
    );
    let (grouped, count) = drive(
        BatchLoop::with_comparison_key(config().with_agent(old.agent().clone()), |v: &String| {
            v.trim().to_owned()
        })
        .unwrap(),
        |_| "same".into(),
        answer,
    );
    assert_eq!(count, 0);
    let agent = grouped.agent();
    let before = agent.root();
    for custom in [false, true] {
        let c = config().with_agent(agent.clone());
        let strict = if custom {
            BatchLoop::with_comparison_key(c, Clone::clone)
        } else {
            BatchLoop::new(c)
        }
        .unwrap();
        // "same" occurs in old evidence, but not in its representative "same ".
        let (changed, count) = drive(strict, |_| "same".into(), || "strict".into());
        assert_eq!(count, 1);
        assert_eq!(changed.results["e"], "strict");
        assert_eq!(changed.agent().root().observations.len(), 2);
        assert_eq!(
            changed.agent().root().observations[1].representative[0].tool_output,
            "same"
        );
        assert_eq!(agent.root(), before);
    }
    assert_eq!(old.agent().root().observations.len(), 1);
    assert_eq!(
        before.observations[0]
            .evidence
            .iter()
            .map(|e| e.outputs[0].tool_output.as_str())
            .collect::<Vec<_>>(),
        ["same ", "same"]
    );
    assert_eq!(
        agent.assets().next().unwrap().messages()[2].content,
        "same "
    );
}

#[test]
fn coarser_keys_choose_first_existing_edge_without_merging_history() {
    let (first, _) = drive(
        BatchLoop::new(config()).unwrap(),
        |_| "A".into(),
        || "first".into(),
    );
    let (old, count) = drive(
        BatchLoop::new(config().with_agent(first.agent().clone())).unwrap(),
        |_| "a".into(),
        || "second".into(),
    );
    assert_eq!(count, 1);
    let agent = old.agent();
    let before = agent.root();
    let (replay, count) = drive(
        BatchLoop::with_comparison_key(config().with_agent(agent.clone()), |v: &String| {
            v.trim().to_lowercase()
        })
        .unwrap(),
        |_| "a ".into(),
        answer,
    );
    assert_eq!(count, 0);
    assert_eq!(replay.results["e"], "first");
    let root = replay.agent().root();
    assert_eq!(root.observations.len(), 2);
    assert_eq!(root.observations[1], before.observations[1]);
    assert_eq!(
        root.observations[0]
            .evidence
            .iter()
            .map(|e| e.outputs[0].tool_output.as_str())
            .collect::<Vec<_>>(),
        ["A", "a "]
    );
    assert_eq!(root.observations[0].representative[0].tool_output, "A");
    assert_eq!(
        replay.agent().assets().next().unwrap().messages()[2].content,
        "A"
    );
    assert_eq!(agent.root(), before);
    assert_eq!(first.agent().root().observations.len(), 1);
    let (exact, count) = drive(
        BatchLoop::new(config().with_agent(replay.into_agent())).unwrap(),
        |_| "a".into(),
        answer,
    );
    assert_eq!(count, 0);
    assert_eq!(exact.results["e"], "second");
}

#[test]
fn child_query_and_execution_ignore_default_custom_mode() {
    let tool = SubagentTool::new(
        ToolSpec::new("child", "", json!({})),
        |_| "child prompt".to_owned(),
        |_| vec![specs()[0].clone()],
        1,
    )
    .with_subagents(vec![]);
    let mut selected: Option<AgentDescriptor<String>> = None;
    let store = RecordStore::new("child-mode");
    let queries = Arc::new(Mutex::new(vec![]));
    for custom in [false, true, false] {
        let candidate = selected.clone();
        let seen = queries.clone();
        let c = LoopConfig::new("parent".to_owned(), vec!["e".into()], vec![])
            .with_subagent(tool.clone())
            .with_record_store(store.clone())
            .with_agent_resolver(move |q| {
                seen.lock().unwrap().push(q.clone());
                Ok(candidate.as_ref().filter(|d| d.query() == *q).cloned())
            });
        let mut loop_ = if custom {
            BatchLoop::with_comparison_key(c, Clone::clone)
        } else {
            BatchLoop::new(c)
        }
        .unwrap();
        let r = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion {
                request_id: r.request_id().into(),
                content: "delegate".into(),
                tool_calls: vec![ToolCall::new("c", "child", json!({}))],
            }))
            .unwrap();
        let mut child_llms = 0;
        while !loop_.is_done() {
            let reply = match loop_.next_request(RequestKind::Any).unwrap() {
                Request::Llm(r) => {
                    if r.messages[0].content == "parent" {
                        Completion::Llm(LlmCompletion::final_answer(
                            r.request_id,
                            r.messages.last().unwrap().content.clone(),
                        ))
                    } else {
                        child_llms += 1;
                        let calls = if r.messages.len() == 1 {
                            vec![ToolCall::new("r", "read", json!({}))]
                        } else {
                            vec![]
                        };
                        Completion::Llm(LlmCompletion {
                            request_id: r.request_id,
                            content: "child answer".into(),
                            tool_calls: calls,
                        })
                    }
                }
                Request::Tool(r) => Completion::Tool(ToolCompletion {
                    request_id: r.request_id,
                    results: vec![("e".into(), "same".into())],
                }),
            };
            loop_.complete(reply).unwrap();
        }
        assert_eq!(child_llms, if selected.is_none() { 2 } else { 0 });
        let result = loop_.result().unwrap();
        let update = &result.child_agents()[0];
        assert_eq!(update.base, selected);
        selected = Some(update.agent().descriptor.clone());
    }
    let queries = queries.lock().unwrap();
    assert_eq!(queries[0], queries[1]);
    assert_eq!(queries[1], queries[2]);
}

#[test]
fn child_structural_identity_and_fresh_budget() {
    let tool = SubagentTool::new(
        ToolSpec::new("child", "", json!({})),
        |_| "child".to_owned(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let mut loop_ = BatchLoop::new(config().with_subagent(tool.clone())).unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id(),
            "done".into(),
        )))
        .unwrap();
    let mut replay = BatchLoop::new(
        config()
            .with_subagent(tool.clone())
            .with_agent(loop_.agent()),
    )
    .unwrap();
    assert!(replay.is_done());
    let mut fresh = BatchLoop::new(
        config()
            .with_subagent(tool)
            .with_budget(Budget::new(1, |_| "stop".into())),
    )
    .unwrap();
    assert!(fresh.agent().root().assistant.is_none());
    let request = fresh.next_request(RequestKind::Any).unwrap();
    fresh
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "done".into(),
        )))
        .unwrap();
    assert_eq!(
        fresh
            .result()
            .unwrap()
            .agent()
            .root()
            .assistant
            .as_ref()
            .unwrap()
            .content,
        "done"
    );
}
