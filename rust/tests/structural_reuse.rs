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
    let (original, count) = drive(
        BatchLoop::new(config().with_recording()).unwrap(),
        |_| "same".into(),
        answer,
    );
    assert_eq!(count, 2);
    let old = original.agent().unwrap().clone();
    let (same, count) = drive(
        BatchLoop::new(config().with_agent(old.clone())).unwrap(),
        |_| ["sa", "me"].concat(),
        answer,
    );
    assert_eq!(count, 0);
    assert_eq!(same.results, original.results);
    let (changed, count) = drive(
        BatchLoop::new(config().with_agent(same.into_agent().unwrap())).unwrap(),
        |_| "different".into(),
        answer,
    );
    assert_eq!(count, 1);
    assert_eq!(changed.agent().unwrap().root().observations.len(), 2);
    assert_eq!(old.root().observations.len(), 1);
    for value in ["same", "different"] {
        let (_, count) = drive(
            BatchLoop::new(config().with_agent(changed.agent().unwrap().clone())).unwrap(),
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
        BatchLoop::with_group_by(config().with_recording(), |s: &String| s.trim().to_owned())
            .unwrap(),
        |_| "same ".into(),
        answer,
    );
    let (replay, count) = drive(
        BatchLoop::with_group_by(
            config().with_agent(old.agent().unwrap().clone()),
            |s: &String| s.trim().to_lowercase(),
        )
        .unwrap(),
        |_| "SAME".into(),
        || panic!("external provider change does not invalidate recorded turns"),
    );
    assert_eq!(count, 0);
    assert_eq!(replay.results, old.results);
    let root = replay.agent().unwrap().root();
    let evidence: Vec<_> = root.observations[0]
        .evidence
        .iter()
        .map(|e| e.outputs[0].tool_output.as_str())
        .collect();
    assert_eq!(evidence, ["same ", "SAME"]);
    // Only default/custom mode is checked; callers select isolated caches/snapshots
    // when their provider or lossy grouping policy requires different reasoning.
    assert!(matches!(
        BatchLoop::new(config().with_agent(old.into_agent().unwrap())),
        Err(LoopError::IncompatibleHistory)
    ));
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
    for parent in ["parent one", "changed parent"] {
        let seen = queries.clone();
        let candidate = selected.clone();
        let config = LoopConfig::new(parent.to_owned(), vec!["new".into()], vec![])
            .with_subagents(vec![tool.clone()])
            .with_record_store(store.clone())
            .with_agent_resolver(move |q| {
                seen.lock().unwrap().push(q.clone());
                Ok((*q == candidate.query()).then(|| candidate.clone()))
            });
        let mut loop_ = BatchLoop::with_group_by(config, |s: &String| s.trim().to_owned()).unwrap();
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
        assert!(loop_.agent().is_none()); // resolver does not record the root
        assert_eq!(
            loop_.result().unwrap().child_agents()[0].base.as_ref(),
            Some(&selected)
        );
    }
    let queries = queries.lock().unwrap();
    assert_eq!(queries[0], queries[1]);
}

#[test]
fn recording_switch_is_not_identity_and_legacy_budgets_remain_supported() {
    let tool = SubagentTool::new(
        ToolSpec::new("child", "", json!({})),
        |_| "child".to_owned(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let mut loop_ = BatchLoop::new(config().with_subagent(tool.clone()).with_recording()).unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id(),
            "done".into(),
        )))
        .unwrap();
    let mut replay = BatchLoop::new(
        config()
            .with_subagent(tool.clone().with_recording())
            .with_agent(loop_.agent().unwrap()),
    )
    .unwrap();
    assert!(replay.is_done());
    let legacy = BatchLoop::new(
        config()
            .with_subagent(tool.clone())
            .with_budget(Budget::new(1, |_| "stop".into())),
    )
    .unwrap();
    assert!(legacy.agent().is_none());
    assert!(matches!(
        BatchLoop::new(
            config()
                .with_subagent(tool.with_recording())
                .with_budget(Budget::new(1, |_| "stop".into()))
        ),
        Err(LoopError::UnsupportedHistory)
    ));
}
