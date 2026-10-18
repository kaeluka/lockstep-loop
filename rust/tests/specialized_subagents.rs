use lockstep_loop::{
    BatchLoop, BatchResult, Budget, Completion, FunctionTool, History, HistoryDescriptor,
    HistoryQuery, HistoryResolver, LlmCompletion, LlmRequest, LoopConfig, LoopError, NodeRef,
    RecordStore, Request, RequestKind, Role, SubagentTool, ToolCall, ToolCompletion, ToolEntry,
    ToolRequest, ToolSpec, run_to_completion,
};
use serde_json::json;
use std::collections::{BTreeMap, HashSet};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

fn read() -> ToolSpec {
    ToolSpec::new(
        "read_file",
        "Read without modifying",
        json!({"type": "object"}),
    )
}
fn status() -> ToolSpec {
    ToolSpec::new("status", "Read status", json!({}))
}
fn definition(name: &str, kind: &str) -> SubagentTool<String> {
    let kind = kind.to_owned();
    SubagentTool::try_new(
        ToolSpec::new(
            name,
            format!("Run {kind}"),
            json!({
                "type": "object", "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"], "additionalProperties": false,
            }),
        ),
        move |args| {
            if args.as_object().is_none_or(|o| o.len() != 1) || !args["file_path"].is_string() {
                return Err(LoopError::InvalidSubagentArguments(
                    "expected exactly one string file_path".into(),
                ));
            }
            Ok(format!("{kind}:{}", args["file_path"].as_str().unwrap()))
        },
        |_| Ok(vec![read()]),
        1,
    )
    .with_subagents(vec![])
}
fn definitions() -> Vec<SubagentTool<String>> {
    vec![
        definition("list_imports", "imports"),
        definition("count_lines", "metrics"),
    ]
}
fn calls(path: &str) -> Vec<ToolCall> {
    vec![
        ToolCall::new("imports", "list_imports", json!({"file_path": path})),
        ToolCall::new("metrics", "count_lines", json!({"file_path": path})),
        ToolCall::new("local", "status", json!({})),
    ]
}
fn config(prompt: &str, envs: &[&str]) -> LoopConfig<String> {
    LoopConfig::new(
        prompt.into(),
        envs.iter().map(|s| (*s).into()).collect(),
        vec![status()],
    )
    .with_subagents(definitions())
}
fn model(r: &LlmRequest<String>, parent_calls: &[ToolCall]) -> LlmCompletion<String> {
    let prompt = &r.messages[0].content;
    assert_eq!(r.messages[0].role, Role::User);
    if prompt.starts_with("parent") {
        if r.messages.len() == 1 {
            return LlmCompletion {
                request_id: r.request_id.clone(),
                content: "delegate".into(),
                tool_calls: parent_calls.to_vec(),
            };
        }
        let observed: Vec<_> = r.messages.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(
            observed
                .iter()
                .map(|m| m.tool_call_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            parent_calls
                .iter()
                .map(|c| c.id.as_str())
                .collect::<Vec<_>>()
        );
        return LlmCompletion::final_answer(
            &r.request_id,
            observed
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("|"),
        );
    }
    let (kind, path) = prompt.split_once(':').unwrap();
    if r.stats.depth < 2 {
        LlmCompletion {
            request_id: r.request_id.clone(),
            content: "inspect".into(),
            tool_calls: vec![ToolCall::new(
                "read",
                "read_file",
                json!({"path": path, "checkpoint": r.stats.depth}),
            )],
        }
    } else {
        LlmCompletion::final_answer(&r.request_id, format!("{kind}:done"))
    }
}
struct Driven {
    result: BatchResult<String>,
    llms: Vec<LlmRequest<String>>,
    tools: Vec<ToolRequest>,
}
fn drive(
    mut loop_: BatchLoop<String>,
    values: &[(&str, &str)],
    parent_calls: &[ToolCall],
    expected_child_llms: usize,
) -> Driven {
    let values: BTreeMap<_, _> = values.iter().copied().collect();
    let (mut llms, mut tools, mut ids) = (vec![], vec![], HashSet::new());
    while !loop_.is_done() {
        let mut pending = vec![];
        while let Some(r) = loop_.next_request(RequestKind::Any) {
            assert!(ids.insert(r.request_id().to_owned()));
            pending.push(r);
        }
        assert!(!pending.is_empty());
        for request in pending.into_iter().rev() {
            match request {
                Request::Llm(r) => {
                    loop_
                        .complete(Completion::Llm(model(&r, parent_calls)))
                        .unwrap();
                    llms.push(r);
                }
                Request::Tool(r) => {
                    assert!(["status", "read_file"].contains(&r.tool_call.name.as_str()));
                    let results = r
                        .env_ids
                        .iter()
                        .rev()
                        .map(|env| {
                            let output = if r.tool_call.name == "status" {
                                "ok".into()
                            } else {
                                let args = &r.tool_call.args;
                                format!(
                                    "{}:{}",
                                    args["path"].as_str().unwrap(),
                                    if args["checkpoint"] == 0 {
                                        "prefix"
                                    } else {
                                        values[env.as_str()]
                                    }
                                )
                            };
                            (env.clone(), output)
                        })
                        .collect();
                    loop_
                        .complete(Completion::Tool(ToolCompletion {
                            request_id: r.request_id.clone(),
                            results,
                        }))
                        .unwrap();
                    tools.push(r);
                }
            }
        }
    }
    assert_eq!(
        llms.iter()
            .filter(|r| !r.messages[0].content.starts_with("parent"))
            .count(),
        expected_child_llms
    );
    assert!(loop_.error().is_none());
    assert_eq!(loop_.outstanding_count(), 0);
    Driven {
        result: loop_.result().unwrap(),
        llms,
        tools,
    }
}
type Backend = Arc<Mutex<BTreeMap<NodeRef, Vec<u8>>>>;
fn publish(history: &History<String>, backend: &Backend) -> HistoryDescriptor<String> {
    let mut backend = backend.lock().unwrap();
    for (r, n) in history.store.records() {
        let encoded = serde_json::to_vec(&n).unwrap();
        if let Some(old) = backend.get(&r) {
            assert_eq!(old, &encoded);
        } else {
            backend.insert(r, encoded);
        }
    }
    serde_json::from_slice(&serde_json::to_vec(&history.descriptor).unwrap()).unwrap()
}
fn store(backend: &Backend) -> RecordStore<String> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let backend = backend.clone();
    RecordStore::with_loader(
        format!(
            "specialized-writer-{}",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ),
        move |r| Arc::new(serde_json::from_slice(&backend.lock().unwrap()[r]).unwrap()),
    )
}
fn initial() -> BatchResult<String> {
    drive(
        BatchLoop::new(config("parent-A", &["old"])).unwrap(),
        &[("old", "same")],
        &calls("foo.py"),
        6,
    )
    .result
}
fn imports(history: &BatchResult<String>) -> History<String> {
    history
        .child_histories
        .iter()
        .find(|u| u.tool_call.name == "list_imports")
        .unwrap()
        .history
        .clone()
}

#[test]
fn multiple_tools_changed_parent_reuses_children_and_exposes_persistable_provenance() {
    let mut first = BatchLoop::new(config("parent-A", &["old"])).unwrap();
    let Request::Llm(r) = first.next_request(RequestKind::Llm).unwrap() else {
        panic!()
    };
    assert_eq!(
        r.tools.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
        ["status", "list_imports", "count_lines"]
    );
    assert_eq!(r.tools[1].parameters["required"], json!(["file_path"]));
    first
        .complete(Completion::Llm(model(&r, &calls("foo.py"))))
        .unwrap();
    let old = drive(first, &[("old", "same")], &calls("foo.py"), 6).result;
    assert_eq!(old.results["old"], "imports:done|metrics:done|ok");
    let backend = Backend::default();
    let mut selected = BTreeMap::new();
    for u in &old.child_histories {
        let d = publish(&u.history, &backend);
        selected.insert(d.prompt.clone(), d);
        assert!(u.base.is_none());
        assert_eq!(u.env_ids, ["old"]);
    }
    assert_eq!(selected.len(), 2);
    let selected_for_resolver = selected.clone();
    let queries = Arc::new(Mutex::new(vec![]));
    let observed = queries.clone();
    let loaded = Arc::new(Mutex::new(vec![]));
    let loads = loaded.clone();
    let records = backend.clone();
    let store = RecordStore::with_loader("parent-b-lazy", move |r| {
        loads.lock().unwrap().push(r.clone());
        Arc::new(serde_json::from_slice(&records.lock().unwrap()[r]).unwrap())
    });
    let config = config("parent-B", &["tomorrow"])
        .with_record_store(store)
        .with_history_resolver(move |q| {
            observed.lock().unwrap().push(q.clone());
            Ok(selected_for_resolver.get(&q.prompt).cloned())
        });
    let mut new_calls = calls("foo.py");
    for call in &mut new_calls {
        call.id = format!("opaque/turn-1/sub:root/{}", call.id);
    }
    let second = drive(
        BatchLoop::new(config).unwrap(),
        &[("tomorrow", "same")],
        &new_calls,
        0,
    );
    assert_eq!(second.llms.len(), 2);
    assert!(
        !loaded
            .lock()
            .unwrap()
            .contains(&old.history.descriptor.root)
    );
    assert!(
        second
            .llms
            .iter()
            .all(|r| r.messages[0].content == "parent-B")
    );
    assert_eq!(
        second
            .tools
            .iter()
            .filter(|r| r.tool_call.name == "read_file")
            .count(),
        4
    );
    assert_eq!(second.result.results["tomorrow"], old.results["old"]);
    assert_eq!(queries.lock().unwrap().len(), 2);
    let parent = &second.result.history;
    for u in &second.result.child_histories {
        let key = u.history.descriptor.prompt.clone();
        assert_eq!(u.base.as_ref(), selected.get(&key));
        assert_ne!(u.history.descriptor.root, u.base.as_ref().unwrap().root);
        let root = parent.root();
        let entry = root
            .subagents
            .iter()
            .find(|s| s.call_id == u.tool_call.id)
            .unwrap();
        assert_eq!(entry.runs.last().unwrap().base, u.base);
        assert_eq!(entry.runs.last().unwrap().history, u.history.descriptor);
        assert_eq!(entry.runs.last().unwrap().env_ids, ["tomorrow"]);
        publish(&u.history, &backend);
    }
    assert!(
        old.child_histories
            .iter()
            .all(|u| u.history.root().env_ids == ["old"])
    );
    let replay_config = super_config("parent-B", &["later"])
        .with_replay(parent.clone())
        .with_history_resolver(|_| panic!("pinned parent must not consult mutable cache pointer"));
    let replay = drive(
        BatchLoop::new(replay_config).unwrap(),
        &[("later", "same")],
        &new_calls,
        0,
    );
    assert!(replay.llms.is_empty());
    for entry in &replay.result.history.root().subagents {
        assert_eq!(entry.runs.len(), 2);
        assert_eq!(
            entry.runs[0].history,
            parent
                .root()
                .subagents
                .iter()
                .find(|s| s.call_id == entry.call_id)
                .unwrap()
                .history
        );
    }
}
// Keep a name not shadowed by local config bindings above.
fn super_config(prompt: &str, envs: &[&str]) -> LoopConfig<String> {
    config(prompt, envs)
}

#[test]
fn changed_arguments_and_new_observations_do_not_split_parent_on_equal_answers() {
    let old = initial();
    let base = imports(&old);
    let backend = Backend::default();
    let descriptor = publish(&base, &backend);
    let resolver = HistoryResolver::new(move |q: &HistoryQuery<String>| {
        Ok((*q == descriptor.query()).then(|| descriptor.clone()))
    });
    let mut config = config("parent-changed", &["A", "B"]).with_record_store(store(&backend));
    config.history_resolver = Some(resolver.clone());
    let c = calls("foo.py");
    let selected_calls = vec![c[0].clone(), c[2].clone()];
    let changed = drive(
        BatchLoop::new(config).unwrap(),
        &[("A", "same"), ("B", "different")],
        &selected_calls,
        1,
    );
    assert_eq!(changed.result.classes.len(), 1);
    assert_eq!(changed.result.classes[0].env_ids, ["A", "B"]);
    let child = &changed.result.child_histories[0].history;
    let checkpoint = child.load(&child.root().observations[0].continuation);
    assert_eq!(checkpoint.observations.len(), 2);
    let outputs: HashSet<_> = checkpoint
        .observations
        .iter()
        .map(|e| e.evidence.last().unwrap().outputs[0].tool_output.as_str())
        .collect();
    assert_eq!(outputs, HashSet::from(["foo.py:same", "foo.py:different"]));
    assert_eq!(
        changed
            .llms
            .iter()
            .filter(|r| r.messages[0].content.starts_with("imports:"))
            .map(|r| r.stats.depth)
            .collect::<Vec<_>>(),
        [2]
    );
    assert_eq!(
        base.load(&base.root().observations[0].continuation)
            .observations
            .len(),
        1
    );
    for path in ["bar.py", "./foo.py"] {
        let mut config = super_config("parent-file", &["old"]).with_record_store(store(&backend));
        config.history_resolver = Some(resolver.clone());
        let result = drive(
            BatchLoop::new(config).unwrap(),
            &[("old", "same")],
            &[calls(path)[0].clone()],
            3,
        );
        assert!(result.result.child_histories[0].base.is_none());
        assert_eq!(
            result.result.child_histories[0].history.descriptor.prompt,
            format!("imports:{path}")
        );
        assert!(
            result
                .tools
                .iter()
                .all(|r| r.tool_call.args["path"] == path)
        );
    }
    let mut config =
        super_config("parent-new-implementation", &["old"]).with_record_store(store(&backend));
    let rebuilt = SubagentTool::new(
        definitions()[0].spec.clone(),
        |a| format!("imports:{}", a["file_path"].as_str().unwrap()),
        |_| vec![read()],
        1,
    )
    .with_subagents(vec![]);
    config.subagents = Some(vec![rebuilt]);
    config.history_resolver = Some(resolver);
    let versioned = drive(
        BatchLoop::new(config).unwrap(),
        &[("old", "same")],
        &[calls("foo.py")[0].clone()],
        0,
    )
    .result;
    assert!(versioned.child_histories[0].base.is_some());
}

#[test]
fn concurrent_reuse_exposes_two_independent_updates_not_last_writer_or_single_flight() {
    let old = initial();
    let base = imports(&old);
    let backend = Backend::default();
    let descriptor = publish(&base, &backend);
    let lookup_base = descriptor.clone();
    let queries = Arc::new(Mutex::new(vec![]));
    let observed = queries.clone();
    let config = config("parent-concurrent", &["new"])
        .with_record_store(store(&backend))
        .with_history_resolver(move |q| {
            observed.lock().unwrap().push(q.clone());
            Ok(Some(lookup_base.clone()))
        });
    let invocations = vec![
        ToolCall::new("a", "list_imports", json!({"file_path": "foo.py"})),
        ToolCall::new(
            "a/turn-1/sub:root/b",
            "list_imports",
            json!({"file_path": "foo.py"}),
        ),
    ];
    let result = drive(
        BatchLoop::new(config).unwrap(),
        &[("new", "changed")],
        &invocations,
        2,
    )
    .result;
    let qs = queries.lock().unwrap();
    assert_eq!(qs.len(), 2);
    assert_eq!(qs[0], qs[1]);
    let updates = &result.child_histories;
    assert_eq!(updates.len(), 2);
    assert_eq!(
        updates
            .iter()
            .map(|u| &u.request_prefix)
            .collect::<HashSet<_>>()
            .len(),
        2
    );
    assert_eq!(
        updates
            .iter()
            .map(|u| &u.history.descriptor.root)
            .collect::<HashSet<_>>()
            .len(),
        2
    );
    assert_eq!(
        updates
            .iter()
            .map(|u| u.tool_call.id.as_str())
            .collect::<HashSet<_>>(),
        HashSet::from(["a", "a/turn-1/sub:root/b"])
    );
    for update in updates {
        assert_eq!(update.base.as_ref(), Some(&descriptor));
        assert_eq!(
            update
                .history
                .load(&update.history.root().observations[0].continuation)
                .observations
                .len(),
            2
        );
        publish(&update.history, &backend);
    }
    assert_eq!(
        base.load(&base.root().observations[0].continuation)
            .observations
            .len(),
        1
    );
}

#[test]
fn concurrent_parent_branches_publish_distinct_child_forks() {
    let old = initial();
    let base = imports(&old);
    let backend = Backend::default();
    let descriptor = publish(&base, &backend);
    let selected = descriptor.clone();
    let config = config("parent-split", &["A", "B"])
        .with_record_store(store(&backend))
        .with_history_resolver(move |_| Ok(Some(selected.clone())));
    let mut loop_ = BatchLoop::new(config).unwrap();
    let root = loop_.next_request(RequestKind::Any).unwrap();
    let split_call = ToolCall::new("split", "status", json!({}));
    let child_call = ToolCall::new("child", "list_imports", json!({"file_path": "foo.py"}));
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: root.request_id().into(),
            content: "split first".into(),
            tool_calls: vec![split_call.clone()],
        }))
        .unwrap();
    let tool = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Tool(ToolCompletion {
            request_id: tool.request_id().into(),
            results: vec![("A".into(), "alpha".into()), ("B".into(), "beta".into())],
        }))
        .unwrap();
    let branches = [
        loop_.next_request(RequestKind::Llm).unwrap(),
        loop_.next_request(RequestKind::Llm).unwrap(),
    ];
    for branch in branches.into_iter().rev() {
        loop_
            .complete(Completion::Llm(LlmCompletion {
                request_id: branch.request_id().into(),
                content: "delegate".into(),
                tool_calls: vec![child_call.clone()],
            }))
            .unwrap();
    }
    let result = drive(
        loop_,
        &[("A", "first variant"), ("B", "second variant")],
        &[split_call, child_call],
        2,
    )
    .result;
    assert_eq!(result.classes.len(), 2);
    assert_eq!(result.child_histories.len(), 2);
    let mut forks = BTreeMap::new();
    for update in &result.child_histories {
        assert_eq!(update.base.as_ref(), Some(&descriptor));
        let h = &update.history;
        let checkpoint = h.load(&h.root().observations[0].continuation);
        let variants: HashSet<String> = checkpoint
            .observations
            .iter()
            .map(|e| e.representative[0].tool_output.clone())
            .collect();
        forks.insert(update.env_ids[0].clone(), variants);
        publish(h, &backend);
    }
    assert_eq!(
        forks["A"],
        HashSet::from(["foo.py:same".into(), "foo.py:first variant".into()])
    );
    assert_eq!(
        forks["B"],
        HashSet::from(["foo.py:same".into(), "foo.py:second variant".into()])
    );
    assert_eq!(
        base.load(&base.root().observations[0].continuation)
            .observations
            .len(),
        1
    );
}

#[test]
fn incompatible_resolver_results_and_malformed_args_leave_fresh_requests_recoverable() {
    let old = initial();
    let backend = Backend::default();
    let descriptor = publish(&imports(&old), &backend);
    let mut legacy = serde_json::to_value(&descriptor).unwrap();
    let object = legacy.as_object_mut().unwrap();
    object.remove("subagents");
    object.insert("subagent".into(), serde_json::Value::Null);
    assert!(serde_json::from_value::<HistoryDescriptor<String>>(legacy).is_err());
    let mut wrong_prompt = descriptor.clone();
    wrong_prompt.prompt = "other".into();
    let mut wrong_nested = descriptor.clone();
    wrong_nested.subagents = old.history.descriptor.subagents.clone();
    let mut wrong_tools = descriptor.clone();
    wrong_tools.tools.clear();
    for bad in [wrong_prompt, wrong_nested, wrong_tools] {
        let config = config("parent-invalid", &["old"])
            .with_record_store(store(&backend))
            .with_history_resolver(move |_| Ok(Some(bad.clone())));
        let mut loop_ = BatchLoop::new(config).unwrap();
        let request = loop_.next_request(RequestKind::Any).unwrap();
        let before = loop_.history();
        assert_eq!(
            loop_.complete(Completion::Llm(LlmCompletion {
                request_id: request.request_id().into(),
                content: "bad".into(),
                tool_calls: vec![calls("foo.py")[0].clone()]
            })),
            Err(LoopError::IncompatibleHistory)
        );
        assert_eq!(loop_.history(), before);
        assert_eq!(loop_.outstanding_llm(), 1);
        assert!(loop_.error().is_none());
        assert!(loop_.next_request(RequestKind::Any).is_none());
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                request.request_id(),
                "corrected".into(),
            )))
            .unwrap();
        assert!(loop_.is_done());
        assert_eq!(loop_.result().unwrap().results["old"], "corrected");
    }
    let config = config("parent-invalid", &["old"])
        .with_record_store(store(&backend))
        .with_history_resolver(move |_| Ok(Some(descriptor.clone())));
    let mut loop_ = BatchLoop::new(config).unwrap();
    let request = loop_.next_request(RequestKind::Any).unwrap();
    let before = loop_.history();
    let c = calls("foo.py");
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "bad".into(),
            tool_calls: vec![
                c[2].clone(),
                c[0].clone(),
                ToolCall::new("broken", "count_lines", json!({}))
            ]
        })),
        Err(LoopError::InvalidSubagentArguments(_))
    ));
    assert_eq!(loop_.history(), before);
    assert_eq!(loop_.outstanding_count(), 1);
    assert!(loop_.next_request(RequestKind::Any).is_none());
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: request.request_id().into(),
            content: "bad".into(),
            tool_calls: vec![c[0].clone(), c[0].clone()]
        })),
        Err(LoopError::DuplicateToolCallId(_))
    ));
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "corrected".into(),
        )))
        .unwrap();
    assert_eq!(
        loop_
            .result()
            .unwrap()
            .history
            .root()
            .assistant
            .as_ref()
            .unwrap()
            .content,
        "corrected"
    );
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion::final_answer(
            request.request_id(),
            "overwrite".into()
        ))),
        Err(LoopError::UnknownRequest(_))
    ));
}

#[test]
fn resolver_error_after_a_prepared_hit_keeps_the_whole_turn_pending() {
    use std::sync::atomic::AtomicBool;
    let old = initial();
    let backend = Backend::default();
    let descriptor = publish(&imports(&old), &backend);
    let broken = Arc::new(AtomicBool::new(true));
    let fault = broken.clone();
    let queries = Arc::new(Mutex::new(vec![]));
    let seen = queries.clone();
    let config = config("parent-retry", &["new"])
        .with_record_store(store(&backend))
        .with_history_resolver(move |q| {
            seen.lock().unwrap().push(q.prompt.clone());
            if q.prompt.starts_with("imports:") {
                return Ok(Some(descriptor.clone()));
            }
            if fault.load(Ordering::Relaxed) {
                return Err(LoopError::HistoryResolution("cache unavailable".into()));
            }
            Ok(None)
        });
    let mut loop_ = BatchLoop::new(config).unwrap();
    let Request::Llm(request) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    let before = loop_.history();
    assert!(matches!(
        loop_.complete(Completion::Llm(model(&request, &calls("foo.py")))),
        Err(LoopError::HistoryResolution(_))
    ));
    assert_eq!(
        *queries.lock().unwrap(),
        ["imports:foo.py", "metrics:foo.py"]
    );
    assert_eq!(loop_.history(), before);
    assert_eq!(loop_.outstanding_count(), 1);
    assert!(loop_.error().is_none());
    assert!(loop_.next_request(RequestKind::Any).is_none());
    broken.store(false, Ordering::Relaxed);
    loop_
        .complete(Completion::Llm(model(&request, &calls("foo.py"))))
        .unwrap();
    let driven = drive(loop_, &[("new", "same")], &calls("foo.py"), 3);
    assert_eq!(driven.result.results["new"], "imports:done|metrics:done|ok");
    assert_eq!(driven.result.child_histories.len(), 2);
}

#[test]
fn registration_legacy_advertisements_and_all_serial_driver_entries() {
    let imports = definitions().remove(0);
    let metrics = definitions().remove(1);
    let base = || LoopConfig::new("p".to_owned(), vec!["e".into()], vec![]);
    assert!(matches!(
        BatchLoop::new(base().with_subagent(imports.clone()).with_subagents(vec![])),
        Err(LoopError::AmbiguousSubagents)
    ));
    assert!(matches!(
        BatchLoop::new(base().with_subagents(vec![imports.clone(), imports.clone()])),
        Err(LoopError::DuplicateToolName(_))
    ));
    let mut conflict = base().with_subagents(vec![imports.clone()]);
    conflict.tool_specs = vec![ToolSpec::new("list_imports", "wrong", json!({}))];
    assert!(matches!(
        BatchLoop::new(conflict),
        Err(LoopError::ConflictingToolSpec(_))
    ));
    let mut duplicate = base();
    duplicate.tool_specs = vec![read(), read()];
    assert!(matches!(
        BatchLoop::new(duplicate),
        Err(LoopError::DuplicateToolName(_))
    ));
    assert!(
        BatchLoop::new(
            base()
                .with_subagents(vec![imports.clone()])
                .with_history_resolver(|_| Ok(None))
        )
        .is_ok()
    );
    assert!(
        BatchLoop::new(
            base()
                .with_subagents(vec![imports.clone()])
                .with_budget(Budget::new(100, |_| "stop".into()))
        )
        .unwrap()
        .agent()
        .root()
        .assistant
        .is_none()
    );
    let mut legacy_config = base().with_subagent(imports.clone());
    legacy_config.tool_specs = vec![imports.spec.clone()];
    let mut legacy = BatchLoop::new(legacy_config).unwrap();
    let Request::Llm(request) = legacy.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(request.tools.len(), 1);
    let i = imports.clone();
    let result = run_to_completion(
        "parent-driver".to_owned(),
        vec!["e".into()],
        move |_| {
            vec![
                ToolEntry::Subagent(i.clone()),
                ToolEntry::Subagent(metrics.clone()),
                ToolEntry::Callable(Arc::new(FunctionTool::new(read(), |_| "value".to_owned()))),
            ]
        },
        |r| model(r, &calls("foo.py")[..2]),
        None,
    )
    .unwrap();
    assert_eq!(result.results["e"], "imports:done|metrics:done");
    assert_eq!(result.agent().root().subagents.len(), 2);
    assert_eq!(result.child_histories.len(), 2);
    assert!(result.child_histories.iter().all(|u| {
        serde_json::to_value(&u.history.descriptor)
            .unwrap()
            .get("custom_grouping")
            .is_none()
    }));
    let runtime_sub = imports.clone();
    assert!(matches!(
        run_to_completion(
            "parent-driver".to_owned(),
            vec!["a".into(), "b".into()],
            move |env| {
                let ordinary = Arc::new(FunctionTool::new(read(), |_| "value".to_owned()));
                let mut entries = vec![
                    ToolEntry::Subagent(runtime_sub.clone()),
                    ToolEntry::Callable(ordinary.clone()),
                ];
                if env == "b" {
                    entries.push(ToolEntry::Callable(ordinary));
                }
                entries
            },
            |r| model(r, &calls("foo.py")[..1]),
            None
        ),
        Err(LoopError::DuplicateToolName(_))
    ));
    let conflict_sub = imports.clone();
    let conflict_tool = Arc::new(FunctionTool::new(imports.spec.clone(), |_| {
        "ordinary".to_owned()
    }));
    assert!(matches!(
        run_to_completion(
            "p".to_owned(),
            vec!["e".into()],
            move |_| vec![
                ToolEntry::Subagent(conflict_sub.clone()),
                ToolEntry::Callable(conflict_tool.clone()),
            ],
            |r| LlmCompletion::final_answer(&r.request_id, "done".into()),
            None
        ),
        Err(LoopError::DuplicateToolName(_))
    ));
    assert!(matches!(
        run_to_completion(
            "p".to_owned(),
            vec!["e".into()],
            move |_| vec![
                ToolEntry::Subagent(imports.clone()),
                ToolEntry::Subagent(imports.clone())
            ],
            |r| LlmCompletion::final_answer(&r.request_id, "done".into()),
            None
        ),
        Err(LoopError::DuplicateToolName(_))
    ));
}

#[test]
fn explicit_nested_allowlists_cannot_reset_depth_or_leak_sibling_reserved_tools() {
    let mut leaf = definition("leaf", "leaf");
    leaf.max_depth = 99;
    let middle = SubagentTool::new(
        ToolSpec::new("middle", "", json!({})),
        |a| format!("middle:{}", a["file_path"].as_str().unwrap()),
        |_| vec![read(), ToolSpec::new("outer", "", json!({}))],
        99,
    )
    .with_subagents(vec![leaf]);
    let outer = SubagentTool::new(
        ToolSpec::new("outer", "", json!({})),
        |a| format!("outer:{}", a["file_path"].as_str().unwrap()),
        |_| vec![read(), ToolSpec::new("sibling", "", json!({}))],
        2,
    )
    .with_subagents(vec![middle]);
    let sibling = definition("sibling", "sibling");
    let mut config = config("parent-nested", &["old"]);
    config.subagents = Some(vec![outer, sibling]);
    let mut loop_ = BatchLoop::new(config).unwrap();
    let r = loop_.next_request(RequestKind::Any).unwrap();
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: "".into(),
            tool_calls: vec![ToolCall::new("o", "outer", json!({"file_path": "foo.py"}))],
        }))
        .unwrap();
    let Request::Llm(child) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(
        child
            .tools
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["read_file", "middle"]
    );
    let before = loop_.history();
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: child.request_id.clone(),
            content: "bad".into(),
            tool_calls: vec![ToolCall::new(
                "s",
                "sibling",
                json!({"file_path": "foo.py"})
            )]
        })),
        Err(LoopError::UnavailableSubagent(_))
    ));
    assert_eq!(loop_.history(), before);
    assert_eq!(loop_.outstanding_llm(), 1);
    loop_
        .complete(Completion::Llm(LlmCompletion {
            request_id: child.request_id,
            content: "".into(),
            tool_calls: vec![ToolCall::new("m", "middle", json!({"file_path": "foo.py"}))],
        }))
        .unwrap();
    let Request::Llm(grand) = loop_.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(
        grand
            .tools
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["read_file"]
    );
    assert!(matches!(
        loop_.complete(Completion::Llm(LlmCompletion {
            request_id: grand.request_id.clone(),
            content: "bad".into(),
            tool_calls: vec![ToolCall::new("l", "leaf", json!({"file_path": "foo.py"}))]
        })),
        Err(LoopError::UnavailableSubagent(_))
    ));
    loop_
        .complete(Completion::Llm(LlmCompletion::final_answer(
            grand.request_id,
            "nested answer".into(),
        )))
        .unwrap();
    for _ in 0..2 {
        let r = loop_.next_request(RequestKind::Any).unwrap();
        loop_
            .complete(Completion::Llm(LlmCompletion::final_answer(
                r.request_id(),
                "nested answer".into(),
            )))
            .unwrap();
    }
    assert!(loop_.is_done());
    let updates = loop_.result().unwrap().child_histories;
    assert_eq!(
        updates
            .iter()
            .map(|u| u.tool_call.name.as_str())
            .collect::<Vec<_>>(),
        ["middle", "outer"]
    );
    assert!(updates[0].history.descriptor.subagents.is_empty());
    assert_eq!(updates[1].history.descriptor.subagents[0].max_depth, 1);
}

#[test]
fn selected_feedback_revision_is_reused_without_automatically_updating_old_parent() {
    let old = drive(
        BatchLoop::new(config("parent-A", &["old"])).unwrap(),
        &[("old", "same")],
        &[calls("foo.py")[0].clone()],
        3,
    )
    .result;
    let child = imports(&old);
    let mut feedback = BatchLoop::new(
        LoopConfig::new(
            child.descriptor.prompt.clone(),
            vec!["old".into()],
            child.descriptor.tools.clone(),
        )
        .with_replay(child.clone())
        .with_feedback("Also inspect the header".into()),
    )
    .unwrap();
    loop {
        match feedback.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                assert_eq!(
                    r.messages.last().unwrap().content,
                    "Also inspect the header"
                );
                feedback
                    .complete(Completion::Llm(LlmCompletion::final_answer(
                        r.request_id,
                        "improved imports".into(),
                    )))
                    .unwrap();
                break;
            }
            Request::Tool(r) => {
                let a = &r.tool_call.args;
                let value = format!(
                    "{}:{}",
                    a["path"].as_str().unwrap(),
                    if a["checkpoint"] == 0 {
                        "prefix"
                    } else {
                        "same"
                    }
                );
                feedback
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results: vec![("old".into(), value)],
                    }))
                    .unwrap();
            }
        }
    }
    let improved = feedback.result().unwrap().history;
    let backend = Backend::default();
    let selected = publish(&improved, &backend);
    let lookup = selected.clone();
    let queries = Arc::new(Mutex::new(0));
    let count = queries.clone();
    let resolver = HistoryResolver::new(move |_: &HistoryQuery<String>| {
        *count.lock().unwrap() += 1;
        Ok(Some(lookup.clone()))
    });
    let mut c = config("parent-revised", &["new"]).with_record_store(store(&backend));
    c.history_resolver = Some(resolver.clone());
    let new = drive(
        BatchLoop::new(c).unwrap(),
        &[("new", "same")],
        &[calls("foo.py")[0].clone()],
        0,
    )
    .result;
    assert_eq!(new.results["new"], "improved imports");
    assert_eq!(new.child_histories[0].base.as_ref(), Some(&selected));
    let before = *queries.lock().unwrap();
    let mut c = config("parent-A", &["new"]).with_replay(old.history.clone());
    c.history_resolver = Some(resolver);
    let replay = drive(
        BatchLoop::new(c).unwrap(),
        &[("new", "same")],
        &[calls("foo.py")[0].clone()],
        0,
    );
    assert!(replay.llms.is_empty());
    assert_eq!(replay.result.results["new"], "imports:done");
    assert_eq!(*queries.lock().unwrap(), before);
    let history = replay.result.history;
    assert_eq!(history.root().subagents[0].runs.len(), 2);
    assert_eq!(
        history.root().subagents[0].runs[0].history,
        old.history.root().subagents[0].history
    );
}

#[test]
fn opaque_byte_prompts_are_verified_by_value_not_lookup_keys() {
    let tool = SubagentTool::new(
        ToolSpec::new("opaque", "", json!({})),
        |a| a["value"].as_str().unwrap().as_bytes().to_vec(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let mut first = BatchLoop::new(
        LoopConfig::new(b"parent-A".to_vec(), vec!["e".into()], vec![])
            .with_subagents(vec![tool.clone()]),
    )
    .unwrap();
    let r = first.next_request(RequestKind::Any).unwrap();
    first
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: vec![],
            tool_calls: vec![ToolCall::new("c", "opaque", json!({"value": "child"}))],
        }))
        .unwrap();
    let r = first.next_request(RequestKind::Any).unwrap();
    first
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id(),
            b"answer".to_vec(),
        )))
        .unwrap();
    let r = first.next_request(RequestKind::Any).unwrap();
    first
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id(),
            b"parent answer".to_vec(),
        )))
        .unwrap();
    let history = first.result().unwrap().child_histories.remove(0).history;
    let d = history.descriptor.clone();
    let mut second = BatchLoop::new(
        LoopConfig::new(b"parent-B".to_vec(), vec!["e".into()], vec![])
            .with_subagents(vec![tool])
            .with_record_store(history.store)
            .with_history_resolver(move |_| Ok(Some(d.clone()))),
    )
    .unwrap();
    let r = second.next_request(RequestKind::Any).unwrap();
    assert_eq!(
        second.complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: vec![],
            tool_calls: vec![ToolCall::new("c", "opaque", json!({"value": "different"}))]
        })),
        Err(LoopError::IncompatibleHistory)
    );
    assert_eq!(second.outstanding_llm(), 1);
    second
        .complete(Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: vec![],
            tool_calls: vec![ToolCall::new("c", "opaque", json!({"value": "child"}))],
        }))
        .unwrap();
    let Request::Llm(r) = second.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(r.messages[0].content, b"parent-B");
    assert_eq!(r.messages.last().unwrap().content, b"answer");
    second
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            b"parent answer".to_vec(),
        )))
        .unwrap();
    assert!(second.is_done());
}
