use lockstep_loop::*;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

fn config() -> AgentConfig<String> {
    AgentConfig::new(
        "greetings".into(),
        vec![ToolSpec::new("read", "Read greeting", json!({}))],
    )
}
fn start(agent: &Agent<String>, env: &str) -> AgentRun<String> {
    agent
        .start(vec![env.into()], RunOptions::default())
        .unwrap()
}
fn drive(run: &mut AgentRun<String>, output: &str) -> usize {
    let mut llms = 0;
    while !run.is_done() {
        let mut requests = vec![];
        while let Some(r) = run.next_request(RequestKind::Any) {
            requests.push(r);
        }
        assert!(!requests.is_empty());
        for r in requests.into_iter().rev() {
            match r {
                Request::Llm(r) => {
                    llms += 1;
                    let calls = if r.messages.len() == 1 {
                        vec![
                            ToolCall::new("a", "read", json!({})),
                            ToolCall::new("b", "read", json!({})),
                        ]
                    } else {
                        vec![]
                    };
                    let content = if calls.is_empty() {
                        r.messages.last().unwrap().content.clone()
                    } else {
                        "inspect".into()
                    };
                    run.complete(Completion::Llm(LlmCompletion {
                        request_id: r.request_id,
                        content,
                        tool_calls: calls,
                    }))
                    .unwrap();
                }
                Request::Tool(r) => {
                    run.complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results: r
                            .env_ids
                            .into_iter()
                            .rev()
                            .map(|e| (e, output.into()))
                            .collect(),
                    }))
                    .unwrap();
                }
            }
        }
    }
    llms
}
fn answer(run: &mut AgentRun<String>, value: &str) {
    let r = run.next_request(RequestKind::Any).unwrap();
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        r.request_id(),
        value.into(),
    )))
    .unwrap();
}
fn call(run: &mut AgentRun<String>, name: &str) {
    let r = run.next_request(RequestKind::Any).unwrap();
    run.complete(Completion::Llm(LlmCompletion {
        request_id: r.request_id().into(),
        content: "inspect".into(),
        tool_calls: vec![ToolCall::new("opaque/sub:1", name, json!({}))],
    }))
    .unwrap();
}

#[test]
fn fresh_sequential_branch_and_snapshot_backed_assets() {
    let agent = Agent::new(config()).unwrap();
    let empty = agent.snapshot();
    let mut run = start(&agent, "a");
    assert_eq!(run.snapshot(), empty);
    assert_eq!(drive(&mut run, "hello"), 2);
    let old = agent.snapshot(); // automatic publication, before result()
    assert_eq!(old, run.snapshot());
    assert_eq!(run.result().unwrap().snapshot, old);
    assert!(empty.root().assistant.is_none());
    let asset = old.assets().next().unwrap();
    assert_eq!(asset.value().unwrap(), "hello");
    assert_eq!(asset.observations()[0].representative.len(), 2);
    assert_eq!(
        drive(&mut start(&agent, "b"), &format!("{}{}", "hel", "lo")),
        0
    );
    assert_eq!(drive(&mut start(&agent, "c"), "hola"), 1);
    assert_eq!(agent.snapshot().root().observations.len(), 2);
    assert_eq!(old.root().observations.len(), 1);
    assert_eq!(asset.observations()[0].evidence.len(), 1);
    for value in ["hello", "hola"] {
        assert_eq!(drive(&mut start(&agent, "again"), value), 0);
    }
    let fork = Agent::from_snapshot(old.clone(), AgentRuntime::default()).unwrap();
    assert_eq!(drive(&mut start(&fork, "fork"), "hello"), 0);
    assert_eq!(fork.snapshot().root().observations.len(), 1);
    assert_eq!(asset.value().unwrap(), "hello");
}

#[test]
fn overlaps_both_completion_orders_last_wins_and_old_inspection_never_republishes() {
    for reverse in [false, true] {
        let agent = Agent::new(config()).unwrap();
        drive(&mut start(&agent, "seed"), "known");
        let base = agent.snapshot();
        let mut left = start(&agent, "left");
        let mut right = start(&agent, "right");
        assert_eq!(agent.snapshot(), base);
        let (first, last, first_value, last_value) = if reverse {
            (&mut right, &mut left, "right", "left")
        } else {
            (&mut left, &mut right, "left", "right")
        };
        assert_eq!(drive(first, first_value), 1);
        let first_snapshot = first.snapshot();
        assert_eq!(agent.snapshot(), first_snapshot);
        assert_eq!(drive(last, last_value), 1);
        let last_snapshot = last.snapshot();
        assert_eq!(agent.snapshot(), last_snapshot);
        for (snapshot, value) in [(&first_snapshot, first_value), (&last_snapshot, last_value)] {
            let outputs: Vec<_> = snapshot
                .root()
                .observations
                .iter()
                .map(|e| e.representative[0].tool_output.clone())
                .collect();
            assert_eq!(outputs, ["known", value]);
            let fork = Agent::from_snapshot(snapshot.clone(), AgentRuntime::default()).unwrap();
            assert_eq!(drive(&mut start(&fork, "again"), value), 0);
        }
        first.result().unwrap();
        first.snapshot();
        assert!(first.is_done());
        assert!(first.next_request(RequestKind::Any).is_none());
        assert!(
            first
                .complete(Completion::Llm(LlmCompletion::final_answer(
                    "bogus",
                    "stale".into()
                )))
                .is_err()
        );
        assert_eq!(agent.snapshot(), last_snapshot);
        assert_eq!(drive(&mut start(&agent, "new"), last_value), 0);
        assert_eq!(base.root().observations.len(), 1);
    }
}

#[test]
fn storage_roundtrip_lazy_restore_uses_snapshot_configuration_and_raw_evidence() {
    let agent = Agent::with_comparison_key(config(), |s: &String| s.trim().to_owned()).unwrap();
    drive(&mut start(&agent, "a"), "hello ");
    drive(&mut start(&agent, "b"), "hello");
    drive(&mut start(&agent, "other"), "different");
    let old = agent.snapshot();
    let mut backend: BTreeMap<_, _> = old
        .store
        .records()
        .into_iter()
        .map(|(r, n)| (r, serde_json::to_vec(&n).unwrap()))
        .collect();
    let descriptor: AgentDescriptor<String> =
        serde_json::from_slice(&serde_json::to_vec(&old.descriptor).unwrap()).unwrap();
    let loaded = Arc::new(Mutex::new(vec![]));
    let loads = loaded.clone();
    let records = backend.clone();
    let snapshot = AgentSnapshot {
        descriptor,
        store: RecordStore::with_loader("restore", move |r| {
            loads.lock().unwrap().push(r.clone());
            Arc::new(serde_json::from_slice(&records[r]).unwrap())
        }),
    };
    let restored = Agent::from_snapshot_by(snapshot, AgentRuntime::default(), |s: &String| {
        s.trim().to_owned()
    })
    .unwrap();
    assert!(loaded.lock().unwrap().is_empty());
    let mut run = start(&restored, "today");
    assert_eq!(drive(&mut run, "hello\n"), 0);
    assert_eq!(run.result().unwrap().results["today"], "hello ");
    assert!(
        !loaded
            .lock()
            .unwrap()
            .contains(&old.root().observations[1].continuation)
    );
    let current = restored.snapshot();
    let values: Vec<_> = current.root().observations[0]
        .evidence
        .iter()
        .map(|e| e.outputs[0].tool_output.clone())
        .collect();
    assert_eq!(values, ["hello ", "hello", "hello\n"]);
    backend.extend(
        current
            .store
            .records()
            .into_iter()
            .map(|(r, n)| (r, serde_json::to_vec(&n).unwrap())),
    );
    let reopened = Agent::from_snapshot_by(
        AgentSnapshot {
            descriptor: serde_json::from_slice(&serde_json::to_vec(&current.descriptor).unwrap())
                .unwrap(),
            store: RecordStore::with_loader("reopen", move |r| {
                Arc::new(serde_json::from_slice(&backend[r]).unwrap())
            }),
        },
        AgentRuntime::default(),
        |s: &String| s.trim().to_owned(),
    )
    .unwrap();
    assert_eq!(drive(&mut start(&reopened, "again"), "hello"), 0);
    assert_eq!(old.root().observations[0].evidence.len(), 2);
}

#[test]
fn final_only_start_publishes_immediately_invalid_completions_are_correctable() {
    let agent = Agent::new(AgentConfig::new("p".to_owned(), vec![])).unwrap();
    let empty = agent.snapshot();
    let mut run = start(&agent, "a");
    let r = run.next_request(RequestKind::Any).unwrap();
    assert!(
        run.complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id().into(),
            results: vec![("a".into(), "wrong".into())]
        }))
        .is_err()
    );
    assert_eq!(run.outstanding_count(), 1);
    assert_eq!(agent.snapshot(), empty);
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        r.request_id(),
        "answer".into(),
    )))
    .unwrap();
    let first = agent.snapshot();
    let mut replay = start(&agent, "b");
    let latest = agent.snapshot(); // no polling/inspection of replay was needed
    assert_eq!(latest.root().env_ids, ["a", "b"]);
    assert!(replay.is_done());
    run.result().unwrap();
    run.snapshot();
    assert_eq!(agent.snapshot(), latest);
    let fork = Agent::from_snapshot(first.clone(), AgentRuntime::default()).unwrap();
    start(&fork, "c");
    assert_eq!(fork.snapshot().root().env_ids, ["a", "c"]);
    assert_eq!(agent.snapshot(), latest);
    assert_eq!(first.root().env_ids, ["a"]);
}

#[test]
fn abandoned_and_failed_runs_do_not_publish_budget_publishes_only_real_history() {
    let agent = Agent::new(config()).unwrap();
    let empty = agent.snapshot();
    let mut abandoned = start(&agent, "a");
    call(&mut abandoned, "read");
    let partial = abandoned.snapshot();
    drop(abandoned);
    assert_eq!(agent.snapshot(), empty);
    let mut bounded = agent
        .start(
            vec!["a".into()],
            RunOptions {
                budget: Some(Budget::new(0, |_| "fallback".into())),
                feedback: None,
            },
        )
        .unwrap();
    assert_eq!(agent.snapshot(), bounded.snapshot());
    assert!(agent.snapshot().root().assistant.is_none());
    assert_eq!(bounded.result().unwrap().results["a"], "fallback");
    assert!(
        agent
            .start(
                vec!["a".into()],
                RunOptions {
                    feedback: Some("no target".into()),
                    budget: None
                }
            )
            .is_err()
    );
    assert_eq!(agent.snapshot(), empty);
    let resumed = Agent::from_snapshot(partial.clone(), AgentRuntime::default()).unwrap();
    let mut run = resumed
        .start(
            vec!["a".into()],
            RunOptions {
                budget: Some(Budget::new(1, |_| "fallback".into())),
                feedback: None,
            },
        )
        .unwrap();
    let r = run.next_request(RequestKind::Any).unwrap();
    let before = run.snapshot();
    assert!(
        run.complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id().into(),
            results: vec![]
        }))
        .is_err()
    );
    assert_eq!(resumed.snapshot(), partial);
    assert_eq!(run.snapshot(), before);
    run.complete(Completion::Tool(ToolCompletion {
        request_id: r.request_id().into(),
        results: vec![("a".into(), "seen".into())],
    }))
    .unwrap();
    let checkpoint = resumed.snapshot();
    assert!(
        checkpoint
            .load(&checkpoint.root().observations[0].continuation)
            .assistant
            .is_none()
    );
    assert_eq!(drive(&mut start(&resumed, "a"), "seen"), 1);
    let known = resumed.snapshot();
    let mut failed = resumed
        .start(
            vec!["a".into()],
            RunOptions {
                feedback: Some("revise".into()),
                budget: None,
            },
        )
        .unwrap();
    let r = failed.next_request(RequestKind::Any).unwrap();
    assert_eq!(
        failed.complete(Completion::Tool(ToolCompletion {
            request_id: r.request_id().into(),
            results: vec![("a".into(), "unseen".into())]
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert!(failed.is_done());
    assert!(failed.result().is_err());
    assert_eq!(resumed.snapshot(), known);
    assert!(
        checkpoint
            .load(&checkpoint.root().observations[0].continuation)
            .assistant
            .is_none()
    );
}

#[test]
fn feedback_budget_publishes_open_continuation_and_keeps_old_asset_anchored() {
    let agent = Agent::new(config()).unwrap();
    answer(&mut start(&agent, "a"), "old");
    let old = agent.snapshot();
    let asset = old.assets().next().unwrap();
    let mut bounded = agent
        .start(
            vec!["a".into()],
            RunOptions {
                feedback: Some("revise".into()),
                budget: Some(Budget::new(1, |_| "fallback".into())),
            },
        )
        .unwrap();
    let opened = agent.snapshot();
    assert_eq!(opened, bounded.snapshot());
    assert!(!opened.assets().next().unwrap().is_complete());
    assert_eq!(opened.assets().next().unwrap().value().unwrap(), "old");
    let mut followup = start(&agent, "b");
    let Request::Llm(r) = followup.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(
        r.messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        ["greetings", "old", "revise"]
    );
    followup
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            "new".into(),
        )))
        .unwrap();
    let newest = agent.snapshot();
    bounded.result().unwrap();
    assert_eq!(agent.snapshot(), newest);
    assert!(old.root().user_continuations.is_empty());
    assert!(asset.is_complete());
    assert_eq!(asset.value().unwrap(), "old");
}

#[test]
fn nested_child_cache_updates_are_snapshots_and_cached_start_publishes() {
    let leaf = SubagentTool::new(
        ToolSpec::new("leaf", "", json!({})),
        |_| "leaf prompt".to_owned(),
        |_| vec![],
        1,
    )
    .with_subagents(vec![]);
    let outer = SubagentTool::new(
        ToolSpec::new("outer", "", json!({})),
        |_| "outer prompt".to_owned(),
        |_| vec![],
        2,
    )
    .with_subagents(vec![leaf]);
    let runtime = AgentRuntime {
        subagents: vec![outer.clone()],
        agent_resolver: None,
    };
    let agent = Agent::new(
        AgentConfig::new("parent".to_owned(), vec![]).with_subagents(vec![outer.clone()]),
    )
    .unwrap();
    let mut run = start(&agent, "a");
    call(&mut run, "outer");
    call(&mut run, "leaf");
    for value in ["leaf value", "outer value", "parent value"] {
        answer(&mut run, value);
    }
    let old = agent.snapshot();
    let result = run.result().unwrap();
    let child = result
        .child_updates
        .iter()
        .find(|u| u.tool_call.name == "outer")
        .unwrap()
        .snapshot
        .clone();
    assert_eq!(old.child(&old.root().subagents[0]), child);
    assert!(Agent::from_snapshot(old.clone(), AgentRuntime::default()).is_err());
    let backend: BTreeMap<_, _> = old
        .store
        .records()
        .into_iter()
        .map(|(r, n)| (r, serde_json::to_vec(&n).unwrap()))
        .collect();
    let loaded = Arc::new(Mutex::new(vec![]));
    let loads = loaded.clone();
    let imported = AgentSnapshot {
        descriptor: serde_json::from_slice(&serde_json::to_vec(&old.descriptor).unwrap()).unwrap(),
        store: RecordStore::with_loader("nested-restore", move |r| {
            loads.lock().unwrap().push(r.clone());
            Arc::new(serde_json::from_slice(&backend[r]).unwrap())
        }),
    };
    let restored = Agent::from_snapshot(imported.clone(), runtime).unwrap();
    assert!(loaded.lock().unwrap().is_empty());
    assert_eq!(
        imported.assets().next().unwrap().value().unwrap(),
        "parent value"
    );
    assert_eq!(loaded.lock().unwrap().len(), 2);
    assert!(!loaded.lock().unwrap().contains(&child.descriptor.root));
    let imported_child = imported.child(&imported.root().subagents[0]);
    assert_eq!(
        imported_child.assets().next().unwrap().value().unwrap(),
        "outer value"
    );
    assert_eq!(loaded.lock().unwrap().len(), 4);
    assert!(
        !loaded
            .lock()
            .unwrap()
            .contains(&child.root().subagents[0].snapshot.root)
    );
    let mut cached = start(&restored, "b");
    assert_eq!(restored.snapshot().root().env_ids, ["a", "b"]);
    assert!(cached.is_done());
    assert_eq!(cached.result().unwrap().child_updates.len(), 2);
    let selected = child.descriptor.clone();
    let parent = Agent::new(
        AgentConfig::new("changed parent".to_owned(), vec![])
            .with_subagents(vec![outer])
            .with_record_store(child.store.clone())
            .with_agent_resolver(AgentResolver::new(move |q| {
                Ok((q == &selected.query()).then(|| selected.clone()))
            })),
    )
    .unwrap();
    let mut other = start(&parent, "c");
    call(&mut other, "outer");
    let Request::Llm(r) = other.next_request(RequestKind::Any).unwrap() else {
        panic!()
    };
    assert_eq!(r.messages[0].content, "changed parent");
    assert_eq!(r.messages.last().unwrap().content, "outer value");
    other
        .complete(Completion::Llm(LlmCompletion::final_answer(
            r.request_id,
            "new parent value".into(),
        )))
        .unwrap();
    assert_eq!(parent.snapshot(), other.snapshot());
    assert_eq!(old.root().env_ids, ["a"]);
    assert_eq!(child.root().env_ids, ["a"]);
}

#[test]
fn last_tool_completion_publishes_without_polling_and_failure_cannot_revert_new_head() {
    let agent = Agent::new(config()).unwrap();
    drive(&mut start(&agent, "old"), "known");
    let old = agent.snapshot();
    let mut failed = agent
        .start(
            vec!["bad".into()],
            RunOptions {
                feedback: Some("revise".into()),
                budget: None,
            },
        )
        .unwrap();
    let mut replay = start(&agent, "new");
    let requests = [
        replay.next_request(RequestKind::Any).unwrap(),
        replay.next_request(RequestKind::Any).unwrap(),
    ];
    for r in requests.into_iter().rev() {
        replay
            .complete(Completion::Tool(ToolCompletion {
                request_id: r.request_id().into(),
                results: vec![("new".into(), "known".into())],
            }))
            .unwrap();
    }
    // No polling or inspection of replay after completing the final tool tuple.
    let newer = agent.snapshot();
    assert_eq!(newer.root().env_ids, ["old", "new"]);
    assert_eq!(
        newer
            .load(&newer.root().observations[0].continuation)
            .env_ids,
        ["old", "new"]
    );
    let first = failed.next_request(RequestKind::Any).unwrap();
    let second = failed.next_request(RequestKind::Any).unwrap();
    failed
        .complete(Completion::Tool(ToolCompletion {
            request_id: first.request_id().into(),
            results: vec![("bad".into(), "unseen".into())],
        }))
        .unwrap();
    assert_eq!(
        failed.complete(Completion::Tool(ToolCompletion {
            request_id: second.request_id().into(),
            results: vec![("bad".into(), "unseen".into())],
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert_eq!(agent.snapshot(), newer);
    assert_eq!(failed.snapshot(), old);
}

#[test]
fn cached_child_rollup_failure_is_terminal_without_publication_and_fresh_errors_are_correctable() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let broken = Arc::new(AtomicBool::new(false));
    let flag = broken.clone();
    let sub = SubagentTool::try_new(
        ToolSpec::new("child", "", json!({})),
        move |args| {
            if flag.load(Ordering::SeqCst) && args["name"] == "second" {
                return Err(LoopError::SubagentPreparation("unavailable".into()));
            }
            Ok(args["name"].as_str().unwrap().to_owned())
        },
        |_| Ok(vec![]),
        1,
    )
    .with_subagents(vec![]);
    let agent = Agent::new(config().with_subagents(vec![sub])).unwrap();
    let mut run = start(&agent, "old");
    call(&mut run, "read");
    let r = run.next_request(RequestKind::Any).unwrap();
    run.complete(Completion::Tool(ToolCompletion {
        request_id: r.request_id().into(),
        results: vec![("old".into(), "known".into())],
    }))
    .unwrap();
    for name in ["first", "second"] {
        let r = run.next_request(RequestKind::Any).unwrap();
        let completion = Completion::Llm(LlmCompletion {
            request_id: r.request_id().into(),
            content: "delegate".into(),
            tool_calls: vec![ToolCall::new(name, "child", json!({"name": name}))],
        });
        if name == "second" {
            let before = run.snapshot();
            let head = agent.snapshot();
            broken.store(true, Ordering::SeqCst);
            assert!(run.complete(completion.clone()).is_err());
            assert_eq!(run.snapshot(), before);
            assert_eq!(run.outstanding_count(), 1);
            assert_eq!(agent.snapshot(), head);
            broken.store(false, Ordering::SeqCst);
        }
        run.complete(completion).unwrap();
        answer(&mut run, &format!("{name} answer"));
    }
    answer(&mut run, "done");
    let old = agent.snapshot();
    let mut replay = start(&agent, "new");
    let r = replay.next_request(RequestKind::Any).unwrap();
    broken.store(true, Ordering::SeqCst);
    assert!(
        replay
            .complete(Completion::Tool(ToolCompletion {
                request_id: r.request_id().into(),
                results: vec![("new".into(), "known".into())],
            }))
            .is_err()
    );
    assert_eq!(agent.snapshot(), old);
    assert!(replay.is_done() && replay.error().is_some());
    broken.store(false, Ordering::SeqCst);
    assert_eq!(drive(&mut start(&agent, "again"), "known"), 0);
    let newer = agent.snapshot();
    assert!(replay.result().is_err());
    replay.snapshot();
    assert_eq!(agent.snapshot(), newer);
}
