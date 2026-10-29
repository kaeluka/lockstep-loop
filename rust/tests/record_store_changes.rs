use lockstep_loop::{
    Agent, AgentConfig, AgentRun, AgentSnapshot, Completion, CursorError, HistoryNode,
    LlmCompletion, Message, NodeRef, RecordBatch, RecordStore, RequestKind, RunOptions,
    SubagentTool, ToolCall, ToolSpec, WriteCursor,
};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Barrier};

fn indices<P>(batch: &RecordBatch<P>) -> Vec<u64> {
    batch.records.iter().map(|(r, _)| r.index).collect()
}

#[test]
fn empty_append_retry_clone_and_old_cursors() {
    let store = RecordStore::<String>::new("writer");
    let empty = store.changes_since(None).unwrap();
    assert!(empty.records.is_empty());
    let empty_again = store.changes_since(Some(&empty.next_cursor)).unwrap();
    assert!(empty_again.records.is_empty());

    let first = store.put(HistoryNode::default());
    let batch = store.changes_since(Some(&empty_again.next_cursor)).unwrap();
    assert_eq!(indices(&batch), [0]);
    assert_eq!(batch.records[0].0, first);
    assert!(Arc::ptr_eq(&batch.records[0].1, &store.load(&first)));

    let retry = store.changes_since(Some(&empty.next_cursor)).unwrap();
    assert_eq!(retry.records, batch.records);
    assert!(Arc::ptr_eq(&retry.records[0].1, &batch.records[0].1));
    let clone = store.clone();
    let cursor: WriteCursor = batch.next_cursor.clone();
    assert!(
        clone
            .changes_since(Some(&cursor))
            .unwrap()
            .records
            .is_empty()
    );
    let second = clone.put(HistoryNode::default());
    let delta = store.changes_since(Some(&cursor)).unwrap();
    assert_eq!(indices(&delta), [1]);
    assert_eq!(delta.records[0].0, second);
    assert_eq!(indices(&clone.changes_since(Some(&cursor)).unwrap()), [1]);
    assert_eq!(
        indices(&store.changes_since(Some(&empty.next_cursor)).unwrap()),
        [0, 1]
    );
    assert_eq!(indices(&store.changes_since(None).unwrap()), [0, 1]);
    assert_eq!(
        store.records(),
        store
            .changes_since(None)
            .unwrap()
            .records
            .into_iter()
            .collect()
    );
    assert!(
        store
            .changes_since(Some(&delta.next_cursor))
            .unwrap()
            .records
            .is_empty()
    );
}

#[test]
fn cursors_identify_the_actual_writer_not_namespace_or_payload_type() {
    let store = RecordStore::<String>::new("same-namespace");
    let cursor = store.changes_since(None).unwrap().next_cursor;
    for namespace in ["different", "same-namespace"] {
        let foreign = RecordStore::<String>::new(namespace);
        assert_eq!(
            foreign.changes_since(Some(&cursor)).unwrap_err(),
            CursorError::ForeignStore
        );
        foreign.put(HistoryNode::default());
        assert_eq!(
            foreign.changes_since(Some(&cursor)).unwrap_err(),
            CursorError::ForeignStore
        );
        assert_eq!(indices(&foreign.changes_since(None).unwrap()), [0]);
    }
    let other_payload = RecordStore::<u8>::new("same-namespace");
    assert_eq!(
        other_payload.changes_since(Some(&cursor)).unwrap_err(),
        CursorError::ForeignStore
    );
    drop(store);
    // A surviving cursor must not accept a new writer after the original dies.
    let replacement = RecordStore::<String>::new("same-namespace");
    assert_eq!(
        replacement.changes_since(Some(&cursor)).unwrap_err(),
        CursorError::ForeignStore
    );
    assert!(!CursorError::ForeignStore.to_string().is_empty());
}

#[test]
fn loaded_records_are_excluded_and_reads_never_call_loader() {
    let base = RecordStore::<String>::new("base");
    let historical = base.put(HistoryNode::default());
    let loaded = base.load(&historical);
    let weak = Arc::downgrade(&loaded);
    let loader_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls = loader_calls.clone();
    let store = RecordStore::with_loader("fresh", move |reference| {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        base.load(reference)
    });
    assert!(store.changes_since(None).unwrap().records.is_empty());
    assert_eq!(loader_calls.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert!(Arc::ptr_eq(&store.load(&historical), &loaded));
    let empty = store.changes_since(None).unwrap();
    assert!(empty.records.is_empty());
    assert!(store.records().is_empty());
    let local = store.put(HistoryNode::default());
    let batch = store.changes_since(Some(&empty.next_cursor)).unwrap();
    assert_eq!(batch.records.len(), 1);
    assert_eq!(batch.records[0].0, local);
    assert_eq!(store.changes_since(None).unwrap().records, batch.records);
    assert!(Arc::ptr_eq(&store.load(&historical), &loaded));
    assert_eq!(loader_calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    drop(loaded);
    drop(store);
    // The live batch/cursors retain their local Arcs, not the cache or loader.
    assert!(weak.upgrade().is_none());
    assert_eq!(batch.records[0].0, local);
}

#[test]
fn small_deltas_share_arcs_without_cloning_payloads_or_exporting_the_prefix() {
    // Intentionally neither Clone nor Debug: the export API needs neither bound.
    struct Payload(u64);
    let store = RecordStore::<Payload>::new("writer");
    for i in 0..4096 {
        store.put(HistoryNode {
            assistant: Some(Message::assistant(Payload(i), vec![])),
            ..HistoryNode::default()
        });
    }
    let prefix = store.changes_since(None).unwrap();
    let first = &prefix.records[0].1;
    let references_before = Arc::strong_count(first);
    let cursor = prefix.next_cursor;
    let last = store.clone().put(HistoryNode {
        assistant: Some(Message::assistant(Payload(4096), vec![])),
        ..HistoryNode::default()
    });
    for _ in 0..8 {
        let delta = store.changes_since(Some(&cursor)).unwrap();
        assert_eq!(indices(&delta), [4096]);
        assert_eq!(delta.records[0].0, last);
        assert_eq!(
            delta.records[0].1.assistant.as_ref().unwrap().content.0,
            4096
        );
        assert!(Arc::ptr_eq(&delta.records[0].1, &store.load(&last)));
        assert_eq!(Arc::strong_count(first), references_before);
        assert!(
            store
                .changes_since(Some(&delta.next_cursor))
                .unwrap()
                .records
                .is_empty()
        );
    }
    // Full export remains non-destructive and works without P: Clone as well.
    assert_eq!(store.records().len(), 4097);
}

#[test]
fn concurrent_batches_have_exact_watermarks_and_independent_readers() {
    const ROUNDS: u64 = 64;
    const WRITERS: u64 = 2;
    const WRITES: u64 = 3;
    let store = RecordStore::<String>::new("concurrent");
    let barrier = Barrier::new(4);
    std::thread::scope(|scope| {
        for _ in 0..WRITERS {
            let store = store.clone();
            let barrier = &barrier;
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    barrier.wait();
                    for _ in 0..WRITES {
                        store.put(HistoryNode::default());
                    }
                    barrier.wait();
                }
            });
        }
        for _ in 0..2 {
            let store = store.clone();
            let barrier = &barrier;
            scope.spawn(move || {
                let mut cursor = store.changes_since(None).unwrap().next_cursor;
                let mut seen = Vec::new();
                for round in 0..ROUNDS {
                    barrier.wait();
                    // This batch races both writers. Its exact size is unspecified.
                    let racing = store.changes_since(Some(&cursor)).unwrap();
                    barrier.wait();
                    // Writers cannot enter the next round until both readers finish.
                    let tail = store.changes_since(Some(&racing.next_cursor)).unwrap();
                    let retry = store.changes_since(Some(&cursor)).unwrap();
                    let mut combined = indices(&racing);
                    combined.extend(indices(&tail));
                    let expected: Vec<_> =
                        (round * WRITERS * WRITES..(round + 1) * WRITERS * WRITES).collect();
                    assert_eq!(combined, expected); // no omissions or duplicates at watermark
                    assert_eq!(indices(&retry), expected);
                    seen.extend(combined);
                    cursor = tail.next_cursor;
                    assert!(
                        store
                            .changes_since(Some(&cursor))
                            .unwrap()
                            .records
                            .is_empty()
                    );
                }
                assert_eq!(seen, (0..ROUNDS * WRITERS * WRITES).collect::<Vec<_>>());
            });
        }
    });
    assert_eq!(store.records().len() as u64, ROUNDS * WRITERS * WRITES);
}

fn answer(run: &mut AgentRun<String>, value: &str) {
    let request = run.next_request(RequestKind::Llm).unwrap();
    run.complete(Completion::Llm(LlmCompletion::final_answer(
        request.request_id(),
        value.into(),
    )))
    .unwrap();
}

fn reachable(snapshot: &AgentSnapshot<String>) -> BTreeSet<NodeRef> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![snapshot.descriptor.root.clone()];
    while let Some(reference) = todo.pop() {
        if !seen.insert(reference.clone()) {
            continue;
        }
        let node = snapshot.load(&reference);
        todo.extend(node.observations.iter().map(|o| o.continuation.clone()));
        todo.extend(
            node.user_continuations
                .iter()
                .map(|u| u.continuation.clone()),
        );
        for child in &node.subagents {
            todo.push(child.snapshot.root.clone());
            for run in &child.runs {
                todo.push(run.snapshot.root.clone());
                todo.extend(run.base.iter().map(|base| base.root.clone()));
            }
        }
    }
    seen
}

#[test]
fn exports_orphans_other_runs_and_path_copy_ancestors_not_just_reachable_records() {
    let store = RecordStore::new("shared");
    let agent =
        Agent::new(AgentConfig::new("prompt".to_owned(), vec![]).with_record_store(store.clone()))
            .unwrap();
    let initial = agent.snapshot();
    let cursor = store.changes_since(None).unwrap().next_cursor;
    let mut left = agent
        .start(vec!["left".into()], RunOptions::default())
        .unwrap();
    let mut right = agent
        .start(vec!["right".into()], RunOptions::default())
        .unwrap();
    answer(&mut left, "left answer");
    let left_snapshot = left.snapshot();
    answer(&mut right, "right answer");
    let captured = agent.snapshot();
    let orphan = store.put(HistoryNode::default());
    let batch = store.changes_since(Some(&cursor)).unwrap();
    let records: BTreeMap<_, _> = batch.records.into_iter().collect();
    assert!(records.contains_key(&left_snapshot.descriptor.root));
    assert!(records.contains_key(&captured.descriptor.root));
    assert!(records.contains_key(&orphan));
    assert!(!records.contains_key(&initial.descriptor.root));
    assert!(records.len() > reachable(&captured).len());
    assert!(initial.root().assistant.is_none());
    assert_eq!(
        left_snapshot.root().assistant.as_ref().unwrap().content,
        "left answer"
    );
    assert_eq!(
        captured.root().assistant.as_ref().unwrap().content,
        "right answer"
    );
}

#[test]
fn capture_snapshot_before_batch_includes_path_copies_and_child_records() {
    let store = RecordStore::new("nested");
    let agent = Agent::new(
        AgentConfig::new("parent".to_owned(), vec![])
            .with_record_store(store.clone())
            .with_subagents(vec![SubagentTool::new(
                ToolSpec::new("child", "", json!({})),
                |_| "child prompt".to_owned(),
                |_| vec![],
                1,
            )]),
    )
    .unwrap();
    let mut run = agent
        .start(vec!["env".into()], RunOptions::default())
        .unwrap();
    let request = run.next_request(RequestKind::Llm).unwrap();
    let before = store.changes_since(None).unwrap();
    run.complete(Completion::Llm(LlmCompletion {
        request_id: request.request_id().into(),
        content: "investigate".into(),
        tool_calls: vec![ToolCall::new("c", "child", json!({}))],
    }))
    .unwrap();

    // Capture FIRST, then export. The earlier batch cannot cover this new root
    // and its open child history, including path-copy allocations.
    let captured = run.snapshot();
    assert!(
        !before
            .records
            .iter()
            .any(|(r, _)| r == &captured.descriptor.root)
    );
    let after = store.changes_since(Some(&before.next_cursor)).unwrap();
    assert!(
        after
            .records
            .iter()
            .any(|(r, _)| r == &captured.descriptor.root)
    );
    let mut durable: BTreeMap<_, _> = before.records.into_iter().collect();
    durable.extend(after.records);
    let cursor = after.next_cursor; // advance only after records are persisted
    let descriptor = captured.descriptor.clone(); // publish after records
    let base_records = durable.clone();
    let restored = AgentSnapshot {
        descriptor,
        store: RecordStore::with_loader("restored", move |r| base_records[r].clone()),
    };
    assert!(
        restored
            .store
            .changes_since(None)
            .unwrap()
            .records
            .is_empty()
    );
    assert_eq!(reachable(&restored), reachable(&captured));
    assert!(
        restored
            .store
            .changes_since(None)
            .unwrap()
            .records
            .is_empty()
    );

    answer(&mut run, "child answer");
    answer(&mut run, "parent answer");
    let final_snapshot = run.snapshot();
    let batch = store.changes_since(Some(&cursor)).unwrap();
    let delta_refs: BTreeSet<_> = batch.records.iter().map(|(r, _)| r.clone()).collect();
    let final_reachable = reachable(&final_snapshot);
    assert!(delta_refs.contains(&final_snapshot.descriptor.root));
    assert!(delta_refs.difference(&final_reachable).next().is_some()); // superseded path copies
    durable.extend(batch.records);
    assert!(final_reachable.iter().all(|r| durable.contains_key(r)));
    assert_eq!(reachable(&restored), reachable(&captured)); // old graph stays intact
    assert_eq!(final_snapshot.root().observations.len(), 1);
    assert!(captured.root().observations.is_empty());
}
