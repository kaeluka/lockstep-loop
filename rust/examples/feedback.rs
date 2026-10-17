//! Run: cargo run -p lockstep-loop --example feedback
//! Caller-side fake executors and shallow-record persistence, no sessions/endpoints.
use lockstep_loop::{
    BatchLoop, BatchResult, Completion, History, LlmCompletion, LlmRequest, LoopConfig, LoopError,
    NodeRef, RecordStore, Request, RequestKind, Role, ToolCall, ToolCompletion, ToolRequest,
    ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

const PROMPT: &str = "Inspect the repository";
const NOTE: &str = "Also inspect /foo and revise if needed";
fn specs() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new("read", "Read entry", json!({})),
        ToolSpec::new("foo", "Inspect /foo", json!({})),
    ]
}
fn config(
    envs: &[&str],
    previous: Option<History<String>>,
    feedback: Option<&str>,
) -> LoopConfig<String> {
    let config = LoopConfig::new(
        PROMPT.into(),
        envs.iter().map(|s| (*s).into()).collect(),
        specs(),
    );
    let mut config = match previous {
        Some(h) => config.with_replay(h, "feedback-example/v1"),
        None => config
            .with_recording("feedback-example/v1")
            .with_record_store(RecordStore::new("initial-writer")),
    };
    if let Some(content) = feedback {
        config = config.with_feedback(content.into());
    }
    config
}
struct Investigation {
    result: BatchResult<String>,
    llms: Vec<LlmRequest<String>>,
    tools: Vec<ToolRequest>,
}
fn investigate(
    envs: &[&str],
    previous: Option<History<String>>,
    feedback: Option<&str>,
) -> Investigation {
    let mut loop_ = BatchLoop::new(config(envs, previous, feedback)).unwrap();
    let (mut llms, mut tools) = (vec![], vec![]);
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                let calls = if r.messages.len() == 1 {
                    vec![ToolCall::new("read", "read", json!({}))]
                } else if r.messages.last().unwrap().role == Role::User {
                    vec![ToolCall::new("foo", "foo", json!({"path": "/foo"}))]
                } else {
                    vec![]
                };
                let answer = if r.messages.last().unwrap().content == "vulnerability" {
                    "found"
                } else {
                    "no issues"
                };
                loop_
                    .complete(Completion::Llm(LlmCompletion {
                        request_id: r.request_id.clone(),
                        content: if calls.is_empty() { answer } else { "inspect" }.into(),
                        tool_calls: calls,
                    }))
                    .unwrap();
                llms.push(r);
            }
            Request::Tool(r) => {
                let results = r
                    .env_ids
                    .iter()
                    .map(|env| {
                        let value = if r.tool_call.name == "read" {
                            if env == "D" { "unrelated" } else { "shared" }
                        } else if env == "B" {
                            "vulnerability"
                        } else {
                            "safe"
                        };
                        (env.clone(), value.into())
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
    Investigation {
        result: loop_.result().unwrap(),
        llms,
        tools,
    }
}

type Backend = BTreeMap<NodeRef, Vec<u8>>;
fn persist(history: &History<String>, backend: &mut Backend) -> Vec<u8> {
    // Records first; publish descriptor only after records are durable.
    for (reference, node) in history.store.records() {
        let bytes = serde_json::to_vec(&node).unwrap();
        if let Some(old) = backend.get(&reference) {
            assert_eq!(old, &bytes);
        } else {
            backend.insert(reference, bytes);
        }
    }
    serde_json::to_vec(&history.descriptor).unwrap()
}
fn restore(descriptor: &[u8], backend: Backend, fresh_writer: &str) -> History<String> {
    History {
        descriptor: serde_json::from_slice(descriptor).unwrap(),
        store: RecordStore::with_loader(fresh_writer, move |reference| {
            Arc::new(serde_json::from_slice(&backend[reference]).unwrap())
        }),
    }
}

fn main() {
    let old = investigate(&["A", "B", "D"], None, None)
        .result
        .history
        .unwrap();
    let revised = investigate(&["A"], Some(old.clone()), Some(NOTE));
    assert_eq!(revised.llms.len(), 2); // both AFTER resolving the old answer
    let messages = &revised.llms[0].messages;
    assert_eq!(messages[messages.len() - 2].content, "no issues");
    assert_eq!(messages.last().unwrap().content, NOTE);
    assert_eq!(revised.result.results["A"], "no issues"); // no reinjection for same answer
    assert!(
        old.load(&old.root().observations[0].continuation)
            .user_continuations
            .is_empty()
    );
    let mut backend = Backend::new();
    let old_descriptor = persist(&old, &mut backend);
    let descriptor = persist(&revised.result.history.unwrap(), &mut backend);
    let inherited = investigate(
        &["A", "B", "C", "D"],
        Some(restore(&descriptor, backend.clone(), "inherited-writer")),
        None,
    );
    assert_eq!(
        inherited.result.results,
        BTreeMap::from([
            ("A".into(), "no issues".into()),
            ("B".into(), "found".into()),
            ("C".into(), "no issues".into()),
            ("D".into(), "no issues".into()),
        ])
    );
    assert_eq!(inherited.llms.len(), 1);
    assert_eq!(inherited.llms[0].env_ids, ["B"]);
    assert!(
        inherited
            .tools
            .iter()
            .filter(|r| r.tool_call.name == "foo")
            .all(|r| r.env_ids == ["A", "B", "C"])
    );
    let descriptor = persist(&inherited.result.history.unwrap(), &mut backend);
    let repeated = investigate(
        &["B"],
        Some(restore(&descriptor, backend.clone(), "repeated-writer")),
        Some(NOTE),
    );
    assert_eq!(
        repeated.llms[0]
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>(),
        [PROMPT, NOTE, NOTE]
    );
    assert_eq!(repeated.result.results["B"], "found");
    let old_again = investigate(
        &["B"],
        Some(restore(&old_descriptor, backend, "old-reader")),
        None,
    );
    assert_eq!(old_again.result.results["B"], "no issues");
    assert!(old_again.llms.is_empty());
    assert!(old_again.tools.iter().all(|r| r.tool_call.name == "read"));

    // A replay miss consumes the valid result and terminates; it never generates a target.
    let mut failed = BatchLoop::new(config(&["changed"], Some(old.clone()), Some(NOTE))).unwrap();
    let request = failed.next_request(RequestKind::Any).unwrap();
    assert_eq!(
        failed.complete(Completion::Tool(ToolCompletion {
            request_id: request.request_id().into(),
            results: vec![("changed".into(), "unseen".into())],
        })),
        Err(LoopError::FeedbackResolutionMiss)
    );
    assert!(failed.is_done());
    assert_eq!(failed.error(), Some(&LoopError::FeedbackResolutionMiss));
    assert_eq!(failed.outstanding_count(), 0);
    assert_eq!(failed.history().unwrap(), old);
    println!(
        "shared feedback: A/B/C inherit; D unchanged; old snapshot reusable; replay miss explicit"
    );
}
