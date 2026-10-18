//! cargo run -p lockstep-loop --example specialized_subagents
//! Caller metadata selection + ordinary RecordStore loading. Initial prompts use USER role.
use lockstep_loop::{
    AgentDescriptor, BatchLoop, BatchResult, Completion, LlmCompletion, LoopConfig, LoopError,
    NodeRef, RecordStore, Request, RequestKind, SubagentTool, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

type Backend = Arc<Mutex<BTreeMap<NodeRef, Vec<u8>>>>;
type Cache = Arc<Mutex<BTreeMap<String, AgentDescriptor<String>>>>;
fn schema() -> Value {
    json!({"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"], "additionalProperties": false})
}
fn read_spec() -> ToolSpec {
    ToolSpec::new("read_file", "Read without modifying.", schema())
}
fn imports_prompt(args: &Value) -> Result<String, LoopError> {
    if args.as_object().is_none_or(|o| o.len() != 1) || !args["file_path"].is_string() {
        return Err(LoopError::InvalidSubagentArguments(
            "expected exactly one string file_path".into(),
        ));
    }
    // Caller-owned template. No global path normalization or fuzzy matching.
    Ok(format!(
        "List imports in {:?}.",
        args["file_path"].as_str().unwrap()
    ))
}
fn imports() -> SubagentTool<String> {
    SubagentTool::try_new(
        ToolSpec::new(
            "list_imports",
            "List packages imported by a file.",
            schema(),
        ),
        imports_prompt,
        |_| Ok(vec![read_spec()]),
        1,
    )
    .with_subagents(vec![])
}
fn investigate(
    parent_prompt: &str,
    env: &str,
    writer: &str,
    cache: &Cache,
    backend: &Backend,
) -> (BatchResult<String>, (usize, usize, usize)) {
    let read_backend = backend.clone();
    let store = RecordStore::with_loader(writer, move |reference| {
        Arc::new(serde_json::from_slice(&read_backend.lock().unwrap()[reference]).unwrap())
    });
    let selected = cache.clone();
    let config = LoopConfig::new(parent_prompt.into(), vec![env.into()], vec![])
        .with_subagents(vec![imports()])
        .with_record_store(store)
        .with_agent_resolver(move |query| {
            // Fixed configuration in this example; core still checks full compatibility.
            Ok(selected.lock().unwrap().get(&query.prompt).cloned())
        });
    let mut loop_ = BatchLoop::new(config).unwrap();
    let (mut parent_llms, mut child_llms, mut child_tools) = (0, 0, 0);
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(r) => {
                let reply = if r.messages[0].content == parent_prompt {
                    parent_llms += 1;
                    if r.messages.len() == 1 {
                        assert_eq!(
                            r.tools.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
                            ["list_imports"]
                        );
                        LlmCompletion {
                            request_id: r.request_id,
                            content: "Inspect imports.".into(),
                            tool_calls: vec![ToolCall::new(
                                "imports",
                                "list_imports",
                                json!({"file_path": "foo.py"}),
                            )],
                        }
                    } else {
                        LlmCompletion::final_answer(
                            r.request_id,
                            r.messages.last().unwrap().content.clone(),
                        )
                    }
                } else {
                    child_llms += 1;
                    assert_eq!(
                        r.messages[0].content,
                        imports_prompt(&json!({"file_path": "foo.py"})).unwrap()
                    );
                    if r.messages.len() == 1 {
                        LlmCompletion {
                            request_id: r.request_id,
                            content: "Read the file.".into(),
                            tool_calls: vec![ToolCall::new(
                                "read",
                                "read_file",
                                json!({"file_path": "foo.py"}),
                            )],
                        }
                    } else {
                        LlmCompletion::final_answer(r.request_id, "os, sys".into())
                    }
                };
                loop_.complete(Completion::Llm(reply)).unwrap(); // fake provider, owned by caller
            }
            Request::Tool(r) => {
                child_tools += 1;
                assert_eq!(r.tool_call.name, "read_file"); // never list_imports
                assert_eq!(r.env_ids, [env]); // inherited repo/commit, no per-file environments
                loop_
                    .complete(Completion::Tool(ToolCompletion {
                        request_id: r.request_id,
                        results: vec![(env.into(), "import os\nimport sys\n".into())],
                    }))
                    .unwrap();
            }
        }
    }
    (
        loop_.result().unwrap(),
        (parent_llms, child_llms, child_tools),
    )
}
fn publish(result: &BatchResult<String>, backend: &Backend) -> AgentDescriptor<String> {
    assert_eq!(result.child_agents().len(), 1); // concurrent updates need explicit caller coordination
    let agent = result.child_agents()[0].agent();
    let mut backend = backend.lock().unwrap();
    // Insert-only shallow records BEFORE publishing any descriptor. No parent
    // descriptor is required to load this child investigation in another run.
    for (reference, node) in agent.store.records() {
        let bytes = serde_json::to_vec(&node).unwrap();
        if let Some(old) = backend.get(&reference) {
            assert_eq!(old, &bytes);
        } else {
            backend.insert(reference, bytes);
        }
    }
    serde_json::from_slice(&serde_json::to_vec(&agent.descriptor).unwrap()).unwrap()
}
fn main() {
    let cache = Cache::default();
    let backend = Backend::default();
    // Reserve globally unique writer namespaces in durable applications. This
    // example uses distinct names in an ephemeral caller-owned storage domain.
    let (first, counts) = investigate(
        "Parent A: analyze dependencies",
        "repo@commit-1",
        "writer-A",
        &cache,
        &backend,
    );
    assert_eq!(counts, (2, 2, 1));
    let original = first.child_agents()[0].agent();
    let selected = publish(&first, &backend);
    cache
        .lock()
        .unwrap()
        .insert(selected.prompt.clone(), selected.clone());
    let (second, counts) = investigate(
        "Parent B: a completely different parent prompt",
        "repo@commit-2",
        "writer-B",
        &cache,
        &backend,
    );
    assert_eq!(counts, (2, 0, 1));
    assert_eq!(second.results["repo@commit-2"], "os, sys");
    let update = &second.child_agents()[0];
    assert_eq!(update.base.as_ref(), Some(&selected));
    assert_ne!(update.agent().descriptor.root, selected.root);
    assert_eq!(original.root().env_ids, ["repo@commit-1"]);
    let parent_root = second.agent().root();
    let provenance = parent_root.subagents[0].runs.last().unwrap();
    assert_eq!(provenance.base.as_ref(), Some(&selected));
    assert_eq!(provenance.agent_descriptor(), &update.agent().descriptor);
    let updated = publish(&second, &backend);
    cache
        .lock()
        .unwrap()
        .insert(updated.prompt.clone(), updated);
    println!("Changed parent: 2 fresh parent LLM calls, 0 child LLM calls, 1 child tool rerun");
}
