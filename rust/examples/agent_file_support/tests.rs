use super::*;
use lockstep_loop::{MessageEnvelope, Metadata, RequestStats, ToolCall};
use serde_json::json;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;

fn bundled() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/agent-examples/source-code-analyst.json")
}
fn request(messages: Vec<MessageEnvelope<Value>>) -> LlmRequest<Value> {
    LlmRequest {
        request_id: "request".into(),
        branch_id: "branch".into(),
        env_ids: vec!["repository".into()],
        messages,
        tools: vec![],
        metadata: Metadata::default(),
        stats: RequestStats {
            depth: 0,
            split_depth: 0,
            seq: 0,
        },
    }
}
fn response(content: &str) -> Value {
    json!({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]})
}
fn tool_response(name: &str, args: Value) -> Value {
    json!({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": null, "reasoning": "opaque reasoning", "reasoning_details": [{"type": "reasoning.encrypted", "data": "opaque", "signature": "sig"}], "tool_calls": [{"id": "call", "type": "function", "function": {"name": name, "arguments": args.to_string()}}]}}]})
}
#[test]
fn definition_links_templates_and_schema_validation() -> Result<()> {
    let definition = Definition::load(&bundled())?;
    assert_eq!(definition.specs.last().unwrap().name, "security_review");
    assert!(definition.specs.last().unwrap().parameters["properties"]["focus"].is_object());
    let instances = Instances::default();
    assert!(
        definition
            .instantiate(&json!({"path": "."}), &instances)
            .is_err()
    );
    assert!(
        definition
            .instantiate(&json!({"path": ".", "question": 1}), &instances)
            .is_err()
    );
    let messages =
        definition.instantiate(&json!({"path": ".", "question": "{{literal}}"}), &instances)?;
    assert!(
        messages[1]
            .content
            .as_str()
            .unwrap()
            .contains("{{literal}}")
    );
    assert_eq!(
        definition::render("{{x}} {{y}}", &json!({"x": true, "y": [1,2]}))?,
        "true [1,2]"
    );
    for bad in ["{{missing}}", "{{x", "x}}", "{{x.y}}"] {
        assert!(definition::render(bad, &json!({})).is_err());
    }
    let mut req = request(messages);
    req.tools = definition.specs.clone();
    assert!(instances.validate_answer(&req, &json!("report")).is_ok());
    assert!(
        instances
            .validate_answer(&req, &json!({"wrong": "type"}))
            .is_err()
    );
    assert!(compile_schema(&json!({"$ref": "https://example.com/schema"})).is_err());
    Ok(())
}
#[test]
fn link_cycles_unknown_versions_and_output_sources_fail() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let mut value: Value = serde_json::from_slice(&fs::read(bundled())?)?;
    value["tools"] =
        json!([{ "kind": "agent", "name": "self", "description": "cycle", "agent": "agent.json" }]);
    let file = dir.path().join("agent.json");
    fs::write(&file, value.to_string())?;
    assert!(Definition::load(&file).is_err());
    value["tools"] = json!([]);
    value["format_version"] = json!(99);
    fs::write(&file, value.to_string())?;
    assert!(Definition::load(&file).is_err());
    value["format_version"] = json!(1);
    value["output"]["source"]["kind"] = json!("unknown");
    fs::write(&file, value.to_string())?;
    assert!(Definition::load(&file).is_err());
    Ok(())
}
#[test]
fn files_ranges_wildcards_and_confinement() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::create_dir(dir.path().join("src"))?;
    fs::write(dir.path().join("a.rs"), "first\nneedle [x]\nthird\n")?;
    fs::write(dir.path().join("src/b.rs"), "needle two\n")?;
    let files = Files::new(dir.path())?;
    let read = files.execute(
        "read_file",
        &json!({"path": "a.rs", "from_line": 2, "num_of_lines": 1}),
    );
    assert_eq!(read["lines"], json!([{"line": 2, "text": "needle [x]"}]));
    assert_eq!(read["truncated"], false);
    let search = files.execute(
        "string_search",
        &json!({"path_wildcard": "**/*.rs", "text_wildcard": "*needle*"}),
    );
    assert_eq!(search["matches"].as_array().unwrap().len(), 2);
    assert_eq!(search["matches"][0]["path"], "a.rs");
    assert!(files::wildcard("* [x]", "needle [x]"));
    assert!(!files::wildcard("[x]", "x"));
    assert!(files::wildcard("?", "é"));
    assert!(!files::path_matches("*.rs", "src/b.rs"));
    assert!(files::path_matches("**/*.rs", "a.rs"));
    assert!(
        files.execute(
            "read_file",
            &json!({"path": "../secret", "from_line": 1, "num_of_lines": 1})
        )["error"]
            .is_string()
    );
    assert!(files.execute("list_dir", &json!({"path": "/"}))["error"].is_string());
    assert!(
        files.execute(
            "read_file",
            &json!({"path": "a.rs", "from_line": 0, "num_of_lines": 1})
        )["error"]
            .is_string()
    );
    assert_eq!(
        files.execute("list_dir", &json!({"path": "."}))["entries"][0]["name"],
        "a.rs"
    );
    fs::write(dir.path().join("large"), "x".repeat(70_000))?;
    assert_eq!(
        files.execute(
            "read_file",
            &json!({"path": "large", "from_line": 1, "num_of_lines": 1})
        )["truncated"],
        true
    );
    Ok(())
}
#[cfg(unix)]
#[test]
fn symlinks_never_escape_or_get_followed() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let outside = tempfile::tempdir()?;
    fs::write(outside.path().join("secret"), "secret")?;
    std::os::unix::fs::symlink(outside.path(), dir.path().join("link"))?;
    let files = Files::new(dir.path())?;
    assert!(
        files.execute(
            "read_file",
            &json!({"path": "link/secret", "from_line": 1, "num_of_lines": 1})
        )["error"]
            .is_string()
    );
    assert!(
        files.execute(
            "string_search",
            &json!({"path_wildcard": "**", "text_wildcard": "*"})
        )["matches"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    Ok(())
}
#[test]
fn provider_preserves_reasoning_and_rejects_incomplete_responses() -> Result<()> {
    let original = tool_response("list_dir", json!({"path": "."}));
    let completion = provider::parse_response("request", original.clone())?;
    let req = request(vec![
        MessageEnvelope::assistant(completion.content, completion.tool_calls),
        MessageEnvelope::tool(json!({"entries": []}), "call"),
    ]);
    let body = provider::request_body(&req, provider::MODEL)?;
    assert_eq!(body["reasoning"]["effort"], "high");
    assert_eq!(body["model"], "deepseek/deepseek-v4.1-flash-1017");
    assert_eq!(
        body["messages"][0]["reasoning_details"],
        original["choices"][0]["message"]["reasoning_details"]
    );
    assert_eq!(body["messages"][0]["reasoning"], "opaque reasoning");
    assert_eq!(body["messages"][1]["tool_call_id"], "call");
    assert!(body.to_string().find("repository").is_none());
    for finish in ["length", "content_filter", "error"] {
        let mut value = response("partial");
        value["choices"][0]["finish_reason"] = json!(finish);
        assert!(provider::parse_response("request", value).is_err());
    }
    let mut malformed = original;
    malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{");
    assert!(provider::parse_response("request", malformed).is_err());
    Ok(())
}

#[test]
fn mocked_http_drives_real_parent_child_and_files() -> Result<()> {
    let dir = tempfile::tempdir()?;
    fs::write(dir.path().join("main.rs"), "fn main() {}\n")?;
    let mut parent: Value = serde_json::from_slice(&fs::read(bundled())?)?;
    let child_source = bundled().with_file_name("security-reviewer.json");
    let mut child: Value = serde_json::from_slice(&fs::read(child_source)?)?;
    parent["manifest"]["model"] = json!("parent-model");
    parent["manifest"]["private_tag"] = json!("not-model-context");
    child["manifest"]["model"] = json!("child-model");
    fs::write(dir.path().join("agent.json"), parent.to_string())?;
    fs::write(dir.path().join("security-reviewer.json"), child.to_string())?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let endpoint = format!("http://{}/chat/completions", listener.local_addr()?);
    let server = std::thread::spawn(move || -> Result<Vec<Value>> {
        let mut bodies = Vec::new();
        let replies = [
            tool_response(
                "security_review",
                json!({"path": "main.rs", "focus": "Check entry point"}),
            ),
            tool_response(
                "read_file",
                json!({"path": "main.rs", "from_line": 1, "num_of_lines": 5}),
            ),
            response("No supported vulnerability in main.rs:1."),
            response("Summary: main.rs:1 is an empty entry point."),
        ];
        for reply in replies {
            let (mut stream, _) = listener.accept()?;
            stream.set_read_timeout(Some(Duration::from_secs(10)))?;
            let mut reader = BufReader::new(&mut stream);
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line)?;
                if line == "\r\n" {
                    break;
                }
                if let Some((key, value)) = line.split_once(':') {
                    if key.eq_ignore_ascii_case("content-length") {
                        length = value.trim().parse()?;
                    }
                    if key.eq_ignore_ascii_case("authorization") {
                        assert_eq!(value.trim(), "Bearer test-key");
                    }
                }
            }
            let mut body = vec![0; length];
            reader.read_exact(&mut body)?;
            bodies.push(serde_json::from_slice(&body)?);
            let reply = reply.to_string();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                reply.len(),
                reply
            )?;
        }
        Ok(bodies)
    });
    let provider =
        OpenRouter::with_endpoint(None, Duration::from_secs(10), "test-key".into(), endpoint)?;
    let result = investigate(
        Definition::load(&dir.path().join("agent.json"))?,
        &json!({"path": ".", "question": "Review entry point"}),
        &Files::new(dir.path())?,
        4,
        None,
        |request| provider.complete(request),
    )?;
    let bodies = server.join().unwrap()?;
    assert_eq!(
        result.results["repository"],
        "Summary: main.rs:1 is an empty entry point."
    );
    assert_eq!(result.child_updates().len(), 1);
    assert_eq!(
        bodies
            .iter()
            .map(|body| body["model"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["parent-model", "child-model", "child-model", "parent-model"]
    );
    assert!(
        bodies
            .iter()
            .all(|body| !body.to_string().contains("not-model-context"))
    );
    assert!(
        bodies[2]["messages"].as_array().unwrap().last().unwrap()["content"]
            .as_str()
            .unwrap()
            .contains("fn main() {}")
    );
    assert_eq!(
        bodies[3]["messages"].as_array().unwrap().last().unwrap()["content"],
        "No supported vulnerability in main.rs:1."
    );
    assert_eq!(bodies[3]["messages"][2]["reasoning"], "opaque reasoning");
    Ok(())
}
#[test]
fn budget_and_invalid_model_arguments_fail_without_fake_answers() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let files = Files::new(dir.path())?;
    let inputs = json!({"path": ".", "question": "Review"});
    let result = investigate(
        Definition::load(&bundled())?,
        &inputs,
        &files,
        1,
        None,
        |request| {
            Ok(LlmCompletion {
                request_id: request.request_id.clone(),
                content: json!("inspect"),
                tool_calls: vec![ToolCall::new("call", "list_dir", json!({"path": "."}))],
            })
        },
    );
    assert!(result.unwrap_err().to_string().contains("budget exhausted"));
    let result = investigate(
        Definition::load(&bundled())?,
        &inputs,
        &files,
        3,
        None,
        |request| {
            Ok(LlmCompletion {
                request_id: request.request_id.clone(),
                content: json!("inspect"),
                tool_calls: vec![ToolCall::new(
                    "call",
                    "read_file",
                    json!({"path": "main.rs", "from_line": 0, "num_of_lines": 1}),
                )],
            })
        },
    );
    assert!(result.unwrap_err().to_string().contains("JSON Schema"));
    Ok(())
}
