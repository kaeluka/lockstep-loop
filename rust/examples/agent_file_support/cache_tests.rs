use super::*;
use lockstep_loop::ToolCall;
use serde_json::json;
use std::fs;

fn fixture() -> Result<(tempfile::TempDir, PathBuf)> {
    let dir = tempfile::tempdir()?;
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/agent-examples");
    for name in ["source-code-analyst.json", "security-reviewer.json"] {
        fs::copy(source.join(name), dir.path().join(name))?;
    }
    fs::write(dir.path().join("main.rs"), "safe\n")?;
    let agent = dir.path().join("source-code-analyst.json");
    Ok((dir, agent))
}

fn inputs(question: &str) -> Value {
    json!({"path": ".", "question": question})
}

fn tool(request: &LlmRequest<Value>, name: &str, args: Value) -> LlmCompletion<Value> {
    LlmCompletion {
        request_id: request.request_id.clone(),
        content: json!({"openrouter_message": {"content": null, "reasoning": "preserved reasoning"}}),
        tool_calls: vec![ToolCall::new("call", name, args)],
    }
}

fn cached_run(
    path: &Path,
    repository: &Path,
    model_name: &str,
    inputs: &Value,
    model: impl FnMut(&LlmRequest<Value>) -> Result<LlmCompletion<Value>>,
) -> Result<BatchResult<Value>> {
    let mut cache = Cache::load(path, repository, model_name)?;
    investigate(
        Definition::load(path)?,
        inputs,
        &Files::new(repository)?,
        10,
        Some(&mut cache),
        model,
    )
}

fn unexpected_model(_: &LlmRequest<Value>) -> Result<LlmCompletion<Value>> {
    bail!("replay unexpectedly requested a model response")
}

#[test]
fn disk_cache_replays_children_and_branches_on_changed_source() -> Result<()> {
    let (dir, path) = fixture()?;
    let bindings = inputs("Review entry point");
    let mut calls = 0;
    let first = cached_run(&path, dir.path(), provider::MODEL, &bindings, |request| {
        calls += 1;
        Ok(match calls {
            1 => tool(
                request,
                "security_review",
                json!({"path": "main.rs", "focus": "Check entry point"}),
            ),
            2 => tool(
                request,
                "read_file",
                json!({"path": "main.rs", "from_line": 1, "num_of_lines": 5}),
            ),
            3 => LlmCompletion::final_answer(&request.request_id, json!("child safe")),
            4 => LlmCompletion::final_answer(&request.request_id, json!("parent safe")),
            _ => bail!("unexpected model request"),
        })
    })?;
    assert_eq!(calls, 4);
    assert_eq!(first.child_updates().len(), 1);
    let state = Cache::path_for(&path)?;
    assert_eq!(
        state,
        dir.path()
            .canonicalize()?
            .join("source-code-analyst.state.json")
    );
    assert!(state.is_file());
    let bundle: Value = serde_json::from_slice(&fs::read(&state)?)?;
    assert!(bundle["records"].as_array().unwrap().len() > 2);
    assert!(bundle.to_string().contains("preserved reasoning"));
    // A fresh loader/writer, not the first run's in-memory handle.
    let replay = cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &bindings,
        unexpected_model,
    )?;
    assert_eq!(replay.results, first.results);

    fs::write(dir.path().join("main.rs"), "risky\n")?;
    calls = 0;
    let changed = cached_run(&path, dir.path(), provider::MODEL, &bindings, |request| {
        calls += 1;
        if calls == 1 {
            assert!(
                request
                    .messages
                    .last()
                    .unwrap()
                    .content
                    .to_string()
                    .contains("risky")
            );
            // Provider-specific reasoning survived the JSON snapshot round trip.
            let body = provider::request_body(request, provider::MODEL)?;
            assert_eq!(body["messages"][2]["reasoning"], "preserved reasoning");
        }
        Ok(LlmCompletion::final_answer(
            &request.request_id,
            json!(if calls == 1 {
                "child risky"
            } else {
                "parent risky"
            }),
        ))
    })?;
    assert_eq!(calls, 2);
    assert_eq!(changed.results["repository"], "parent risky");
    assert_eq!(first.results["repository"], "parent safe");
    // Both observation paths survive repeated disk publication.
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &bindings,
        unexpected_model,
    )?;
    fs::write(dir.path().join("main.rs"), "safe\n")?;
    let old_path = cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &bindings,
        unexpected_model,
    )?;
    assert_eq!(old_path.results["repository"], "parent safe");
    Ok(())
}

#[test]
fn cache_keeps_inputs_and_isolates_models_repositories_and_child_recipes() -> Result<()> {
    let (dir, path) = fixture()?;
    for question in ["one", "two"] {
        let mut calls = 0;
        cached_run(
            &path,
            dir.path(),
            provider::MODEL,
            &inputs(question),
            |request| {
                calls += 1;
                Ok(LlmCompletion::final_answer(
                    &request.request_id,
                    json!(question),
                ))
            },
        )?;
        assert_eq!(calls, 1);
    }
    for question in ["one", "two"] {
        let result = cached_run(
            &path,
            dir.path(),
            provider::MODEL,
            &inputs(question),
            unexpected_model,
        )?;
        assert_eq!(result.results["repository"], question);
    }
    let mut definition: Value = serde_json::from_slice(&fs::read(&path)?)?;
    definition["manifest"]["version"] = json!("changed metadata only");
    definition["name"] = json!("renamed");
    fs::write(&path, serde_json::to_vec(&definition)?)?;
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("one"),
        unexpected_model,
    )?;

    let other_repository = tempfile::tempdir()?;
    let mut fresh_calls = 0;
    for (root, model_name) in [
        (dir.path(), "other-model"),
        (other_repository.path(), provider::MODEL),
    ] {
        cached_run(&path, root, model_name, &inputs("one"), |request| {
            fresh_calls += 1;
            Ok(LlmCompletion::final_answer(
                &request.request_id,
                json!("isolated"),
            ))
        })?;
    }
    assert_eq!(fresh_calls, 2);
    let child_path = dir.path().join("security-reviewer.json");
    let mut child: Value = serde_json::from_slice(&fs::read(&child_path)?)?;
    child["initial_messages"][0]["template"] = json!("A changed child instruction.");
    fs::write(child_path, serde_json::to_vec(&child)?)?;
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("one"),
        |request| {
            fresh_calls += 1;
            Ok(LlmCompletion::final_answer(
                &request.request_id,
                json!("new child contract"),
            ))
        },
    )?;
    assert_eq!(fresh_calls, 3);
    Ok(())
}

#[test]
fn failures_do_not_publish_or_replace_cache() -> Result<()> {
    let (dir, path) = fixture()?;
    let state = Cache::path_for(&path)?;
    assert!(
        cached_run(
            &path,
            dir.path(),
            provider::MODEL,
            &inputs("failed first"),
            |_| { bail!("provider unavailable") }
        )
        .is_err()
    );
    assert!(!state.exists());
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("ok"),
        |request| {
            Ok(LlmCompletion::final_answer(
                &request.request_id,
                json!("ok"),
            ))
        },
    )?;
    let before = fs::read(&state)?;
    assert!(
        cached_run(
            &path,
            dir.path(),
            provider::MODEL,
            &inputs("invalid output"),
            |request| { Ok(LlmCompletion::final_answer(&request.request_id, json!(123))) }
        )
        .is_err()
    );
    assert_eq!(fs::read(&state)?, before);
    let mut cache = Cache::load(&path, dir.path(), provider::MODEL)?;
    assert!(
        investigate(
            Definition::load(&path)?,
            &inputs("budget"),
            &Files::new(dir.path())?,
            1,
            Some(&mut cache),
            |request| Ok(tool(request, "list_dir", json!({"path": "."}))),
        )
        .is_err()
    );
    assert_eq!(fs::read(&state)?, before);
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("ok"),
        unexpected_model,
    )?;
    Ok(())
}

#[test]
fn generated_cache_is_not_a_tool_observation() -> Result<()> {
    let (dir, path) = fixture()?;
    let state = Cache::path_for(&path)?;
    let files = Files::new(dir.path())?.excluding(&state);
    let listing = files.execute("list_dir", &json!({"path": "."}));
    let search = files.execute(
        "string_search",
        &json!({"path_wildcard": "**/*.json", "text_wildcard": "*"}),
    );
    let mut calls = 0;
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("list"),
        |request| {
            calls += 1;
            Ok(if calls == 1 {
                tool(request, "list_dir", json!({"path": "."}))
            } else {
                LlmCompletion::final_answer(&request.request_id, json!("listed"))
            })
        },
    )?;
    assert_eq!(files.execute("list_dir", &json!({"path": "."})), listing);
    assert_eq!(
        files.execute(
            "string_search",
            &json!({"path_wildcard": "**/*.json", "text_wildcard": "*"})
        ),
        search
    );
    assert!(
        files.execute(
            "read_file",
            &json!({"path": "source-code-analyst.state.json", "from_line": 1, "num_of_lines": 1})
        )["error"]
            .is_string()
    );
    // Also exclude atomic-save staging files during concurrent invocations.
    fs::write(dir.path().join(".lockstep-cache-test"), "staging")?;
    assert_eq!(files.execute("list_dir", &json!({"path": "."})), listing);
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("list"),
        unexpected_model,
    )?;
    Ok(())
}

#[test]
fn corrupt_cache_fails_explicitly_instead_of_becoming_a_miss() -> Result<()> {
    let (dir, path) = fixture()?;
    cached_run(
        &path,
        dir.path(),
        provider::MODEL,
        &inputs("ok"),
        |request| {
            Ok(LlmCompletion::final_answer(
                &request.request_id,
                json!("ok"),
            ))
        },
    )?;
    let state = Cache::path_for(&path)?;
    let valid: Value = serde_json::from_slice(&fs::read(&state)?)?;
    fs::write(&state, "not JSON")?;
    assert!(Cache::load(&path, dir.path(), provider::MODEL).is_err());
    for modification in 0..5 {
        let mut value = valid.clone();
        match modification {
            0 => value["format_version"] = json!(999),
            1 => value["records"] = json!([]),
            2 => {
                let duplicate = value["records"][0].clone();
                value["records"].as_array_mut().unwrap().push(duplicate);
            }
            3 | 4 => {
                let mut reference = value["records"][0][0].clone();
                if modification == 4 {
                    reference["namespace"] = json!("missing");
                }
                value["records"][0][1]["observations"] = json!([{
                    "representative": [], "evidence": [], "continuation": reference
                }]);
            }
            _ => unreachable!(),
        }
        fs::write(&state, serde_json::to_vec(&value)?)?;
        assert!(
            Cache::load(&path, dir.path(), provider::MODEL).is_err(),
            "modification {modification}"
        );
    }
    Ok(())
}
