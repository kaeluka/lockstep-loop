use super::*;
use lockstep_loop::Metadata;
use serde_json::json;
use std::fs;

#[test]
fn model_selection_defaults_validates_and_allows_explicit_override() -> Result<()> {
    let empty = Metadata::default();
    let custom: Metadata = json!({"model": "custom", "tags": ["private"]})
        .as_object()
        .unwrap()
        .clone()
        .into();
    assert_eq!(provider::model_for(&empty, None)?, provider::MODEL);
    assert_eq!(provider::model_for(&custom, None)?, "custom");
    assert_eq!(provider::model_for(&custom, Some("override"))?, "override");
    for bad in [
        json!({"model": 1}),
        json!({"model": null}),
        json!({"model": " "}),
    ] {
        assert!(provider::model_for(&bad.as_object().unwrap().clone().into(), None).is_err());
    }
    assert!(provider::model_for(&empty, Some("")).is_err());
    Ok(())
}

#[test]
fn child_model_policy_isolates_cache_but_tags_do_not() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../docs/agent-examples");
    for name in ["source-code-analyst.json", "security-reviewer.json"] {
        fs::copy(source.join(name), dir.path().join(name))?;
    }
    let path = dir.path().join("source-code-analyst.json");
    let child_path = dir.path().join("security-reviewer.json");
    let inputs = json!({"path": ".", "question": "review"});
    let definition = Definition::load(&path)?;
    let original_contract = definition.cache_contract();
    let original_policy = definition.model_policy(None)?;
    let mut cache = Cache::load(&path, dir.path(), &original_policy)?;
    investigate(
        definition,
        &inputs,
        &Files::new(dir.path())?,
        10,
        Some(&mut cache),
        |req| {
            Ok(LlmCompletion::final_answer(
                &req.request_id,
                json!("old answer"),
            ))
        },
    )?;
    let mut child: Value = serde_json::from_slice(&fs::read(&child_path)?)?;
    child["manifest"]["model"] = json!("different-child");
    fs::write(&child_path, child.to_string())?;
    let definition = Definition::load(&path)?;
    // Core structural compatibility is unchanged; only host policy isolates models.
    assert_eq!(definition.cache_contract(), original_contract);
    let new_policy = definition.model_policy(None)?;
    assert_ne!(new_policy, original_policy);
    assert_eq!(
        definition.model_policy(Some(provider::MODEL))?,
        original_policy
    );
    let mut cache = Cache::load(&path, dir.path(), &new_policy)?;
    let mut calls = 0;
    investigate(
        definition,
        &inputs,
        &Files::new(dir.path())?,
        10,
        Some(&mut cache),
        |req| {
            calls += 1;
            Ok(LlmCompletion::final_answer(
                &req.request_id,
                json!("new answer"),
            ))
        },
    )?;
    assert_eq!(calls, 1);
    child["manifest"]["tags"] = json!(["different-tag"]);
    fs::write(&child_path, child.to_string())?;
    let definition = Definition::load(&path)?;
    assert_eq!(definition.model_policy(None)?, new_policy);
    let mut cache = Cache::load(&path, dir.path(), &new_policy)?;
    let result = investigate(
        definition,
        &inputs,
        &Files::new(dir.path())?,
        10,
        Some(&mut cache),
        |_| bail!("metadata tags must not invalidate cache"),
    )?;
    assert_eq!(result.results["repository"], "new answer");
    Ok(())
}
