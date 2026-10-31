//! A deliberately small agent-file host, not a core library loader.
//! See docs/agent-examples/README.md for usage and security/format limitations.
#[path = "agent_file_support/cache.rs"]
mod cache;
#[path = "agent_file_support/definition.rs"]
mod definition;
#[path = "agent_file_support/files.rs"]
mod files;
#[path = "agent_file_support/provider.rs"]
mod provider;

use anyhow::{Context, Result, bail, ensure};
use cache::Cache;
use definition::{Definition, Instances, compile_schema, validate};
use files::Files;
use lockstep_loop::{
    Agent, AgentConfig, BatchResult, Completion, LlmCompletion, LlmRequest, Request, RequestKind,
    RunOptions, ToolCompletion,
};
use provider::OpenRouter;
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

fn investigate(
    definition: Arc<Definition>,
    inputs: &Value,
    files: &Files,
    max_turns: usize,
    cache: Option<&mut Cache>,
    mut model: impl FnMut(&LlmRequest<Value>) -> Result<LlmCompletion<Value>>,
) -> Result<BatchResult<Value>> {
    let instances = Instances::default();
    let initial = definition.instantiate(inputs, &instances)?;
    let config = AgentConfig::new(initial, definition.specs.clone())
        .with_subagents(definition.subagents(&instances))
        .with_metadata(definition.metadata.clone());
    let contract = definition.cache_contract();
    let agent = match cache.as_ref() {
        Some(cache) => cache.agent(config, &contract)?,
        None => Agent::new(config)?,
    };
    let files = match cache.as_ref() {
        Some(cache) => files.clone().excluding(cache.path()),
        None => files.clone(),
    };
    // The handle is for routing only and is never sent to OpenRouter.
    let mut run = agent.start(vec!["repository".into()], RunOptions::default())?;
    let mut turns = 0;
    while !run.is_done() {
        match run
            .next_request(RequestKind::Any)
            .context("investigation stalled without a work request")?
        {
            Request::Llm(request) => {
                ensure!(
                    turns < max_turns,
                    "global model-turn budget exhausted ({max_turns}); no final answer was fabricated"
                );
                turns += 1;
                eprintln!(
                    "model turn {turns}/{max_turns}, branch turn {}",
                    request.stats.depth
                );
                let completion = model(&request)?;
                ensure!(
                    completion.request_id == request.request_id,
                    "model completion has wrong request ID"
                );
                let mut ids = HashSet::new();
                for call in &completion.tool_calls {
                    ensure!(ids.insert(&call.id), "duplicate tool call ID");
                    let spec = request
                        .tools
                        .iter()
                        .find(|spec| spec.name == call.name)
                        .with_context(|| {
                            format!("model requested unavailable tool {}", call.name)
                        })?;
                    validate(
                        &compile_schema(&spec.parameters)?,
                        &call.args,
                        "tool arguments",
                    )?;
                }
                if completion.tool_calls.is_empty() {
                    instances.validate_answer(&request, &completion.content)?;
                }
                run.complete(Completion::Llm(completion))?;
            }
            Request::Tool(request) => {
                eprintln!("tool {} {}", request.tool_call.name, request.tool_call.args);
                let results = request
                    .env_ids
                    .into_iter()
                    .map(|env| {
                        (
                            env,
                            files.execute(&request.tool_call.name, &request.tool_call.args),
                        )
                    })
                    .collect();
                run.complete(Completion::Tool(ToolCompletion {
                    request_id: request.request_id,
                    results,
                }))?;
            }
        }
    }
    let result = run.result().context("investigation failed")?;
    for answer in result.results.values() {
        definition.validate_output(answer)?;
    }
    if let Some(cache) = cache {
        cache.publish(&result.snapshot, contract)?;
    }
    Ok(result)
}

struct Options {
    agent: PathBuf,
    root: PathBuf,
    inputs: Value,
    model: Option<String>,
    max_turns: usize,
    timeout: Duration,
}
fn options() -> Result<Option<Options>> {
    let mut agent = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/agent-examples/source-code-analyst.json");
    let mut root = PathBuf::from(".");
    let mut inputs = None;
    let mut model = None;
    let mut max_turns = 1000;
    let mut seconds = 180;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "Usage: cargo run -p lockstep-loop --example agent_file -- [OPTIONS]\n\n  --agent PATH          Agent JSON definition (default: bundled source-code analyst)\n  --root PATH           Stable repository root (default: .)\n  --inputs JSON         Required input bindings as a JSON object\n  --inputs-file PATH    Read input bindings from a JSON file instead\n  --model ID            Override model for ALL agents (otherwise manifest.model; fallback: {})\n  --max-turns N         Total parent + child model requests (default: 1000)\n  --timeout-secs N      Per-request HTTP timeout (default: 180)\n\nReads ~/secret/lockstep-loop.openrouter.key. Sends inspected source to OpenRouter.\nUses standard DeepSeek thinking (reasoning.effort=high).\nAutomatically loads/saves <agent-stem>.state.json beside the agent definition.",
                provider::MODEL
            );
            return Ok(None);
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--agent" => agent = value.into(),
            "--root" => root = value.into(),
            "--inputs" | "--inputs-file" => {
                ensure!(inputs.is_none(), "provide inputs only once");
                let text = if flag == "--inputs-file" {
                    std::fs::read_to_string(value).context("read inputs file")?
                } else {
                    value
                };
                inputs = Some(serde_json::from_str(&text).context("decode input JSON")?);
            }
            "--model" => model = Some(value),
            "--max-turns" => max_turns = value.parse().context("invalid max-turns")?,
            "--timeout-secs" => seconds = value.parse().context("invalid timeout-secs")?,
            _ => bail!("unknown option {flag}; use --help"),
        }
    }
    ensure!(
        max_turns > 0 && seconds > 0,
        "budget and timeout must be positive"
    );
    ensure!(
        model.as_ref().is_none_or(|model| !model.trim().is_empty()),
        "model must not be empty"
    );
    Ok(Some(Options {
        agent,
        root,
        inputs: inputs.context("--inputs or --inputs-file is required")?,
        model,
        max_turns,
        timeout: Duration::from_secs(seconds),
    }))
}
fn main() -> Result<()> {
    let Some(options) = options()? else {
        return Ok(());
    };
    let definition = Definition::load(&options.agent)?;
    // Validate local configuration before loading credentials or making a network request.
    definition.instantiate(&options.inputs, &Instances::default())?;
    let files = Files::new(Path::new(&options.root))?;
    eprintln!(
        "agent {}, model {} (standard thinking: high); inspected source will be sent to OpenRouter",
        definition.name,
        provider::model_for(&definition.metadata, options.model.as_deref())?
    );
    let policy = definition.model_policy(options.model.as_deref())?;
    let mut cache = Cache::load(&options.agent, &options.root, &policy)?;
    // A fully cached investigation needs neither credentials nor network access.
    let mut provider = None;
    let result = investigate(
        definition,
        &options.inputs,
        &files,
        options.max_turns,
        Some(&mut cache),
        |request| {
            if provider.is_none() {
                provider = Some(OpenRouter::new(options.model.clone(), options.timeout)?);
            }
            provider
                .as_ref()
                .expect("provider initialized")
                .complete(request)
        },
    )?;
    let answer = result
        .results
        .get("repository")
        .and_then(Value::as_str)
        .context("completed investigation has no string answer")?;
    println!("{answer}");
    Ok(())
}

#[cfg(test)]
#[path = "agent_file_support/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "agent_file_support/cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
#[path = "agent_file_support/model_tests.rs"]
mod model_tests;
