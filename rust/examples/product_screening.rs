//! Product-specification screening: shared work, then replay on new evidence.
//! Run: cargo run -p lockstep-loop --example product_screening
//! Scripted replies demonstrate orchestration, not product-certification advice.

use lockstep_loop::{
    Agent, BatchLoop, BatchResult, Completion, LlmCompletion, LlmRequest, LoopConfig, LoopError,
    Request, RequestKind, Role, ToolCall, ToolCompletion, ToolSpec,
};
use serde_json::json;
use std::collections::BTreeMap;

const PROMPT: &str = "Check this sensor against our toy outdoor requirements: enclosure rating \
    IP65 or IP67, and an operating range covering -10 to 40 degrees Celsius. \
    Read the ratings before concluding.";

fn ratings(enclosure: &str, min_c: i64, max_c: i64) -> String {
    // Full relevant evidence in a stable representation, not an output hash.
    json!({"enclosure": enclosure, "min_c": min_c, "max_c": max_c}).to_string()
}

fn scripted_model(request: &LlmRequest<String>) -> LlmCompletion<String> {
    // Replace with an LLM adapter. Subject IDs are not part of the conversation.
    if request.messages.len() == 1 {
        return LlmCompletion {
            request_id: request.request_id.clone(),
            content: "Read the product ratings.".into(),
            tool_calls: vec![ToolCall::new("ratings", "read_ratings", json!({}))],
        };
    }
    let observation = request.messages.last().unwrap();
    assert_eq!(observation.role, Role::Tool);
    let evidence: serde_json::Value = serde_json::from_str(&observation.content).unwrap();
    let meets_requirements = matches!(evidence["enclosure"].as_str(), Some("IP65" | "IP67"))
        && evidence["min_c"].as_i64().unwrap() <= -10
        && evidence["max_c"].as_i64().unwrap() >= 40;
    LlmCompletion::final_answer(
        request.request_id.clone(),
        if meets_requirements {
            "meets toy requirements"
        } else {
            "does not meet toy requirements"
        }
        .into(),
    )
}

fn screen(
    evidence: BTreeMap<String, String>,
    agent: Option<Agent<String>>,
) -> Result<(BatchResult<String>, usize, usize), LoopError> {
    let mut config = LoopConfig::new(
        PROMPT.into(),
        evidence.keys().cloned().collect(),
        vec![ToolSpec::new(
            "read_ratings",
            "Read the sensor's enclosure rating and minimum/maximum temperatures in Celsius.",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
        )],
    );
    config = if let Some(previous) = agent {
        config.with_agent(previous)
    } else {
        config
    };
    // Default record namespaces suffice for this single-process, in-memory demo.
    // Durable callers reserve namespaces and persist records as in history_replay.rs.
    let mut loop_ = BatchLoop::new(config)?;
    let (mut llm_calls, mut tool_executions) = (0, 0);
    while !loop_.is_done() {
        match loop_.next_request(RequestKind::Any).unwrap() {
            Request::Llm(request) => {
                llm_calls += 1;
                loop_.complete(Completion::Llm(scripted_model(&request)))?;
            }
            Request::Tool(request) => {
                assert_eq!(request.tool_call.name, "read_ratings");
                assert_eq!(request.tool_call.args, json!({}));
                tool_executions += request.env_ids.len();
                let results = request
                    .env_ids
                    .into_iter()
                    .map(|env| {
                        let output = evidence[&env].clone();
                        (env, output)
                    })
                    .collect();
                loop_.complete(Completion::Tool(ToolCompletion {
                    request_id: request.request_id,
                    results,
                }))?;
            }
        }
    }
    Ok((loop_.result()?, llm_calls, tool_executions))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let outdoor = ratings("IP65", -20, 50);
    let (initial, llms, reads) = screen(
        BTreeMap::from([
            ("product-a/v1".into(), outdoor.clone()),
            ("product-b/v1".into(), outdoor.clone()),
            ("product-c/v1".into(), ratings("IP20", 0, 40)),
        ]),
        None,
    )?;
    assert_eq!((llms, reads, initial.classes.len()), (3, 3, 2));
    let old = initial.agent().clone();
    println!(
        "Initial: {llms} LLM calls, {reads} tool executions, {} classes",
        initial.classes.len()
    );

    let (matching, llms, reads) = screen(
        BTreeMap::from([("product-d/v1".into(), outdoor)]),
        Some(old.clone()),
    )?;
    assert_eq!((llms, reads), (0, 1));
    assert_eq!(matching.results["product-d/v1"], "meets toy requirements");
    println!("Matching new product: {llms} LLM calls, {reads} tool execution");

    let (changed, llms, reads) = screen(
        BTreeMap::from([("product-a/v2".into(), ratings("IP65", 0, 50))]),
        Some(matching.into_agent()),
    )?;
    assert_eq!((llms, reads), (1, 1));
    assert_eq!(
        changed.results["product-a/v2"],
        "does not meet toy requirements"
    );
    let extended = changed.agent();
    assert_eq!(old.assets().count(), 2);
    assert_eq!(extended.assets().count(), 3);
    println!("Revised specifications: {llms} LLM call, {reads} tool execution");
    for asset in extended.assets() {
        println!("{:?} -> {}", asset.env_ids(), asset.value().unwrap());
    }
    Ok(())
}
