use anyhow::{Context, Result, bail, ensure};
use lockstep_loop::{LlmCompletion, LlmRequest, Metadata, Role, ToolCall};
use reqwest::blocking::Client;
use serde_json::{Value, json};
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

pub const MODEL: &str = "deepseek/deepseek-v4.1-flash-1017";
const ENDPOINT: &str = "https://openrouter.ai/api/v1/chat/completions";
const MAX_RESPONSE: u64 = 8 * 1_048_576;

pub struct OpenRouter {
    client: Client,
    key: String,
    model_override: Option<String>,
    endpoint: String,
}
impl OpenRouter {
    pub fn new(model_override: Option<String>, timeout: Duration) -> Result<Self> {
        let home = std::env::var_os("HOME").context("HOME is not set")?;
        let path = PathBuf::from(home).join("secret/lockstep-loop.openrouter.key");
        let key = std::fs::read_to_string(&path)
            .with_context(|| format!("read OpenRouter key at {}", path.display()))?;
        let key = key.trim().to_owned();
        ensure!(
            !key.is_empty() && !key.contains(['\r', '\n']),
            "OpenRouter key file must contain one nonempty key"
        );
        Self::with_endpoint(model_override, timeout, key, ENDPOINT.into())
    }
    pub fn with_endpoint(
        model_override: Option<String>,
        timeout: Duration,
        key: String,
        endpoint: String,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            .connect_timeout(Duration::from_secs(15))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            client,
            key,
            model_override,
            endpoint,
        })
    }
    pub fn complete(&self, request: &LlmRequest<Value>) -> Result<LlmCompletion<Value>> {
        let model = model_for(request.metadata(), self.model_override.as_deref())?;
        eprintln!("OpenRouter model {model}");
        let body = request_body(request, model)?;
        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.key)
            .json(&body)
            .send()
            .context("OpenRouter request failed")?;
        // Do not echo provider error bodies or authorization headers into logs.
        ensure!(
            response.status().is_success(),
            "OpenRouter returned HTTP {} (no model fallback or automatic retry)",
            response.status()
        );
        let mut bytes = Vec::new();
        response
            .take(MAX_RESPONSE + 1)
            .read_to_end(&mut bytes)
            .context("read OpenRouter response")?;
        ensure!(
            bytes.len() as u64 <= MAX_RESPONSE,
            "OpenRouter response exceeds 8 MiB"
        );
        let value = serde_json::from_slice(&bytes).context("OpenRouter returned invalid JSON")?;
        parse_response(&request.request_id, value)
    }
}

/// Per-agent selection is host policy; the core only carries opaque metadata.
pub fn model_for<'a>(metadata: &'a Metadata, override_model: Option<&'a str>) -> Result<&'a str> {
    let model = match override_model {
        Some(model) => model,
        None => match metadata.get("model") {
            Some(value) => value.as_str().context("manifest.model must be a string")?,
            None => MODEL,
        },
    };
    ensure!(!model.trim().is_empty(), "model must not be empty");
    Ok(model)
}

pub fn request_body(request: &LlmRequest<Value>, model: &str) -> Result<Value> {
    let mut messages = Vec::new();
    for envelope in &request.messages {
        let message = match envelope.role {
            Role::System | Role::User => {
                json!({"role": if envelope.role == Role::System { "system" } else { "user" }, "content": envelope.content})
            }
            Role::Tool => {
                json!({"role": "tool", "tool_call_id": envelope.tool_call_id.as_ref().context("tool message missing call ID")?, "content": text_content(&envelope.content)?})
            }
            Role::Assistant => {
                let mut message = if let Some(original) = envelope.content.get("openrouter_message")
                {
                    original.clone()
                } else {
                    json!({"role": "assistant", "content": envelope.content})
                };
                ensure!(message.is_object(), "invalid stored assistant message");
                message["role"] = json!("assistant");
                if !envelope.tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(envelope.tool_calls.iter().map(|call| Ok(json!({"id": call.id, "type": "function", "function": {"name": call.name, "arguments": serde_json::to_string(&call.args)?}}))).collect::<Result<_>>()?);
                }
                message
            }
        };
        messages.push(message);
    }
    let mut body = json!({"model": model, "messages": messages, "stream": false, "max_tokens": 16384, "reasoning": {"effort": "high"}});
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(|spec| json!({"type": "function", "function": {"name": spec.name, "description": spec.description, "parameters": spec.parameters}})).collect());
        body["tool_choice"] = json!("auto");
    }
    Ok(body)
}
fn text_content(value: &Value) -> Result<String> {
    Ok(match value {
        Value::String(text) => text.clone(),
        value => serde_json::to_string(value)?,
    })
}
pub fn parse_response(request_id: &str, value: Value) -> Result<LlmCompletion<Value>> {
    ensure!(
        value.get("error").is_none(),
        "OpenRouter returned an API error envelope"
    );
    let choices = value["choices"]
        .as_array()
        .context("OpenRouter response has no choices")?;
    ensure!(choices.len() == 1, "expected exactly one OpenRouter choice");
    let choice = &choices[0];
    let finish = choice["finish_reason"]
        .as_str()
        .context("missing finish_reason")?;
    ensure!(
        matches!(finish, "stop" | "tool_calls"),
        "OpenRouter response was not complete: finish_reason={finish}"
    );
    let message = &choice["message"];
    ensure!(
        message["role"] == "assistant",
        "expected assistant response"
    );
    ensure!(
        message.get("refusal").is_none_or(Value::is_null),
        "model refused the request"
    );
    let mut calls = Vec::new();
    if let Some(raw) = message.get("tool_calls").filter(|v| !v.is_null()) {
        let raw = raw.as_array().context("tool_calls must be an array")?;
        for call in raw {
            ensure!(call["type"] == "function", "unsupported tool call type");
            let id = call["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("tool call missing ID")?;
            let name = call["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .context("tool call missing name")?;
            let args = call["function"]["arguments"]
                .as_str()
                .context("tool arguments must be JSON text")?;
            calls.push(ToolCall::new(
                id,
                name,
                serde_json::from_str(args).context("tool arguments are not valid JSON")?,
            ));
        }
    }
    let content = if calls.is_empty() {
        ensure!(finish == "stop", "tool_calls finish without calls");
        Value::String(
            message["content"]
                .as_str()
                .context("final answer must have string content")?
                .to_owned(),
        )
    } else {
        ensure!(
            finish == "tool_calls",
            "tool calls require tool_calls finish reason"
        );
        if !message["content"].is_null() && !message["content"].is_string() {
            bail!("assistant tool-turn content must be text or null");
        }
        // Preserve reasoning/reasoning_details (including signatures) verbatim.
        // Final answers use a plain JSON string so child folding returns the declared asset.
        json!({"openrouter_message": message})
    };
    Ok(LlmCompletion {
        request_id: request_id.into(),
        content,
        tool_calls: calls,
    })
}
