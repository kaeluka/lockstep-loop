use anyhow::{Context, Result, bail, ensure};
use lockstep_loop::{LlmRequest, LoopError, MessageEnvelope, Metadata, SubagentTool, ToolSpec};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MAX_DEPTH: usize = 8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDefinition {
    format_version: u32,
    name: String,
    manifest: serde_json::Map<String, Value>,
    inputs: Value,
    initial_messages: Vec<Template>,
    output: Output,
    tools: Vec<FileTool>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Template {
    role: String,
    template: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Output {
    description: String,
    schema: Value,
    source: Source,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Source {
    LastAnswer,
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum FileTool {
    External {
        name: String,
        description: String,
        input_schema: Value,
    },
    Agent {
        name: String,
        description: String,
        agent: PathBuf,
    },
}

pub struct Definition {
    pub name: String,
    pub metadata: Metadata,
    inputs: Value,
    input_validator: jsonschema::Validator,
    templates: Vec<Template>,
    output: Value,
    output_validator: Arc<jsonschema::Validator>,
    pub specs: Vec<ToolSpec>,
    children: Vec<(ToolSpec, Arc<Definition>)>,
}

#[derive(Clone, Default)]
pub struct Instances(Arc<Mutex<Vec<Instance>>>);
struct Instance {
    messages: Vec<MessageEnvelope<Value>>,
    specs: Vec<ToolSpec>,
    output: Value,
    validator: Arc<jsonschema::Validator>,
}

pub fn compile_schema(schema: &Value) -> Result<jsonschema::Validator> {
    // No network/file retrieval: only document-local JSON pointers are supported.
    fn check(value: &Value) -> Result<()> {
        match value {
            Value::Object(map) => {
                for (key, value) in map {
                    if key == "$ref" || key == "$dynamicRef" {
                        ensure!(
                            value.as_str().is_some_and(|s| s.starts_with('#')),
                            "only document-local schema references are supported"
                        );
                    }
                    check(value)?;
                }
            }
            Value::Array(values) => {
                for value in values {
                    check(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    check(schema)?;
    if let Some(dialect) = schema.get("$schema") {
        ensure!(
            dialect == "https://json-schema.org/draft/2020-12/schema",
            "only JSON Schema Draft 2020-12 is supported"
        );
    }
    jsonschema::draft202012::options()
        .build(schema)
        .context("invalid JSON Schema")
}

pub fn validate(validator: &jsonschema::Validator, value: &Value, label: &str) -> Result<()> {
    if let Some(error) = validator.iter_errors(value).next() {
        // Do not print the instance value: inputs/tool arguments could be sensitive.
        bail!("{label} fails JSON Schema at {}", error.instance_path());
    }
    Ok(())
}

impl Definition {
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        Self::load_inner(path, &mut Vec::new())
    }
    fn load_inner(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Arc<Self>> {
        ensure!(stack.len() < MAX_DEPTH, "agent nesting exceeds {MAX_DEPTH}");
        let path = path
            .canonicalize()
            .with_context(|| format!("resolve agent {}", path.display()))?;
        ensure!(
            !stack.contains(&path),
            "cyclic subagent link at {}",
            path.display()
        );
        let bytes =
            std::fs::read(&path).with_context(|| format!("read agent {}", path.display()))?;
        ensure!(
            bytes.len() <= 1_048_576,
            "agent definition is larger than 1 MiB"
        );
        let file: FileDefinition = serde_json::from_slice(&bytes)
            .with_context(|| format!("decode agent {}", path.display()))?;
        ensure!(
            file.format_version == 1,
            "unsupported agent format version {}",
            file.format_version
        );
        ensure!(!file.name.is_empty(), "agent name must not be empty");
        ensure!(
            !file.initial_messages.is_empty(),
            "initial_messages must not be empty"
        );
        ensure!(
            file.inputs.get("type").and_then(Value::as_str) == Some("object"),
            "inputs must be an object-valued JSON Schema"
        );
        for template in &file.initial_messages {
            ensure!(
                matches!(template.role.as_str(), "system" | "user" | "assistant"),
                "unsupported initial message role {}",
                template.role
            );
        }
        let _ = (&file.output.description, &file.output.source);
        // This interpretation belongs to the example host, never to the core.
        let metadata: Metadata = file.manifest.into();
        super::provider::model_for(&metadata, None)?;
        let input_validator = compile_schema(&file.inputs)?;
        let output_validator = Arc::new(compile_schema(&file.output.schema)?);
        stack.push(path.clone());
        let mut names = HashSet::new();
        let mut specs = Vec::new();
        let mut children = Vec::new();
        for tool in file.tools {
            let spec = match tool {
                FileTool::External {
                    name,
                    description,
                    input_schema,
                } => {
                    ensure!(
                        matches!(name.as_str(), "read_file" | "string_search" | "list_dir"),
                        "unknown external tool {name}"
                    );
                    compile_schema(&input_schema)?;
                    ToolSpec::new(name, description, input_schema)
                }
                FileTool::Agent {
                    name,
                    description,
                    agent,
                } => {
                    let child = Self::load_inner(
                        &path
                            .parent()
                            .context("agent has no parent directory")?
                            .join(agent),
                        stack,
                    )?;
                    let spec = ToolSpec::new(name, description, child.inputs.clone());
                    children.push((spec.clone(), child));
                    spec
                }
            };
            ensure!(
                names.insert(spec.name.clone()),
                "duplicate tool {}",
                spec.name
            );
            specs.push(spec);
        }
        stack.pop();
        Ok(Arc::new(Self {
            name: file.name,
            metadata,
            inputs: file.inputs,
            input_validator,
            templates: file.initial_messages,
            output: file.output.schema,
            output_validator,
            specs,
            children,
        }))
    }

    pub fn instantiate(
        &self,
        inputs: &Value,
        instances: &Instances,
    ) -> Result<Vec<MessageEnvelope<Value>>> {
        validate(&self.input_validator, inputs, "agent inputs")?;
        let messages = self
            .templates
            .iter()
            .map(|template| {
                let content = Value::String(render(&template.template, inputs)?);
                Ok(match template.role.as_str() {
                    "system" => MessageEnvelope::system(content),
                    "user" => MessageEnvelope::user(content),
                    "assistant" => MessageEnvelope::assistant(content, Vec::new()),
                    _ => unreachable!("roles checked when loading"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        instances
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("instance registry poisoned"))?
            .push(Instance {
                messages: messages.clone(),
                specs: self.specs.clone(),
                output: self.output.clone(),
                validator: self.output_validator.clone(),
            });
        Ok(messages)
    }

    // Factories are not serializable. Compare their declarative recipes as well as
    // the core's concrete structural configuration, including linked children.
    // Names, manifest tags/versions, and link locations are not reuse keys.
    pub fn cache_contract(&self) -> Value {
        serde_json::json!({
            "inputs": self.inputs,
            "templates": self.templates,
            "output": self.output,
            "tools": self.specs,
            "children": self.children.iter().map(|(spec, child)| {
                serde_json::json!({"tool": spec, "definition": child.cache_contract()})
            }).collect::<Vec<_>>()
        })
    }

    /// Host-owned model isolation, including all linked child model routes.
    /// A homogeneous tree retains the previous single-model cache policy.
    pub fn model_policy(&self, override_model: Option<&str>) -> Result<String> {
        fn collect(
            definition: &Definition,
            override_model: Option<&str>,
            path: Vec<String>,
            routes: &mut Vec<(Vec<String>, String)>,
        ) -> Result<()> {
            routes.push((
                path.clone(),
                super::provider::model_for(&definition.metadata, override_model)?.to_owned(),
            ));
            for (spec, child) in &definition.children {
                let mut child_path = path.clone();
                child_path.push(spec.name.clone());
                collect(child, override_model, child_path, routes)?;
            }
            Ok(())
        }
        let mut routes = Vec::new();
        collect(self, override_model, Vec::new(), &mut routes)?;
        if routes.iter().all(|(_, model)| model == &routes[0].1) {
            Ok(routes[0].1.clone())
        } else {
            Ok(serde_json::to_string(&routes)?)
        }
    }

    pub fn validate_output(&self, answer: &Value) -> Result<()> {
        validate(&self.output_validator, answer, "agent output")
    }

    pub fn subagents(&self, instances: &Instances) -> Vec<SubagentTool<Value>> {
        self.children
            .iter()
            .map(|(spec, child)| {
                let nested = child.subagents(instances);
                let child = child.clone();
                let metadata = child.metadata.clone();
                let specs = child.specs.clone();
                let instances = instances.clone();
                SubagentTool::try_new(
                    spec.clone(),
                    move |args| {
                        child
                            .instantiate(args, &instances)
                            .map_err(|error| LoopError::InvalidSubagentArguments(error.to_string()))
                    },
                    move |_| Ok(specs.clone()),
                    MAX_DEPTH,
                )
                .with_subagents(nested)
                .with_metadata(metadata)
            })
            .collect()
    }
}

impl Instances {
    pub fn validate_answer(&self, request: &LlmRequest<Value>, answer: &Value) -> Result<()> {
        let registry = self
            .0
            .lock()
            .map_err(|_| anyhow::anyhow!("instance registry poisoned"))?;
        let matches: Vec<_> = registry
            .iter()
            .filter(|i| request.messages.starts_with(&i.messages) && request.tools == i.specs)
            .collect();
        ensure!(
            !matches.is_empty(),
            "cannot identify output schema for model request"
        );
        ensure!(
            matches.iter().all(|i| i.output == matches[0].output),
            "ambiguous output contracts for identical initial conversation/tool configuration"
        );
        validate(&matches[0].validator, answer, "agent output")
    }
}

pub fn render(template: &str, inputs: &Value) -> Result<String> {
    let mut rest = template;
    let mut out = String::new();
    while let Some(start) = rest.find("{{") {
        out.push_str(&rest[..start]);
        rest = &rest[start + 2..];
        let end = rest.find("}}").context("unclosed template reference")?;
        let key = &rest[..end];
        ensure!(
            !key.is_empty() && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "template references must be simple input names"
        );
        let value = inputs
            .get(key)
            .with_context(|| format!("missing template input {key}"))?;
        match value {
            Value::String(text) => out.push_str(text),
            value => out.push_str(&serde_json::to_string(value)?),
        }
        rest = &rest[end + 2..];
    }
    ensure!(!rest.contains("}}"), "unmatched template closing delimiter");
    out.push_str(rest);
    Ok(out)
}
