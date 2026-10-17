//! Provider-agnostic lockstep agent orchestration.
//!
//! A [`BatchLoop`] starts one conversation for an equivalence class of
//! environments. Tool calls fan out over every environment in the class; the
//! class splits only when tool results differ. The caller owns all LLM and
//! tool execution and may complete requests in any order.
//!
//! An immutable [`Agent`] contains the initial prompt, compatible execution
//! contract and accumulated trie. Its [`Asset`] views are leaf investigations
//! backed by their complete conversations, not just published values or
//! environment classes. [`History`] remains an alias; persistence fields stay
//! compatible. Feedback still resolves an agent snapshot + environment by replay.

mod history;
use history::PathStep;
pub use history::{
    Agent, AgentDescriptor, AgentOptions, AgentQuery, AgentResolver, Asset, Assets, ChildAgentRun,
    ChildAgentUpdate, ChildHistoryRun, ChildHistoryUpdate, EnvObservation, History,
    HistoryDescriptor, HistoryNode, HistoryOptions, HistoryQuery, HistoryResolver, NodeRef,
    Observation, OutputRecord, RecordStore, SubagentContract, SubagentHistory, SubagentRecord,
    UserContinuation,
};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::sync::Arc;
use thiserror::Error;

type BudgetFallback<P> = dyn Fn(usize) -> P + Send + Sync;
type PromptBuilder<P> = dyn Fn(&Value) -> Result<P, LoopError> + Send + Sync;
type SubToolSpecFactory = dyn Fn(&str) -> Result<Vec<ToolSpec>, LoopError> + Send + Sync;
type GroupFunction<P, K> = dyn Fn(&P) -> K + Send + Sync;
type ToolFunction<P> = dyn Fn(&Value) -> P + Send + Sync;

/// Provider-neutral tool description.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A tool invocation requested by the model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub args: Value,
}

impl ToolCall {
    pub fn new(id: impl Into<String>, name: impl Into<String>, args: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            args,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    Tool,
}

/// Provider-neutral conversation message. Payloads are opaque to the loop.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message<P> {
    pub role: Role,
    pub content: P,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl<P> Message<P> {
    pub fn user(content: P) -> Self {
        Self {
            role: Role::User,
            content,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant(content: P, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool(content: P, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content,
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Facts the caller can compose into a scheduling policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestStats {
    pub depth: usize,
    pub split_depth: usize,
    pub seq: u64,
}

/// One shared LLM completion requested for a whole equivalence class.
#[derive(Clone, Debug, PartialEq)]
pub struct LlmRequest<P> {
    pub request_id: String,
    pub branch_id: String,
    pub env_ids: Vec<String>,
    pub messages: Vec<Message<P>>,
    pub tools: Vec<ToolSpec>,
    pub stats: RequestStats,
}

/// One tool call to execute once per environment in `env_ids`.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolRequest {
    pub request_id: String,
    pub branch_id: String,
    pub tool_call: ToolCall,
    pub env_ids: Vec<String>,
    pub stats: RequestStats,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Request<P> {
    Llm(LlmRequest<P>),
    Tool(ToolRequest),
}

impl<P> Request<P> {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Llm(request) => &request.request_id,
            Self::Tool(request) => &request.request_id,
        }
    }

    pub fn stats(&self) -> RequestStats {
        match self {
            Self::Llm(request) => request.stats,
            Self::Tool(request) => request.stats,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    Any,
    Llm,
    Tool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LlmCompletion<P> {
    pub request_id: String,
    pub content: P,
    pub tool_calls: Vec<ToolCall>,
}

impl<P> LlmCompletion<P> {
    pub fn final_answer(request_id: impl Into<String>, content: P) -> Self {
        Self {
            request_id: request_id.into(),
            content,
            tool_calls: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCompletion<P> {
    pub request_id: String,
    pub results: Vec<(String, P)>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Completion<P> {
    Llm(LlmCompletion<P>),
    Tool(ToolCompletion<P>),
}

impl<P> Completion<P> {
    pub fn request_id(&self) -> &str {
        match self {
            Self::Llm(completion) => &completion.request_id,
            Self::Tool(completion) => &completion.request_id,
        }
    }
}

/// Caller-supplied budget behavior for a generic payload type.
///
/// Rust cannot fabricate the Python implementation's string sentinel for an
/// arbitrary `P`, so the caller supplies the payload to use on exhaustion.
pub struct Budget<P> {
    pub max_steps_per_branch: usize,
    exhausted: Arc<BudgetFallback<P>>,
}

impl<P> Budget<P> {
    pub fn new(
        max_steps_per_branch: usize,
        exhausted: impl Fn(usize) -> P + Send + Sync + 'static,
    ) -> Self {
        Self {
            max_steps_per_branch,
            exhausted: Arc::new(exhausted),
        }
    }
}

impl<P> Clone for Budget<P> {
    fn clone(&self) -> Self {
        Self {
            max_steps_per_branch: self.max_steps_per_branch,
            exhausted: Arc::clone(&self.exhausted),
        }
    }
}

impl<P> Debug for Budget<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Budget")
            .field("max_steps_per_branch", &self.max_steps_per_branch)
            .finish_non_exhaustive()
    }
}

/// Reserved specialized tool implemented by a nested [`BatchLoop`].
/// The caller deterministically constructs initial USER content from structured
/// arguments; schemas are advertisements, not runtime argument validation.
pub struct SubagentTool<P> {
    pub spec: ToolSpec,
    pub max_depth: usize,
    pub history_contract: Option<String>,
    /// None preserves bounded self-recursion; Some is an explicit child allowlist.
    pub subagents: Option<Vec<SubagentTool<P>>>,
    build_prompt: Arc<PromptBuilder<P>>,
    sub_tool_specs: Arc<SubToolSpecFactory>,
}

impl<P> SubagentTool<P> {
    pub fn new(
        spec: ToolSpec,
        build_prompt: impl Fn(&Value) -> P + Send + Sync + 'static,
        sub_tool_specs: impl Fn(&str) -> Vec<ToolSpec> + Send + Sync + 'static,
        max_depth: usize,
    ) -> Self {
        Self::try_new(
            spec,
            move |args| Ok(build_prompt(args)),
            move |env| Ok(sub_tool_specs(env)),
            max_depth,
        )
    }

    /// Fallible factories validate structured args without panicking or consuming a request.
    pub fn try_new(
        spec: ToolSpec,
        build_prompt: impl Fn(&Value) -> Result<P, LoopError> + Send + Sync + 'static,
        sub_tool_specs: impl Fn(&str) -> Result<Vec<ToolSpec>, LoopError> + Send + Sync + 'static,
        max_depth: usize,
    ) -> Self {
        Self {
            spec,
            max_depth,
            history_contract: None,
            subagents: None,
            build_prompt: Arc::new(build_prompt),
            sub_tool_specs: Arc::new(sub_tool_specs),
        }
    }

    pub fn with_agent_contract(mut self, contract: impl Into<String>) -> Self {
        self.history_contract = Some(contract.into());
        self
    }

    /// Compatibility spelling of with_agent_contract.
    pub fn with_history_contract(self, contract: impl Into<String>) -> Self {
        self.with_agent_contract(contract)
    }

    pub fn agent_contract(&self) -> Option<&str> {
        self.history_contract.as_deref()
    }

    pub fn with_subagents(mut self, subagents: Vec<SubagentTool<P>>) -> Self {
        self.subagents = Some(subagents);
        self
    }

    fn contract(&self) -> SubagentContract {
        SubagentContract {
            spec: self.spec.clone(),
            max_depth: self.max_depth,
            history_contract: self.history_contract.clone(),
            subagents: self
                .subagents
                .as_ref()
                .map(|s| s.iter().map(Self::contract).collect()),
        }
    }

    fn with_max_depth(&self, max_depth: usize) -> Self {
        Self {
            spec: self.spec.clone(),
            max_depth,
            history_contract: self.history_contract.clone(),
            subagents: self.subagents.clone(),
            build_prompt: Arc::clone(&self.build_prompt),
            sub_tool_specs: Arc::clone(&self.sub_tool_specs),
        }
    }
}

impl<P> Clone for SubagentTool<P> {
    fn clone(&self) -> Self {
        self.with_max_depth(self.max_depth)
    }
}

impl<P> Debug for SubagentTool<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubagentTool")
            .field("spec", &self.spec)
            .field("max_depth", &self.max_depth)
            .field("history_contract", &self.history_contract)
            .field("subagents", &self.subagents)
            .finish_non_exhaustive()
    }
}

/// Configuration shared by default and custom-grouping constructors.
#[derive(Clone, Debug)]
pub struct LoopConfig<P> {
    pub prompt: P,
    pub env_ids: Vec<String>,
    pub tool_specs: Vec<ToolSpec>,
    pub subagent: Option<SubagentTool<P>>,
    pub subagents: Option<Vec<SubagentTool<P>>>,
    /// Compatibility field; configure agent lookup with with_agent_resolver.
    pub history_resolver: Option<AgentResolver<P>>,
    pub record_store: Option<RecordStore<P>>,
    pub budget: Option<Budget<P>>,
    /// Compatibility field; configure recorded agents with with_agent/with_recording.
    pub history: Option<AgentOptions<P>>,
    /// Some(payload) is one intervention; even Some(None) for an optional payload.
    pub feedback: Option<P>,
}

impl<P> LoopConfig<P> {
    pub fn new(prompt: P, env_ids: Vec<String>, tool_specs: Vec<ToolSpec>) -> Self {
        Self {
            prompt,
            env_ids,
            tool_specs,
            subagent: None,
            subagents: None,
            history_resolver: None,
            record_store: None,
            budget: None,
            history: None,
            feedback: None,
        }
    }

    /// Enable recording; the caller versions the complete reuse contract.
    pub fn with_recording(mut self, contract: impl Into<String>) -> Self {
        self.history = Some(AgentOptions {
            contract: contract.into(),
            previous: None,
            store: None,
            prompt_eq: None,
        });
        self
    }

    /// Reuse a snapshot under an explicitly attested compatible contract.
    pub fn with_agent(mut self, previous: Agent<P>, contract: impl Into<String>) -> Self
    where
        P: PartialEq,
    {
        self.history = Some(AgentOptions {
            contract: contract.into(),
            previous: Some(previous),
            store: None,
            prompt_eq: Some(PartialEq::eq),
        });
        self
    }

    /// Compatibility spelling of with_agent; still resolves environments through replay.
    pub fn with_replay(self, previous: Agent<P>, contract: impl Into<String>) -> Self
    where
        P: PartialEq,
    {
        self.with_agent(previous, contract)
    }

    /// Select the shared record resolver/writer for root and child recording.
    /// A conflicting store on an existing history is rejected during construction.
    pub fn with_record_store(mut self, store: RecordStore<P>) -> Self {
        self.record_store = Some(store);
        self
    }

    /// Replay one environment to an existing answer, append once, then resume.
    /// The new continuation is inherited by everyone reaching that path.
    pub fn with_feedback(mut self, content: P) -> Self {
        self.feedback = Some(content);
        self
    }

    pub fn with_subagents(mut self, subagents: Vec<SubagentTool<P>>) -> Self {
        self.subagents = Some(subagents);
        self
    }

    pub fn with_agent_resolver(
        mut self,
        resolver: impl Fn(&AgentQuery<P>) -> Result<Option<AgentDescriptor<P>>, LoopError>
        + Send
        + Sync
        + 'static,
    ) -> Self
    where
        P: PartialEq,
    {
        self.history_resolver = Some(AgentResolver::new(resolver));
        self
    }

    /// Compatibility spelling of with_agent_resolver.
    pub fn with_history_resolver(
        self,
        resolver: impl Fn(&AgentQuery<P>) -> Result<Option<AgentDescriptor<P>>, LoopError>
        + Send
        + Sync
        + 'static,
    ) -> Self
    where
        P: PartialEq,
    {
        self.with_agent_resolver(resolver)
    }

    pub fn with_subagent(mut self, subagent: SubagentTool<P>) -> Self {
        self.subagent = Some(subagent);
        self
    }

    pub fn with_budget(mut self, budget: Budget<P>) -> Self {
        self.budget = Some(budget);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitEdge {
    /// Result-group debug representation; distinct edges may have identical labels.
    pub group_key: String,
    pub node: SplitNode,
}

/// Immutable result trie node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitNode {
    pub env_ids: Vec<String>,
    pub children: Vec<SplitEdge>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnvClass {
    pub id: String,
    pub env_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BatchResult<P> {
    pub tree: SplitNode,
    pub classes: Vec<EnvClass>,
    pub results: BTreeMap<String, P>,
    /// Compatibility field; prefer agent() for the recorded agent snapshot.
    pub history: Option<Agent<P>>,
    /// Compatibility field; prefer child_agents().
    pub child_histories: Vec<ChildAgentUpdate<P>>,
}

impl<P> BatchResult<P> {
    pub fn agent(&self) -> Option<&Agent<P>> {
        self.history.as_ref()
    }
    pub fn into_agent(self) -> Option<Agent<P>> {
        self.history
    }
    pub fn child_agents(&self) -> &[ChildAgentUpdate<P>] {
        &self.child_histories
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum LoopError {
    #[error("use subagent or subagents, not both")]
    AmbiguousSubagents,
    #[error("duplicate tool/subagent name {0:?}")]
    DuplicateToolName(String),
    #[error("conflicting ordinary/subagent spec for {0:?}")]
    ConflictingToolSpec(String),
    #[error("child history_contract for {0:?} must be non-empty (required with history_resolver)")]
    MissingChildHistoryContract(String),
    #[error("subagent {0:?} is unavailable in this child/depth")]
    UnavailableSubagent(String),
    #[error("invalid subagent arguments: {0}")]
    InvalidSubagentArguments(String),
    #[error("subagent preparation failed: {0}")]
    SubagentPreparation(String),
    #[error("history resolution failed: {0}")]
    HistoryResolution(String),
    #[error("incompatible history contract, prompt, tools, grouping, subagents or record store")]
    IncompatibleHistory,
    #[error("recorded history does not yet support budgets")]
    UnsupportedHistory,
    #[error("feedback requires a supplied history and exactly one environment")]
    InvalidFeedback,
    #[error("no recorded completed conversation on the feedback resolution path")]
    FeedbackResolutionMiss,
    #[error("expected one user continuation after a final assistant answer")]
    InvalidUserContinuation,
    #[error("resolve a recorded single-environment run before appending user context")]
    UnresolvedConversation,
    #[error("env_ids must be non-empty")]
    EmptyEnvIds,
    #[error("env_ids must be unique; duplicate {0:?}")]
    DuplicateEnvId(String),
    #[error("subagent.max_depth must be >= 1")]
    InvalidSubagentDepth,
    #[error("no outstanding request with id {0:?}")]
    UnknownRequest(String),
    #[error("{request_kind} request {request_id} must be answered with {completion_kind}")]
    WrongCompletionKind {
        request_id: String,
        request_kind: &'static str,
        completion_kind: &'static str,
    },
    #[error("tool completion for {0} must return exactly one result per env in the request")]
    InvalidToolCompletion(String),
    #[error("duplicate tool call id {0:?} in one assistant turn")]
    DuplicateToolCallId(String),
    #[error("the loop is not done")]
    NotDone,
    #[error("no runnable request but the loop is not done")]
    Stalled,
    #[error("env {env_id:?} has no tool named {tool_name:?}")]
    MissingTool { env_id: String, tool_name: String },
}

#[derive(Clone)]
struct Branch<P> {
    branch_id: String,
    env_ids: Vec<String>,
    messages: Vec<Message<P>>,
    split_depth: usize,
    steps: usize,
    history_path: Vec<PathStep>,
    turn_calls: Vec<ToolCall>,
    turn_results: HashMap<String, BTreeMap<String, P>>,
}

#[derive(Clone)]
struct BranchRecord {
    branch_id: String,
    parent_id: Option<String>,
    group_key: Option<String>,
    env_ids: Vec<String>,
}

#[derive(Clone)]
struct FinalBranch<P> {
    history_path: Vec<PathStep>,
    branch_id: String,
    env_ids: Vec<String>,
    content: P,
}

struct ChildLink<P, K> {
    child: Box<BatchLoop<P, K>>,
    branch_id: String,
    call: ToolCall,
    base: Option<AgentDescriptor<P>>,
    prior_runs: Vec<Arc<ChildAgentRun<P>>>,
}

struct PreparedChildren<P, K> {
    next_child: u64,
    children: IndexMap<String, (String, ChildLink<P, K>)>,
}

struct Registration<P> {
    specs: Vec<ToolSpec>,
    subagents: IndexMap<String, SubagentTool<P>>,
    contracts: Vec<SubagentContract>,
}

fn register<P>(
    specs: Vec<ToolSpec>,
    definitions: Vec<SubagentTool<P>>,
    independent: bool,
) -> Result<Registration<P>, LoopError> {
    let mut by_name = HashMap::new();
    for spec in &specs {
        if by_name.insert(spec.name.clone(), spec.clone()).is_some() {
            return Err(LoopError::DuplicateToolName(spec.name.clone()));
        }
    }
    let mut advertised = specs;
    let mut subagents = IndexMap::new();
    let mut contracts = Vec::new();
    for sub in definitions {
        if subagents.contains_key(&sub.spec.name) {
            return Err(LoopError::DuplicateToolName(sub.spec.name));
        }
        if sub.max_depth == 0 {
            return Err(LoopError::InvalidSubagentDepth);
        }
        if sub.history_contract.as_deref() == Some("")
            || (independent && sub.history_contract.is_none())
        {
            return Err(LoopError::MissingChildHistoryContract(sub.spec.name));
        }
        if let Some(existing) = by_name.get(&sub.spec.name) {
            if existing != &sub.spec {
                return Err(LoopError::ConflictingToolSpec(sub.spec.name));
            }
        } else {
            advertised.push(sub.spec.clone());
        }
        contracts.push(sub.contract());
        subagents.insert(sub.spec.name.clone(), sub);
    }
    Ok(Registration {
        specs: advertised,
        subagents,
        contracts,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExactRequestKind {
    Llm,
    Tool,
}

struct ChildRoute {
    child_key: String,
    kind: ExactRequestKind,
}

/// The provider-agnostic partition-refinement state machine.
///
/// Methods take `&mut self`; callers that need cross-thread serving can put
/// the loop behind their preferred `Mutex` without paying for synchronization
/// they do not need.
pub struct BatchLoop<P, K = P> {
    specs: Vec<ToolSpec>,
    subagents: IndexMap<String, SubagentTool<P>>,
    history_resolver: Option<AgentResolver<P>>,
    record_store: Option<RecordStore<P>>,
    unavailable_subagents: HashSet<String>,
    group_by: Arc<GroupFunction<P, K>>,
    budget: Option<Budget<P>>,
    id_prefix: String,

    seq: u64,
    next_branch: u64,
    next_child: u64,
    branches: HashMap<String, Branch<P>>,
    partition: Vec<BranchRecord>, // legacy per-run partition summary
    recorded: Option<Agent<P>>,
    feedback: Option<P>,
    resolve_only: bool,
    resolution_base: Option<Agent<P>>,
    error: Option<LoopError>,
    prompt_eq: Option<fn(&P, &P) -> bool>,
    custom_grouping: bool,
    finals: Vec<FinalBranch<P>>,
    child_histories: Vec<ChildAgentUpdate<P>>,
    ready_llm: VecDeque<LlmRequest<P>>,
    ready_tool: VecDeque<ToolRequest>,
    outstanding: HashMap<String, Request<P>>,
    children: IndexMap<String, ChildLink<P, K>>,
    child_routes: HashMap<String, ChildRoute>,
}

impl<P> BatchLoop<P, P>
where
    P: Clone + Eq + Hash + Debug + 'static,
{
    /// Construct a loop that groups tool results by value equality.
    pub fn new(config: LoopConfig<P>) -> Result<Self, LoopError> {
        Self::build(
            config,
            Arc::new(Clone::clone),
            String::new(),
            false,
            false,
            HashSet::new(),
        )
    }
}

impl<P, K> BatchLoop<P, K>
where
    P: Clone + 'static,
    K: Eq + Hash + Debug + 'static,
{
    /// Construct a loop with a caller-provided lossy grouping key.
    pub fn with_group_by(
        config: LoopConfig<P>,
        group_by: impl Fn(&P) -> K + Send + Sync + 'static,
    ) -> Result<Self, LoopError> {
        Self::build(
            config,
            Arc::new(group_by),
            String::new(),
            true,
            false,
            HashSet::new(),
        )
    }

    fn build(
        config: LoopConfig<P>,
        group_by: Arc<GroupFunction<P, K>>,
        id_prefix: String,
        custom_grouping: bool,
        resolve_only: bool,
        mut unavailable_subagents: HashSet<String>,
    ) -> Result<Self, LoopError> {
        if config.env_ids.is_empty() {
            return Err(LoopError::EmptyEnvIds);
        }
        let mut seen = HashSet::new();
        for env in &config.env_ids {
            if !seen.insert(env.clone()) {
                return Err(LoopError::DuplicateEnvId(env.clone()));
            }
        }
        if config.subagent.is_some() && config.subagents.is_some() {
            return Err(LoopError::AmbiguousSubagents);
        }
        let definitions = config
            .subagents
            .unwrap_or_else(|| config.subagent.into_iter().collect());
        let registry = register(
            config.tool_specs,
            definitions,
            config.history_resolver.is_some(),
        )?;
        unavailable_subagents.retain(|name| !registry.subagents.contains_key(name));
        let independent = config.history_resolver.is_some()
            || registry
                .subagents
                .values()
                .any(|s| s.history_contract.is_some());
        if independent && config.budget.is_some() {
            return Err(LoopError::UnsupportedHistory);
        }

        if config.feedback.is_some()
            && (config.env_ids.len() != 1
                || config.history.as_ref().is_none_or(|h| h.previous.is_none()))
        {
            return Err(LoopError::InvalidFeedback);
        }
        let resolve_only = resolve_only || config.feedback.is_some();
        let prompt_eq = config
            .history
            .as_ref()
            .and_then(|h| h.prompt_eq)
            .or_else(|| config.history_resolver.as_ref().map(|r| r.prompt_eq));
        let recorded = if let Some(options) = config.history {
            if options
                .store
                .as_ref()
                .zip(config.record_store.as_ref())
                .is_some_and(|(a, b)| !a.same_store(b))
            {
                return Err(LoopError::IncompatibleHistory);
            }
            if config.budget.is_some() {
                return Err(LoopError::UnsupportedHistory);
            }
            if options.contract.is_empty() {
                return Err(LoopError::IncompatibleHistory);
            }
            if let Some(previous) = options.previous {
                if config
                    .record_store
                    .as_ref()
                    .is_some_and(|s| !s.same_store(&previous.store))
                    || options
                        .store
                        .as_ref()
                        .is_some_and(|s| !s.same_store(&previous.store))
                {
                    return Err(LoopError::IncompatibleHistory);
                }
                if previous.descriptor.contract != options.contract
                    || !options
                        .prompt_eq
                        .is_some_and(|eq| eq(&previous.descriptor.prompt, &config.prompt))
                    || previous.descriptor.tools != registry.specs
                    || previous.descriptor.custom_grouping != custom_grouping
                    || previous.descriptor.subagents != registry.contracts
                {
                    return Err(LoopError::IncompatibleHistory);
                }
                Some(previous)
            } else {
                let store = options
                    .store
                    .or_else(|| config.record_store.clone())
                    .unwrap_or_default();
                Some(Agent {
                    descriptor: AgentDescriptor {
                        contract: options.contract,
                        prompt: config.prompt.clone(),
                        tools: registry.specs.clone(),
                        custom_grouping,
                        root: store.put(HistoryNode::default()),
                        subagents: registry.contracts,
                    },
                    store,
                })
            }
        } else {
            None
        };
        let record_store = recorded
            .as_ref()
            .map(|h| h.store.clone())
            .or(config.record_store)
            .or_else(|| independent.then(RecordStore::default));
        let prompt = recorded
            .as_ref()
            .map_or(config.prompt, |h| h.descriptor.prompt.clone());
        let root = Branch {
            branch_id: "root".to_owned(),
            env_ids: config.env_ids.clone(),
            messages: vec![Message::user(prompt)],
            split_depth: 0,
            steps: 0,
            history_path: Vec::new(),
            turn_calls: Vec::new(),
            turn_results: HashMap::new(),
        };
        let root_record = BranchRecord {
            branch_id: root.branch_id.clone(),
            parent_id: None,
            group_key: None,
            env_ids: root.env_ids.clone(),
        };
        let mut loop_ = Self {
            specs: registry.specs,
            subagents: registry.subagents,
            history_resolver: config.history_resolver,
            record_store,
            unavailable_subagents,
            group_by,
            budget: config.budget,
            id_prefix,
            seq: 0,
            next_branch: 0,
            next_child: 0,
            branches: HashMap::new(),
            partition: vec![root_record],
            resolution_base: if resolve_only { recorded.clone() } else { None },
            recorded,
            feedback: config.feedback,
            resolve_only,
            error: None,
            prompt_eq,
            custom_grouping,
            finals: Vec::new(),
            child_histories: Vec::new(),
            ready_llm: VecDeque::new(),
            ready_tool: VecDeque::new(),
            outstanding: HashMap::new(),
            children: IndexMap::new(),
            child_routes: HashMap::new(),
        };
        loop_.schedule_branch(root);
        if let Some(error) = loop_.error {
            return Err(error);
        }
        Ok(loop_)
    }

    /// Pop one runnable request. `None` is not a termination signal; use
    /// [`BatchLoop::is_done`] to distinguish blocked from finished.
    pub fn next_request(&mut self, kind: RequestKind) -> Option<Request<P>> {
        self.rollup_children();
        if self.error.is_some() {
            return None;
        }
        if let Some(request) = self.pop_child_request(kind) {
            return Some(request);
        }

        let request = match kind {
            RequestKind::Llm => self.ready_llm.pop_front().map(Request::Llm),
            RequestKind::Tool => self.ready_tool.pop_front().map(Request::Tool),
            RequestKind::Any => self.pop_any(),
        }?;
        self.outstanding
            .insert(request.request_id().to_owned(), request.clone());
        Some(request)
    }

    /// Deliver a completion for any outstanding root or nested request.
    /// Validation errors leave pending requests correctable. Child factories and
    /// constructors prepare once before the assistant turn is fulfilled. Detected
    /// nested incompatibility, a resolution miss, or invalid user continuation is
    /// terminal: pending work is cancelled, is_done() becomes true, and result()/
    /// error() report the failure. Arbitrary callback panics are not recovered.
    pub fn complete(&mut self, completion: Completion<P>) -> Result<(), LoopError> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let request_id = completion.request_id().to_owned();
        if let Some(route) = self.child_routes.get(&request_id) {
            let child_key = route.child_key.clone();
            let child = &mut self
                .children
                .get_mut(&child_key)
                .expect("child route must reference a live child")
                .child;
            let result = child.complete(completion);
            if let Some(error) = child.error.clone() {
                self.fail(error.clone());
                return Err(error);
            }
            result?;
            self.child_routes.remove(&request_id);
            self.rollup_children();
            return self.error.clone().map_or(Ok(()), Err);
        }

        let request = self
            .outstanding
            .get(&request_id)
            .cloned()
            .ok_or_else(|| LoopError::UnknownRequest(request_id.clone()))?;
        self.validate_completion(&request, &completion)?;

        match (request, completion) {
            (Request::Llm(request), Completion::Llm(completion)) => {
                self.handle_llm(request, completion)?
            }
            (Request::Tool(request), Completion::Tool(completion)) => {
                self.handle_tool(request, completion)
            }
            _ => unreachable!("completion kind was validated"),
        }
        self.outstanding.remove(&request_id);
        self.error.clone().map_or(Ok(()), Err)
    }

    /// True on success or terminal failure; result() returns the stored error on failure.
    pub fn is_done(&mut self) -> bool {
        self.rollup_children();
        self.ready_llm.is_empty()
            && self.ready_tool.is_empty()
            && self.outstanding.is_empty()
            && self.child_routes.is_empty()
            && self.children.is_empty()
    }

    pub fn outstanding_llm(&self) -> usize {
        self.outstanding
            .values()
            .filter(|request| matches!(request, Request::Llm(_)))
            .count()
            + self
                .child_routes
                .values()
                .filter(|route| route.kind == ExactRequestKind::Llm)
                .count()
    }

    pub fn outstanding_tool(&self) -> usize {
        self.outstanding
            .values()
            .filter(|request| matches!(request, Request::Tool(_)))
            .count()
            + self
                .child_routes
                .values()
                .filter(|route| route.kind == ExactRequestKind::Tool)
                .count()
    }

    pub fn outstanding_count(&self) -> usize {
        self.outstanding.len() + self.child_routes.len()
    }

    /// Build the immutable trie, final classes, and per-environment results.
    /// A terminal resolution/history failure returns its stored error, not a partial result.
    pub fn result(&mut self) -> Result<BatchResult<P>, LoopError> {
        let done = self.is_done();
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if !done {
            return Err(LoopError::NotDone);
        }
        Ok(self.finalize())
    }

    /// Terminal resolution/history error, not a correctable completion validation error.
    pub fn error(&self) -> Option<&LoopError> {
        self.error.as_ref()
    }

    fn fail(&mut self, error: LoopError) {
        self.error = Some(error);
        if let Some(base) = self.resolution_base.take() {
            self.recorded = Some(base);
        }
        self.ready_llm.clear();
        self.ready_tool.clear();
        self.outstanding.clear();
        self.child_routes.clear();
        self.children.clear();
        self.branches.clear();
        self.finals.clear();
    }

    /// A structurally shared snapshot; builders and pending requests are not persisted.
    pub fn agent(&self) -> Option<Agent<P>> {
        let mut snapshot = self.recorded.clone()?;
        for link in self.children.values() {
            let child = link
                .child
                .agent()
                .expect("recorded parent has recorded child agents");
            let branch = &self.branches[&link.branch_id];
            let current = snapshot.node(&branch.history_path);
            if current.subagents.iter().any(|old| {
                old.call_id == link.call.id
                    && old.history.root == child.descriptor.root
                    && old.runs.len() == link.prior_runs.len() + 1
            }) {
                continue;
            }
            let mut node = current.as_ref().clone();
            let mut runs = link.prior_runs.clone();
            runs.push(Arc::new(ChildAgentRun {
                env_ids: branch.env_ids.clone(),
                base: link.base.clone(),
                history: child.descriptor.clone(),
            }));
            let entry = SubagentRecord {
                call_id: link.call.id.clone(),
                history: child.descriptor,
                runs,
            };
            if let Some(old) = node
                .subagents
                .iter_mut()
                .find(|s| s.call_id == link.call.id)
            {
                *old = entry;
            } else {
                node.subagents.push(entry);
            }
            snapshot.replace_node(&branch.history_path, node);
        }
        Some(snapshot)
    }

    /// Compatibility spelling of agent().
    pub fn history(&self) -> Option<Agent<P>> {
        self.agent()
    }

    /// Fork an open shared-path continuation of an already completed run. This
    /// legacy helper does not resume this loop. Prefer with_feedback for strict
    /// replay-only resolution before resumption. No endpoint ID is accepted.
    pub fn append_user_message(&mut self, content: P) -> Result<Agent<P>, LoopError> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        if !self.is_done() || self.finals.len() != 1 || self.finals[0].env_ids.len() != 1 {
            return Err(LoopError::UnresolvedConversation);
        }
        let mut snapshot = self.history().ok_or(LoopError::UnresolvedConversation)?;
        let path = &self.finals[0].history_path;
        let mut node = snapshot.node(path).as_ref().clone();
        node.user_continuations.push(UserContinuation {
            message: Message::user(content),
            continuation: snapshot.store.put(HistoryNode::default()),
            resolved_env_id: Some(self.finals[0].env_ids[0].clone()),
        });
        snapshot.replace_node(path, node);
        Ok(snapshot)
    }

    fn validate_completion(
        &self,
        request: &Request<P>,
        completion: &Completion<P>,
    ) -> Result<(), LoopError> {
        match (request, completion) {
            (Request::Llm(_), Completion::Tool(completion)) => {
                Err(LoopError::WrongCompletionKind {
                    request_id: completion.request_id.clone(),
                    request_kind: "LLM",
                    completion_kind: "LlmCompletion",
                })
            }
            (Request::Tool(_), Completion::Llm(completion)) => {
                Err(LoopError::WrongCompletionKind {
                    request_id: completion.request_id.clone(),
                    request_kind: "tool",
                    completion_kind: "ToolCompletion",
                })
            }
            (Request::Llm(_), Completion::Llm(completion)) => {
                let mut ids = HashSet::new();
                for call in &completion.tool_calls {
                    if self.unavailable_subagents.contains(&call.name) {
                        return Err(LoopError::UnavailableSubagent(call.name.clone()));
                    }
                    if !ids.insert(&call.id) {
                        return Err(LoopError::DuplicateToolCallId(call.id.clone()));
                    }
                }
                Ok(())
            }
            (Request::Tool(request), Completion::Tool(completion)) => {
                let returned: HashSet<&str> = completion
                    .results
                    .iter()
                    .map(|(env, _)| env.as_str())
                    .collect();
                let expected: HashSet<&str> = request.env_ids.iter().map(String::as_str).collect();
                if completion.results.len() != request.env_ids.len() || returned != expected {
                    return Err(LoopError::InvalidToolCompletion(
                        completion.request_id.clone(),
                    ));
                }
                Ok(())
            }
        }
    }

    fn pop_child_request(&mut self, kind: RequestKind) -> Option<Request<P>> {
        let keys: Vec<String> = self.children.keys().cloned().collect();
        for child_key in keys {
            let request = self
                .children
                .get_mut(&child_key)
                .expect("collected child key must still exist")
                .child
                .next_request(kind);
            if let Some(request) = request {
                let exact_kind = match &request {
                    Request::Llm(_) => ExactRequestKind::Llm,
                    Request::Tool(_) => ExactRequestKind::Tool,
                };
                self.child_routes.insert(
                    request.request_id().to_owned(),
                    ChildRoute {
                        child_key,
                        kind: exact_kind,
                    },
                );
                return Some(request);
            }
        }
        None
    }

    fn rollup_children(&mut self) {
        while self.error.is_none() {
            let done_keys: Vec<String> = self
                .children
                .iter_mut()
                .filter_map(|(key, link)| link.child.is_done().then(|| key.clone()))
                .collect();
            if let Some(error) = self
                .children
                .values()
                .find_map(|link| link.child.error.clone())
            {
                self.fail(error);
                return;
            }
            self.recorded = self.history();
            if done_keys.is_empty() {
                return;
            }
            for key in done_keys {
                let mut link = self
                    .children
                    .shift_remove(&key)
                    .expect("done child must still exist");
                let child_result = link.child.result().expect("done child must have a result");
                let mut branch = self
                    .branches
                    .remove(&link.branch_id)
                    .expect("parent branch must wait for its child");
                branch
                    .turn_results
                    .insert(link.call.id.clone(), child_result.results);
                self.child_histories.extend(child_result.child_histories);
                if let Some(history) = child_result.history {
                    self.child_histories.push(ChildAgentUpdate {
                        request_prefix: link.child.id_prefix.clone(),
                        branch_id: branch.branch_id.clone(),
                        depth: branch.steps,
                        tool_call: link.call,
                        env_ids: branch.env_ids.clone(),
                        base: link.base,
                        history,
                    });
                }
                if branch.turn_results.len() == branch.turn_calls.len() {
                    self.advance_turn(branch);
                    if self.error.is_some() {
                        return;
                    }
                } else {
                    self.branches.insert(branch.branch_id.clone(), branch);
                }
            }
            // Replayed parent turns can spawn children already finished by replay.
            // Settle them without returning a false stalled frontier.
        }
    }

    fn pop_any(&mut self) -> Option<Request<P>> {
        match (self.ready_llm.front(), self.ready_tool.front()) {
            (None, None) => None,
            (Some(_), None) => self.ready_llm.pop_front().map(Request::Llm),
            (None, Some(_)) => self.ready_tool.pop_front().map(Request::Tool),
            (Some(llm), Some(tool)) if llm.stats.seq <= tool.stats.seq => {
                self.ready_llm.pop_front().map(Request::Llm)
            }
            (Some(_), Some(_)) => self.ready_tool.pop_front().map(Request::Tool),
        }
    }

    fn next_stats(&mut self, branch: &Branch<P>) -> RequestStats {
        self.seq += 1;
        RequestStats {
            depth: branch.steps,
            split_depth: branch.split_depth,
            seq: self.seq,
        }
    }

    fn schedule_branch(&mut self, mut branch: Branch<P>) {
        if self.error.is_some() {
            return;
        }
        while let Some(history) = &self.recorded {
            let node = history.node(&branch.history_path);
            if !node.user_continuations.is_empty()
                && (node.user_continuations.len() != 1
                    || !node
                        .assistant
                        .as_ref()
                        .is_some_and(|m| m.tool_calls.is_empty())
                    || node.user_continuations[0].message.role != Role::User
                    || !node.user_continuations[0].message.tool_calls.is_empty()
                    || node.user_continuations[0].message.tool_call_id.is_some())
            {
                self.fail(LoopError::InvalidUserContinuation);
                return;
            }
            let Some(message) = node.assistant.clone() else {
                break;
            };
            let Some(next) = self.accept_assistant(branch, message) else {
                return;
            };
            branch = next;
        }
        if self.resolve_only {
            self.fail(LoopError::FeedbackResolutionMiss);
            return;
        }
        if let Some(budget) = &self.budget
            && branch.steps >= budget.max_steps_per_branch
        {
            let content = (budget.exhausted)(branch.steps);
            self.finish_branch(branch, content);
            return;
        }
        let stats = self.next_stats(&branch);
        self.ready_llm.push_back(LlmRequest {
            request_id: format!("{}llm-{}", self.id_prefix, stats.seq),
            branch_id: branch.branch_id.clone(),
            env_ids: branch.env_ids.clone(),
            messages: branch.messages.clone(),
            tools: self.specs.clone(),
            stats,
        });
        self.branches.insert(branch.branch_id.clone(), branch);
    }

    fn request_tool(&mut self, branch: &Branch<P>, call: ToolCall) {
        let stats = self.next_stats(branch);
        self.ready_tool.push_back(ToolRequest {
            request_id: format!("{}tool-{}", self.id_prefix, stats.seq),
            branch_id: branch.branch_id.clone(),
            tool_call: call,
            env_ids: branch.env_ids.clone(),
            stats,
        });
    }

    fn finish_branch(&mut self, branch: Branch<P>, content: P) {
        self.finals.push(FinalBranch {
            history_path: branch.history_path,
            branch_id: branch.branch_id,
            env_ids: branch.env_ids,
            content,
        });
    }

    fn handle_llm(
        &mut self,
        request: LlmRequest<P>,
        completion: LlmCompletion<P>,
    ) -> Result<(), LoopError> {
        let branch = self
            .branches
            .get(&request.branch_id)
            .expect("LLM request must reference an active branch");
        let message = Message::assistant(completion.content, completion.tool_calls);
        let prepared = self.prepare_assistant_children(branch, &message)?;
        if let Some(branch) = self.accept_prepared(branch.clone(), message, prepared) {
            self.schedule_branch(branch);
        }
        Ok(())
    }

    fn prepare_assistant_children(
        &self,
        branch: &Branch<P>,
        message: &Message<P>,
    ) -> Result<PreparedChildren<P, K>, LoopError> {
        let mut children = IndexMap::new();
        let mut next_child = self.next_child;
        for call in &message.tool_calls {
            if self.unavailable_subagents.contains(&call.name) {
                return Err(LoopError::UnavailableSubagent(call.name.clone()));
            }
            if self.subagents.contains_key(&call.name) {
                next_child = next_child
                    .checked_add(1)
                    .expect("child invocation counter exhausted");
                let key = format!("sub:{next_child}/");
                let child = self.prepare_child(branch, call.clone(), &key)?;
                children.insert(call.id.clone(), (key, child));
            }
        }
        Ok(PreparedChildren {
            next_child,
            children,
        })
    }

    fn accept_assistant(&mut self, branch: Branch<P>, message: Message<P>) -> Option<Branch<P>> {
        match self.prepare_assistant_children(&branch, &message) {
            Ok(prepared) => self.accept_prepared(branch, message, prepared),
            Err(error) => {
                self.fail(error);
                None
            }
        }
    }

    fn accept_prepared(
        &mut self,
        mut branch: Branch<P>,
        message: Message<P>,
        mut prepared: PreparedChildren<P, K>,
    ) -> Option<Branch<P>> {
        self.branches.remove(&branch.branch_id);
        if let Some(history) = &mut self.recorded {
            let mut node = history.node(&branch.history_path).as_ref().clone();
            node.assistant.get_or_insert_with(|| message.clone());
            node.env_ids.extend(branch.env_ids.clone());
            history.replace_node(&branch.history_path, node);
        }
        branch.steps += 1;
        branch.messages.push(message.clone());
        if message.tool_calls.is_empty() {
            if let Some(history) = &mut self.recorded {
                let mut node = history.node(&branch.history_path).as_ref().clone();
                if node.user_continuations.is_empty()
                    && let Some(content) = self.feedback.take()
                {
                    node.user_continuations.push(UserContinuation {
                        message: Message::user(content),
                        continuation: history.store.put(HistoryNode::default()),
                        resolved_env_id: Some(branch.env_ids[0].clone()),
                    });
                    history.replace_node(&branch.history_path, node.clone());
                    self.resolve_only = false;
                    self.resolution_base = None;
                }
                if let Some(edge) = node.user_continuations.first() {
                    branch.messages.push(edge.message.clone());
                    branch.history_path.push(PathStep::User(0));
                    return Some(branch);
                }
            }
            self.finish_branch(branch, message.content);
            return None;
        }

        branch.turn_calls = message.tool_calls;
        branch.turn_results.clear();
        self.next_child = prepared.next_child;
        for call in branch.turn_calls.clone() {
            if let Some((key, child)) = prepared.children.shift_remove(&call.id) {
                self.children.insert(key, child);
            } else {
                self.request_tool(&branch, call);
            }
        }
        self.branches.insert(branch.branch_id.clone(), branch);
        // Attach newly spawned (possibly still open) children once, so repeated
        // snapshots and rejected completions retain the same root reference.
        self.recorded = self.history();
        None
    }

    fn handle_tool(&mut self, request: ToolRequest, completion: ToolCompletion<P>) {
        let mut branch = self
            .branches
            .remove(&request.branch_id)
            .expect("tool request must reference an active branch");
        branch.turn_results.insert(
            request.tool_call.id,
            completion.results.into_iter().collect(),
        );
        if branch.turn_results.len() == branch.turn_calls.len() {
            self.advance_turn(branch);
        } else {
            self.branches.insert(branch.branch_id.clone(), branch);
        }
    }

    fn prepare_child(
        &self,
        branch: &Branch<P>,
        call: ToolCall,
        child_key: &str,
    ) -> Result<ChildLink<P, K>, LoopError> {
        let subagent = &self.subagents[&call.name];
        let declared = subagent
            .subagents
            .clone()
            .unwrap_or_else(|| vec![subagent.clone()]);
        let nested: Vec<_> = if subagent.max_depth > 1 {
            declared
                .iter()
                .map(|s| s.with_max_depth(s.max_depth.min(subagent.max_depth - 1)))
                .collect()
        } else {
            vec![]
        };
        let mut unavailable = self.unavailable_subagents.clone();
        unavailable.extend(self.subagents.keys().cloned());
        unavailable.extend(declared.iter().map(|s| s.spec.name.clone()));
        for s in &nested {
            unavailable.remove(&s.spec.name);
        }
        let raw_specs = (subagent.sub_tool_specs)(&branch.env_ids[0])?
            .into_iter()
            .filter(|spec| !unavailable.contains(&spec.name))
            .collect();
        let registry = register(raw_specs, nested.clone(), self.history_resolver.is_some())?;
        let prompt = (subagent.build_prompt)(&call.args)?;
        let contract = subagent.history_contract.clone().or_else(|| {
            self.recorded
                .as_ref()
                .map(|h| h.descriptor.contract.clone())
        });
        let mut previous = None;
        let mut prior_runs = Vec::new();
        if let Some(history) = &self.recorded {
            let node = history.node(&branch.history_path);
            if let Some(entry) = node.subagents.iter().find(|s| s.call_id == call.id) {
                previous = Some(Agent {
                    descriptor: entry.agent_descriptor().clone(),
                    store: history.store.clone(),
                });
                prior_runs = entry.runs.clone();
                if prior_runs.is_empty() {
                    prior_runs.push(Arc::new(ChildAgentRun {
                        env_ids: vec![],
                        base: None,
                        history: entry.history.clone(),
                    }));
                }
            }
        }
        // Recorded parent invocations pin a revision. Lookup selects only new
        // invocations, never silently updating a completed parent's dependencies.
        if self.resolve_only && previous.is_none() {
            return Err(LoopError::FeedbackResolutionMiss);
        }
        if previous.is_none()
            && let Some(resolver) = &self.history_resolver
        {
            let query = AgentQuery {
                contract: contract.clone().expect("independent child has a contract"),
                prompt: prompt.clone(),
                tools: registry.specs.clone(),
                custom_grouping: self.custom_grouping,
                subagents: registry.contracts,
            };
            if let Some(descriptor) = (resolver.resolve)(&query)? {
                previous = Some(Agent {
                    descriptor,
                    store: self
                        .record_store
                        .clone()
                        .expect("independent recording store"),
                });
            }
        }
        let base = previous.as_ref().map(|h| h.descriptor.clone());
        let history_options = contract.map(|contract| AgentOptions {
            contract,
            previous,
            store: self.record_store.clone(),
            prompt_eq: self.prompt_eq,
        });
        let config = LoopConfig {
            prompt,
            env_ids: branch.env_ids.clone(),
            tool_specs: registry.specs,
            subagent: None,
            subagents: Some(nested),
            history_resolver: self.history_resolver.clone(),
            record_store: self.record_store.clone(),
            budget: self.budget.clone(),
            history: history_options,
            feedback: None,
        };
        // The locally reserved key uses only internal invocation counters, not
        // opaque provider IDs, and is published only after all children prepare.
        let child = Self::build(
            config,
            Arc::clone(&self.group_by),
            format!("{}{child_key}", self.id_prefix),
            self.custom_grouping,
            self.resolve_only,
            unavailable,
        )?;
        Ok(ChildLink {
            child: Box::new(child),
            branch_id: branch.branch_id.clone(),
            call,
            base,
            prior_runs,
        })
    }

    fn advance_turn(&mut self, mut branch: Branch<P>) {
        let mut groups: IndexMap<Vec<K>, Vec<String>> = IndexMap::new();
        for env in &branch.env_ids {
            let key = branch
                .turn_calls
                .iter()
                .map(|call| {
                    let result = branch
                        .turn_results
                        .get(&call.id)
                        .and_then(|results| results.get(env))
                        .expect("every call must have one result per env");
                    (self.group_by)(result)
                })
                .collect::<Vec<_>>();
            groups.entry(key).or_default().push(env.clone());
        }

        let split = groups.len() > 1;
        for (key, env_ids) in groups {
            let mut messages = branch.messages.clone();
            let mut history_path = branch.history_path.clone();
            if let Some(history) = &mut self.recorded {
                let mut node = history.node(&history_path).as_ref().clone();
                let evidence: Vec<_> = env_ids
                    .iter()
                    .map(|env| EnvObservation {
                        env_id: env.clone(),
                        outputs: branch
                            .turn_calls
                            .iter()
                            .map(|c| OutputRecord {
                                tool_output: branch.turn_results[&c.id][env].clone(),
                            })
                            .collect(),
                    })
                    .collect();
                let matched = node.observations.iter().position(|edge| {
                    let old_key: Vec<K> = edge
                        .representative
                        .iter()
                        .map(|o| (self.group_by)(&o.tool_output))
                        .collect();
                    old_key == key
                });
                if self.resolve_only && matched.is_none() {
                    self.fail(LoopError::FeedbackResolutionMiss);
                    return;
                }
                let index = matched.unwrap_or(node.observations.len());
                if matched.is_none() {
                    node.observations.push(Observation {
                        representative: evidence[0].outputs.clone(),
                        evidence: Vec::new(),
                        continuation: history.store.put(HistoryNode::default()),
                    });
                }
                let edge = &mut node.observations[index];
                edge.evidence.extend(evidence);
                for (call, output) in branch.turn_calls.iter().zip(&edge.representative) {
                    messages.push(Message::tool(output.tool_output.clone(), call.id.clone()));
                }
                history.replace_node(&history_path, node);
                history_path.push(PathStep::Observation(index));
            } else {
                Self::append_tool_messages(
                    &mut messages,
                    &branch.turn_calls,
                    &branch.turn_results,
                    &env_ids[0],
                );
            }
            if !split {
                branch.messages = messages;
                branch.history_path = history_path;
                branch.turn_calls.clear();
                branch.turn_results.clear();
                self.schedule_branch(branch);
                return;
            }
            self.next_branch += 1;
            let branch_id = format!("{}/b{}", branch.branch_id, self.next_branch);
            let child = Branch {
                branch_id: branch_id.clone(),
                env_ids: env_ids.clone(),
                messages,
                split_depth: branch.split_depth + 1,
                steps: branch.steps,
                history_path,
                turn_calls: Vec::new(),
                turn_results: HashMap::new(),
            };
            self.partition.push(BranchRecord {
                branch_id,
                parent_id: Some(branch.branch_id.clone()),
                group_key: Some(format!("{key:?}")),
                env_ids,
            });
            self.schedule_branch(child);
            if self.error.is_some() {
                return;
            }
        }
    }

    fn append_tool_messages(
        messages: &mut Vec<Message<P>>,
        calls: &[ToolCall],
        results: &HashMap<String, BTreeMap<String, P>>,
        representative_env: &str,
    ) {
        for call in calls {
            let result = results
                .get(&call.id)
                .and_then(|by_env| by_env.get(representative_env))
                .expect("representative result must exist")
                .clone();
            messages.push(Message::tool(result, call.id.clone()));
        }
    }

    fn finalize(&self) -> BatchResult<P> {
        let mut children_of: HashMap<String, Vec<SplitEdge>> = HashMap::new();
        let mut root = None;
        for record in self.partition.iter().rev() {
            let mut children = children_of.remove(&record.branch_id).unwrap_or_default();
            children.reverse();
            let node = SplitNode {
                env_ids: record.env_ids.clone(),
                children,
            };
            if let Some(parent_id) = &record.parent_id {
                children_of
                    .entry(parent_id.clone())
                    .or_default()
                    .push(SplitEdge {
                        group_key: record
                            .group_key
                            .clone()
                            .expect("non-root record must have a group key"),
                        node,
                    });
            } else {
                root = Some(node);
            }
        }

        let classes = self
            .finals
            .iter()
            .map(|branch| EnvClass {
                id: branch.branch_id.clone(),
                env_ids: branch.env_ids.clone(),
            })
            .collect();
        let mut results = BTreeMap::new();
        for branch in &self.finals {
            for env in &branch.env_ids {
                results.insert(env.clone(), branch.content.clone());
            }
        }
        BatchResult {
            tree: root.expect("partition always contains a root"),
            classes,
            results,
            history: self.recorded.clone(),
            child_histories: self.child_histories.clone(),
        }
    }
}

/// A callable environment-local tool used by the synchronous driver.
pub trait Tool<P>: Send + Sync {
    fn spec(&self) -> ToolSpec;
    fn call(&self, args: &Value) -> P;
}

/// Convenience implementation of [`Tool`] backed by a closure.
pub struct FunctionTool<P> {
    spec: ToolSpec,
    call: Arc<ToolFunction<P>>,
}

impl<P> FunctionTool<P> {
    pub fn new(spec: ToolSpec, call: impl Fn(&Value) -> P + Send + Sync + 'static) -> Self {
        Self {
            spec,
            call: Arc::new(call),
        }
    }
}

impl<P> Tool<P> for FunctionTool<P> {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn call(&self, args: &Value) -> P {
        (self.call)(args)
    }
}

/// One entry returned by a synchronous driver's per-environment tool factory.
pub enum ToolEntry<P> {
    Callable(Arc<dyn Tool<P>>),
    Subagent(SubagentTool<P>),
}

impl<P> Clone for ToolEntry<P> {
    fn clone(&self) -> Self {
        match self {
            Self::Callable(tool) => Self::Callable(Arc::clone(tool)),
            Self::Subagent(subagent) => Self::Subagent(subagent.clone()),
        }
    }
}

/// Synchronous convenience driver with value-equality grouping.
pub fn run_to_completion<P, TF, LF>(
    prompt: P,
    env_ids: Vec<String>,
    tools: TF,
    complete_llm: LF,
    budget: Option<Budget<P>>,
) -> Result<BatchResult<P>, LoopError>
where
    P: Clone + Eq + Hash + Debug + 'static,
    TF: Fn(&str) -> Vec<ToolEntry<P>>,
    LF: FnMut(&LlmRequest<P>) -> LlmCompletion<P>,
{
    run_to_completion_inner(
        prompt,
        env_ids,
        tools,
        complete_llm,
        Clone::clone,
        budget,
        false,
    )
}

/// Synchronous convenience driver with a caller-provided grouping key.
pub fn run_to_completion_by<P, K, TF, LF, G>(
    prompt: P,
    env_ids: Vec<String>,
    tools: TF,
    complete_llm: LF,
    group_by: G,
    budget: Option<Budget<P>>,
) -> Result<BatchResult<P>, LoopError>
where
    P: Clone + 'static,
    K: Eq + Hash + Debug + 'static,
    TF: Fn(&str) -> Vec<ToolEntry<P>>,
    LF: FnMut(&LlmRequest<P>) -> LlmCompletion<P>,
    G: Fn(&P) -> K + Send + Sync + 'static,
{
    run_to_completion_inner(prompt, env_ids, tools, complete_llm, group_by, budget, true)
}

fn run_to_completion_inner<P, K, TF, LF, G>(
    prompt: P,
    env_ids: Vec<String>,
    tools: TF,
    mut complete_llm: LF,
    group_by: G,
    budget: Option<Budget<P>>,
    custom_grouping: bool,
) -> Result<BatchResult<P>, LoopError>
where
    P: Clone + 'static,
    K: Eq + Hash + Debug + 'static,
    TF: Fn(&str) -> Vec<ToolEntry<P>>,
    LF: FnMut(&LlmRequest<P>) -> LlmCompletion<P>,
    G: Fn(&P) -> K + Send + Sync + 'static,
{
    if env_ids.is_empty() {
        return Err(LoopError::EmptyEnvIds);
    }
    let first_entries = tools(&env_ids[0]);
    let mut specs = Vec::new();
    let mut subagents = Vec::new();
    let mut names = HashSet::new();
    for entry in first_entries {
        let spec = match entry {
            ToolEntry::Callable(tool) => tool.spec(),
            ToolEntry::Subagent(candidate) => {
                let spec = candidate.spec.clone();
                subagents.push(candidate);
                spec
            }
        };
        if !names.insert(spec.name.clone()) {
            return Err(LoopError::DuplicateToolName(spec.name));
        }
        specs.push(spec);
    }

    let mut config = LoopConfig::new(prompt, env_ids, specs).with_subagents(subagents);
    config.budget = budget;
    let mut loop_ = BatchLoop::<P, K>::build(
        config,
        Arc::new(group_by),
        String::new(),
        custom_grouping,
        false,
        HashSet::new(),
    )?;

    while !loop_.is_done() {
        let request = loop_
            .next_request(RequestKind::Any)
            .ok_or(LoopError::Stalled)?;
        match request {
            Request::Llm(request) => {
                loop_.complete(Completion::Llm(complete_llm(&request)))?;
            }
            Request::Tool(request) => {
                let mut results = Vec::with_capacity(request.env_ids.len());
                for env in &request.env_ids {
                    let mut names = HashSet::new();
                    let mut selected = None;
                    for entry in tools(env) {
                        let name = match &entry {
                            ToolEntry::Callable(tool) => tool.spec().name,
                            ToolEntry::Subagent(sub) => sub.spec.name.clone(),
                        };
                        if !names.insert(name.clone()) {
                            return Err(LoopError::DuplicateToolName(name));
                        }
                        if name == request.tool_call.name
                            && let ToolEntry::Callable(tool) = entry
                        {
                            selected = Some(tool);
                        }
                    }
                    let tool = selected.ok_or_else(|| LoopError::MissingTool {
                        env_id: env.clone(),
                        tool_name: request.tool_call.name.clone(),
                    })?;
                    results.push((env.clone(), tool.call(&request.tool_call.args)));
                }
                loop_.complete(Completion::Tool(ToolCompletion {
                    request_id: request.request_id,
                    results,
                }))?;
            }
        }
    }
    loop_.result()
}
