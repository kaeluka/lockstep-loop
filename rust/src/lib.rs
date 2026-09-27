//! Provider-agnostic lockstep agent orchestration.
//!
//! A [`BatchLoop`] starts one conversation for an equivalence class of
//! environments. Tool calls fan out over every environment in the class; the
//! class splits only when tool results differ. The caller owns all LLM and
//! tool execution and may complete requests in any order.

use indexmap::IndexMap;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::sync::Arc;
use thiserror::Error;

type BudgetFallback<P> = dyn Fn(usize) -> P + Send + Sync;
type PromptBuilder<P> = dyn Fn(&Value) -> P + Send + Sync;
type SubToolSpecFactory = dyn Fn(&str) -> Vec<ToolSpec> + Send + Sync;
type GroupFunction<P, K> = dyn Fn(&P) -> K + Send + Sync;
type ToolFunction<P> = dyn Fn(&Value) -> P + Send + Sync;

/// Provider-neutral tool description.
#[derive(Clone, Debug, PartialEq)]
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
#[derive(Clone, Debug, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Tool,
}

/// Provider-neutral conversation message. Payloads are opaque to the loop.
#[derive(Clone, Debug, PartialEq)]
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

/// Reserved tool implemented by a nested [`BatchLoop`].
pub struct SubagentTool<P> {
    pub spec: ToolSpec,
    pub max_depth: usize,
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
        Self {
            spec,
            max_depth,
            build_prompt: Arc::new(build_prompt),
            sub_tool_specs: Arc::new(sub_tool_specs),
        }
    }

    fn with_max_depth(&self, max_depth: usize) -> Self {
        Self {
            spec: self.spec.clone(),
            max_depth,
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
    pub budget: Option<Budget<P>>,
}

impl<P> LoopConfig<P> {
    pub fn new(prompt: P, env_ids: Vec<String>, tool_specs: Vec<ToolSpec>) -> Self {
        Self {
            prompt,
            env_ids,
            tool_specs,
            subagent: None,
            budget: None,
        }
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
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LoopError {
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
    branch_id: String,
    env_ids: Vec<String>,
    content: P,
}

struct ChildLink<P, K> {
    child: Box<BatchLoop<P, K>>,
    branch_id: String,
    call: ToolCall,
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
    subagent: Option<SubagentTool<P>>,
    group_by: Arc<GroupFunction<P, K>>,
    budget: Option<Budget<P>>,
    id_prefix: String,

    seq: u64,
    next_branch: u64,
    branches: HashMap<String, Branch<P>>,
    history: Vec<BranchRecord>,
    finals: Vec<FinalBranch<P>>,
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
        Self::build(config, Arc::new(Clone::clone), String::new())
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
        Self::build(config, Arc::new(group_by), String::new())
    }

    fn build(
        config: LoopConfig<P>,
        group_by: Arc<GroupFunction<P, K>>,
        id_prefix: String,
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
        if config
            .subagent
            .as_ref()
            .is_some_and(|subagent| subagent.max_depth == 0)
        {
            return Err(LoopError::InvalidSubagentDepth);
        }

        let root = Branch {
            branch_id: "root".to_owned(),
            env_ids: config.env_ids.clone(),
            messages: vec![Message::user(config.prompt)],
            split_depth: 0,
            steps: 0,
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
            specs: config.tool_specs,
            subagent: config.subagent,
            group_by,
            budget: config.budget,
            id_prefix,
            seq: 0,
            next_branch: 0,
            branches: HashMap::new(),
            history: vec![root_record],
            finals: Vec::new(),
            ready_llm: VecDeque::new(),
            ready_tool: VecDeque::new(),
            outstanding: HashMap::new(),
            children: IndexMap::new(),
            child_routes: HashMap::new(),
        };
        loop_.schedule_branch(root);
        Ok(loop_)
    }

    /// Pop one runnable request. `None` is not a termination signal; use
    /// [`BatchLoop::is_done`] to distinguish blocked from finished.
    pub fn next_request(&mut self, kind: RequestKind) -> Option<Request<P>> {
        self.rollup_children();
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
    /// Validation errors leave the loop state untouched.
    pub fn complete(&mut self, completion: Completion<P>) -> Result<(), LoopError> {
        let request_id = completion.request_id().to_owned();
        if let Some(route) = self.child_routes.get(&request_id) {
            let child_key = route.child_key.clone();
            self.children
                .get_mut(&child_key)
                .expect("child route must reference a live child")
                .child
                .complete(completion)?;
            self.child_routes.remove(&request_id);
            self.rollup_children();
            return Ok(());
        }

        let request = self
            .outstanding
            .get(&request_id)
            .cloned()
            .ok_or_else(|| LoopError::UnknownRequest(request_id.clone()))?;
        self.validate_completion(&request, &completion)?;
        self.outstanding.remove(&request_id);

        match (request, completion) {
            (Request::Llm(request), Completion::Llm(completion)) => {
                self.handle_llm(request, completion)
            }
            (Request::Tool(request), Completion::Tool(completion)) => {
                self.handle_tool(request, completion)
            }
            _ => unreachable!("completion kind was validated"),
        }
        Ok(())
    }

    /// True only when no root or nested work is runnable or outstanding.
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
    pub fn result(&mut self) -> Result<BatchResult<P>, LoopError> {
        if !self.is_done() {
            return Err(LoopError::NotDone);
        }
        Ok(self.finalize())
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
        let done_keys: Vec<String> = self
            .children
            .iter_mut()
            .filter_map(|(key, link)| link.child.is_done().then(|| key.clone()))
            .collect();
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
                .insert(link.call.id, child_result.results);
            if branch.turn_results.len() == branch.turn_calls.len() {
                self.advance_turn(branch);
            } else {
                self.branches.insert(branch.branch_id.clone(), branch);
            }
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

    fn schedule_branch(&mut self, branch: Branch<P>) {
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
            branch_id: branch.branch_id,
            env_ids: branch.env_ids,
            content,
        });
    }

    fn handle_llm(&mut self, request: LlmRequest<P>, completion: LlmCompletion<P>) {
        let mut branch = self
            .branches
            .remove(&request.branch_id)
            .expect("LLM request must reference an active branch");
        branch.steps += 1;
        branch.messages.push(Message::assistant(
            completion.content.clone(),
            completion.tool_calls.clone(),
        ));
        if completion.tool_calls.is_empty() {
            self.finish_branch(branch, completion.content);
            return;
        }

        branch.turn_calls = completion.tool_calls;
        branch.turn_results.clear();
        for call in branch.turn_calls.clone() {
            if self
                .subagent
                .as_ref()
                .is_some_and(|subagent| call.name == subagent.spec.name)
            {
                self.spawn_child(&branch, call);
            } else {
                self.request_tool(&branch, call);
            }
        }
        self.branches.insert(branch.branch_id.clone(), branch);
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

    fn spawn_child(&mut self, branch: &Branch<P>, call: ToolCall) {
        let subagent = self
            .subagent
            .as_ref()
            .expect("subagent dispatch requires a configured subagent");
        let child_subagent =
            (subagent.max_depth > 1).then(|| subagent.with_max_depth(subagent.max_depth - 1));
        let config = LoopConfig {
            prompt: (subagent.build_prompt)(&call.args),
            env_ids: branch.env_ids.clone(),
            tool_specs: (subagent.sub_tool_specs)(&branch.env_ids[0]),
            subagent: child_subagent,
            budget: self.budget.clone(),
        };
        let child = Self::build(
            config,
            Arc::clone(&self.group_by),
            format!("{}sub:{}/{}/", self.id_prefix, branch.branch_id, call.id),
        )
        .expect("child config is derived from a valid parent config");
        self.children.insert(
            format!("{}:{}", branch.branch_id, call.id),
            ChildLink {
                child: Box::new(child),
                branch_id: branch.branch_id.clone(),
                call,
            },
        );
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

        if groups.len() == 1 {
            let representative = branch.env_ids[0].clone();
            Self::append_tool_messages(
                &mut branch.messages,
                &branch.turn_calls,
                &branch.turn_results,
                &representative,
            );
            branch.turn_calls.clear();
            branch.turn_results.clear();
            self.schedule_branch(branch);
            return;
        }

        for (key, env_ids) in groups {
            self.next_branch += 1;
            let branch_id = format!("{}/b{}", branch.branch_id, self.next_branch);
            let mut messages = branch.messages.clone();
            Self::append_tool_messages(
                &mut messages,
                &branch.turn_calls,
                &branch.turn_results,
                &env_ids[0],
            );
            let child = Branch {
                branch_id: branch_id.clone(),
                env_ids: env_ids.clone(),
                messages,
                split_depth: branch.split_depth + 1,
                steps: branch.steps,
                turn_calls: Vec::new(),
                turn_results: HashMap::new(),
            };
            self.history.push(BranchRecord {
                branch_id,
                parent_id: Some(branch.branch_id.clone()),
                group_key: Some(format!("{key:?}")),
                env_ids,
            });
            self.schedule_branch(child);
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
        for record in self.history.iter().rev() {
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
            tree: root.expect("history always contains a root"),
            classes,
            results,
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
    run_to_completion_by(prompt, env_ids, tools, complete_llm, Clone::clone, budget)
}

/// Synchronous convenience driver with a caller-provided grouping key.
pub fn run_to_completion_by<P, K, TF, LF, G>(
    prompt: P,
    env_ids: Vec<String>,
    tools: TF,
    mut complete_llm: LF,
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
    if env_ids.is_empty() {
        return Err(LoopError::EmptyEnvIds);
    }
    let first_entries = tools(&env_ids[0]);
    let mut specs = Vec::new();
    let mut subagent = None;
    for entry in first_entries {
        match entry {
            ToolEntry::Callable(tool) => specs.push(tool.spec()),
            ToolEntry::Subagent(candidate) if subagent.is_none() => {
                specs.push(candidate.spec.clone());
                subagent = Some(candidate);
            }
            ToolEntry::Subagent(_) => {}
        }
    }

    let mut config = LoopConfig::new(prompt, env_ids, specs);
    config.subagent = subagent;
    config.budget = budget;
    let mut loop_ = BatchLoop::with_group_by(config, group_by)?;

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
                    let entries = tools(env);
                    let tool = entries.into_iter().find_map(|entry| match entry {
                        ToolEntry::Callable(tool) if tool.spec().name == request.tool_call.name => {
                            Some(tool)
                        }
                        _ => None,
                    });
                    let tool = tool.ok_or_else(|| LoopError::MissingTool {
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
