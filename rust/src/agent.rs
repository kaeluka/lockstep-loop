//! Mutable handles over immutable snapshot values. No provider or persistence I/O.
use crate::{
    AgentDescriptor, AgentResolver, AgentSnapshot, BatchLoop, BatchResult, Budget,
    ComparisonKeyFunction, Completion, HistoryNode, LoopConfig, LoopError, RecordStore, Request,
    RequestKind, SubagentTool, ToolSpec, register,
};
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// Caller-owned executable dependencies, never serialized or compared by identity.
#[derive(Clone, Debug)]
pub struct AgentRuntime<P> {
    pub subagents: Vec<SubagentTool<P>>,
    pub agent_resolver: Option<AgentResolver<P>>,
}
impl<P> Default for AgentRuntime<P> {
    fn default() -> Self {
        Self {
            subagents: vec![],
            agent_resolver: None,
        }
    }
}

/// Fresh-agent configuration; budgets and feedback belong to individual runs.
#[derive(Clone, Debug)]
pub struct AgentConfig<P> {
    pub prompt: P,
    pub tool_specs: Vec<ToolSpec>,
    pub runtime: AgentRuntime<P>,
    pub record_store: RecordStore<P>,
}
impl<P> AgentConfig<P> {
    pub fn new(prompt: P, tool_specs: Vec<ToolSpec>) -> Self {
        Self {
            prompt,
            tool_specs,
            runtime: AgentRuntime::default(),
            record_store: RecordStore::default(),
        }
    }
    pub fn with_subagents(mut self, subagents: Vec<SubagentTool<P>>) -> Self {
        self.runtime.subagents = subagents;
        self
    }
    pub fn with_agent_resolver(mut self, resolver: AgentResolver<P>) -> Self {
        self.runtime.agent_resolver = Some(resolver);
        self
    }
    pub fn with_record_store(mut self, store: RecordStore<P>) -> Self {
        self.record_store = store;
        self
    }
}

/// Per-run intervention and accepted-assistant-turn limit.
#[derive(Clone, Debug)]
pub struct RunOptions<P> {
    /// Some(payload) is an intervention, even Some(None) for optional payloads.
    pub feedback: Option<P>,
    pub budget: Option<Budget<P>>,
}
impl<P> Default for RunOptions<P> {
    fn default() -> Self {
        Self {
            feedback: None,
            budget: None,
        }
    }
}

/// Mutable owner of a current immutable snapshot and runtime dependencies.
///
/// Multiple live runs are allowed, without exclusive borrows of this handle.
/// This handle deliberately does not implement Clone. To fork an independent
/// head, use `Agent::from_snapshot(agent.snapshot(), runtime)` (or the custom-key
/// variant). Records may be shared; future publications affect only their owner.
/// Payload clones must have immutable value semantics.
pub struct Agent<P, K = P> {
    head: Arc<Mutex<AgentSnapshot<P>>>,
    runtime: AgentRuntime<P>,
    comparison_key: Arc<ComparisonKeyFunction<P, K>>,
}

impl<P> Agent<P, P>
where
    P: Clone + Eq + Hash + Debug + 'static,
{
    pub fn new(config: AgentConfig<P>) -> Result<Self, LoopError> {
        Self::with_comparison_key(config, Clone::clone)
    }
    /// Restore without repeating prompt/specs or loading any history records.
    pub fn from_snapshot(
        snapshot: AgentSnapshot<P>,
        runtime: AgentRuntime<P>,
    ) -> Result<Self, LoopError> {
        Self::from_snapshot_by(snapshot, runtime, Clone::clone)
    }
}

impl<P, K> Agent<P, K>
where
    P: Clone + PartialEq + 'static,
    K: Eq + Hash + Debug + 'static,
{
    /// The key applies to both current outputs and stored representatives.
    pub fn with_comparison_key(
        config: AgentConfig<P>,
        comparison_key: impl Fn(&P) -> K + Send + Sync + 'static,
    ) -> Result<Self, LoopError> {
        let registry = register(config.tool_specs, config.runtime.subagents.clone())?;
        let snapshot = AgentSnapshot {
            descriptor: AgentDescriptor {
                prompt: config.prompt,
                tools: registry.specs,
                subagents: registry.contracts,
                root: config.record_store.put(HistoryNode::default()),
            },
            store: config.record_store,
        };
        Ok(Self {
            head: Arc::new(Mutex::new(snapshot)),
            runtime: config.runtime,
            comparison_key: Arc::new(comparison_key),
        })
    }

    /// Restore with a current grouping function; no callable identity/mode gate.
    /// Matching subagent factories must be supplied; callbacks are never persisted.
    /// Prompt and ordered tool specs come exclusively from the snapshot.
    pub fn from_snapshot_by(
        snapshot: AgentSnapshot<P>,
        runtime: AgentRuntime<P>,
        comparison_key: impl Fn(&P) -> K + Send + Sync + 'static,
    ) -> Result<Self, LoopError> {
        let registry = register(snapshot.descriptor.tools.clone(), runtime.subagents.clone())?;
        if registry.specs != snapshot.descriptor.tools
            || registry.contracts != snapshot.descriptor.subagents
        {
            return Err(LoopError::IncompatibleHistory);
        }
        Ok(Self {
            head: Arc::new(Mutex::new(snapshot)),
            runtime,
            comparison_key: Arc::new(comparison_key),
        })
    }

    /// Capture the current root without traversing records.
    pub fn snapshot(&self) -> AgentSnapshot<P> {
        self.head.lock().expect("agent head poisoned").clone()
    }

    /// Start from the current root. Successful completion publishes exactly once.
    ///
    /// # Warning
    /// Overlapping runs start from the agent state available when each run begins.
    /// Their histories are not merged. The last run to complete replaces the
    /// agent's current state, potentially excluding experience from other runs.
    /// Retain each run's snapshot to preserve independent results, or finish each
    /// run before starting the next to accumulate experience sequentially.
    ///
    /// Invalid completions remain correctable. Terminal failures and abandoned
    /// runs do not publish. Budget completion publishes real history, not fallback.
    pub fn start(
        &self,
        environments: Vec<String>,
        options: RunOptions<P>,
    ) -> Result<AgentRun<P, K>, LoopError> {
        let snapshot = self.snapshot();
        let mut config = LoopConfig::new(
            snapshot.descriptor.prompt.clone(),
            environments,
            snapshot.descriptor.tools.clone(),
        )
        .with_snapshot(snapshot)
        .with_subagents(self.runtime.subagents.clone());
        config.agent_resolver = self.runtime.agent_resolver.clone();
        config.budget = options.budget;
        config.feedback = options.feedback;
        let engine = BatchLoop::build(
            config,
            self.comparison_key.clone(),
            String::new(),
            false,
            HashSet::new(),
        )?;
        let mut run = AgentRun {
            owner: self.head.clone(),
            engine,
            published: false,
        };
        run.settle();
        if let Some(error) = run.error() {
            return Err(error.clone());
        }
        Ok(run)
    }
}

/// Mutable request/completion state with an independent immutable snapshot.
///
/// Successful completion automatically publishes once, including cached starts.
/// Inspection of older runs cannot republish them. Snapshots contain validated
/// facts, not pending requests or a crash recovery protocol. Request IDs are local
/// to a run; route completions to their originating run. Use your own Mutex for
/// cross-thread advancement of one run; no lock spans a run's lifetime.
pub struct AgentRun<P, K = P> {
    owner: Arc<Mutex<AgentSnapshot<P>>>,
    engine: BatchLoop<P, K>,
    published: bool,
}
impl<P, K> AgentRun<P, K>
where
    P: Clone + 'static,
    K: Eq + Hash + Debug + 'static,
{
    fn settle(&mut self) -> bool {
        let done = self.engine.is_done();
        if done && self.engine.error().is_none() && !self.published {
            *self.owner.lock().expect("agent head poisoned") = self.engine.snapshot();
            self.published = true;
        }
        done
    }
    pub fn is_done(&mut self) -> bool {
        self.settle()
    }
    pub fn next_request(&mut self, kind: RequestKind) -> Option<Request<P>> {
        let request = self.engine.next_request(kind);
        self.settle();
        request
    }
    pub fn complete(&mut self, completion: Completion<P>) -> Result<(), LoopError> {
        self.engine.complete(completion)?;
        self.settle();
        self.error().cloned().map_or(Ok(()), Err)
    }
    /// Capture validated history; never publishes or advances requests.
    pub fn snapshot(&self) -> AgentSnapshot<P> {
        self.engine.snapshot()
    }
    pub fn result(&mut self) -> Result<BatchResult<P>, LoopError> {
        self.settle();
        self.engine.result()
    }
    pub fn error(&self) -> Option<&LoopError> {
        self.engine.error()
    }
    pub fn outstanding_llm(&self) -> usize {
        self.engine.outstanding_llm()
    }
    pub fn outstanding_tool(&self) -> usize {
        self.engine.outstanding_tool()
    }
    pub fn outstanding_count(&self) -> usize {
        self.engine.outstanding_count()
    }
}
