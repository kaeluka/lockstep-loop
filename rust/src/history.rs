//! Immutable Agent snapshots, contextual Asset views, and shallow history records.
//! Payload clones must have immutable value semantics. No core storage I/O.

use crate::{LoopError, Message, Role, ToolCall, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// An exact storage address, NOT a conversation/session identity or content hash.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeRef {
    pub namespace: String,
    pub index: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputRecord<P> {
    pub tool_output: P,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvObservation<P> {
    pub env_id: String,
    pub outputs: Vec<OutputRecord<P>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Observation<P> {
    pub representative: Vec<OutputRecord<P>>,
    pub evidence: Vec<EnvObservation<P>>,
    pub continuation: NodeRef,
}

/// Shared-path continuation followed by every environment reaching this endpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserContinuation<P> {
    pub message: Message<P>,
    pub continuation: NodeRef,
    /// Resolution provenance only, NOT an inheritance/selection predicate.
    pub resolved_env_id: Option<String>,
}

/// Structural subagent configuration; legacy type name, no caller version key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentContract {
    pub spec: ToolSpec,
    pub max_depth: usize,
    /// None means bounded self-recursion; Some is the explicit child allowlist.
    pub subagents: Option<Vec<SubagentContract>>,
}

/// AgentQuery: child structural configuration, without parent/invocation/environment IDs.
/// External provider/grouping policy isolation is caller-owned. The legacy struct
/// name is retained; self-describing Serde ignores old contract labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryQuery<P> {
    pub prompt: P,
    pub tools: Vec<ToolSpec>,
    pub custom_grouping: bool,
    pub subagents: Vec<SubagentContract>,
}

type Resolver<P> =
    dyn Fn(&HistoryQuery<P>) -> Result<Option<HistoryDescriptor<P>>, LoopError> + Send + Sync;

/// AgentResolver: caller-owned agent revision selection, using the RecordStore loader.
pub struct HistoryResolver<P> {
    pub(crate) resolve: Arc<Resolver<P>>,
    pub(crate) prompt_eq: fn(&P, &P) -> bool,
}
impl<P: PartialEq> HistoryResolver<P> {
    pub fn new(
        resolve: impl Fn(&HistoryQuery<P>) -> Result<Option<HistoryDescriptor<P>>, LoopError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            resolve: Arc::new(resolve),
            prompt_eq: PartialEq::eq,
        }
    }
}
impl<P> Clone for HistoryResolver<P> {
    fn clone(&self) -> Self {
        Self {
            resolve: self.resolve.clone(),
            prompt_eq: self.prompt_eq,
        }
    }
}
impl<P> fmt::Debug for HistoryResolver<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HistoryResolver { .. }")
    }
}

/// AgentDescriptor: small persistence envelope; publish AFTER its referenced records.
/// Legacy struct name retained; self-describing Serde ignores old contract labels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryDescriptor<P> {
    pub prompt: P,
    pub tools: Vec<ToolSpec>,
    pub custom_grouping: bool,
    pub root: NodeRef,
    pub subagents: Vec<SubagentContract>,
}

/// SubagentRecord: parent-call provenance, not the complete child agent itself.
/// Use Agent::child to inspect the child agent and its assets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentHistory<P> {
    pub call_id: String,
    pub history: HistoryDescriptor<P>,
    #[serde(default = "Vec::new")]
    pub runs: Vec<Arc<ChildHistoryRun<P>>>,
}

/// ChildAgentRun: one invocation's environments, selected base and resulting revision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChildHistoryRun<P> {
    pub env_ids: Vec<String>,
    pub base: Option<HistoryDescriptor<P>>,
    pub history: HistoryDescriptor<P>,
}

/// ChildAgentUpdate: one completed invocation, not a merged cache or conversation ID.
#[derive(Clone, Debug, PartialEq)]
pub struct ChildHistoryUpdate<P> {
    pub request_prefix: String,
    pub branch_id: String,
    pub depth: usize,
    pub tool_call: ToolCall,
    pub env_ids: Vec<String>,
    pub base: Option<HistoryDescriptor<P>>,
    pub history: Agent<P>,
}

/// Shared prefix or historical endpoint; an Asset also needs its agent/path context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryNode<P> {
    pub assistant: Option<Message<P>>,
    pub env_ids: Vec<String>,
    pub observations: Vec<Observation<P>>,
    pub subagents: Vec<SubagentHistory<P>>,
    pub user_continuations: Vec<UserContinuation<P>>,
}

impl<P> Default for HistoryNode<P> {
    fn default() -> Self {
        Self {
            assistant: None,
            env_ids: Vec::new(),
            observations: Vec::new(),
            subagents: Vec::new(),
            user_continuations: Vec::new(),
        }
    }
}

type Loader<P> = dyn Fn(&NodeRef) -> Arc<HistoryNode<P>> + Send + Sync;
struct Records<P> {
    writes: BTreeMap<NodeRef, Arc<HistoryNode<P>>>,
    cache: BTreeMap<NodeRef, Arc<HistoryNode<P>>>,
}

/// Append-only writes plus an optional lazy caller loader; no backend or GC.
/// Allocate a fresh namespace for EVERY durable writer. The loader must resolve
/// valid immutable records and must not re-enter the store. Loader panics/missing
/// records are fatal infrastructure failures, not recoverable completion errors.
pub struct RecordStore<P> {
    namespace: String,
    records: Arc<Mutex<Records<P>>>,
    loader: Option<Arc<Loader<P>>>,
}
impl<P> Clone for RecordStore<P> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            records: self.records.clone(),
            loader: self.loader.clone(),
        }
    }
}
impl<P> fmt::Debug for RecordStore<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordStore")
            .field("namespace", &self.namespace)
            .finish_non_exhaustive()
    }
}
impl<P> Default for RecordStore<P> {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        // Convenience for process-local use. Durable callers supply their own
        // globally unique namespace via new/with_loader, not this local counter.
        Self::new(format!(
            "memory-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
}
impl<P> RecordStore<P> {
    pub fn new(namespace: impl Into<String>) -> Self {
        let namespace = namespace.into();
        assert!(!namespace.is_empty(), "record namespace must be non-empty");
        Self {
            namespace,
            records: Arc::new(Mutex::new(Records {
                writes: BTreeMap::new(),
                cache: BTreeMap::new(),
            })),
            loader: None,
        }
    }
    pub fn with_loader(
        namespace: impl Into<String>,
        loader: impl Fn(&NodeRef) -> Arc<HistoryNode<P>> + Send + Sync + 'static,
    ) -> Self {
        let mut store = Self::new(namespace);
        store.loader = Some(Arc::new(loader));
        store
    }
    pub(crate) fn same_store(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.records, &other.records)
    }
    pub fn put(&self, node: HistoryNode<P>) -> NodeRef {
        let mut records = self.records.lock().expect("record store poisoned");
        let reference = NodeRef {
            namespace: self.namespace.clone(),
            index: records.writes.len() as u64,
        };
        assert!(
            !records.cache.contains_key(&reference),
            "writer namespace collides with loaded records"
        );
        records.writes.insert(reference.clone(), Arc::new(node));
        reference
    }
    pub fn load(&self, reference: &NodeRef) -> Arc<HistoryNode<P>> {
        let mut records = self.records.lock().expect("record store poisoned");
        if let Some(node) = records
            .writes
            .get(reference)
            .or_else(|| records.cache.get(reference))
        {
            return node.clone();
        }
        assert_ne!(
            reference.namespace, self.namespace,
            "writer namespace reused for existing or missing records"
        );
        let node = self
            .loader
            .as_ref()
            .expect("no loader for referenced record")(reference);
        records.cache.insert(reference.clone(), node.clone());
        node
    }
    /// Local writes only. Persist these before publishing any new descriptor.
    pub fn records(&self) -> BTreeMap<NodeRef, Arc<HistoryNode<P>>> {
        self.records
            .lock()
            .expect("record store poisoned")
            .writes
            .clone()
    }
}

/// Complete immutable agent: prompt, structural configuration and accumulated trie.
/// This is not a tool definition or a running BatchLoop. Persist its descriptor
/// and records, not a live resolver or asset view. History is a compatibility alias.
#[derive(Clone, Debug)]
pub struct Agent<P> {
    pub descriptor: AgentDescriptor<P>,
    pub store: RecordStore<P>,
}
impl<P: PartialEq> PartialEq for Agent<P> {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}
#[derive(Clone, Copy)]
pub(crate) enum PathStep {
    Observation(usize),
    User(usize),
}

impl<P> Agent<P> {
    pub fn prompt(&self) -> &P {
        &self.descriptor.prompt
    }
    pub fn root(&self) -> Arc<HistoryNode<P>> {
        self.load(&self.descriptor.root)
    }
    pub fn load(&self, reference: &NodeRef) -> Arc<HistoryNode<P>> {
        self.store.load(reference)
    }
    /// Lazy leaf inspection. Child agents are separate tries; use child(entry)
    /// to inspect them. This never executes tools or creates records.
    pub fn assets(&self) -> Assets<'_, P> {
        Assets {
            agent: self,
            stack: vec![(self.descriptor.root.clone(), 0, None)],
            path: vec![],
        }
    }
    pub(crate) fn node(&self, path: &[PathStep]) -> Arc<HistoryNode<P>> {
        let mut node = self.root();
        for step in path {
            let reference = match step {
                PathStep::Observation(i) => &node.observations[*i].continuation,
                PathStep::User(i) => &node.user_continuations[*i].continuation,
            };
            node = self.load(reference);
        }
        node
    }
}
impl<P: Clone> HistoryDescriptor<P> {
    pub fn query(&self) -> AgentQuery<P> {
        AgentQuery {
            prompt: self.prompt.clone(),
            tools: self.tools.clone(),
            custom_grouping: self.custom_grouping,
            subagents: self.subagents.clone(),
        }
    }
}

impl<P: Clone> Agent<P> {
    pub fn child(&self, entry: &SubagentRecord<P>) -> Agent<P> {
        Agent {
            descriptor: entry.agent_descriptor().clone(),
            store: self.store.clone(),
        }
    }
    pub(crate) fn replace_node(&mut self, path: &[PathStep], node: HistoryNode<P>) {
        let mut ancestors = Vec::new();
        let mut parent = self.root();
        for step in path {
            let reference = match step {
                PathStep::Observation(i) => &parent.observations[*i].continuation,
                PathStep::User(i) => &parent.user_continuations[*i].continuation,
            };
            let next = self.load(reference);
            ancestors.push((parent, *step));
            parent = next;
        }
        let mut reference = self.store.put(node);
        for (ancestor, step) in ancestors.into_iter().rev() {
            let mut node = ancestor.as_ref().clone();
            match step {
                PathStep::Observation(i) => node.observations[i].continuation = reference,
                PathStep::User(i) => node.user_continuations[i].continuation = reference,
            }
            reference = self.store.put(node);
        }
        self.descriptor.root = reference;
    }
}

/// AgentOptions: recording/replay configuration, not a second agent snapshot model.
#[derive(Clone, Debug)]
pub struct HistoryOptions<P> {
    pub previous: Option<Agent<P>>,
    pub store: Option<RecordStore<P>>,
    pub(crate) prompt_eq: Option<fn(&P, &P) -> bool>,
}

impl<P> SubagentHistory<P> {
    pub fn agent_descriptor(&self) -> &AgentDescriptor<P> {
        &self.history
    }
}
impl<P> ChildHistoryRun<P> {
    pub fn agent_descriptor(&self) -> &AgentDescriptor<P> {
        &self.history
    }
}
impl<P> ChildHistoryUpdate<P> {
    pub fn agent(&self) -> &Agent<P> {
        &self.history
    }
    pub fn into_agent(self) -> Agent<P> {
        self.history
    }
}

struct AssetStep<P> {
    node: Arc<HistoryNode<P>>,
    via: Option<PathStep>,
}
impl<P> Clone for AssetStep<P> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            via: self.via,
        }
    }
}

/// Read-only leaf investigation, in its agent/path context. No IDs, serialization
/// or resume operation: feedback still resolves agent snapshot + environment.
/// Asset equality is not defined by the published value.
pub struct Asset<'a, P> {
    agent: &'a Agent<P>,
    path: Vec<AssetStep<P>>,
}

/// Lazy depth-first inspection of one agent's leaves. Paths copy only record
/// references once per yielded view; no payload cloning or eager subtree loading.
pub struct Assets<'a, P> {
    agent: &'a Agent<P>,
    stack: Vec<(NodeRef, usize, Option<PathStep>)>,
    path: Vec<AssetStep<P>>,
}
impl<'a, P> Iterator for Assets<'a, P> {
    type Item = Asset<'a, P>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((reference, depth, via)) = self.stack.pop() {
            self.path.truncate(depth);
            let node = self.agent.load(&reference);
            self.path.push(AssetStep {
                node: node.clone(),
                via,
            });
            if node.observations.is_empty() && node.user_continuations.is_empty() {
                return Some(Asset {
                    agent: self.agent,
                    path: self.path.clone(),
                });
            }
            for (i, edge) in node.user_continuations.iter().enumerate().rev() {
                self.stack.push((
                    edge.continuation.clone(),
                    depth + 1,
                    Some(PathStep::User(i)),
                ));
            }
            for (i, edge) in node.observations.iter().enumerate().rev() {
                self.stack.push((
                    edge.continuation.clone(),
                    depth + 1,
                    Some(PathStep::Observation(i)),
                ));
            }
        }
        None
    }
}
impl<P> Asset<'_, P> {
    pub fn agent(&self) -> &Agent<P> {
        self.agent
    }
    pub fn node(&self) -> &HistoryNode<P> {
        &self.path.last().expect("asset has a path").node
    }
    /// Shared prefix through leaf, including child-agent provenance records.
    pub fn nodes(&self) -> Vec<&HistoryNode<P>> {
        self.path.iter().map(|step| step.node.as_ref()).collect()
    }
    /// Selected checkpoints, with both representative and actual per-env evidence.
    pub fn observations(&self) -> Vec<&Observation<P>> {
        self.path
            .iter()
            .enumerate()
            .filter_map(|(i, step)| match step.via {
                Some(PathStep::Observation(edge)) => {
                    Some(&self.path[i - 1].node.observations[edge])
                }
                _ => None,
            })
            .collect()
    }
    pub fn is_complete(&self) -> bool {
        self.node()
            .assistant
            .as_ref()
            .is_some_and(|m| m.role == Role::Assistant && m.tool_calls.is_empty())
    }
    /// Latest terminal assistant reply, including an ancestor before open feedback work.
    pub fn published_message(&self) -> Option<&Message<P>> {
        self.path
            .iter()
            .rev()
            .filter_map(|step| step.node.assistant.as_ref())
            .find(|m| m.role == Role::Assistant && m.tool_calls.is_empty())
    }
    pub fn has_value(&self) -> bool {
        self.published_message().is_some()
    }
    /// None means no publication yet; Some(&None) is a real null optional payload.
    pub fn value(&self) -> Option<&P> {
        self.published_message().map(|m| &m.content)
    }
    /// Known path memberships, not a global/latest environment partition. The
    /// latest observation bounds membership; later node visits add evidence.
    /// Feedback provenance is never an environment gate. Repeated exact evidence
    /// remains available in observations/nodes; partial in-flight tuples are not recorded.
    pub fn env_ids(&self) -> Vec<String> {
        let mut visits = Vec::new();
        let mut evidence = Vec::new();
        for (i, step) in self.path.iter().enumerate().rev() {
            visits.push(&step.node.env_ids);
            if let Some(PathStep::Observation(edge)) = step.via {
                evidence.extend(
                    self.path[i - 1].node.observations[edge]
                        .evidence
                        .iter()
                        .map(|e| e.env_id.as_str()),
                );
                break;
            }
        }
        let mut seen = HashSet::new();
        evidence
            .into_iter()
            .chain(
                visits
                    .into_iter()
                    .rev()
                    .flat_map(|v| v.iter().map(String::as_str)),
            )
            .filter(|env| seen.insert(*env))
            .map(str::to_owned)
            .collect()
    }
}
impl<P: Clone> Asset<'_, P> {
    /// The model-visible conversation, using historical representatives.
    pub fn messages(&self) -> Vec<Message<P>> {
        let mut messages = vec![Message::user(self.agent.descriptor.prompt.clone())];
        for (i, step) in self.path.iter().enumerate() {
            match step.via {
                Some(PathStep::Observation(edge)) => {
                    let parent = &self.path[i - 1].node;
                    let calls = &parent
                        .assistant
                        .as_ref()
                        .expect("observation follows assistant")
                        .tool_calls;
                    let outputs = &parent.observations[edge].representative;
                    assert_eq!(calls.len(), outputs.len(), "malformed observation tuple");
                    for (call, output) in calls.iter().zip(outputs) {
                        messages.push(Message::tool(output.tool_output.clone(), call.id.clone()));
                    }
                }
                Some(PathStep::User(edge)) => messages.push(
                    self.path[i - 1].node.user_continuations[edge]
                        .message
                        .clone(),
                ),
                None => {}
            }
            if let Some(message) = &step.node.assistant {
                messages.push(message.clone());
            }
        }
        messages
    }
}

// One model. Legacy type names retained; manual contract fields removed.
// These aliases are the preferred domain vocabulary, not duplicate wrappers.
pub use Agent as History;
pub use ChildHistoryRun as ChildAgentRun;
pub use ChildHistoryUpdate as ChildAgentUpdate;
pub use HistoryDescriptor as AgentDescriptor;
pub use HistoryOptions as AgentOptions;
pub use HistoryQuery as AgentQuery;
pub use HistoryResolver as AgentResolver;
pub use SubagentHistory as SubagentRecord;
