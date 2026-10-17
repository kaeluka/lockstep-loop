//! Immutable shallow records, lazy loading, and caller-owned persistence.
//! Payload clones must have immutable value semantics. No core storage I/O.

use crate::{Message, ToolSpec};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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

/// Represented but never implicitly selected or inherited by replay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserContinuation<P> {
    pub message: Message<P>,
    pub continuation: NodeRef,
    /// Resolution provenance only, NOT an inheritance/selection predicate.
    pub resolved_env_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentContract {
    pub spec: ToolSpec,
    pub max_depth: usize,
}

/// Small persistence envelope: publish only AFTER its referenced records.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryDescriptor<P> {
    pub contract: String,
    pub prompt: P,
    pub tools: Vec<ToolSpec>,
    pub custom_grouping: bool,
    pub root: NodeRef,
    pub subagent: Option<SubagentContract>,
    pub requires_user_policy: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubagentHistory<P> {
    pub call_id: String,
    pub history: HistoryDescriptor<P>,
}

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

/// One immutable snapshot plus its resolver. Serialize descriptor and records,
/// not the resolver. No recursively nested persistence format is needed.
#[derive(Clone, Debug)]
pub struct History<P> {
    pub descriptor: HistoryDescriptor<P>,
    pub store: RecordStore<P>,
}
impl<P: PartialEq> PartialEq for History<P> {
    fn eq(&self, other: &Self) -> bool {
        self.descriptor == other.descriptor
    }
}
impl<P> History<P> {
    pub fn root(&self) -> Arc<HistoryNode<P>> {
        self.load(&self.descriptor.root)
    }
    pub fn load(&self, reference: &NodeRef) -> Arc<HistoryNode<P>> {
        self.store.load(reference)
    }
    pub(crate) fn node(&self, path: &[usize]) -> Arc<HistoryNode<P>> {
        let mut node = self.root();
        for index in path {
            node = self.load(&node.observations[*index].continuation);
        }
        node
    }
}
impl<P: Clone> History<P> {
    pub(crate) fn replace_node(&mut self, path: &[usize], node: HistoryNode<P>) {
        let mut ancestors = Vec::new();
        let mut parent = self.root();
        for index in path {
            let next = self.load(&parent.observations[*index].continuation);
            ancestors.push((parent, *index));
            parent = next;
        }
        let mut reference = self.store.put(node);
        for (ancestor, index) in ancestors.into_iter().rev() {
            let mut node = ancestor.as_ref().clone();
            node.observations[index].continuation = reference;
            reference = self.store.put(node);
        }
        self.descriptor.root = reference;
    }
}

#[derive(Clone, Debug)]
pub struct HistoryOptions<P> {
    pub contract: String,
    pub previous: Option<History<P>>,
    pub store: Option<RecordStore<P>>,
    pub(crate) prompt_eq: Option<fn(&P, &P) -> bool>,
}
