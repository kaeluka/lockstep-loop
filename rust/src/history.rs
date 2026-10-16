//! Immutable, storage-neutral recorded conversations. No persistence I/O.
//!
//! Payloads must behave as immutable values: `Clone` must not expose mutable
//! aliases (including interior mutability). Callers own codecs and contract IDs.

use crate::{Message, ToolSpec};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    pub continuation: Arc<HistoryNode<P>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryNode<P> {
    pub assistant: Option<Message<P>>,
    pub env_ids: Vec<String>,
    pub observations: Vec<Observation<P>>,
}

impl<P> Default for HistoryNode<P> {
    fn default() -> Self {
        Self {
            assistant: None,
            env_ids: Vec::new(),
            observations: Vec::new(),
        }
    }
}

/// Clone captures a snapshot sharing all unchanged nodes through `Arc`.
/// The contract ID attests exact prompt, tool semantics and grouping policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct History<P> {
    pub contract: String,
    pub prompt: P,
    pub tools: Vec<ToolSpec>,
    pub custom_grouping: bool,
    pub root: Arc<HistoryNode<P>>,
}

#[derive(Clone, Debug)]
pub struct HistoryOptions<P> {
    pub contract: String,
    pub previous: Option<History<P>>,
    pub(crate) prompt_eq: Option<fn(&P, &P) -> bool>,
}

impl<P> History<P> {
    pub(crate) fn node(&self, path: &[usize]) -> &HistoryNode<P> {
        let mut node = self.root.as_ref();
        for index in path {
            node = &node.observations[*index].continuation;
        }
        node
    }
}

impl<P: Clone> History<P> {
    pub(crate) fn node_mut(&mut self, path: &[usize]) -> &mut HistoryNode<P> {
        let mut node = Arc::make_mut(&mut self.root);
        for index in path {
            node = Arc::make_mut(&mut node.observations[*index].continuation);
        }
        node
    }
}
