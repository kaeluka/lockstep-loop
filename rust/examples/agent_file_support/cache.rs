//! Example-owned JSON snapshot bundles; not a stable core wire format.
use anyhow::{Context, Result, ensure};
use lockstep_loop::{
    Agent, AgentConfig, AgentDescriptor, AgentSnapshot, HistoryNode, NodeRef, RecordStore,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    model: String,
    repository: PathBuf,
    contract: Value,
    descriptor: AgentDescriptor<Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    format_version: u32,
    entries: Vec<Entry>,
    records: Vec<(NodeRef, Arc<HistoryNode<Value>>)>,
}

pub struct Cache {
    path: PathBuf,
    model: String,
    repository: PathBuf,
    entries: Vec<Entry>,
    store: RecordStore<Value>,
}

impl Cache {
    pub fn path_for(agent: &Path) -> Result<PathBuf> {
        Ok(agent
            .canonicalize()
            .context("resolve cache's agent definition")?
            .with_extension("state.json"))
    }

    pub fn load(agent: &Path, repository: &Path, model: &str) -> Result<Self> {
        let path = Self::path_for(agent)?;
        let bundle = match File::open(&path) {
            Ok(file) => {
                let bundle: Bundle =
                    serde_json::from_reader(BufReader::new(file)).with_context(|| {
                        format!(
                            "decode cache {}; move it aside to start fresh",
                            path.display()
                        )
                    })?;
                ensure!(
                    bundle.format_version == FORMAT_VERSION,
                    "unsupported cache format in {}",
                    path.display()
                );
                bundle
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Bundle {
                format_version: FORMAT_VERSION,
                entries: Vec::new(),
                records: Vec::new(),
            },
            Err(error) => {
                return Err(error).with_context(|| format!("open cache {}", path.display()));
            }
        };
        let mut records = BTreeMap::new();
        for (reference, node) in bundle.records {
            ensure!(
                records.insert(reference, node).is_none(),
                "duplicate cache record reference"
            );
        }
        validate_graph(&bundle.entries, &records)?;
        // Process-local RecordStore defaults are not suitable for durable writers.
        let namespace = loop {
            let candidate = uuid::Uuid::new_v4().to_string();
            if records
                .keys()
                .all(|reference| reference.namespace != candidate)
            {
                break candidate;
            }
        };
        let store = RecordStore::with_loader(namespace, move |reference| {
            records
                .get(reference)
                .expect("cache references validated before restoration")
                .clone()
        });
        Ok(Self {
            path,
            model: model.to_owned(),
            repository: repository
                .canonicalize()
                .context("resolve cached repository")?,
            entries: bundle.entries,
            store,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn matches(&self, entry: &Entry, contract: &Value, snapshot: &AgentSnapshot<Value>) -> bool {
        entry.model == self.model
            && entry.repository == self.repository
            && entry.contract == *contract
            && entry.descriptor.initial_messages == snapshot.descriptor.initial_messages
            && entry.descriptor.tools == snapshot.descriptor.tools
            && entry.descriptor.subagents == snapshot.descriptor.subagents
    }

    pub fn agent(&self, config: AgentConfig<Value>, contract: &Value) -> Result<Agent<Value>> {
        let runtime = config.runtime.clone();
        let fresh = Agent::new(config.with_record_store(self.store.clone()))?;
        let expected = fresh.snapshot();
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| self.matches(entry, contract, &expected))
        {
            eprintln!("reusing history from {}", self.path.display());
            return Ok(Agent::from_snapshot(
                AgentSnapshot {
                    descriptor: entry.descriptor.clone(),
                    store: self.store.clone(),
                },
                runtime,
            )?);
        }
        eprintln!(
            "no matching history in {}; starting fresh",
            self.path.display()
        );
        Ok(fresh)
    }

    pub fn publish(&mut self, snapshot: &AgentSnapshot<Value>, contract: Value) -> Result<()> {
        let mut entries = self.entries.clone();
        let entry = Entry {
            model: self.model.clone(),
            repository: self.repository.clone(),
            contract: contract.clone(),
            descriptor: snapshot.descriptor.clone(),
        };
        if let Some(index) = entries
            .iter()
            .position(|old| self.matches(old, &contract, snapshot))
        {
            entries[index] = entry;
        } else {
            entries.push(entry);
        }

        // Export reachable records, including child invocation bases/results, not
        // the writer's allocation log. Keep roots for other inputs/configurations.
        let mut records = BTreeMap::new();
        for entry in &entries {
            let root = AgentSnapshot {
                descriptor: entry.descriptor.clone(),
                store: self.store.clone(),
            };
            let mut pending = vec![root.descriptor.root.clone()];
            while let Some(reference) = pending.pop() {
                if records.contains_key(&reference) {
                    continue;
                }
                let node = root.load(&reference);
                pending.extend(references(&node).into_iter().cloned());
                records.insert(reference, node);
            }
        }
        let bundle = Bundle {
            format_version: FORMAT_VERSION,
            entries,
            records: records.into_iter().collect(),
        };
        let parent = self
            .path
            .parent()
            .context("cache has no parent directory")?;
        let mut temporary = tempfile::Builder::new()
            .prefix(".lockstep-cache-")
            .tempfile_in(parent)
            .context("create temporary cache beside agent definition")?;
        {
            let mut writer = BufWriter::new(temporary.as_file_mut());
            serde_json::to_writer(&mut writer, &bundle).context("encode history cache")?;
            writer.flush()?;
        }
        temporary.as_file().sync_all()?;
        temporary
            .persist(&self.path)
            .context("atomically publish history cache")?;
        #[cfg(unix)]
        File::open(parent)?.sync_all()?;
        self.entries = bundle.entries;
        eprintln!("saved history to {}", self.path.display());
        Ok(())
    }
}

fn references(node: &HistoryNode<Value>) -> Vec<&NodeRef> {
    node.observations
        .iter()
        .map(|edge| &edge.continuation)
        .chain(
            node.user_continuations
                .iter()
                .map(|edge| &edge.continuation),
        )
        .chain(node.subagents.iter().flat_map(|child| {
            std::iter::once(&child.snapshot.root).chain(child.runs.iter().flat_map(|run| {
                run.base
                    .iter()
                    .map(|base| &base.root)
                    .chain(std::iter::once(&run.snapshot.root))
            }))
        }))
        .collect()
}

fn validate_graph(
    entries: &[Entry],
    records: &BTreeMap<NodeRef, Arc<HistoryNode<Value>>>,
) -> Result<()> {
    // Check missing references and cycles before entering the core's infallible
    // loader. This is not a validator for arbitrary untrusted history semantics.
    for entry in entries {
        ensure!(
            records.contains_key(&entry.descriptor.root),
            "missing cache root"
        );
    }
    let mut complete = HashSet::new();
    let mut active = HashSet::new();
    for root in records.keys() {
        let mut stack = vec![(root, false)];
        while let Some((reference, exiting)) = stack.pop() {
            if exiting {
                active.remove(reference);
                complete.insert(reference);
                continue;
            }
            if complete.contains(reference) {
                continue;
            }
            ensure!(active.insert(reference), "cycle in cache records");
            let node = records
                .get(reference)
                .context("missing referenced cache record")?;
            stack.push((reference, true));
            for next in references(node).into_iter().rev() {
                stack.push((next, false));
            }
        }
    }
    Ok(())
}
