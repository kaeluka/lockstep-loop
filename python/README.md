# lockstep-loop for Python

Share an investigation across similar, read-only environments, branching only
when observations differ. An environment can be a document bundle, a product
specification, a dataset release, or a repository snapshot—not necessarily a
filesystem. [Start with the visual guide](../docs/concepts.md) or explore
[use cases and tool designs](../docs/use-cases.md).

`Agent` is the complete immutable prompt + structural configuration + accumulated trie
(`History` remains an alias). `Asset` is a read-only leaf investigation backed by
its full conversation, not just an answer or an environment class. See the
[glossary and compatibility mapping](../docs/agent-assets.md).

## How it works

- All environments start in one equivalence class sharing **one** LLM
  conversation.
- When the model makes a tool call, the call is evaluated on **every**
  environment still in the class (SIMD-style fan-out).
- Results are compared by value equality (or by keys computed with a
  caller-provided `comparison_key`) and the class is refined — one
  sub-conversation per distinct result. Branches then advance
  independently as their completions arrive.
- The process repeats per branch until done, yielding a prefix-sharing trie
  whose leaves are **equivalence classes of environments**. LLM work follows
  distinct observation paths; tools still run per environment. Equal final
  answers do not merge previously split classes.

Two invariants make this sound: tools must be **pure** (results depend only
on environment + arguments), and the model must never see which environment
it is in (tool specs identical across environments).

## Status

Working prototype. The core is a provider-agnostic orchestrator: `BatchLoop`
owns partition refinement and the trie; the caller owns all LLM and tool
execution. Pull work with `next_request(kind)` — `"llm"`, `"tool"`, or
`"any"` — serve it however you like (any order, any concurrency), and feed
results back with `complete()` as they arrive, so a slow provider or tool
delays only its own branch. Both request kinds carry their env set (the
branch's equivalence class), and requests carry a `stats` block (depth,
split depth, sequence) with facts the caller can compose into a serving
policy, e.g. "deepest currently available branch first" for prompt-cache
warmth. All methods are internally synchronized, so separate LLM and tool
consumer loops can share one `BatchLoop`. A generator adapter (`run_batch`)
remains for pump-style drivers. Payloads are opaque: the loop is generic in
the content type, so multimodal conversations (content blocks, image bytes)
work unchanged — tool results only need to be hashable for value-equality
grouping.

For the common serial case there is a convenience driver. This integration sketch
assumes your `my_provider` adapter and `make_tools` factory; it is not standalone.
Convert provider tool calls to `lockstep_loop.ToolCall` values before completion:

```python
from lockstep_loop import LlmCompletion, run_to_completion

def complete_llm(req):  # the only provider seam
    resp = my_provider.chat(messages=req.messages, tools=req.tools)
    return LlmCompletion(request_id=req.request_id, content=resp.text,
                         tool_calls=resp.tool_calls)

result = run_to_completion(
    prompt="Assess whether the supplier's submitted evidence is complete.",
    env_ids=["packet-a/v1", "packet-b/v1", "packet-c/v2"],
    tools=make_tools,          # env_id -> tool impls (specs must be identical)
    complete_llm=complete_llm,
)
# result.tree, result.classes, result.results (env_id -> final message)
```

`make_tools(env_id)` binds reads to that frozen evidence packet. The tool specs
are the same for every packet; routing IDs stay outside model-visible content.
The library supplies neither document extraction nor compliance judgments.

For a complete, provider-free product-ratings example:

```sh
PYTHONPATH=src python examples/product_screening.py
```

[Source](examples/product_screening.py): three products share one initial LLM turn,
ratings split them into two classes, and a matching new product replays with zero
new LLM calls. The scripted replies demonstrate orchestration, not model quality.

For parallel serving, drive `BatchLoop` yourself (scheduling sketch):

```python
loop = BatchLoop(prompt=..., env_ids=..., tool_specs=[...])
while not loop.done:
    if (req := loop.next_request("any")) is None:
        await any_completion_arrives()  # outstanding work, nothing runnable
    else:
        spawn_somewhere(req)
# elsewhere, as completions arrive: loop.complete(completion)
result = loop.result()
```

## Comparison keys

`comparison_key` computes a hashable key from each raw tool output. It defaults
to identity: exact value equality. The same function is applied to **both fresh
outputs and stored representative outputs**, for partition grouping and replay
matching. For example, `comparison_key=lambda output: output.strip()` deliberately
ignores surrounding whitespace when comparing, not when presenting evidence.
Keys never rewrite model-visible payloads: recorded conversations reuse the original
representative, and full raw per-environment outputs remain in the evidence.
Callables/lambdas are never compared by identity. The current function determines
matches, including when its semantics differ from the recording's function.

**Source-level API break:** the `group_by=` keyword is now `comparison_key=` on
`BatchLoop`, `run_batch`, and `run_to_completion`; the old keyword is not an alias.
This rename adds no persisted fields. See the migration notes below for the
separate removal of manual contract labels and `custom_grouping` metadata.

## Recording and replay

Record an agent on the direct loop API; existing constructors without these
options retain their behavior. Legacy `history` spellings remain aliases:

```python
loop = BatchLoop(
    prompt="Assess the evidence packet.", env_ids=["packet-a/v1"], tool_specs=specs,
    recording=True,
)
# Drive the ordinary requests/completions as above.
snapshot = loop.agent             # immutable agent, including fulfilled turns
# After completion: snapshot = loop.result().agent
next_loop = BatchLoop(
    prompt="Assess the evidence packet.", env_ids=["packet-a/v2"], tool_specs=specs,
    agent=snapshot,  # implies replay + recording
)
```

`next_loop` may immediately emit tools (or already be done for a recorded final
answer). Always use `done`, not the assumption that the first request is an LLM
request. `stats.depth` counts reused assistant turns too; `split_depth` describes
only the new execution's partition. Independent calls can finish in any order.

Run the complete, executable [caller-side example](examples/history_replay.py):

```sh
PYTHONPATH=src python examples/history_replay.py
# unchanged: 0 LLM calls; changed: 1; both paths reusable
```

It records a non-splitting turn, persists **individual shallow records** plus a
`AgentDescriptor`, lazily restores, and branches on changed observations. It
publishes records before descriptors, checks insert-only collisions, and reuses
both new and old published roots. Pickle is only for **trusted** data. Substitute
your own codecs/backend; do not pickle a live `Agent`/resolver or an `Asset` view.

```python
# Save each record with your own codec, then publish the descriptor atomically:
for reference, record in snapshot.store.records().items():
    backend.insert_immutable(reference, encode_node(record))
published = encode_descriptor(snapshot.descriptor)

# Fresh writer namespace (UUID default); no traversal or matching in this callback:
store = RecordStore(load=lambda ref: decode_node(backend.read(ref)))
restored = Agent(decode_descriptor(published), store)
```

`RecordStore(namespace=..., load=...)` is an in-memory append-only write set/cache,
not a backend. `BatchLoop(record_store=store, recording=True)` uses a chosen
store for a new recording; replay uses the store attached to `Agent`. Allocate
a globally unique namespace per persistent writer. New records use exact
`NodeRef(namespace, index)` addresses, never hashes. Persist only local writes;
the new root may still reference old backend records. Loader errors/missing or
corrupt records are fatal, not completion-validation errors. See the
[complete persistence protocol](../docs/history-reference.md#individually-addressable-records-and-lazy-persistence).

Inspect `snapshot.root.assistant`, `root.observations[i].representative`, and
`observations[i].evidence[j].outputs[k].tool_output`. Load a continuation with
`snapshot.load(edge.continuation)`; its `NodeRef` is only a storage address.
Output order matches call order. Full payloads/arguments survive; custom grouping
keeps exact per-environment evidence separate from historical representative
context. Snapshot extension path-copies records, leaving prior roots unchanged.

**Subagents are recorded and replayed**, including mixed ordinary/subagent turns,
concurrent child calls and configured nesting depth. `node.subagents` entries
contain the assistant call ID, latest child descriptor, and immutable `runs`
(selected base, resulting revision, and inherited env IDs). Inspect the latest
child with `snapshot.child(entry)`. Child tool observations are rerun; only
child final answers enter the parent's result tuple. Divergent child traces with
identical answers do not split the parent. Active child snapshots retain fulfilled
turns, not partial tool collections.

**Compatibility:** replay checks equal prompts, ordered tool specs and nested
subagent specs/allowlists/depth. Child prompts and specs
are checked on entry. Pure tools are always re-executed: changed implementation
with equal observations reuses reasoning; different observations branch. No manual
version key is needed. Provider/adapter settings and custom grouping semantics
cannot be inferred; callers own external cache isolation and snapshot selection
when those policies change. Payloads, arguments and schemas are **borrowed immutable values**: do not
mutate nested contents through any alias after handing them to the loop. Convert
mutable provider objects to stable values or make your own copies first.

Default `comparison_key` is identity. Neither default/custom mode nor callback identity
is persisted or compared. The current function compares **both** fresh raw outputs
and stored representative outputs. Stricter keys can miss despite a match in old
per-environment evidence; coarser keys can reuse the first matching existing edge
in stored observation order, even if several now match. Replay preserves that
edge's historical representative context and appends full fresh evidence. It does
not merge historical edges, retroactively regroup evidence, or mutate old roots.
The callback must accept historical outputs and remain stable during execution.

### Inspecting assets

```python
snapshot = result.agent
for asset in snapshot.assets():
    print(asset.env_ids, asset.is_complete)
    if asset.has_value:
        print(asset.value)  # latest terminal assistant payload; may be None
    print(asset.messages)   # prompt through leaf, including tool calls and feedback
    print(asset.observations)  # full actual per-env evidence plus representatives
```

An open continuation can have a previous value without a new completed answer.
The view retains its agent/path context; an isolated node or `EnvClass` is not an
asset. Known `env_ids` reflect the latest selected checkpoint and later visits,
not a global latest environment partition. `Asset` is immutable, has no ID or
resume method, and is not independently persisted. Feedback still takes an agent
snapshot and environment and reruns tools. Iteration is lazy/read-only and does not
walk child-agent tries automatically; inspect `snapshot.child(entry).assets()`.
See [view semantics and costs](../docs/agent-assets.md#inspecting-assets).

```sh
PYTHONPATH=src python examples/agent_assets.py
```

The [end-to-end example](examples/agent_assets.py) loads an agent, resolves a new
environment, inspects its asset, continues with feedback, and saves the new agent
while retaining old and open snapshots. No new stored leaf identities are added.

### Specialized tools and independent child caches

```python
from lockstep_loop import BatchLoop, RecordStore, SubagentTool, ToolSpec

study_design = SubagentTool(
    name="assess_design", description="Assess a study-design criterion.",
    parameters={"type": "object", "properties": {"criterion": {"type": "string"}},
                "required": ["criterion"], "additionalProperties": False},
    build_prompt=lambda args: f"Assess study design against criterion {args['criterion']!r}.",
    sub_tool_specs=lambda env: study_tool_specs,
    subagents=(),  # explicit leaf
)
loop = BatchLoop(
    prompt="Screen the study against our review criteria.", env_ids=["study-packet/v2"],
    tool_specs=ordinary_parent_specs, subagents=[study_design, another_specialist],
    recording=True,
    agent_resolver=lambda query: selected.get(query.prompt),  # fixed config in this sketch
    record_store=RecordStore(load=load_node),
)
```

The subagent specs are advertised automatically; no duplicate entries are needed
in `tool_specs`. The child prompt is an initial **user** message, not a system
message. The caller must construct it deterministically and validate structured
args (the example above's minimal lambda raises on a missing key; the executable
example validates keys/types). The core neither evaluates JSON Schema nor
canonicalizes criterion names or paraphrases delegation. Distinct criteria yield
distinct prompts/tries under this template. Child environment IDs are inherited
unchanged: the specialist reads the same study packet as the parent.

The full [independent-cache example](examples/specialized_subagents.py) uses file
imports as a small fixture for this same protocol:

```sh
PYTHONPATH=src python examples/specialized_subagents.py
# Changed parent: 2 fresh parent LLM calls, 0 child LLM calls, 1 child tool rerun
```

An agent resolver receives `AgentQuery(prompt, tools, subagents)`
and returns a caller-selected **AgentDescriptor**, or None for a miss. Lookup is
independent of parent prompt/identity, invocation/call IDs, and env membership.
No child key is required. A recorded parent or resolver enables child recording;
otherwise `SubagentTool(recording=True, ...)` optionally records standalone children.
That switch is not part of structural identity. Fresh roots still require
`BatchLoop(recording=True)`; supplying an `Agent` implies replay plus recording.
Child grouping is inherited. External provider/grouping policy isolation belongs
to the caller, not a core version field.

The resolver only selects a revision; the existing shared `RecordStore` loads its
nodes lazily. Its loader must resolve all selected descriptor references. For
in-memory reuse, share the same store; for persistence, save immutable record
writes first, publish child descriptors, and attach a new writer with a loader
covering those records. Do not return a live `Agent` from the metadata callback
or assume the descriptor carries its records. Full prompt/spec/nested
configuration compatibility is checked even if the caller indexes only by a
hash or prompt. An incompatible hit raises, never silently falls back.

`result.child_agents` contains `ChildAgentUpdate` records for every completed
recorded invocation (descendants included when their containing child finishes):
`tool_call`, `env_ids`, selected `base`, resulting `agent`, parent `branch_id`/`depth`,
and run-local `request_prefix`. This is provenance, not a new conversation identity.
Persist `update.agent.descriptor` and `update.agent.store.records()` with the normal
codecs. Store writes may include parent/orphan records; the child descriptor is an
independent entry point with no parent descriptor dependency. Two concurrent hits
on the same base expose two separate updates, not a hidden mutable cache merge.
Do not blindly collapse them into a dict unless you intend to select one; caller
coordination/root publication is explicit.

A **recorded parent invocation is pinned** to its recorded child revision and
bypasses the resolver. Only new invocations consult the external cache. Thus child
feedback can be selected by a new parent investigation without automatically
rewriting completed parents. Child feedback's shared-path semantics are preserved;
there is no assumption that one prompt implies one revision. Parent entries retain
all prior `runs`; `entry.agent_descriptor` exposes the latest/current run (wire
field `history`). Root agent selection remains the explicit `agent=` argument. Strict feedback resolution
cannot synthesize missing invocation provenance from an external cache hit.

**Registration/nesting:** duplicate definitions and duplicate ordinary spec names
are errors. Conflicting ordinary/subagent specs are errors; an exact matching spec
is accepted as a redundant legacy advertisement, not as a second executor. Serial
`run_to_completion` now recognizes all subagent entries and rejects actual callable/
subagent name collisions. Use direct `BatchLoop` for resolver-driven serving.
Mixing `subagent=` and `subagents=` (even an empty list) is rejected. Singular callers
remain supported. A definition's `subagents` is its explicit child allowlist; `()`
means leaf, and None preserves bounded self-recursion, not parent siblings. Each
nested definition is capped to the remaining `max_depth`; known unavailable
reserved names are omitted/rejected, never ordinary tool requests.

Resolver/factory failures during advancement preserve the pending Python work via
the existing journal, including mixed turns. Callbacks are not rerun at commit;
caller side effects and orphan record writes are not rolled back. Budgets cannot
be combined with recorded children. No environment remapping, dependency
invalidation, logical asset registry, single-flight, or root-merging service is
introduced.

**API/schema migration:** manual contract arguments, fields, and builders have
been removed, not renamed. Legacy pickle class aliases and provenance `history`
fields remain; old `contract`/`history_contract` labels and `custom_grouping` mode
are discarded on pickle load and omitted on new writes. Remove mode flags from
caller cache indices too. Custom codecs may need migration. Immutable fixtures still load and replay. Earlier singular
`subagent` schemas still need caller migration. See the
[exact mapping and wire guarantees](../docs/agent-assets.md#preferred-names-and-compatibility).

### Feedback and shared-path resumption

```python
from lockstep_loop import BatchLoop, Feedback, FeedbackResolutionError

loop = BatchLoop(
    prompt="Assess the evidence packet.", env_ids=["packet-a/v1"], tool_specs=specs,
    agent=snapshot,
    feedback=Feedback("Also check certificate exclusions and revise if needed."),
)
# Serve ordinary requests/completions. Resolution reruns tools but emits NO new
# LLM work until an existing completed conversation has been found and feedback appended.
# revised = loop.result()  # only after completion; revised.agent is a new snapshot
```

Run the full caller-side [feedback example](examples/feedback.py):

```sh
PYTHONPATH=src python examples/feedback.py
```

The operation requires a supplied agent and **exactly one environment to locate
the target**, not to scope the improvement. In the new snapshot, **everyone who
reaches that observation path inherits the feedback**: old siblings and new
revision IDs alike. Additional observations can split them. Other paths and all
old snapshots remain unchanged. There are no session/endpoint handles or
answer-value triggers. The source environment field is provenance, never a gate.
A repeated explicit operation replays through prior feedback to the revised
endpoint and appends again, including identical messages. The same final answer
is valid and terminates without reinjection.

`Feedback(content)` makes presence explicit: `feedback=None` means absent, while
`feedback=Feedback(None)` submits an actual null payload. Content is the normal
opaque `P` and obeys the same borrowed-immutable ownership contract. Feedback is
stored in `node.user_continuations[0].message`; load its continuation reference
with the existing resolver. The old answer remains at that node. Pending new
assistant turns are open nodes, never overwritten historical answers. No parallel
persistence format or core I/O is added.

**Failure protocol:** an unseen observation or unfulfilled selected endpoint
raises `FeedbackResolutionError`, possibly during construction or after tool
results arrive. No fresh reasoning manufactures a target. The loop becomes done,
`loop.error` is set, outstanding/queued work is cancelled, `next_request()` returns
None, and `result()`/further completions raise the stored error. On a resolution
miss `loop.agent` is the supplied root again (orphan writes may remain in the
store). Invalid completions instead leave requests correctable and `error` unset.
An open prior feedback continuation is not a completed target for another explicit
operation; ordinary replay can resume it without adding more feedback. Imported
multiple user successors or malformed user edges fail with `InvalidHistoryError`;
separate roots are not implicitly merged.

Feedback targets the given loop's no-tool assistant endpoint. Cached nested child
tools are rerun during strict resolution and may not request fresh LLM work to
manufacture a target; after injection, fresh nested work is supported normally.
No targeting/routing API for a nested asset or result-tool publication is added.
Lossy custom grouping keeps its existing representative/evidence semantics and
shares feedback across that entire grouped path. The legacy `append_user_message`
helper forks an already completed live run to an open snapshot without resuming
that loop; use constructor feedback for strict resolution and resumption.

**Caveat:** feedback changes the investigation for the entire path, not a private
annotation on one environment. Express transferable instructions (“check whether
X”), not unqualified subject-specific assertions, or retain a separate snapshot.
There is no automatic rewriting/classification, scope switch or environment
binding. Budgets remain rejected for recorded execution, including both resolution
and the new continuation. “Once” is per live operation, not crash-safe publication.

**Transactional callback preparation:** all child prompt/spec factories (including
spec iterators) and child constructors run before the parent assistant turn is
fulfilled or its ordinary requests are queued. A callback exception preserves the
pending request, history root, queues/routes, branch messages/steps and counters;
retry with a corrected completion. This also covers callbacks reached by a tool
completion, nested child rollup or polling cached work. Commit uses the prepared
children without rerunning factories. An undo journal touches only affected
builders/entries, sharing immutable history and retaining partition/final entries;
there is no whole-execution, sibling, or historical-message copying. Queue pulls
and `done` checks with no rollup work do not visit conversations. Child history
links are maintained incrementally, so `agent` (legacy `history`) does not flush mutable state.
Callbacks may read it reentrantly and see only the committed pre-transition root;
reentrant advancement is rejected. Caller side effects, warmed loader caches and
unreachable append-only records are not rolled back. Explicit terminal
failures, such as `FeedbackResolutionError`, retain their cancellation semantics.
See the [transaction boundary](../docs/history-reference.md#transactional-child-preparation-and-callback-failures).

**Limits:** root agent recording/replay requires direct `BatchLoop`; serial drivers
can expose recorded child-agent updates. Recording still rejects budgets. No core
storage I/O, async loader, eviction/GC, independent-root merge, or crash recovery.
Stores retain local writes and cached loads; deep path copying can have quadratic
write volume. Pending tool collections/queues are not persisted. Valid imported
records are trusted; no stable cross-version wire format is promised. Invalid
completions leave requests correctable, including nested routes. Branch IDs use
collision-free run-local counters; legacy tree labels retain `str(result_tuple)`.
Python mapping label collisions append ` [collision N]` without dropping groups
or changing noncolliding labels. Asset views are inspection only; result-tool
publication, DAG scheduling and UI remain out of scope. Multiple specialized
subagent tools and independent child-agent lookup are supported as described above.
See the [shared design and limitations](../docs/history-reference.md#immutable-recorded-history-and-incremental-replay).

## Development

Non-timing guards in `tests/test_transition_cost.py` check that individual serving
operations do not traverse/copy sibling or historical conversations. To reproduce
the shared-tool-call / all-distinct-results / immediate-final-answer workload:

```sh
PYTHONPATH=src python benchmarks/split_completion.py --repeats 3
```

This reports 100/400/1000-environment timings including construction and the plain
`while not loop.done: next_request(); complete(...)` serving loop. Result validation
is outside the timer. Recording is disabled, and no external executors are used.

From this `python/` directory:

```sh
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```
