# Design intent

lockstep-loop is a provider-neutral partition-refinement state machine with Python
and Rust implementations. The caller owns meaning and execution; the library owns
conversation structure, refinement, and transparent reuse.

## Responsibility boundaries

The caller defines messages, stable environments, tool implementations and
specifications, and equivalence policy. It executes model and tool requests,
adapts provider protocols, chooses compatible history for external policy changes,
and decides which conversation-derived values are assets.

The library treats message contents as opaque. It tracks environment handles,
coordinates requests and completions, refines groups from complete ordered tool
results, and maintains an immutable history trie. It reuses recorded model
responses, re-executes their tools through the caller, and requests fresh model
work only where the selected continuation lacks a response.

The library preserves exact evidence and provenance. Equivalence governs sharing,
not permission to discard evidence. The caller decides which sharing is
appropriate; library-maintained environment hygiene remains an explicit open
design issue below, not an implemented isolation guarantee.

Storage is shared responsibility: the library owns the history's structure and
semantics—immutable records, references, snapshots, and traversal. The caller owns
how and where those records are persisted: filesystem, cloud blob storage,
database, codecs, and durable publication. The library exposes records and loads
references through caller-provided hooks; it implements replay, but no backend
I/O. Model clients, execution infrastructure, and storage backends stay outside
the core.

## Domain concepts and operations

### Environment

An environment is a caller-owned reference or handle identifying a stable
execution context—for example, an `owner/repo@commit` string, not the repository
itself.

- All environments expose identical tool specifications.
- Tools are read-only, with stable observations during an investigation.
- The library tracks and routes the handle; it is not exposed to the model.

**Operation:** the caller executes a tool call using the environment handle and
returns its output. The library does not perform this execution.

### Tool call

A tool call is an assistant's request to invoke a named tool with specific
arguments.

- It contains a call ID, tool name, and arguments—not an environment handle.
- The same call is executed against every environment in the current refinement
  group. The library tracks and routes the environment handles; the model never
  sees them.
- The call ID correlates outputs with the call; it does not identify an execution
  environment.

**Operation:** the library requests execution of the call against every environment
in the current refinement group. The caller performs execution and returns the
per-environment outputs; the library accepts the completion.

### Refinement group

A refinement group is a set of environments sharing one conversation path during
an investigation.

- Initially, all supplied environments belong to one group.
- Every tool call is executed against every environment in the group.
- After all calls in a turn complete, each environment has an ordered tuple of
  outputs. Tuple order follows tool-call order, not completion order.
- Environments with equivalent tuples stay together; differing tuples split the
  group.
- Groups never merge during an investigation, even if their final answers agree.

**Operation:** refine a group into one or more groups according to tool result
equivalence.

### Turn tool results

Turn tool results are the complete, ordered tuple of tool outputs produced by one
environment for one assistant turn.

- Outputs follow tool-call order, regardless of completion order.
- Partial results are not yet turn tool results.
- Exact outputs are preserved, even when grouping considers different tuples
  equivalent.

**Operation:** assemble turn tool results once every tool call in the turn has
returned an output for that environment.

### Tool result equivalence

Tool result equivalence is the caller-defined policy deciding which turn tool
results may share subsequent reasoning.

- Defaults to exact output equality.
- Compares corresponding outputs in tool-call order.
- Applies to both fresh outputs and historical representatives during replay.
- Never discards exact outputs from the recorded per-environment evidence.
- Matching uses historical representatives, not any matching sample in old
  evidence. A stricter policy can therefore miss an existing path.
- If a coarser policy matches several continuations, the first in stored order
  wins; existing continuations and their representative context are preserved.
- Hashes may accelerate comparison but never replace equality verification.
  Neither callback identity nor default/custom mode defines compatibility.

**Operation:** determine whether two turn tool results are equivalent.

#### Representative turn tool results and substitution

Representative turn tool results are the complete tuple used in the shared
model-visible conversation when a refinement group's actual results are equivalent.
For a new continuation, select one environment's complete tuple, not a mixture of
outputs from different environments. The current implementation selects the first
environment in the group. Replay retains the historical representative.

**Operation:** select representative turn tool results when creating a
continuation; retain them when replaying it. Preserve each environment's actual
results as distinct evidence, without requiring separate physical storage.

Representative substitution is an intentional part of equivalence-based sharing,
not a bug or oversight. If one environment returns audio saying "yes" and another
returns audio saying "yeah", a policy may treat them as equivalent. The shared
continuation receives only one recording. For the other environment, the model
reasons from an output that its tool did not actually return. The representative
is actual recorded content, not synthesized content or a uniquely correct
canonical answer.

Equivalence expresses the caller's decision that outputs are interchangeable for
subsequent reasoning. It determines which distinctions—such as exact wording,
tone, or speaker identity—the shared conversation preserves. Preserving exact
per-environment evidence makes that choice inspectable; it does not change the
content used for shared reasoning. With default value equality, representatives
and grouped outputs are equal under that equality.

Overequating can cross intended sharing boundaries. For example, in a deployment
serving different users, an overly broad policy could lead to conversations using
messages originating from different users. Content equivalence alone does not
establish that sharing across those users is appropriate; the caller owns that
policy and the selection of compatible history.

**Open design question (not a decided API):** should equivalence be indexed by an
environment key, so sharing can be scoped by caller-defined context? This remains
to be explored; environment handles would still not be exposed to the model.

**TODO: environment hygiene under equivalence.** Equivalent tool outputs should
be retained in the trie for every environment, with the library maintaining
correct environment attribution and appropriate separation of conversation
content. Exact per-environment output storage already exists; it does not by
itself provide this hygiene, because shared model-visible conversations use
representatives that can originate in another environment. Define how reuse,
conversation construction, and inspection maintain these boundaries, including
how reused model responses may depend on another environment's outputs. Do not
treat this TODO as an implemented isolation guarantee.

### Message

A message is an opaque, immutable, caller-defined value.

- The library stores and passes messages without interpreting their contents.
- The only content-dependent operation is comparison: caller-supplied equivalence,
  falling back to value equality. The library delegates comparison rather than
  inspecting message contents itself.
- No textual format, modality, or internal structure is assumed. A message may
  contain text, images, audio, video, or structured multimodal content.
- Opacity to the library is a domain guarantee, not merely an implementation detail.
- The caller must preserve immutable value semantics for messages, tool arguments,
  and schemas, including nested values and aliases. Frozen wrappers alone do not
  provide this guarantee; cloning must not permit later mutation of recorded facts.

### Model response

A model response contains a message and zero or more ordered tool calls.
It is shared by a refinement group and preserved in history.

**Operation: take an assistant turn.** Obtain a model response for the group's
current conversation—reuse a recorded response when available, otherwise request
fresh model work. The response then determines whether to execute tools or finish
the group.

### Conversation

A conversation is an ordered exchange along one history path.

- Begins with one or more initial messages—for example, a fixed system message
  followed by one or several user messages.
- Continues through model responses, tool outputs, and user follow-ups.
- Message contents remain opaque to the library.

**Operation:** extend the conversation without changing its existing prefix.

### History trie

A history trie is the accumulated collection of recorded conversations and their
tool-result evidence, sharing common prefixes. It is a domain concept as well as
an efficient representation of many similar conversations. A conversation is one
path through the history trie, exposed as an ordered exchange; it does not require
independent storage.

- Records every tool-result checkpoint, including turns that do not split a
  refinement group.
- Shares common prefixes and preserves previously explored paths.
- Enables transparent reuse when investigating new environments.
- Uses one immutable record model for execution history, inspection, and
  persistence. Extending a trie never mutates captured roots or records.
- Preserves full model messages, tool calls/arguments, and outputs in extensible
  records, not just hashes or grouping keys.
- References identify exact records; hash/debug-string collisions must not
  conflate records or invalidate older snapshots.
- Actual tool outputs retain their environment attribution, but the shared
  conversation and its reasoning can use representatives from other environments.
  "Per-environment evidence" does not imply environment-isolated conversations;
  see the environment-hygiene TODO under tool result equivalence.

**User-facing operation:** inspect recorded conversations and evidence. Users
submit environments for investigation; replay and history extension are internal
mechanisms, not separate modes users must select.

### Agent

An agent combines one or more initial messages, available tool specifications,
and its current history.

**User-facing operation: investigate environments.** Supply environment handles.
The library uses the agent's history transparently, requesting model work only
where needed and tool execution to determine each environment's path. A successful
investigation updates the agent's history and returns results for the supplied
environments.

The caller executes model and tool requests; the agent coordinates them.

History reuse requires equal initial messages, ordered tool specifications, and
nested subagent specifications, allowlists, and depth limits. Actual child initial
messages and specifications are checked on entry. Independent child lookup uses
the child's structural configuration, not parent or environment identity.
Do not introduce manual contract/version/namespace keys for reasoning reuse.
Changed pure tool implementations reuse reasoning when their results match and
branch when they differ. Provider/adapter policy and cache isolation remain
caller-owned, not inferred from structural compatibility.

#### Investigate environments (operation)

Investigating environments is an operation on an agent, not a separate domain
entity. Executing the operation:

- Captures the agent's current history at the start.
- Tracks refinement groups and outstanding model/tool requests.
- Transparently reuses recorded reasoning and adds new paths as needed.
- Produces results for the supplied environments.
- Publishes its resulting snapshot automatically and exactly once on successful
  completion, including immediate cached or budget-limited completion. Inspecting
  an old execution never republishes it.
- Failed or abandoned executions never replace the agent's current snapshot.
- Overlapping executions are allowed: **LAST COMPLETION WINS WITHOUT MERGING**.
  No queues or lifetime locks are required; retain independent snapshots to keep
  results excluded by a later publication.
- Budgets limit accepted model responses per refinement group, including reused
  responses. Exhaustion returns a fallback, never a fabricated model answer;
  preserve the real checkpoint so a later investigation can continue.
- Every execution, including child investigations and convenience drivers, records
  a snapshot. Child invocation provenance anchors snapshots, never mutable agents.

**User-facing operations:** start an investigation, obtain work requests, submit
completions, and retrieve results.

The current code's run handle represents the operation's temporary execution
state. Pending requests, refinement groups, and collected results do not require
an additional core domain entity.

### Work request

A work request describes work the investigation needs the caller to perform.

- A model request asks for a model response for a refinement group's conversation,
  using the available tool specifications.
- A tool request asks for one tool call to be executed against every environment
  in its refinement group.
- Each request carries an identifier to correlate the returned completion.
- Environment handles are available to the caller for routing, never exposed to
  the model.
- Independent tool requests may complete in any order; refinement waits for
  complete turn tool results.

**User-facing operations:** obtain pending work requests from an investigation,
perform the requested work, and submit the results back to the library as
completions associated with their request identifiers.

### Completion

A completion is the caller's submission of results for a work request.

- Identifies the request it fulfills.
- Contains either a model response or the requested per-environment tool outputs.
- Must satisfy the request's structural requirements before being accepted.
- Invalid submissions leave the request pending and correctable.
- An accepted completion fulfills the request exactly once; retries or corrections
  must not overwrite fulfilled historical facts.

**User-facing operation:** submit a completion to advance the investigation.

### Snapshot

A snapshot captures an agent's initial messages, tool specifications, and history
at a particular revision.

- Remains unchanged as later investigations extend the agent's history.
- Contains recorded facts, not outstanding requests or partially collected results.
- Does not include executable callbacks or model/tool execution infrastructure.
- Can serve as the starting point for an independent agent.
- Snapshot capture and restoration support caller-provided lazy stores/loaders
  symmetrically. Restore structural configuration from the snapshot and attach
  runtime behavior explicitly, without making the caller implement replay.
- Conversation views and child provenance anchor snapshots, not mutable agents.
- Python supports direct trusted pickle of the reachable history as a convenience;
  both languages retain immutable descriptor/record APIs for lazy or incremental
  persistence. See the persistence reference for codecs and format limitations.

**User-facing operations:** capture a snapshot, inspect it, persist it using
caller-owned storage, and restore an agent from it with runtime behavior
explicitly reattached.

Restoring a snapshot reuses recorded history; it does not resume an interrupted
investigation's pending work.

### Asset

An asset is the value the caller cares about, derived from a conversation.

The library provides conversation accessors for common ways to obtain that value:

- `get_last_answer_asset()` retrieves the latest model answer.
- `get_final_tool_input_asset(tool_name: str)` retrieves the input delivered through the
  named final-answer tool, such as a typed `submitFinalAnswer` tool. The answer is
  the tool input, not its output.

The caller determines which value constitutes the asset. Derivation can be
expressed as a function from conversation to asset; this does not require a
library-managed projection callback or a separate asset wrapper. The conversation
provides the source material and provenance. A conversation may not yet contain
the answer or final-tool input the caller wants.

**User-facing operation:** access the desired value from a recorded conversation.

This is the intended domain distinction; the current code's Asset type instead
provides a snapshot-anchored conversation view. The accessor names above describe
the intended API, not a claim that it is already implemented.

## Scope and development

The domain definitions describe the intended design; current API differences are
documented explicitly. Keep canonical names without deprecated aliases or legacy
migration scaffolding. Leave room for continuing conversations and downstream use
of assets, but do not add DAG scheduling, dependency updates, feedback routing,
asset-publication machinery, or a UI as part of this work.

Preserve request/completion behavior and regression coverage when changing the
implementation. Document deliberate limitations, especially nested agents, lazy
loading, compatibility policy, and Python/Rust differences.

- [History and persistence reference](docs/history-reference.md): current APIs,
  record publication/loading, Python pickle, child histories, and limitations.
- [Verification](docs/verification.md#required-regression-coverage): required
  behavioral coverage and implementation evidence.

Development commands:
- Python: `cd python && uv run pytest && uv run mypy src && uv run ruff check .`
- Rust: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo fmt --all --check`
