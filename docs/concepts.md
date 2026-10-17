# One investigation, many environments

Imagine asking: **“Inspect this repository for unsafe path handling.”** You want
answers for several commits or forks, not four copies of the same investigation.
Later, you want tomorrow's commit to benefit from what you already learned.

lockstep-loop shares reasoning where observations match, records the evidence,
and grows new paths where they don't. You choose the model, run the tools, and
save the experience.

## An agent is a recipe *and* its experience

![An agent contains its initial prompt and contract plus the entire investigation tree. A, B, and C share entry/config observations; D branches to its own no-issues asset. Inspecting /foo subdivides A/B from C. A/B share one asset. The blue root-to-leaf path is one conversation.](diagrams/agent-trie.svg)

**Agent** means the whole trie: its initial prompt/investigation recipe, compatible
reuse contract, and accumulated experience. Not just a prompt, and not the executor
running it. A trie is a tree whose paths share their common beginnings.

**Conversation** means one root-to-endpoint path: the prompt, assistant messages,
tool calls and observations, and any later user feedback.

**Asset** means a leaf investigation state backed by that full conversation and
its published value. The value alone is not the asset: A/B and D above both say
“no issues” but have different histories. One leaf can represent several
environments. An open continuation is also an investigation state; it need not
have a new completed value yet.

> **Today's API:** `Agent` is the complete immutable snapshot (`History` remains
> an alias). `agent.assets()` exposes read-only, path-backed `Asset` views, not
> independent endpoint handles. Values are terminal assistant replies **with no
> tool calls**, not designated result-tool arguments. An open asset can retain an
> older value without having a new completed answer.
>
> [Inspect agents/assets and migrate API names →](agent-assets.md)

## Subdivide only when evidence requires it

A, B, C, and D are caller-resolved, immutable repository snapshots. They initially
share one conversation. The assistant asks to read an entry file and configuration;
your application runs both calls in every environment.

Compare the **ordered tuple** `(entry output, config output)`. A/B/C match and
continue together; D differs and branches. The next tool turn inspects `/foo`,
subdividing A/B from C. Repeat this at every turn: recursive subdivision, not one
up-front clustering pass. Independent calls may finish in any order; comparison
waits for the whole turn and uses call order, not arrival order.

Classes **never remerge within an execution**, even if later observations or final
answers agree. Sharing reduces LLM work along common prefixes; it does not remove
per-environment tool work or guarantee unique generated tokens globally.

The drawing abbreviates evidence; the recorded agent does not. It keeps full
messages, call arguments, and actual per-environment tool outputs, not hash-only
summaries. Default grouping uses value equality. An explicit custom grouping policy
can share unequal outputs, while retaining exact evidence separately from the
representative content used in the conversation.

## Tomorrow's commit follows yesterday's reasoning

![A recorded assistant turn supplies tool calls that are rerun on a new stable commit. A known ordered result tuple selects a recorded continuation and repeats replay. An unseen tuple requests fresh LLM reasoning and adds a branch. Both paths survive in a new immutable snapshot.](diagrams/replay.svg)

A new commit is a **new environment**, not automatically a new investigation.
Reuse the agent under the same prompt and compatible tool/grouping contract.
Rerun its recorded tools, follow matching observations, and reuse assistant turns.
A matching completed path needs **zero new LLM requests**. At the first unseen
observation, ask for new reasoning. Even a turn that originally split nobody is
a recorded checkpoint where tomorrow's commit can diverge.

Extending experience creates a **new immutable agent snapshot**. Both paths stay
reusable; yesterday's snapshot remains unchanged. Your application persists those
snapshots — the core owns no provider or storage. Change the recipe or tool
semantics and compatibility must be reconsidered; this is not fuzzy prompt caching.

## Feedback continues the investigation

Suppose an earlier snapshot stopped after reading entry/config and answered
“no issues”. You realize it should also inspect `/foo`.

![The old snapshot ends at a shared A/B asset with value no issues. In a new snapshot, replay resolves A's completed path, retains the old answer, appends the user message Also inspect /foo, and opens a continuation. B and future matching commits inherit it; other paths and the old snapshot are unaffected.](diagrams/feedback.svg)

Provide **snapshot + one stable environment + explicit feedback**. The core
replays tools to locate the existing completed endpoint, then appends your user
message. It does not request fresh LLM work merely to find a target. An unseen or
unfinished target fails explicitly; ordinary replay can instead grow unseen paths.

In the **new snapshot**, every environment reaching that path inherits the note:
A, B, and future commits alike. Additional tools may split them. Unrelated paths
and the entire old snapshot are untouched. The former leaf is now a historical
endpoint leading to an open continuation, not an answer overwritten in place.

“Still no issues” is a valid revised answer. Feedback is **not a value-to-message
map**: an identical answer elsewhere doesn't trigger it. Each new explicit
intervention appends again at the latest completed endpoint. There are no separate
public session/endpoint handles that bypass replay.

Use transferable instructions (“check whether X”), not assertions that are only
true of one commit, or keep a separate snapshot.

## Subagents contain divergence — and can contain invalidation

![A shared parent asks a child agent to inspect imports in foo.py. The child follows different source-evidence paths for A and B, but both return imports pathlib. Child paths remain distinct and retain full evidence; equal child values allow the parent to remain shared.](diagrams/subagents.svg)

Source text can differ without changing what a parent needs to know. A child
investigates those details and returns a value. Its paths may diverge extensively;
if its final values agree, that child call does not split the parent. Different
parent-visible tuples still split normally. This is **containment, not remerging**.

There is a second payoff: **contain invalidation**. Give the child a
fixed or deterministically constructed recipe — for example, “List imports in
`foo.py`” built from a structured file argument. With independently reusable child
agents, changing a parent's wording need not discard the child's experience.
Different file paths mean different recipes/tries, not fuzzy prompt matches.

**Supported now:** multiple named `SubagentTool` definitions, deterministic
`build_prompt`, bounded child allowlists, nested replay, and independent child-agent
lookup across changed parent prompts. The initial prompt is a user message, not a
separate system-role API. A caller's resolver selects a compatible child revision
for a new invocation; recorded parent invocations remain pinned to their recorded
child revision. Each invocation exposes its base/result agent snapshots for
inspection and caller-owned persistence; concurrent updates are not silently merged.
See the [specialized-agent reference](history-reference.md#specialized-tools-and-independently-reusable-child-histories).

## Why this design?

Keep expensive reasoning shared, make every conclusion explainable, and improve
investigations without erasing their past. The boundary stays small: the core
chooses what work is needed; your application chooses how and where to execute it.
You are responsible for pure/read-only tools, stable environments, identical specs,
no model-visible environment identities, and immutable payload contents.

**Future, not implemented:** dependency-DAG push updates ([issue #1](https://github.com/kaeluka/lockstep-loop/issues/1))
and designated result-tool publication. A changed child value does not automatically
update an already completed parent.

**Next:** [Agent/Asset glossary and inspection](agent-assets.md) ·
[Python API](../python/README.md) · [Rust API](../rust/README.md) ·
[Runnable replay and feedback demos](../README.md#try-it) ·
[Storage and compatibility reference](history-reference.md)
