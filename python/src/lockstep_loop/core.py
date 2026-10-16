"""Core types and orchestrator for lockstep-loop.

lockstep-loop runs a single agent prompt across many similar, read-only
environments without generating shared tokens twice:

1. All environments start in one equivalence class sharing a single LLM
   conversation.
2. When the model makes a tool call, the call is evaluated on every
   environment still in the class (SIMD-style fan-out).
3. Tool results are compared and the class is refined: one sub-conversation
   per distinct result.  Branches then continue independently.
4. The process repeats per branch until the loop finishes, yielding a
   prefix-sharing trie whose leaves are equivalence classes of environments.

Cost scales with the number of distinct tool-call traces, not with the
number of environments.

Provider-agnostic by construction: the library never calls an LLM (or a
tool) itself.  ``BatchLoop`` hands out requests — LLM calls and tool
fan-outs — and the caller answers them in whatever order they finish.  A
slow provider (or tool) delays only its own branch.

Payloads are opaque: the loop is generic in the content type ``P``, so
multimodal conversations (content blocks, image bytes) work unchanged.
Tool results are grouped by value equality and therefore must be hashable,
unless a ``group_by`` canonicalizer is given.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast, overload

#: Conversation payload type: message content, tool results, final answers.
P = TypeVar("P")

#: Covariant variant for the ``Tool`` protocol (P only appears in return position).
P_co = TypeVar("P_co", covariant=True)


class Tool(Protocol[P_co]):
    """A single environment's implementation of one tool.

    Invariants (the caller's responsibility):

    * Tools must be read-only / pure: a result may depend only on the
      environment and the call arguments.
    * ``name``, ``description`` and ``parameters`` must be identical across
      all environments and must not leak the environment id — otherwise
      conversations cannot be shared.
    """

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, Any]:
        """JSON Schema describing the call arguments."""
        ...

    def __call__(self, args: Mapping[str, Any]) -> P_co: ...


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool description.

    Handed to the caller inside every ``LlmRequest``; the caller's provider
    adapter converts it to whatever schema its LLM API expects.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    """Unique within one assistant turn; ties tool result messages back to the call."""

    name: str
    args: Mapping[str, Any]


@dataclass(frozen=True)
class Message(Generic[P]):
    """A provider-neutral conversation message.

    ``tool_calls`` is only set on assistant messages; ``tool_call_id`` only
    on tool messages.  ``content`` is opaque to the loop.
    """

    role: Literal["user", "assistant", "tool"]
    content: P
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class RequestStats:
    """Facts about a request, for the caller's scheduling decisions.

    The library reports facts and never computes priorities: the caller
    composes these with its own serving facts (prefix recency, cost, ...)
    to pick serving order — e.g. "deepest currently available branch first"
    for prompt-cache warmth.
    """

    depth: int
    """LLM steps this branch has taken so far."""

    split_depth: int
    """Divergences on the path from the root to this branch."""

    seq: int
    """Monotone issue counter, for deterministic tie-breaks."""


@dataclass(frozen=True)
class LlmRequest(Generic[P]):
    """The loop needs one LLM completion for a branch.

    ``env_ids`` is the branch's equivalence class — the whole set of envs
    this completion rides on."""

    request_id: str
    branch_id: str
    env_ids: tuple[str, ...]
    messages: tuple[Message[P], ...]
    tools: tuple[ToolSpec, ...]
    stats: RequestStats


@dataclass(frozen=True)
class ToolRequest:
    """The loop needs one tool call fanned out over a set of environments.

    The caller executes ``tool_call`` once per id in ``env_ids`` and returns
    every result in a single ``ToolCompletion``.  The results drive
    partition refinement: environments with equal results stay in one
    conversation.
    """

    request_id: str
    branch_id: str
    tool_call: ToolCall
    env_ids: tuple[str, ...]
    stats: RequestStats


#: Work the loop asks the caller to perform.
Request: TypeAlias = LlmRequest[P] | ToolRequest


@dataclass(frozen=True)
class LlmCompletion(Generic[P]):
    """The model's reply to an ``LlmRequest``.

    Empty ``tool_calls`` ends the branch and ``content`` becomes its final
    answer.  ``content`` is opaque to the loop.
    """

    request_id: str
    content: P
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolCompletion(Generic[P]):
    """Fanned-out results answering a ``ToolRequest``.

    Exactly one ``(env_id, result)`` pair per env in the request.  Results
    are compared by value equality and must therefore be hashable, unless
    the loop was constructed with a ``group_by`` canonicalizer — the escape
    hatch for grouping values that are deliberately *not* equal.
    """

    request_id: str
    results: tuple[tuple[str, P], ...]


#: The caller's answer to a ``Request``.
Completion: TypeAlias = LlmCompletion[P] | ToolCompletion[P]


@dataclass(frozen=True)
class SubagentTool(Generic[P]):
    """Reserved tool the core executes itself by spawning a nested ``BatchLoop``.

    When the model calls this tool, the loop spawns a child ``BatchLoop``
    over the calling branch's environments instead of emitting a
    ``ToolRequest``.  When the child finishes, its per-env final answers
    are folded in as the tool results, so environments with equal answers
    stay grouped — the "homogenize a cheap investigation" pattern.

    ``__call__`` is intentionally absent: the core routes calls to this
    name into child spawning, so a SubagentTool must never reach a tool
    executor.  Detection is by ``isinstance`` at construction; dispatch at
    call time is by name.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    build_prompt: Callable[[Mapping[str, Any]], P]
    """Build the child loop's root prompt from the tool-call arguments."""
    sub_tool_specs: Callable[[str], Sequence[ToolSpec]]
    """Spec provider for the child loop's tools (one env id -> all specs)."""
    max_depth: int = 1
    """How many levels below this loop sub-agents may still spawn (children get max_depth-1)."""


@dataclass(frozen=True)
class Budget:
    """Limits for the run.

    Budget is shared while branches share a conversation and becomes
    per-branch after a split, so a pathological environment cannot burn
    its siblings' budget.
    """

    max_steps_per_branch: int | None = None


@dataclass(frozen=True)
class SplitNode:
    """An immutable node in the prefix-sharing trie.

    Represents the set of environments whose conversations are identical so
    far. Children use opaque, collision-free split labels. Actual tool evidence
    and replay continuations live in ``History``, not this partition summary.

    Nodes are never mutated: a split builds new child nodes and copies the
    path from the split point back to the root (structural sharing).  Treat
    ``children`` as read-only.
    """

    env_ids: tuple[str, ...]
    children: Mapping[str, SplitNode] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvClass:
    """A final equivalence class of environments."""

    id: str
    env_ids: tuple[str, ...]


@dataclass(frozen=True)
class OutputRecord(Generic[P]):
    """Exact tool evidence, in assistant call order (never a grouping key)."""

    tool_output: P


@dataclass(frozen=True)
class EnvObservation(Generic[P]):
    env_id: str
    outputs: tuple[OutputRecord[P], ...]


@dataclass(frozen=True)
class Observation(Generic[P]):
    """Representative conversation content and all actual per-env observations."""

    representative: tuple[OutputRecord[P], ...]
    evidence: tuple[EnvObservation[P], ...]
    continuation: HistoryNode[P]


@dataclass(frozen=True)
class HistoryNode(Generic[P]):
    assistant: Message[P] | None = None
    env_ids: tuple[str, ...] = ()
    observations: tuple[Observation[P], ...] = ()


@dataclass(frozen=True)
class History(Generic[P]):
    """Storage-neutral snapshot. Payloads/args/schemas are borrowed immutable values.

    The caller must not mutate them, including through requests or snapshots.
    ``contract`` attests tool semantics, grouping policy and provider settings;
    it is a caller-managed version, not a hash computed by this library.
    Replay additionally checks prompt and tool-spec equality.
    """

    contract: str
    prompt: P
    tools: tuple[ToolSpec, ...]
    custom_grouping: bool
    root: HistoryNode[P] = field(default_factory=HistoryNode)


def _history_node(root: HistoryNode[P], path: tuple[int, ...]) -> HistoryNode[P]:
    for index in path:
        root = root.observations[index].continuation
    return root


def _replace_history_node(
    root: HistoryNode[P], path: tuple[int, ...], node: HistoryNode[P]
) -> HistoryNode[P]:
    ancestors = []
    for index in path:
        ancestors.append((root, index))
        root = root.observations[index].continuation
    for ancestor, index in reversed(ancestors):
        updated = replace(ancestor.observations[index], continuation=node)
        node = replace(
            ancestor,
            observations=(
                *ancestor.observations[:index],
                updated,
                *ancestor.observations[index + 1 :],
            ),
        )
    return node


@dataclass(frozen=True)
class BatchResult(Generic[P]):
    """Output of a completed ``BatchLoop``."""

    tree: SplitNode
    """Legacy partition summary; use history for replay."""

    classes: tuple[EnvClass, ...]
    """Final partition of environments by full tool-call trace."""

    results: Mapping[str, P]
    """Final assistant message per environment (identical within a class)."""

    history: History[P] | None = None


@dataclass
class _Branch(Generic[P]):
    """Mutable internal state for one live conversation branch."""

    branch_id: str
    parent_id: str | None
    group_key: str | None
    """Edge label from the parent (None for the root)."""
    env_ids: tuple[str, ...]
    messages: list[Message[P]]
    split_depth: int
    steps: int = 0
    history_path: tuple[int, ...] = ()
    final_content: P | None = None
    turn_calls: tuple[ToolCall, ...] = ()
    turn_results: dict[str, dict[str, P]] = field(default_factory=dict)
    """tool_call_id -> (env_id -> result) for the current assistant turn."""


@dataclass
class _ChildLink(Generic[P]):
    """A parent branch suspended on a spawned child loop."""

    child: BatchLoop[P]
    branch_id: str
    call: ToolCall


class BatchLoop(Generic[P]):
    """Orchestrates one prompt across many environments in lockstep.

    The loop owns partition refinement and the trie; the caller owns all
    execution.  A typical driver:

        loop = BatchLoop(prompt=..., env_ids=..., tool_specs=...)
        while not loop.done:
            request = loop.next_request("any")
            if request is None:
                wait_for_completions()  # work outstanding, nothing runnable
                continue
            serve(request)              # maybe on another thread
        # ...as completions arrive, from any thread:
        loop.complete(completion)
        result = loop.result()

    Thread safety: all public methods are internally synchronized, so
    separate LLM and tool consumer loops (and completion collectors) can
    share one ``BatchLoop``.

    Ordering: each request kind has its own FIFO queue, always drained in
    issue order.  ``kind="any"`` merges the two queue heads by sequence
    number, so given identical queue state it is deterministic; under
    concurrent pulling and completing, the *cross-kind* interleaving is
    best-effort — the frontier moves while you decide.  Sequential drivers
    get exact, replayable issue order.

    Grouping: tool results are compared by value equality, so payloads must
    be hashable.  ``group_by`` is the escape hatch for lossy grouping —
    environments whose canonical keys match share a conversation even when
    their raw payloads differ (the conversation then records a
    representative's payload).

    Invariants:
        * No token is generated twice for environments that remain in the
          same equivalence class.
        * Classes only refine (split), never merge.
    """

    def __init__(
        self,
        *,
        prompt: P,
        env_ids: Sequence[str],
        tool_specs: Sequence[ToolSpec],
        subagent: SubagentTool[P] | None = None,
        group_by: Callable[[P], Hashable] | None = None,
        budget: Budget | None = None,
        history_contract: str | None = None,
        history: History[P] | None = None,
        _id_prefix: str = "",
    ) -> None:
        envs = tuple(env_ids)
        if not envs:
            raise ValueError("env_ids must be non-empty")
        if len(set(envs)) != len(envs):
            raise ValueError("env_ids must be unique")
        if subagent is not None and subagent.max_depth < 1:
            raise ValueError("subagent.max_depth must be >= 1")
        self._specs = tuple(tool_specs)
        if history is not None and history_contract is None:
            raise ValueError("replay requires an explicit history_contract")
        if history_contract is not None:
            if not history_contract:
                raise ValueError("history_contract must be non-empty")
            if subagent is not None or budget is not None:
                raise ValueError("recorded history does not yet support subagents or budgets")
            if history is not None and (
                history.contract != history_contract
                or history.prompt != prompt
                or history.tools != self._specs
                or history.custom_grouping != (group_by is not None)
            ):
                raise ValueError("incompatible history contract, prompt, tools or grouping mode")
            history = history or History(
                history_contract, prompt, self._specs, group_by is not None
            )
        self._recorded = history
        self._next_branch = 0
        self._subagent = subagent
        self._id_prefix = _id_prefix
        self._group_by: Callable[[P], Hashable] = (
            group_by if group_by is not None else lambda v: cast(Hashable, v)
        )
        self._budget = budget
        self._children: dict[str, _ChildLink[P]] = {}
        self._child_ids: dict[str, BatchLoop[P]] = {}

        self._lock = threading.Lock()
        self._seq = 0
        self._branches: dict[str, _Branch[P]] = {}
        self._partition: list[_Branch[P]] = []  # legacy per-run partition summary
        self._finals: list[_Branch[P]] = []
        self._ready_llm: deque[LlmRequest[P]] = deque()
        self._ready_tool: deque[ToolRequest] = deque()
        self._outstanding: dict[str, Request[P]] = {}
        self._out_llm = 0
        self._out_tool = 0

        root: _Branch[P] = _Branch(
            branch_id="root",
            parent_id=None,
            group_key=None,
            env_ids=envs,
            messages=[Message(role="user", content=history.prompt if history else prompt)],
            split_depth=0,
        )
        self._branches[root.branch_id] = root
        self._partition.append(root)
        self._request_llm(root)

    @overload
    def next_request(self, kind: Literal["llm"]) -> LlmRequest[P] | None: ...

    @overload
    def next_request(self, kind: Literal["tool"]) -> ToolRequest | None: ...

    @overload
    def next_request(self, kind: Literal["any"] = "any") -> Request[P] | None: ...

    def next_request(self, kind: Literal["any", "llm", "tool"] = "any") -> Request[P] | None:
        """Pop one runnable request, or None if none of that kind is runnable.

        None is not a termination signal — work may still be outstanding, or
        queued under another kind.  ``done`` is the only termination signal.
        Separate consumer modules can each pull their own kind; an LLM pool
        and a tool executor never see each other's requests.
        """
        with self._lock:
            self._rollup_children()
            req = self._pop_children(kind)
            if req is not None:
                return req
            if kind == "llm":
                request: Request[P] | None = self._ready_llm.popleft() if self._ready_llm else None
            elif kind == "tool":
                request = self._ready_tool.popleft() if self._ready_tool else None
            else:
                request = self._pop_any()
            if request is None:
                return None
            self._outstanding[request.request_id] = request
            if isinstance(request, LlmRequest):
                self._out_llm += 1
            else:
                self._out_tool += 1
            return request

    def complete(self, completion: Completion[P]) -> None:
        """Deliver a result for an outstanding request.

        Unknown/answered IDs, duplicate call IDs and invalid environment result
        sets raise ValueError; wrong completion kinds raise TypeError. Validation
        (including tool grouping) runs before fulfillment, leaving pending work
        recoverable. Caller callbacks must be pure; side effects are not rolled back.
        """
        with self._lock:
            child = self._child_ids.get(completion.request_id)
            if child is not None:
                child.complete(completion)
                del self._child_ids[completion.request_id]
                if isinstance(completion, LlmCompletion):
                    self._out_llm -= 1
                else:
                    self._out_tool -= 1
                self._rollup_children()
                return
            request = self._outstanding.get(completion.request_id)
            if request is None:
                raise ValueError(f"no outstanding request with id {completion.request_id!r}")
            if isinstance(request, LlmRequest):
                if not isinstance(completion, LlmCompletion):
                    raise TypeError(
                        f"LlmRequest {request.request_id} must be answered with LlmCompletion"
                    )
                ids = [call.id for call in completion.tool_calls]
                if len(set(ids)) != len(ids):
                    raise ValueError("duplicate tool call id in one assistant turn")
                self._handle_llm(request, completion)
                del self._outstanding[completion.request_id]
                self._out_llm -= 1
            else:
                if not isinstance(completion, ToolCompletion):
                    raise TypeError(
                        f"ToolRequest {request.request_id} must be answered with ToolCompletion"
                    )
                self._handle_tool(request, completion)
                del self._outstanding[completion.request_id]
                self._out_tool -= 1

    @property
    def history(self) -> History[P] | None:
        """Capture an immutable, structurally shared root, including fulfilled turns."""
        with self._lock:
            return self._recorded

    def _set_history_node(self, path: tuple[int, ...], node: HistoryNode[P]) -> None:
        assert self._recorded is not None
        self._recorded = replace(
            self._recorded,
            root=_replace_history_node(self._recorded.root, path, node),
        )

    @property
    def done(self) -> bool:
        """True when nothing is runnable and nothing is outstanding (recursively)."""
        with self._lock:
            self._rollup_children()
            return (
                not self._ready_llm
                and not self._ready_tool
                and not self._outstanding
                and not self._child_ids
                and not self._children
            )

    @property
    def outstanding_llm(self) -> int:
        """LLM requests handed out but not yet completed."""
        with self._lock:
            return self._out_llm

    @property
    def outstanding_tool(self) -> int:
        """Tool requests handed out but not yet completed."""
        with self._lock:
            return self._out_tool

    @property
    def outstanding_count(self) -> int:
        """All requests handed out but not yet completed."""
        with self._lock:
            return self._out_llm + self._out_tool

    def result(self) -> BatchResult[P]:
        """The final snapshot: split trie, equivalence classes, results.

        Raises RuntimeError unless the loop is done.
        """
        with self._lock:
            self._rollup_children()
            if not (
                not self._ready_llm
                and not self._ready_tool
                and not self._outstanding
                and not self._child_ids
                and not self._children
            ):
                raise RuntimeError("the loop is not done")
            return self._finalize()

    # -- internals (all called with self._lock held) --

    def _pop_children(self, kind: Literal["any", "llm", "tool"]) -> Request[P] | None:
        """Pull one request from a live child loop (children first)."""
        for link in self._children.values():
            req: Request[P] | None = link.child.next_request(kind)
            if req is not None:
                self._child_ids[req.request_id] = link.child
                if isinstance(req, LlmRequest):
                    self._out_llm += 1
                else:
                    self._out_tool += 1
                return req
        return None

    def _rollup_children(self) -> None:
        """Fold finished children into their parent's suspended branch."""
        for key, link in list(self._children.items()):
            if link.child.done:
                branch = self._branches[link.branch_id]
                branch.turn_results[link.call.id] = dict(link.child.result().results)
                del self._children[key]
                if len(branch.turn_results) == len(branch.turn_calls):
                    self._advance_turn(branch)

    def _is_subagent_call(self, call: ToolCall) -> bool:
        return self._subagent is not None and call.name == self._subagent.name

    def _spawn_child(self, branch: _Branch[P], call: ToolCall) -> None:
        sub = self._subagent
        assert sub is not None
        child_subagent: SubagentTool[P] | None = None
        if sub.max_depth > 1:
            child_subagent = replace(sub, max_depth=sub.max_depth - 1)
        child: BatchLoop[P] = BatchLoop(
            prompt=sub.build_prompt(call.args),
            env_ids=branch.env_ids,
            tool_specs=tuple(sub.sub_tool_specs(branch.env_ids[0])),
            subagent=child_subagent,
            group_by=self._group_by,
            budget=self._budget,
            _id_prefix=f"{self._id_prefix}sub:{branch.branch_id}/{call.id}/",
        )
        self._children[f"{branch.branch_id}:{call.id}"] = _ChildLink(
            child=child, branch_id=branch.branch_id, call=call
        )

    def _pop_any(self) -> Request[P] | None:
        llm_head = self._ready_llm[0] if self._ready_llm else None
        tool_head = self._ready_tool[0] if self._ready_tool else None
        if llm_head is None:
            return self._ready_tool.popleft() if self._ready_tool else None
        if tool_head is None:
            return self._ready_llm.popleft()
        if llm_head.stats.seq <= tool_head.stats.seq:
            return self._ready_llm.popleft()
        return self._ready_tool.popleft()

    def _stats_for(self, branch: _Branch[P]) -> RequestStats:
        self._seq += 1
        return RequestStats(
            depth=branch.steps,
            split_depth=branch.split_depth,
            seq=self._seq,
        )

    def _finish(self, branch: _Branch[P], content: P) -> None:
        branch.final_content = content
        del self._branches[branch.branch_id]
        self._finals.append(branch)

    def _request_llm(self, branch: _Branch[P]) -> None:
        if self._recorded is not None:
            node = _history_node(self._recorded.root, branch.history_path)
            if node.assistant is not None:
                self._accept_assistant(branch, node.assistant)
                return
        max_steps = self._budget.max_steps_per_branch if self._budget else None
        if max_steps is not None and branch.steps >= max_steps:
            self._finish(
                branch,
                cast(P, f"(budget exhausted after {branch.steps} steps)"),
            )
            return
        stats = self._stats_for(branch)
        self._ready_llm.append(
            LlmRequest(
                request_id=f"{self._id_prefix}llm-{stats.seq}",
                branch_id=branch.branch_id,
                env_ids=branch.env_ids,
                messages=tuple(branch.messages),
                tools=self._specs,
                stats=stats,
            )
        )

    def _request_tool(self, branch: _Branch[P], call: ToolCall) -> None:
        stats = self._stats_for(branch)
        self._ready_tool.append(
            ToolRequest(
                request_id=f"{self._id_prefix}tool-{stats.seq}",
                branch_id=branch.branch_id,
                tool_call=call,
                env_ids=branch.env_ids,
                stats=stats,
            )
        )

    def _handle_llm(self, request: LlmRequest[P], completion: LlmCompletion[P]) -> None:
        branch = self._branches[request.branch_id]
        self._accept_assistant(
            branch,
            Message(role="assistant", content=completion.content, tool_calls=completion.tool_calls),
        )

    def _accept_assistant(self, branch: _Branch[P], message: Message[P]) -> None:
        if self._recorded is not None:
            node = _history_node(self._recorded.root, branch.history_path)
            self._set_history_node(
                branch.history_path,
                replace(
                    node,
                    assistant=node.assistant if node.assistant is not None else message,
                    env_ids=(*node.env_ids, *branch.env_ids),
                ),
            )
        branch.steps += 1
        branch.messages.append(message)
        if not message.tool_calls:
            self._finish(branch, message.content)
            return
        branch.turn_calls = message.tool_calls
        branch.turn_results = {}
        for call in message.tool_calls:
            if self._is_subagent_call(call):
                self._spawn_child(branch, call)
            else:
                self._request_tool(branch, call)

    def _handle_tool(self, request: ToolRequest, completion: ToolCompletion[P]) -> None:
        branch = self._branches[request.branch_id]
        returned = [env for env, _ in completion.results]
        if set(returned) != set(request.env_ids) or len(returned) != len(request.env_ids):
            raise ValueError(
                f"ToolCompletion for {request.request_id} must return exactly one "
                "result per env in the request"
            )
        # All fallible grouping/lookup happens before committing even this partial result.
        for _, output in completion.results:
            hash(self._group_by(output))
        candidate = {**branch.turn_results, request.tool_call.id: dict(completion.results)}
        plan = None
        if len(candidate) == len(branch.turn_calls):
            plan = self._plan_turn(branch, candidate)
        branch.turn_results = candidate
        if plan is not None:
            self._advance_turn(branch, plan)

    def _plan_turn(
        self, branch: _Branch[P], results: dict[str, dict[str, P]]
    ) -> list[tuple[list[str], int | None]]:
        groups: dict[tuple[Hashable, ...], list[str]] = {}
        for env in branch.env_ids:
            key = tuple(self._group_by(results[c.id][env]) for c in branch.turn_calls)
            groups.setdefault(key, []).append(env)
        node = _history_node(self._recorded.root, branch.history_path) if self._recorded else None
        plan: list[tuple[list[str], int | None]] = []
        for key, envs in groups.items():
            match = None
            if node is not None:
                for index, edge in enumerate(node.observations):
                    if key == tuple(self._group_by(o.tool_output) for o in edge.representative):
                        match = index
                        break
            plan.append((envs, match))
        return plan

    def _advance_turn(
        self, branch: _Branch[P], plan: list[tuple[list[str], int | None]] | None = None
    ) -> None:
        if plan is None:
            plan = self._plan_turn(branch, branch.turn_results)
        split = len(plan) > 1
        for envs, match in plan:
            path = branch.history_path
            messages = list(branch.messages)
            if self._recorded is not None:
                node = _history_node(self._recorded.root, path)
                evidence = tuple(
                    EnvObservation(
                        env,
                        tuple(
                            OutputRecord(branch.turn_results[c.id][env]) for c in branch.turn_calls
                        ),
                    )
                    for env in envs
                )
                edges = list(node.observations)
                if match is None:
                    match = len(edges)
                    edges.append(Observation(evidence[0].outputs, evidence, HistoryNode()))
                else:
                    edges[match] = replace(
                        edges[match], evidence=(*edges[match].evidence, *evidence)
                    )
                self._set_history_node(path, replace(node, observations=tuple(edges)))
                # Reuse the historical representative, not today's lossy-equivalent output.
                for call, output in zip(
                    branch.turn_calls, edges[match].representative, strict=True
                ):
                    messages.append(
                        Message(role="tool", content=output.tool_output, tool_call_id=call.id)
                    )
                path = (*path, match)
            else:
                self._append_tool_messages(
                    messages, branch.turn_calls, branch.turn_results, envs[0]
                )
            if split:
                self._next_branch += 1
                label = f"b{self._next_branch}"
                child: _Branch[P] = _Branch(
                    branch_id=f"{branch.branch_id}/{label}",
                    parent_id=branch.branch_id,
                    group_key=label,
                    env_ids=tuple(envs),
                    messages=messages,
                    split_depth=branch.split_depth + 1,
                    steps=branch.steps,
                    history_path=path,
                )
                self._branches[child.branch_id] = child
                self._partition.append(child)
                self._request_llm(child)
            else:
                branch.messages = messages
                branch.history_path = path
                branch.turn_calls = ()
                branch.turn_results = {}
                self._request_llm(branch)
        if split:
            del self._branches[branch.branch_id]

    @staticmethod
    def _append_tool_messages(
        messages: list[Message[P]],
        calls: tuple[ToolCall, ...],
        results: dict[str, dict[str, P]],
        representative_env: str,
    ) -> None:
        # With default value-equality grouping every env in the class holds an
        # equal value, so any representative is exact.  A lossy group_by lets
        # unequal values share a class; the conversation then records the
        # representative's payload.
        for call in calls:
            result = results[call.id][representative_env]
            messages.append(Message(role="tool", content=result, tool_call_id=call.id))

    def _finalize(self) -> BatchResult[P]:
        children_of: dict[str, dict[str, SplitNode]] = {}
        root_node: SplitNode | None = None
        for b in reversed(self._partition):
            node = SplitNode(env_ids=b.env_ids, children=children_of.get(b.branch_id, {}))
            if b.parent_id is None:
                root_node = node
            else:
                assert b.group_key is not None
                children_of.setdefault(b.parent_id, {})[b.group_key] = node
        assert root_node is not None
        classes = tuple(EnvClass(id=b.branch_id, env_ids=b.env_ids) for b in self._finals)
        results: dict[str, P] = {}
        for b in self._finals:
            for env in b.env_ids:
                results[env] = cast(P, b.final_content)
        return BatchResult(tree=root_node, classes=classes, results=results, history=self._recorded)


def run_batch(
    *,
    prompt: P,
    env_ids: Sequence[str],
    tool_specs: Sequence[ToolSpec],
    group_by: Callable[[P], Hashable] | None = None,
    budget: Budget | None = None,
) -> Generator[Request[P] | None, Completion[P] | None, BatchResult[P]]:
    """``BatchLoop`` as a generator, for pump-style drivers.

    Prime with ``next(gen)`` to get the first request; ``gen.send(completion)``
    delivers one completion and returns the next runnable request (or None
    when nothing is runnable but requests are outstanding); ``gen.send(None)``
    pops another queued request without delivering a completion.  When no work
    remains, ``send`` raises ``StopIteration`` whose ``value`` is the
    ``BatchResult``.
    """
    loop: BatchLoop[P] = BatchLoop(
        prompt=prompt,
        env_ids=env_ids,
        tool_specs=tool_specs,
        group_by=group_by,
        budget=budget,
    )

    def pump() -> Generator[Request[P] | None, Completion[P] | None, BatchResult[P]]:
        incoming: Completion[P] | None = None
        while True:
            if incoming is not None:
                loop.complete(incoming)
                incoming = None
            request = loop.next_request()
            if request is not None:
                incoming = yield request
            elif not loop.done:
                incoming = yield None
            else:
                return loop.result()

    return pump()


def run_to_completion(
    *,
    prompt: P,
    env_ids: Sequence[str],
    tools: Callable[[str], Sequence[Tool[P] | SubagentTool[P]]],
    complete_llm: Callable[[LlmRequest[P]], LlmCompletion[P]],
    complete_tool: Callable[[ToolRequest], ToolCompletion[P]] | None = None,
    group_by: Callable[[P], Hashable] | None = None,
    budget: Budget | None = None,
) -> BatchResult[P]:
    """Drive a ``BatchLoop`` synchronously, one request at a time.

    ``tools`` is a factory producing each environment's tool implementations;
    the first environment's specs describe the tools to the model, so specs
    must be identical across environments.  Any ``SubagentTool`` in the list
    is detected by ``isinstance`` and made the loop's subagent (the first one
    wins); it never reaches the tool executor.  ``complete_llm`` is the only
    provider dependency: adapt any LLM API to it.  ``complete_tool`` defaults
    to executing tools directly via the ``tools`` factory.  For parallel
    serving, drive ``BatchLoop`` yourself instead.
    """
    envs = tuple(env_ids)
    if not envs:
        raise ValueError("env_ids must be non-empty")
    specs: list[ToolSpec] = []
    subagent: SubagentTool[P] | None = None
    for t in tools(envs[0]):
        if isinstance(t, SubagentTool):
            if subagent is None:
                subagent = t
                specs.append(
                    ToolSpec(
                        name=t.name,
                        description=t.description,
                        parameters=dict(t.parameters),
                    )
                )
        else:
            specs.append(
                ToolSpec(name=t.name, description=t.description, parameters=dict(t.parameters))
            )
    if complete_tool is None:
        complete_tool = _local_tool_executor(tools)
    loop: BatchLoop[P] = BatchLoop(
        prompt=prompt,
        env_ids=envs,
        tool_specs=specs,
        subagent=subagent,
        group_by=group_by,
        budget=budget,
    )
    while not loop.done:
        request = loop.next_request()
        if request is None:
            raise RuntimeError("no runnable request but the loop is not done")
        if isinstance(request, LlmRequest):
            loop.complete(complete_llm(request))
        else:
            loop.complete(complete_tool(request))
    return loop.result()


def _local_tool_executor(
    tools: Callable[[str], Sequence[Tool[P] | SubagentTool[P]]],
) -> Callable[[ToolRequest], ToolCompletion[P]]:
    def execute(request: ToolRequest) -> ToolCompletion[P]:
        results: list[tuple[str, P]] = []
        for env in request.env_ids:
            impls = {t.name: t for t in tools(env) if not isinstance(t, SubagentTool)}
            impl = impls.get(request.tool_call.name)
            if impl is None:
                raise ValueError(f"env {env!r} has no tool named {request.tool_call.name!r}")
            results.append((env, impl(request.tool_call.args)))
        return ToolCompletion(request_id=request.request_id, results=tuple(results))

    return execute
