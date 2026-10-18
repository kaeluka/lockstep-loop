"""Core types and orchestrator for lockstep-loop.

lockstep-loop runs an agent's initial prompt across similar, read-only
environments, sharing LLM work while observation paths remain equivalent:

1. All environments start in one equivalence class sharing a single LLM
   conversation.
2. When the model makes a tool call, the call is evaluated on every
   environment still in the class (SIMD-style fan-out).
3. Tool results are compared and the class is refined: one sub-conversation
   per distinct result.  Branches then continue independently.
4. The process repeats per branch until the loop finishes. With recording,
   the immutable Agent includes prompt, structural configuration, and accumulated
   conversation trie; its Asset views expose leaf investigations and their complete paths.
   EnvClass and SplitNode remain partition-only summaries, not assets or agents.

LLM work is shared by observation path; tools still execute per environment.
Separate paths may publish the same value without remerging their conversations.

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

from .history import (
    Agent,
    AgentDescriptor,
    AgentQuery,
    AgentResolver,
    ChildAgentRun,
    ChildAgentUpdate,
    EnvObservation,
    HistoryNode,
    Observation,
    OutputRecord,
    RecordStore,
    SubagentContract,
    SubagentRecord,
    UserContinuation,
    _HistoryPath,
)

#: Conversation payload type: message content, tool results, final answers.
P = TypeVar("P")
_R = TypeVar("_R")
_T = TypeVar("_T")

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
    """Deterministically build the child's initial USER content; validate args here."""
    sub_tool_specs: Callable[[str], Sequence[ToolSpec]]
    """Spec provider for the child loop's tools (one env id -> all specs)."""
    max_depth: int = 1
    """Total spawn depth below this call site; nested definitions are clamped to the remainder."""
    recording: bool = False
    """Record children even without a recorded parent or resolver; not a cache key."""
    subagents: tuple[SubagentTool[P], ...] | None = None
    """Child allowlist. None preserves bounded self-recursion; () makes this a leaf."""


def _subagent_contract(sub: SubagentTool[P]) -> SubagentContract:
    return SubagentContract(
        ToolSpec(sub.name, sub.description, sub.parameters),
        sub.max_depth,
        None if sub.subagents is None else tuple(_subagent_contract(s) for s in sub.subagents),
    )


def _registration(
    specs: Sequence[ToolSpec], definitions: Sequence[SubagentTool[P]]
) -> tuple[tuple[ToolSpec, ...], dict[str, SubagentTool[P]], tuple[SubagentContract, ...]]:
    advertised = list(specs)
    by_name: dict[str, ToolSpec] = {}
    for spec in advertised:
        if spec.name in by_name:
            raise ValueError(f"duplicate tool name {spec.name!r}")
        by_name[spec.name] = spec
    registered: dict[str, SubagentTool[P]] = {}
    for sub in definitions:
        if sub.name in registered:
            raise ValueError(f"duplicate subagent name {sub.name!r}")
        if sub.max_depth < 1:
            raise ValueError("subagent.max_depth must be >= 1")
        spec = ToolSpec(sub.name, sub.description, sub.parameters)
        if sub.name in by_name:
            if by_name[sub.name] != spec:
                raise ValueError(f"conflicting ordinary/subagent spec for {sub.name!r}")
            # Exact redundant advertisements are allowed for legacy callers.
        else:
            advertised.append(spec)
            by_name[sub.name] = spec
        registered[sub.name] = sub
    return tuple(advertised), registered, tuple(_subagent_contract(s) for s in definitions)


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
    far. Children retain result-group representations as labels, disambiguated
    on representation collisions. Exact evidence and continuations live in
    ``Agent``, not this partition summary.

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
class BatchResult(Generic[P]):
    """Output of a completed ``BatchLoop``."""

    tree: SplitNode
    """Legacy partition summary; use history for replay."""

    classes: tuple[EnvClass, ...]
    """Final partition of environments by full tool-call trace."""

    results: Mapping[str, P]
    """Final assistant message per environment (identical within a class)."""

    history: Agent[P] | None = None
    child_histories: tuple[ChildAgentUpdate[P], ...] = ()

    @property
    def agent(self) -> Agent[P] | None:
        """Updated immutable agent snapshot; history is the compatibility field."""
        return self.history

    @property
    def child_agents(self) -> tuple[ChildAgentUpdate[P], ...]:
        return self.child_histories


@dataclass(frozen=True)
class Feedback(Generic[P]):
    """An explicit one-time intervention; Feedback(None) is not absence."""

    content: P


class FeedbackResolutionError(RuntimeError):
    """Replay could not locate an existing completed conversation. Terminal failure."""


class InvalidHistoryError(RuntimeError):
    """The selected path contains unsupported/ambiguous user continuations."""


_ExecutionError: TypeAlias = FeedbackResolutionError | InvalidHistoryError


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
    history_path: _HistoryPath = ()
    final_content: P | None = None
    turn_calls: tuple[ToolCall, ...] = ()
    turn_results: dict[str, dict[str, P]] = field(default_factory=dict)
    """tool_call_id -> (env_id -> result) for the current assistant turn."""


@dataclass(frozen=True)
class _ChildLink(Generic[P]):
    """A parent branch suspended on a spawned child loop."""

    child: BatchLoop[P]
    branch_id: str
    call: ToolCall
    ordinal: int
    base: AgentDescriptor[P] | None
    prior_runs: tuple[ChildAgentRun[P], ...]


class _Transition(Generic[P]):
    """Undo only touched builders/entries; never copy an execution or its history.

    One journal follows a completion through nested routing and parent rollup.
    Until commit, public history reads return the pre-transition immutable root.
    """

    def __init__(self) -> None:
        self.undo: list[Callable[[], None]] = []
        self.loops: list[BatchLoop[P]] = []
        self.roots: dict[int, Agent[P] | None] = {}
        self.branches: set[int] = set()
        self.removed_children: dict[int, BatchLoop[P]] = {}

    def attach(self, loop: BatchLoop[P]) -> None:
        if loop._transaction is self:
            return
        assert loop._transaction is None
        loop._transaction = self
        self.loops.append(loop)

    def watch_loop(self, loop: BatchLoop[P]) -> None:
        if id(loop) in self.roots:
            return
        self.roots[id(loop)] = loop._recorded
        before = (
            loop._recorded,
            loop._seq,
            loop._next_branch,
            loop._next_child,
            loop._out_llm,
            loop._out_tool,
            loop._feedback,
            loop._resolve_only,
            loop._resolution_base,
            loop._error,
        )

        def restore() -> None:
            (
                loop._recorded,
                loop._seq,
                loop._next_branch,
                loop._next_child,
                loop._out_llm,
                loop._out_tool,
                loop._feedback,
                loop._resolve_only,
                loop._resolution_base,
                loop._error,
            ) = before

        self.undo.append(restore)

    def watch_branch(self, branch: _Branch[P]) -> None:
        if id(branch) in self.branches:
            return
        self.branches.add(id(branch))
        # Messages only append, or are replaced with a new list on refinement.
        # Results dictionaries are likewise replaced, never edited in place.
        before = (
            branch.messages,
            branch.steps,
            branch.history_path,
            branch.final_content,
            branch.turn_calls,
            branch.turn_results,
        )
        length = len(branch.messages)

        def restore() -> None:
            (
                branch.messages,
                branch.steps,
                branch.history_path,
                branch.final_content,
                branch.turn_calls,
                branch.turn_results,
            ) = before
            del branch.messages[length:]

        self.undo.append(restore)

    def rollback(self) -> None:
        for restore in reversed(self.undo):
            restore()
        # Child iteration order is serving order. Restoring deleted dict entries
        # alone would move them to the end. Reorder references ONLY on rollback,
        # using internal invocation ordinals (never provider IDs).
        for loop in self.removed_children.values():
            ordered = sorted(loop._children.items(), key=lambda item: item[1].ordinal)
            loop._children.clear()
            loop._children.update(ordered)

    def close(self) -> None:
        for loop in self.loops:
            loop._transaction = None
        self.undo.clear()
        self.loops.clear()
        # Keep the immutable root anchors until this journal itself is released:
        # an inspection may already hold a reference to it while we detach loops.
        self.branches.clear()
        self.removed_children.clear()


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

    Recording: fresh roots opt in with ``recording=True``; supplying an ``Agent``
    implies replay and recording. A recorded parent or resolver records children;
    otherwise a child definition's ``recording=True`` is an optional switch, not
    part of cache identity. External provider/grouping policy is caller-owned.

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
        subagents: Sequence[SubagentTool[P]] | None = None,
        history_resolver: AgentResolver[P] | None = None,
        group_by: Callable[[P], Hashable] | None = None,
        budget: Budget | None = None,
        recording: bool = False,
        history: Agent[P] | None = None,
        record_store: RecordStore[P] | None = None,
        agent: Agent[P] | None = None,
        agent_resolver: AgentResolver[P] | None = None,
        feedback: Feedback[P] | None = None,
        _id_prefix: str = "",
        _resolve_only: bool = False,
        _defer_failure: bool = False,
        _unavailable_subagents: frozenset[str] = frozenset(),
    ) -> None:
        # Aliases converge immediately to one state/resolver. Never guess
        # between two supplied snapshots or maintain parallel representations.
        if agent is not None:
            if history is not None:
                raise ValueError("use agent or history, not both")
            history = agent
        if agent_resolver is not None:
            if history_resolver is not None:
                raise ValueError("use agent_resolver or history_resolver, not both")
            history_resolver = agent_resolver
        if history is not None and not isinstance(history, Agent):
            raise TypeError("agent/history must be an Agent snapshot, not an asset or endpoint")
        envs = tuple(env_ids)
        if not envs:
            raise ValueError("env_ids must be non-empty")
        if len(set(envs)) != len(envs):
            raise ValueError("env_ids must be unique")
        if subagent is not None and subagents is not None:
            raise ValueError("use subagent or subagents, not both")
        definitions = (
            tuple(subagents) if subagents is not None else ((subagent,) if subagent else ())
        )
        self._specs, self._subagents, sub_contracts = _registration(tool_specs, definitions)
        self._history_resolver = history_resolver
        self._unavailable_subagents = _unavailable_subagents - self._subagents.keys()
        independent = history_resolver is not None or any(s.recording for s in definitions)
        if independent and budget is not None:
            raise ValueError("recorded child histories do not support budgets")
        if feedback is not None and not isinstance(feedback, Feedback):
            raise TypeError("feedback must be Feedback(content), or None for no intervention")
        if feedback is not None and (history is None or len(envs) != 1):
            raise ValueError("feedback requires a supplied history and exactly one environment")
        if history is not None and "subagents" not in vars(history.descriptor):
            # Old pickles bypass __init__: do not let the new empty-tuple class
            # default silently erase their former singular registration.
            raise ValueError("migrate legacy history descriptor from subagent to subagents")
        if recording or history is not None:
            if budget is not None:
                raise ValueError("recorded history does not yet support budgets")
            if history is not None and (
                history.descriptor.prompt != prompt
                or history.descriptor.tools != self._specs
                or history.descriptor.custom_grouping != (group_by is not None)
                or history.descriptor.subagents != sub_contracts
            ):
                raise ValueError(
                    "incompatible history prompt, tools, grouping "
                    "or subagent configuration"
                )
            if history is None:
                store = record_store if record_store is not None else RecordStore[P]()
                history = Agent(
                    AgentDescriptor(
                        prompt,
                        self._specs,
                        group_by is not None,
                        store.put(HistoryNode()),
                        sub_contracts,
                    ),
                    store,
                )
            elif record_store is not None and record_store is not history.store:
                raise ValueError("replay uses the snapshot's record store")
        self._record_store = history.store if history else record_store
        if self._record_store is None and independent:
            self._record_store = RecordStore()
        self._custom_grouping = group_by is not None
        self._recorded = history
        self._feedback = feedback
        self._resolve_only = feedback is not None or _resolve_only
        self._resolution_base = history if self._resolve_only else None
        self._error: _ExecutionError | None = None
        self._next_branch = 0
        self._next_child = 0
        self._id_prefix = _id_prefix
        self._group_by: Callable[[P], Hashable] = (
            group_by if group_by is not None else lambda v: cast(Hashable, v)
        )
        self._budget = budget
        self._children: dict[str, _ChildLink[P]] = {}
        self._child_ids: dict[str, _ChildLink[P]] = {}

        self._lock = threading.RLock()
        self._transaction: _Transition[P] | None = None
        self._advancing = False
        self._seq = 0
        self._branches: dict[str, _Branch[P]] = {}
        self._partition: list[_Branch[P]] = []  # legacy per-run partition summary
        self._finals: list[_Branch[P]] = []
        self._child_histories: list[ChildAgentUpdate[P]] = []
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
            messages=[
                Message(role="user", content=history.descriptor.prompt if history else prompt)
            ],
            split_depth=0,
        )
        self._branches[root.branch_id] = root
        self._partition.append(root)
        self._run_atomic(lambda: self._request_llm(root))
        if not _defer_failure:
            self._raise_if_failed()

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

        def advance() -> Request[P] | None:
            self._rollup_children()
            if self._error is not None:
                return None
            req = self._pop_children(kind)
            if req is not None:
                return req
            if kind == "llm":
                request: Request[P] | None = (
                    self._popleft(self._ready_llm) if self._ready_llm else None
                )
            elif kind == "tool":
                request = self._popleft(self._ready_tool) if self._ready_tool else None
            else:
                request = self._pop_any()
            if request is None:
                return None
            self._put(self._outstanding, request.request_id, request)
            self._watch_loop()
            if isinstance(request, LlmRequest):
                self._out_llm += 1
            else:
                self._out_tool += 1
            return request

        with self._lock:
            # Without child rollup this is only a FIFO pull: no caller code runs.
            return self._run_atomic(advance, journal=bool(self._children))

    def complete(self, completion: Completion[P]) -> None:
        """Deliver a result for an outstanding request.

        Unknown/answered IDs, duplicate call IDs and invalid environment result
        sets raise ValueError; wrong completion kinds raise TypeError. Validation
        (including tool grouping) runs before fulfillment, leaving pending work
        recoverable. Child factories/constructors are prepared once before an
        assistant turn is recorded. Callback exceptions during advancement leave
        runtime builders, history roots and pending work unchanged; publishing a
        prepared transition does not rerun callbacks. Explicit stored terminal
        failures instead cancel pending work and result() raises their error.
        Caller side effects and unreachable append-only store writes are not rolled back.
        """

        def advance() -> None:
            self._raise_if_failed()
            link = self._child_ids.get(completion.request_id)
            if link is not None:
                child = link.child
                # A child final answer becomes a parent tool output. Validate its
                # grouping contract before fulfilling the child's final request.
                if isinstance(completion, LlmCompletion) and not completion.tool_calls:
                    hash(self._group_by(completion.content))
                try:
                    self._in_child(link, lambda child: child.complete(completion))
                except (FeedbackResolutionError, InvalidHistoryError):
                    if child.error is not None:
                        self._fail(child.error)
                    raise
                self._delete(self._child_ids, completion.request_id)
                self._watch_loop()
                if isinstance(completion, LlmCompletion):
                    self._out_llm -= 1
                else:
                    self._out_tool -= 1
                self._rollup_children()
                self._raise_if_failed()
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
                if any(c.name in self._unavailable_subagents for c in completion.tool_calls):
                    raise ValueError("subagent is unavailable in this child/depth")
                self._handle_llm(request, completion)
                self._raise_if_failed()
                self._delete(self._outstanding, completion.request_id)
                self._watch_loop()
                self._out_llm -= 1
            else:
                if not isinstance(completion, ToolCompletion):
                    raise TypeError(
                        f"ToolRequest {request.request_id} must be answered with ToolCompletion"
                    )
                self._handle_tool(request, completion)
                self._raise_if_failed()
                self._delete(self._outstanding, completion.request_id)
                self._watch_loop()
                self._out_tool -= 1

        with self._lock:
            self._run_atomic(advance)

    def _run_atomic(self, action: Callable[[], _R], *, journal: bool = True) -> _R:
        if self._advancing:
            raise RuntimeError("callbacks may inspect history, but must not reenter advancement")
        self._advancing = True
        try:
            if self._transaction is not None or not journal:
                # A child participates in its parent's journal; pure queue/state
                # operations at the root need no undo log at all.
                return action()
            transaction: _Transition[P] = _Transition()
            transaction.attach(self)
            try:
                return action()
            except (FeedbackResolutionError, InvalidHistoryError):
                # Explicit terminal failures commit cancellation; exceptions raised
                # by caller code without a stored loop error remain recoverable.
                if self._error is None:
                    transaction.rollback()
                raise
            except BaseException:
                transaction.rollback()
                raise
            finally:
                transaction.close()
        finally:
            self._advancing = False

    def _watch_loop(self) -> None:
        if self._transaction is not None:
            self._transaction.watch_loop(self)

    def _watch_branch(self, branch: _Branch[P]) -> None:
        if self._transaction is not None:
            self._transaction.watch_branch(branch)

    def _put(self, mapping: dict[str, _T], key: str, value: _T) -> None:
        if self._transaction is not None:
            existed = key in mapping
            old = mapping.get(key)

            def restore() -> None:
                if existed:
                    mapping[key] = cast(_T, old)
                else:
                    del mapping[key]

            self._transaction.undo.append(restore)
        mapping[key] = value

    def _delete(self, mapping: dict[str, _T], key: str) -> None:
        old = mapping.pop(key)
        if self._transaction is not None:
            self._transaction.undo.append(lambda: mapping.__setitem__(key, old))

    def _append(self, values: list[_T] | deque[_T], value: _T) -> None:
        values.append(value)
        if self._transaction is not None:

            def restore() -> None:
                values.pop()

            self._transaction.undo.append(restore)

    def _popleft(self, values: deque[_T]) -> _T:
        value = values.popleft()
        if self._transaction is not None:
            self._transaction.undo.append(lambda: values.appendleft(value))
        return value

    def _in_child(self, link: _ChildLink[P], action: Callable[[BatchLoop[P]], _R]) -> _R:
        if self._transaction is not None:
            self._transaction.attach(link.child)
        before = link.child._recorded
        result = action(link.child)
        if link.child._error is None and link.child._recorded is not before:
            self._sync_child_history(link)
        return result

    @property
    def error(self) -> _ExecutionError | None:
        """Terminal resolution/history failure, if any; validation errors are not terminal."""
        with self._lock:
            return self._error

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def _fail(self, error: _ExecutionError) -> None:
        self._watch_loop()
        if self._transaction is not None:
            before = (
                self._ready_llm,
                self._ready_tool,
                self._outstanding,
                self._child_ids,
                self._children,
                self._branches,
                self._finals,
            )

            def restore() -> None:
                (
                    self._ready_llm,
                    self._ready_tool,
                    self._outstanding,
                    self._child_ids,
                    self._children,
                    self._branches,
                    self._finals,
                ) = before

            self._transaction.undo.append(restore)
        self._error = error
        if self._resolution_base is not None:
            self._recorded = self._resolution_base
        self._ready_llm = deque()
        self._ready_tool = deque()
        self._outstanding = {}
        self._child_ids = {}
        self._children = {}
        self._branches = {}
        self._finals = []
        self._out_llm = self._out_tool = 0

    @property
    def agent(self) -> Agent[P] | None:
        """Capture the committed immutable agent snapshot, even during a caller callback.

        Child roots are linked incrementally on advancement, not by this getter.
        A reentrant read must never expose a root that might still roll back.
        """
        with self._lock:
            transaction = self._transaction
            if transaction is not None:
                return transaction.roots.get(id(self), self._recorded)
            return self._recorded

    @property
    def history(self) -> Agent[P] | None:
        """Compatibility spelling of agent; both return the same snapshot."""
        return self.agent

    def append_user_message(self, content: P) -> Agent[P]:
        """Fork an OPEN continuation after resolving one environment by replay.

        This legacy helper forks an already completed run; it does not resume
        this loop. All environments reaching the path in the returned snapshot
        inherit the user message. Prefer the constructor's feedback option to
        enforce replay-only resolution before resumption. No endpoint ID exists.
        """
        with self._lock:
            if self._advancing:
                raise RuntimeError(
                    "callbacks may inspect history, but must not reenter advancement"
                )
            self._raise_if_failed()
            if self._recorded is None or len(self._finals) != 1 or self._branches:
                raise ValueError("resolve a recorded single-environment run first")
            branch = self._finals[0]
            if len(branch.env_ids) != 1:
                raise ValueError("user continuations require one resolved environment")
            node = self._recorded._node(branch.history_path)
            edge = UserContinuation(
                Message(role="user", content=content),
                self._recorded.store.put(HistoryNode()),
                resolved_env_id=branch.env_ids[0],
            )
            snapshot = self._recorded._replace_node(
                branch.history_path,
                replace(node, user_continuations=(*node.user_continuations, edge)),
            )
            return snapshot

    def _set_history_node(self, path: _HistoryPath, node: HistoryNode[P]) -> None:
        assert self._recorded is not None
        self._watch_loop()
        self._recorded = self._recorded._replace_node(path, node)

    @property
    def done(self) -> bool:
        """True on success or terminal failure; result() raises on failure."""

        def advance() -> bool:
            self._rollup_children()
            return self._quiescent()

        with self._lock:
            return self._run_atomic(advance, journal=bool(self._children))

    def _quiescent(self) -> bool:
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

        Raises RuntimeError unless the loop is done, or the stored terminal
        execution error when resolution failed. No partial revised result is published.
        """

        def advance() -> BatchResult[P]:
            self._rollup_children()
            self._raise_if_failed()
            if not (
                not self._ready_llm
                and not self._ready_tool
                and not self._outstanding
                and not self._child_ids
                and not self._children
            ):
                raise RuntimeError("the loop is not done")
            return self._finalize()

        with self._lock:
            return self._run_atomic(advance)

    # -- internals (all called with self._lock held) --

    def _pop_children(self, kind: Literal["any", "llm", "tool"]) -> Request[P] | None:
        """Pull one request from a live child loop (children first)."""
        for link in self._children.values():
            child = link.child
            if not child._children and not (
                (kind != "tool" and child._ready_llm) or (kind != "llm" and child._ready_tool)
            ):
                continue
            req = self._in_child(link, lambda child: child.next_request(kind))
            if req is not None:
                self._put(self._child_ids, req.request_id, link)
                self._watch_loop()
                if isinstance(req, LlmRequest):
                    self._out_llm += 1
                else:
                    self._out_tool += 1
                return req
        return None

    def _sync_child_history(self, link: _ChildLink[P]) -> None:
        if self._recorded is None:
            return
        # Internal transition code uses the working child root. Public history
        # reads deliberately return the committed root while a journal is active.
        history = link.child._recorded
        assert history is not None
        branch = self._branches[link.branch_id]
        node = self._recorded._node(branch.history_path)
        entries = list(node.subagents)
        index = next((i for i, e in enumerate(entries) if e.call_id == link.call.id), None)
        run = ChildAgentRun(branch.env_ids, link.base, history.descriptor)
        entry = SubagentRecord(link.call.id, history.descriptor, (*link.prior_runs, run))
        if index is None:
            entries.append(entry)
        elif (
            entries[index].history.root == history.descriptor.root
            and len(entries[index].runs) == len(link.prior_runs) + 1
        ):
            return
        else:
            entries[index] = entry
        self._set_history_node(branch.history_path, replace(node, subagents=tuple(entries)))

    def _rollup_children(self) -> None:
        """Fold finished children into their parent's suspended branch."""
        while self._children and self._error is None:
            finished = []
            for key, link in self._children.items():
                # Leaf polling cannot advance anything. Do not journal or visit
                # that child's conversations just to inspect its queue state.
                done = (
                    self._in_child(link, lambda child: child.done)
                    if link.child._children
                    else link.child._quiescent()
                )
                if link.child._error is not None:
                    self._fail(link.child._error)
                    return
                if done:
                    finished.append((key, link))
            if not finished:
                return
            for key, link in finished:
                branch = self._branches[link.branch_id]
                result = self._in_child(link, lambda child: child.result())
                results = dict(result.results)
                candidate = {**branch.turn_results, link.call.id: results}
                plan = (
                    self._plan_turn(branch, candidate)
                    if len(candidate) == len(branch.turn_calls)
                    else None
                )
                self._watch_branch(branch)
                branch.turn_results = candidate
                for update in result.child_histories:
                    self._append(self._child_histories, update)
                if result.history is not None:
                    self._append(
                        self._child_histories,
                        ChildAgentUpdate(
                            link.child._id_prefix,
                            branch.branch_id,
                            branch.steps,
                            link.call,
                            branch.env_ids,
                            link.base,
                            result.history,
                        ),
                    )
                if self._transaction is not None:
                    self._transaction.removed_children[id(self)] = self
                self._delete(self._children, key)
                if plan is not None:
                    self._advance_turn(branch, plan)
                    if self._error is not None:
                        return
            # Replayed parent turns may spawn children already finished by replay.
            # Settle those too, without producing a false stalled frontier.

    def _is_subagent_call(self, call: ToolCall) -> bool:
        return call.name in self._subagents

    def _prepare_child(
        self, branch: _Branch[P], call: ToolCall, child_key: str, ordinal: int
    ) -> _ChildLink[P] | _ExecutionError:
        """Build once; distinguish detected failure from an exception in caller code."""
        sub = self._subagents[call.name]
        declared = sub.subagents if sub.subagents is not None else (sub,)
        nested = (
            tuple(replace(s, max_depth=min(s.max_depth, sub.max_depth - 1)) for s in declared)
            if sub.max_depth > 1
            else ()
        )
        reserved = self._unavailable_subagents | self._subagents.keys() | {s.name for s in declared}
        allowed = {s.name for s in nested}
        unavailable = frozenset(reserved - allowed)
        prompt = sub.build_prompt(call.args)
        child_specs, _, child_contracts = _registration(
            tuple(s for s in sub.sub_tool_specs(branch.env_ids[0]) if s.name not in unavailable),
            nested,
        )
        recording = (
            sub.recording or self._recorded is not None or self._history_resolver is not None
        )
        previous = None
        prior_runs: tuple[ChildAgentRun[P], ...] = ()
        if self._recorded is not None:
            node = self._recorded._node(branch.history_path)
            entry = next((e for e in node.subagents if e.call_id == call.id), None)
            if entry is not None:
                previous = Agent(entry.agent_descriptor, self._recorded.store)
                prior_runs = entry.runs or (ChildAgentRun((), None, entry.agent_descriptor),)
        # A recorded parent pins its revision. External cache changes must not
        # reinterpret it, nor manufacture a missing feedback-resolution target.
        if self._resolve_only and previous is None:
            return FeedbackResolutionError("no recorded subagent conversation")
        if previous is None and self._history_resolver is not None:
            assert self._record_store is not None
            selected = self._history_resolver(
                AgentQuery(prompt, child_specs, self._custom_grouping, child_contracts)
            )
            if selected is not None:
                if not isinstance(selected, AgentDescriptor):
                    raise TypeError(
                        "agent_resolver (history_resolver) must return AgentDescriptor or None"
                    )
                previous = Agent(selected, self._record_store)
        child: BatchLoop[P] = BatchLoop(
            prompt=prompt,
            env_ids=branch.env_ids,
            tool_specs=child_specs,
            subagents=nested,
            history_resolver=self._history_resolver,
            group_by=self._group_by if self._custom_grouping else None,
            budget=self._budget,
            recording=recording,
            history=previous,
            record_store=self._record_store,
            _id_prefix=f"{self._id_prefix}{child_key}",
            _resolve_only=self._resolve_only,
            _defer_failure=True,
            _unavailable_subagents=unavailable,
        )
        if child._error is not None:
            return child._error
        return _ChildLink(
            child=child,
            branch_id=branch.branch_id,
            call=call,
            ordinal=ordinal,
            base=previous.descriptor if previous else None,
            prior_runs=prior_runs,
        )

    def _pop_any(self) -> Request[P] | None:
        llm_head = self._ready_llm[0] if self._ready_llm else None
        tool_head = self._ready_tool[0] if self._ready_tool else None
        if llm_head is None:
            return self._popleft(self._ready_tool) if self._ready_tool else None
        if tool_head is None:
            return self._popleft(self._ready_llm)
        if llm_head.stats.seq <= tool_head.stats.seq:
            return self._popleft(self._ready_llm)
        return self._popleft(self._ready_tool)

    def _stats_for(self, branch: _Branch[P]) -> RequestStats:
        self._watch_loop()
        self._seq += 1
        return RequestStats(
            depth=branch.steps,
            split_depth=branch.split_depth,
            seq=self._seq,
        )

    def _finish(self, branch: _Branch[P], content: P) -> None:
        self._watch_branch(branch)
        branch.final_content = content
        self._delete(self._branches, branch.branch_id)
        self._append(self._finals, branch)

    def _request_llm(self, branch: _Branch[P]) -> None:
        if self._error is not None:
            return
        while self._recorded is not None:
            node = self._recorded._node(branch.history_path)
            if node.user_continuations and (
                len(node.user_continuations) != 1
                or node.assistant is None
                or node.assistant.tool_calls
                or node.user_continuations[0].message.role != "user"
                or node.user_continuations[0].message.tool_calls
                or node.user_continuations[0].message.tool_call_id is not None
            ):
                self._fail(
                    InvalidHistoryError("expected one user continuation after a final answer")
                )
                return
            if node.assistant is None:
                break
            if not self._accept_assistant(branch, node.assistant):
                return
        if self._resolve_only:
            self._fail(FeedbackResolutionError("no recorded completed conversation at this path"))
            return
        max_steps = self._budget.max_steps_per_branch if self._budget else None
        if max_steps is not None and branch.steps >= max_steps:
            self._finish(
                branch,
                cast(P, f"(budget exhausted after {branch.steps} steps)"),
            )
            return
        stats = self._stats_for(branch)
        self._append(
            self._ready_llm,
            LlmRequest(
                request_id=f"{self._id_prefix}llm-{stats.seq}",
                branch_id=branch.branch_id,
                env_ids=branch.env_ids,
                messages=tuple(branch.messages),
                tools=self._specs,
                stats=stats,
            ),
        )

    def _request_tool(self, branch: _Branch[P], call: ToolCall) -> None:
        stats = self._stats_for(branch)
        self._append(
            self._ready_tool,
            ToolRequest(
                request_id=f"{self._id_prefix}tool-{stats.seq}",
                branch_id=branch.branch_id,
                tool_call=call,
                env_ids=branch.env_ids,
                stats=stats,
            ),
        )

    def _handle_llm(self, request: LlmRequest[P], completion: LlmCompletion[P]) -> None:
        branch = self._branches[request.branch_id]
        if self._accept_assistant(
            branch,
            Message(role="assistant", content=completion.content, tool_calls=completion.tool_calls),
        ):
            self._request_llm(branch)

    def _accept_assistant(self, branch: _Branch[P], message: Message[P]) -> bool:
        """Return True to schedule the next node after a user continuation."""
        if any(c.name in self._unavailable_subagents for c in message.tool_calls):
            self._fail(InvalidHistoryError("recorded call targets an unavailable subagent"))
            return False
        prepared: dict[str, tuple[str, _ChildLink[P]]] = {}
        next_child = self._next_child
        for call in message.tool_calls:
            if self._is_subagent_call(call):
                # Reserve locally. A rejected turn must not issue IDs, attach
                # children, or enqueue its ordinary calls either.
                next_child += 1
                key = f"sub:{next_child}/"
                child = self._prepare_child(branch, call, key, next_child)
                if isinstance(child, (FeedbackResolutionError, InvalidHistoryError)):
                    self._fail(child)
                    return False
                prepared[call.id] = (key, child)
        self._watch_branch(branch)
        if self._recorded is not None:
            node = self._recorded._node(branch.history_path)
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
            if self._recorded is not None:
                node = self._recorded._node(branch.history_path)
                if not node.user_continuations and self._feedback is not None:
                    edge = UserContinuation(
                        Message(role="user", content=self._feedback.content),
                        self._recorded.store.put(HistoryNode()),
                        resolved_env_id=branch.env_ids[0],
                    )
                    node = replace(node, user_continuations=(edge,))
                    self._set_history_node(branch.history_path, node)
                    self._feedback = None  # once per explicit operation, never by answer equality
                    self._resolve_only = False
                    self._resolution_base = None
                if node.user_continuations:
                    branch.messages.append(node.user_continuations[0].message)
                    branch.history_path = (*branch.history_path, ("user", 0))
                    return True
            self._finish(branch, message.content)
            return False
        branch.turn_calls = message.tool_calls
        branch.turn_results = {}
        self._watch_loop()
        self._next_child = next_child
        for call in message.tool_calls:
            if call.id in prepared:
                key, link = prepared[call.id]
                self._put(self._children, key, link)
                self._sync_child_history(link)
            else:
                self._request_tool(branch, call)
        return False

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
        self._watch_branch(branch)
        branch.turn_results = candidate
        if plan is not None:
            self._advance_turn(branch, plan)

    def _plan_turn(
        self, branch: _Branch[P], results: dict[str, dict[str, P]]
    ) -> list[tuple[list[str], int | None, str]]:
        groups: dict[tuple[Hashable, ...], list[str]] = {}
        for env in branch.env_ids:
            key = tuple(self._group_by(results[c.id][env]) for c in branch.turn_calls)
            groups.setdefault(key, []).append(env)
        node = self._recorded._node(branch.history_path) if self._recorded else None
        plan: list[tuple[list[str], int | None, str]] = []
        labels: set[str] = set()
        reserved = {str(key) for key in groups} if len(groups) > 1 else set()
        for key, envs in groups.items():
            match = None
            if node is not None:
                for index, edge in enumerate(node.observations):
                    if key == tuple(self._group_by(o.tool_output) for o in edge.representative):
                        match = index
                        break
            label = str(key) if len(groups) > 1 else ""
            if label in labels:
                suffix = 2
                while f"{label} [collision {suffix}]" in labels | reserved:
                    suffix += 1
                label = f"{label} [collision {suffix}]"
            labels.add(label)
            plan.append((envs, match, label))
        return plan

    def _advance_turn(
        self, branch: _Branch[P], plan: list[tuple[list[str], int | None, str]] | None = None
    ) -> None:
        if plan is None:
            plan = self._plan_turn(branch, branch.turn_results)
        if self._resolve_only and any(match is None for _, match, _ in plan):
            self._fail(FeedbackResolutionError("unseen tool observation while resolving feedback"))
            return
        split = len(plan) > 1
        for envs, match, group_label in plan:
            path = branch.history_path
            messages = list(branch.messages)
            if self._recorded is not None:
                node = self._recorded._node(path)
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
                    edges.append(
                        Observation(
                            evidence[0].outputs, evidence, self._recorded.store.put(HistoryNode())
                        )
                    )
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
                path = (*path, ("observation", match))
            else:
                self._append_tool_messages(
                    messages, branch.turn_calls, branch.turn_results, envs[0]
                )
            if split:
                self._watch_loop()
                self._next_branch += 1
                label = f"b{self._next_branch}"
                child: _Branch[P] = _Branch(
                    branch_id=f"{branch.branch_id}/{label}",
                    parent_id=branch.branch_id,
                    group_key=group_label,
                    env_ids=tuple(envs),
                    messages=messages,
                    split_depth=branch.split_depth + 1,
                    steps=branch.steps,
                    history_path=path,
                )
                self._put(self._branches, child.branch_id, child)
                self._append(self._partition, child)
                self._request_llm(child)
            else:
                branch.messages = messages
                branch.history_path = path
                branch.turn_calls = ()
                branch.turn_results = {}
                self._request_llm(branch)
            if self._error is not None:
                return
        if split:
            self._delete(self._branches, branch.branch_id)

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
        return BatchResult(
            tree=root_node,
            classes=classes,
            results=results,
            history=self._recorded,
            child_histories=tuple(self._child_histories),
        )


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
    must be identical across environments. Every ``SubagentTool`` is registered
    as a reserved tool; none reaches the ordinary tool executor. Duplicate names,
    including callable/subagent collisions, are rejected.  ``complete_llm`` is the only
    provider dependency: adapt any LLM API to it.  ``complete_tool`` defaults
    to executing tools directly via the ``tools`` factory.  For parallel
    serving, drive ``BatchLoop`` yourself instead.
    """
    envs = tuple(env_ids)
    if not envs:
        raise ValueError("env_ids must be non-empty")
    specs: list[ToolSpec] = []
    subagents: list[SubagentTool[P]] = []
    names: set[str] = set()
    for t in tools(envs[0]):
        if t.name in names:
            raise ValueError(f"duplicate tool implementation {t.name!r}")
        names.add(t.name)
        specs.append(ToolSpec(t.name, t.description, dict(t.parameters)))
        if isinstance(t, SubagentTool):
            subagents.append(t)
    if complete_tool is None:
        complete_tool = _local_tool_executor(tools)
    loop: BatchLoop[P] = BatchLoop(
        prompt=prompt,
        env_ids=envs,
        tool_specs=specs,
        subagents=subagents,
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
            impls: dict[str, Tool[P]] = {}
            names: set[str] = set()
            for tool in tools(env):
                if tool.name in names:
                    raise ValueError(f"duplicate tool implementation {tool.name!r}")
                names.add(tool.name)
                if not isinstance(tool, SubagentTool):
                    impls[tool.name] = tool
            impl = impls.get(request.tool_call.name)
            if impl is None:
                raise ValueError(f"env {env!r} has no tool named {request.tool_call.name!r}")
            results.append((env, impl(request.tool_call.args)))
        return ToolCompletion(request_id=request.request_id, results=tuple(results))

    return execute
