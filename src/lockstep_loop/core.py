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

import hashlib
import threading
from collections import deque
from collections.abc import Callable, Generator, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast, overload

#: Conversation payload type: message content, tool results, final answers.
P = TypeVar("P")

#: Covariant variant for the ``Tool`` protocol (P only appears in return position).
P_co = TypeVar("P_co", covariant=True)


class Tool(Protocol[P_co]):
    """A single environment's implementation of one tool.

    Only ``run_to_completion``'s default executor needs concrete tools;
    ``BatchLoop`` itself only ever sees ``ToolSpec`` metadata.

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

    env_count: int
    """Environments riding on this request's branch."""

    seq: int
    """Monotone issue counter, for deterministic tie-breaks."""


@dataclass(frozen=True)
class LlmRequest(Generic[P]):
    """The loop needs one LLM completion for a branch."""

    request_id: str
    branch_id: str
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
    far.  Children are keyed by the tool-result group key that caused the
    split.

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
    """The split trie; the persisted form that makes re-runs cheap."""

    classes: tuple[EnvClass, ...]
    """Final partition of environments by full tool-call trace."""

    results: Mapping[str, P]
    """Final assistant message per environment (identical within a class)."""


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
    final_content: P | None = None
    turn_calls: tuple[ToolCall, ...] = ()
    turn_results: dict[str, dict[str, P]] = field(default_factory=dict)
    """tool_call_id -> (env_id -> result) for the current assistant turn."""


def _short_hash(key: tuple[Hashable, ...]) -> str:
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]


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
        group_by: Callable[[P], Hashable] | None = None,
        budget: Budget | None = None,
    ) -> None:
        envs = tuple(env_ids)
        if not envs:
            raise ValueError("env_ids must be non-empty")
        if len(set(envs)) != len(envs):
            raise ValueError("env_ids must be unique")
        self._specs = tuple(tool_specs)
        self._group_by: Callable[[P], Hashable] = (
            group_by if group_by is not None else lambda v: cast(Hashable, v)
        )
        self._budget = budget

        self._lock = threading.Lock()
        self._seq = 0
        self._branches: dict[str, _Branch[P]] = {}
        self._history: list[_Branch[P]] = []  # every branch ever created, in order
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
            messages=[Message(role="user", content=prompt)],
            split_depth=0,
        )
        self._branches[root.branch_id] = root
        self._history.append(root)
        self._request_llm(root)

    @overload
    def next_request(self, kind: Literal["llm"]) -> LlmRequest[P] | None: ...

    @overload
    def next_request(self, kind: Literal["tool"]) -> ToolRequest | None: ...

    @overload
    def next_request(self, kind: Literal["any"] = "any") -> Request[P] | None: ...

    def next_request(
        self, kind: Literal["any", "llm", "tool"] = "any"
    ) -> Request[P] | None:
        """Pop one runnable request, or None if none of that kind is runnable.

        None is not a termination signal — work may still be outstanding, or
        queued under another kind.  ``done`` is the only termination signal.
        Separate consumer modules can each pull their own kind; an LLM pool
        and a tool executor never see each other's requests.
        """
        with self._lock:
            if kind == "llm":
                req: Request[P] | None = (
                    self._ready_llm.popleft() if self._ready_llm else None
                )
            elif kind == "tool":
                req = self._ready_tool.popleft() if self._ready_tool else None
            else:
                req = self._pop_any()
            if req is None:
                return None
            self._outstanding[req.request_id] = req
            if isinstance(req, LlmRequest):
                self._out_llm += 1
            else:
                self._out_tool += 1
            return req

    def complete(self, completion: Completion[P]) -> None:
        """Deliver a result for an outstanding request.

        Raises ValueError for unknown or already-answered request ids and
        TypeError when the completion type does not match the request.  Both
        error paths leave the loop's state untouched.
        """
        with self._lock:
            request = self._outstanding.get(completion.request_id)
            if request is None:
                raise ValueError(
                    f"no outstanding request with id {completion.request_id!r}"
                )
            if isinstance(request, LlmRequest):
                if not isinstance(completion, LlmCompletion):
                    raise TypeError(
                        f"LlmRequest {request.request_id} must be answered "
                        "with LlmCompletion"
                    )
                del self._outstanding[completion.request_id]
                self._out_llm -= 1
                self._handle_llm(request, completion)
            else:
                if not isinstance(completion, ToolCompletion):
                    raise TypeError(
                        f"ToolRequest {request.request_id} must be answered "
                        "with ToolCompletion"
                    )
                del self._outstanding[completion.request_id]
                self._out_tool -= 1
                self._handle_tool(request, completion)

    @property
    def done(self) -> bool:
        """True when nothing is runnable and nothing is outstanding."""
        with self._lock:
            return (
                not self._ready_llm and not self._ready_tool and not self._outstanding
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
            if not (
                not self._ready_llm and not self._ready_tool and not self._outstanding
            ):
                raise RuntimeError("the loop is not done")
            return self._finalize()

    # -- internals (all called with self._lock held) --

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
            env_count=len(branch.env_ids),
            seq=self._seq,
        )

    def _finish(self, branch: _Branch[P], content: P) -> None:
        branch.final_content = content
        del self._branches[branch.branch_id]
        self._finals.append(branch)

    def _request_llm(self, branch: _Branch[P]) -> None:
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
                request_id=f"llm-{stats.seq}",
                branch_id=branch.branch_id,
                messages=tuple(branch.messages),
                tools=self._specs,
                stats=stats,
            )
        )

    def _request_tool(self, branch: _Branch[P], call: ToolCall) -> None:
        stats = self._stats_for(branch)
        self._ready_tool.append(
            ToolRequest(
                request_id=f"tool-{stats.seq}",
                branch_id=branch.branch_id,
                tool_call=call,
                env_ids=branch.env_ids,
                stats=stats,
            )
        )

    def _handle_llm(self, request: LlmRequest[P], completion: LlmCompletion[P]) -> None:
        branch = self._branches[request.branch_id]
        branch.steps += 1
        branch.messages.append(
            Message(
                role="assistant", content=completion.content, tool_calls=completion.tool_calls
            )
        )
        if not completion.tool_calls:
            self._finish(branch, completion.content)
            return
        branch.turn_calls = completion.tool_calls
        branch.turn_results = {}
        for call in completion.tool_calls:
            self._request_tool(branch, call)

    def _handle_tool(self, request: ToolRequest, completion: ToolCompletion[P]) -> None:
        branch = self._branches[request.branch_id]
        returned = [env for env, _ in completion.results]
        if set(returned) != set(request.env_ids) or len(returned) != len(request.env_ids):
            raise ValueError(
                f"ToolCompletion for {request.request_id} must return exactly one "
                "result per env in the request"
            )
        branch.turn_results[request.tool_call.id] = dict(completion.results)
        if len(branch.turn_results) < len(branch.turn_calls):
            return  # other calls of this assistant turn are still in flight
        self._advance_turn(branch)

    def _advance_turn(self, branch: _Branch[P]) -> None:
        # All calls of the turn have results on every env: refine the partition
        # by the tuple of results (one group key per call).
        groups: dict[tuple[Hashable, ...], list[str]] = {}
        for env in branch.env_ids:
            key = tuple(
                self._group_by(branch.turn_results[call.id][env])
                for call in branch.turn_calls
            )
            groups.setdefault(key, []).append(env)
        if len(groups) == 1:
            self._append_tool_messages(
                branch.messages, branch.turn_calls, branch.turn_results, branch.env_ids[0]
            )
            branch.turn_calls = ()
            branch.turn_results = {}
            self._request_llm(branch)
            return
        for key, envs in groups.items():
            child: _Branch[P] = _Branch(
                branch_id=f"{branch.branch_id}/{_short_hash(key)}",
                parent_id=branch.branch_id,
                group_key=str(key),
                env_ids=tuple(envs),
                messages=list(branch.messages),
                split_depth=branch.split_depth + 1,
                steps=branch.steps,
            )
            self._append_tool_messages(
                child.messages, branch.turn_calls, branch.turn_results, envs[0]
            )
            self._branches[child.branch_id] = child
            self._history.append(child)
            self._request_llm(child)
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
        for b in reversed(self._history):
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
            assert b.final_content is not None
            for env in b.env_ids:
                results[env] = b.final_content
        return BatchResult(tree=root_node, classes=classes, results=results)


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
    tools: Callable[[str], Sequence[Tool[P]]],
    complete_llm: Callable[[LlmRequest[P]], LlmCompletion[P]],
    complete_tool: Callable[[ToolRequest], ToolCompletion[P]] | None = None,
    group_by: Callable[[P], Hashable] | None = None,
    budget: Budget | None = None,
) -> BatchResult[P]:
    """Drive a ``BatchLoop`` synchronously, one request at a time.

    ``tools`` is a factory producing each environment's tool implementations;
    the first environment's specs describe the tools to the model, so specs
    must be identical across environments.  ``complete_llm`` is the only
    provider dependency: adapt any LLM API to it.  ``complete_tool`` defaults
    to executing tools directly via the ``tools`` factory.  For parallel
    serving, drive ``BatchLoop`` yourself instead.
    """
    envs = tuple(env_ids)
    if not envs:
        raise ValueError("env_ids must be non-empty")
    specs = tuple(
        ToolSpec(name=t.name, description=t.description, parameters=dict(t.parameters))
        for t in tools(envs[0])
    )
    if complete_tool is None:
        complete_tool = _local_tool_executor(tools)
    loop: BatchLoop[P] = BatchLoop(
        prompt=prompt, env_ids=envs, tool_specs=specs, group_by=group_by, budget=budget
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
    tools: Callable[[str], Sequence[Tool[P]]],
) -> Callable[[ToolRequest], ToolCompletion[P]]:
    def execute(request: ToolRequest) -> ToolCompletion[P]:
        results: list[tuple[str, P]] = []
        for env in request.env_ids:
            impls = {t.name: t for t in tools(env)}
            impl = impls.get(request.tool_call.name)
            if impl is None:
                raise ValueError(f"env {env!r} has no tool named {request.tool_call.name!r}")
            results.append((env, impl(request.tool_call.args)))
        return ToolCompletion(request_id=request.request_id, results=tuple(results))

    return execute
