"""Core types and entrypoint for lockstep-loop.

lockstep-loop runs a single agent prompt across many similar, read-only
environments without generating shared tokens twice:

1. All environments start in one equivalence class sharing a single LLM
   conversation.
2. When the model makes a tool call, the call is evaluated on every
   environment still in the class (SIMD-style fan-out).
3. Results are canonicalized (``group_by``) and the class is refined: one
   sub-conversation per distinct result key.  Branches then continue
   independently.
4. The process repeats per branch until the loop finishes, yielding a
   prefix-sharing trie whose leaves are equivalence classes of environments.

Cost scales with the number of distinct tool-call traces, not with the
number of environments.

Provider-agnostic by construction: the library never calls an LLM (or a
tool) itself.  ``run_batch`` is a generator that *yields* requests — LLM
completions and tool fan-outs — and the caller answers them in whatever
order they finish.  A slow provider (or tool) delays only its own branch.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

#: What a tool returns for a single environment.
ToolResult: TypeAlias = str | Mapping[str, Any]


class Tool(Protocol):
    """A single environment's implementation of one tool.

    Invariants (enforced by ``run_batch``):

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

    def __call__(self, args: Mapping[str, Any]) -> ToolResult: ...


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool description.

    Extracted from ``Tool`` and handed to the caller inside every
    ``LlmRequest``; the caller's provider adapter converts it to whatever
    schema its LLM API expects.
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
class Message:
    """A provider-neutral conversation message.

    ``tool_calls`` is only set on assistant messages; ``tool_call_id`` only
    on tool messages.
    """

    role: Literal["user", "assistant", "tool"]
    content: str
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
class LlmRequest:
    """The loop needs one LLM completion for a branch."""

    request_id: str
    branch_id: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    stats: RequestStats


@dataclass(frozen=True)
class ToolRequest:
    """The loop needs one tool call fanned out over a set of environments.

    The caller executes ``tool_call`` once per id in ``env_ids`` and returns
    every result in a single ``ToolCompletion``.  The results drive
    partition refinement: environments whose results canonicalize to equal
    group keys stay in one conversation.
    """

    request_id: str
    branch_id: str
    tool_call: ToolCall
    env_ids: tuple[str, ...]
    stats: RequestStats


#: Work the loop asks the caller to perform.
Request: TypeAlias = LlmRequest | ToolRequest


@dataclass(frozen=True)
class LlmCompletion:
    """The model's reply to an ``LlmRequest``.

    Empty ``tool_calls`` ends the branch and ``content`` becomes its final
    answer.
    """

    request_id: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ToolCompletion:
    """Fanned-out results answering a ``ToolRequest``.

    Exactly one ``(env_id, result)`` pair per env in the request.
    """

    request_id: str
    results: tuple[tuple[str, ToolResult], ...]


#: The caller's answer to a ``Request``.
Completion: TypeAlias = LlmCompletion | ToolCompletion


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
    far.  Children are keyed by the canonicalized tool-result group key that
    caused the split.

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
class BatchResult:
    """Output of a completed ``run_batch`` invocation."""

    tree: SplitNode
    """The split trie; the persisted form that makes re-runs cheap."""

    classes: tuple[EnvClass, ...]
    """Final partition of environments by full tool-call trace."""

    results: Mapping[str, str]
    """Final assistant message per environment (identical within a class)."""


@dataclass
class _Branch:
    """Mutable internal state for one live conversation branch."""

    branch_id: str
    parent_id: str | None
    group_key: str | None
    """Edge label from the parent (None for the root)."""
    env_ids: tuple[str, ...]
    messages: list[Message]
    split_depth: int
    steps: int = 0
    final_content: str | None = None
    turn_calls: tuple[ToolCall, ...] = ()
    turn_results: dict[str, dict[str, ToolResult]] = field(default_factory=dict)
    """tool_call_id -> (env_id -> result) for the current assistant turn."""


def _default_group_by(result: ToolResult) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, sort_keys=True, default=repr)


def _render(result: ToolResult) -> str:
    return result if isinstance(result, str) else json.dumps(result, sort_keys=True, default=repr)


def _short_hash(key: tuple[str, ...]) -> str:
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]


def _validate_tool_specs(
    env_ids: tuple[str, ...], tools: Callable[[str], Sequence[Tool]]
) -> tuple[ToolSpec, ...]:
    """Extract tool specs; raise unless they are identical across environments."""
    first: tuple[ToolSpec, ...] | None = None
    for env in env_ids:
        specs = tuple(
            ToolSpec(name=t.name, description=t.description, parameters=dict(t.parameters))
            for t in tools(env)
        )
        if first is None:
            first = specs
        elif specs != first:
            raise ValueError(
                f"tool specs of env {env!r} differ from env {env_ids[0]!r}; "
                "tool specs must be identical across environments"
            )
    assert first is not None
    return first


def run_batch(
    *,
    prompt: str,
    env_ids: Sequence[str],
    tools: Callable[[str], Sequence[Tool]],
    group_by: Callable[[ToolResult], str] | None = None,
    budget: Budget | None = None,
) -> Generator[Request | None, Completion | None, BatchResult]:
    """Run one prompt across many environments in lockstep.

    Args:
        prompt: The single prompt shared by every environment.
        env_ids: Identifiers of the environments to run against.
        tools: Factory producing each environment's tool implementations.
            Tool specs (name/description/parameters) must be identical across
            environments; only behavior may differ.
        group_by: Canonicalizes a tool result into a group key.  Environments
            whose results share a key stay in one conversation; distinct keys
            split it.  ``None`` means "use the default canonicalizer":
            strings as-is, mappings as sorted JSON.  Normalize volatile
            fields here, or near-identical environments split on noise.
        budget: Per-branch limits.

    Returns:
        A generator that *is* the loop.  Pump it:

        * prime with ``next(gen)`` to get the first request;
        * ``gen.send(completion)`` delivers one completion and returns the
          next runnable request.  Deliver completions **in arrival order** —
          the protocol is deliberately out-of-order so a slow provider call
          delays only its own branch;
        * the generator yields ``None`` when nothing is runnable but
          requests are still outstanding — keep waiting for completions;
        * ``gen.send(None)`` pops one more queued request without delivering
          a completion (use to drain after several branches became runnable);
        * when no work remains, ``send`` raises ``StopIteration`` whose
          ``value`` is the ``BatchResult``.

        At most one LLM request is outstanding per branch; a branch's tool
        requests per assistant turn equal the number of calls in that turn.

    Invariants:
        * No token is generated twice for environments that remain in the
          same equivalence class.
        * Classes only refine (split), never merge.
    """
    envs = tuple(env_ids)
    if not envs:
        raise ValueError("env_ids must be non-empty")
    if len(set(envs)) != len(envs):
        raise ValueError("env_ids must be unique")
    specs = _validate_tool_specs(envs, tools)
    return _loop(
        prompt=prompt,
        env_ids=envs,
        specs=specs,
        group_by=group_by if group_by is not None else _default_group_by,
        budget=budget,
    )


def _loop(
    *,
    prompt: str,
    env_ids: tuple[str, ...],
    specs: tuple[ToolSpec, ...],
    group_by: Callable[[ToolResult], str],
    budget: Budget | None,
) -> Generator[Request | None, Completion | None, BatchResult]:
    seq = 0
    branches: dict[str, _Branch] = {}
    history: list[_Branch] = []  # every branch ever created, in creation order
    finals: list[_Branch] = []
    ready: deque[Request] = deque()
    outstanding: dict[str, Request] = {}

    root = _Branch(
        branch_id="root",
        parent_id=None,
        group_key=None,
        env_ids=env_ids,
        messages=[Message(role="user", content=prompt)],
        split_depth=0,
    )
    branches[root.branch_id] = root
    history.append(root)

    def stats_for(branch: _Branch) -> RequestStats:
        nonlocal seq
        seq += 1
        return RequestStats(
            depth=branch.steps,
            split_depth=branch.split_depth,
            env_count=len(branch.env_ids),
            seq=seq,
        )

    def finish(branch: _Branch, content: str) -> None:
        branch.final_content = content
        del branches[branch.branch_id]
        finals.append(branch)

    def request_llm(branch: _Branch) -> None:
        max_steps = budget.max_steps_per_branch if budget is not None else None
        if max_steps is not None and branch.steps >= max_steps:
            finish(branch, f"(budget exhausted after {branch.steps} steps)")
            return
        stats = stats_for(branch)
        ready.append(
            LlmRequest(
                request_id=f"llm-{stats.seq}",
                branch_id=branch.branch_id,
                messages=tuple(branch.messages),
                tools=specs,
                stats=stats,
            )
        )

    def request_tool(branch: _Branch, call: ToolCall) -> None:
        stats = stats_for(branch)
        ready.append(
            ToolRequest(
                request_id=f"tool-{stats.seq}",
                branch_id=branch.branch_id,
                tool_call=call,
                env_ids=branch.env_ids,
                stats=stats,
            )
        )

    def handle_llm(request: LlmRequest, completion: LlmCompletion) -> None:
        branch = branches[request.branch_id]
        branch.steps += 1
        branch.messages.append(
            Message(
                role="assistant", content=completion.content, tool_calls=completion.tool_calls
            )
        )
        if not completion.tool_calls:
            finish(branch, completion.content)
            return
        branch.turn_calls = completion.tool_calls
        branch.turn_results = {}
        for call in completion.tool_calls:
            request_tool(branch, call)

    def handle_tool(request: ToolRequest, completion: ToolCompletion) -> None:
        branch = branches[request.branch_id]
        returned = [env for env, _ in completion.results]
        if set(returned) != set(request.env_ids) or len(returned) != len(request.env_ids):
            raise ValueError(
                f"ToolCompletion for {request.request_id} must return exactly one "
                "result per env in the request"
            )
        branch.turn_results[request.tool_call.id] = dict(completion.results)
        if len(branch.turn_results) < len(branch.turn_calls):
            return  # other calls of this assistant turn are still in flight
        advance_turn(branch)

    def append_tool_messages(
        messages: list[Message],
        calls: tuple[ToolCall, ...],
        results: dict[str, dict[str, ToolResult]],
        representative_env: str,
    ) -> None:
        # Envs in one class share a canonicalized result per call.  Raw results
        # may still differ under a lossy group_by; we take the representative's.
        for call in calls:
            result = results[call.id][representative_env]
            messages.append(Message(role="tool", content=_render(result), tool_call_id=call.id))

    def advance_turn(branch: _Branch) -> None:
        # All calls of the turn have results on every env: refine the partition
        # by the tuple of canonicalized results (one group key per call).
        groups: dict[tuple[str, ...], list[str]] = {}
        for env in branch.env_ids:
            key = tuple(group_by(branch.turn_results[call.id][env]) for call in branch.turn_calls)
            groups.setdefault(key, []).append(env)
        if len(groups) == 1:
            append_tool_messages(branch.messages, branch.turn_calls, branch.turn_results,
                                 branch.env_ids[0])
            branch.turn_calls = ()
            branch.turn_results = {}
            request_llm(branch)
            return
        for key, envs in groups.items():
            child = _Branch(
                branch_id=f"{branch.branch_id}/{_short_hash(key)}",
                parent_id=branch.branch_id,
                group_key=json.dumps(key),
                env_ids=tuple(envs),
                messages=list(branch.messages),
                split_depth=branch.split_depth + 1,
                steps=branch.steps,
            )
            append_tool_messages(child.messages, branch.turn_calls, branch.turn_results, envs[0])
            branches[child.branch_id] = child
            history.append(child)
            request_llm(child)
        del branches[branch.branch_id]

    def finalize() -> BatchResult:
        children_of: dict[str, dict[str, SplitNode]] = {}
        root_node: SplitNode | None = None
        for b in reversed(history):
            node = SplitNode(env_ids=b.env_ids, children=children_of.get(b.branch_id, {}))
            if b.parent_id is None:
                root_node = node
            else:
                assert b.group_key is not None
                children_of.setdefault(b.parent_id, {})[b.group_key] = node
        assert root_node is not None
        classes = tuple(EnvClass(id=b.branch_id, env_ids=b.env_ids) for b in finals)
        results: dict[str, str] = {}
        for b in finals:
            assert b.final_content is not None
            for env in b.env_ids:
                results[env] = b.final_content
        return BatchResult(tree=root_node, classes=classes, results=results)

    request_llm(root)

    incoming: Completion | None = None
    while True:
        if incoming is not None:
            request = outstanding.pop(incoming.request_id, None)
            if request is None:
                raise ValueError(f"no outstanding request with id {incoming.request_id!r}")
            if isinstance(request, LlmRequest):
                if not isinstance(incoming, LlmCompletion):
                    raise TypeError(
                        f"LlmRequest {request.request_id} must be answered with LlmCompletion"
                    )
                handle_llm(request, incoming)
            else:
                if not isinstance(incoming, ToolCompletion):
                    raise TypeError(
                        f"ToolRequest {request.request_id} must be answered with ToolCompletion"
                    )
                handle_tool(request, incoming)
            incoming = None
        if ready:
            next_request = ready.popleft()
            outstanding[next_request.request_id] = next_request
            incoming = yield next_request
        elif outstanding:
            incoming = yield None
        else:
            return finalize()


def run_to_completion(
    *,
    prompt: str,
    env_ids: Sequence[str],
    tools: Callable[[str], Sequence[Tool]],
    complete_llm: Callable[[LlmRequest], LlmCompletion],
    complete_tool: Callable[[ToolRequest], ToolCompletion] | None = None,
    group_by: Callable[[ToolResult], str] | None = None,
    budget: Budget | None = None,
) -> BatchResult:
    """Drive ``run_batch`` synchronously, one request at a time.

    ``complete_llm`` is the only provider dependency: adapt any LLM API to
    it.  ``complete_tool`` defaults to executing tools directly via the
    ``tools`` factory.  For parallel serving (the whole point of the
    protocol), pump ``run_batch`` yourself instead.
    """
    if complete_tool is None:
        complete_tool = _local_tool_executor(tools)
    gen = run_batch(
        prompt=prompt, env_ids=env_ids, tools=tools, group_by=group_by, budget=budget
    )
    try:
        request = next(gen)
        while True:
            while request is None:
                request = gen.send(None)
            completion: Completion
            if isinstance(request, LlmRequest):
                completion = complete_llm(request)
            else:
                completion = complete_tool(request)
            request = gen.send(completion)
    except StopIteration as stop:
        result: BatchResult = stop.value
        return result


def _local_tool_executor(
    tools: Callable[[str], Sequence[Tool]],
) -> Callable[[ToolRequest], ToolCompletion]:
    def execute(request: ToolRequest) -> ToolCompletion:
        results: list[tuple[str, ToolResult]] = []
        for env in request.env_ids:
            impls = {t.name: t for t in tools(env)}
            impl = impls.get(request.tool_call.name)
            if impl is None:
                raise ValueError(f"env {env!r} has no tool named {request.tool_call.name!r}")
            results.append((env, impl(request.tool_call.args)))
        return ToolCompletion(request_id=request.request_id, results=tuple(results))

    return execute
