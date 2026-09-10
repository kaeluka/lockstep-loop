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
            volatile fields must be normalized away here, or near-identical
            environments split on noise.
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
    raise NotImplementedError("run_batch is a skeleton; the loop is not implemented yet")


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
    raise NotImplementedError("run_to_completion is a skeleton; the loop is not implemented yet")
