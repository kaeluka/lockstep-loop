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
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

#: What a tool returns for a single environment.
ToolResult: TypeAlias = str | Mapping[str, Any]


class Tool(Protocol):
    """A single environment's implementation of one tool.

    Invariants (to be enforced by ``run_batch``):

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


@dataclass
class Budget:
    """Limits for the run.

    Budget is shared while branches share a conversation and becomes
    per-branch after a split, so a pathological environment cannot burn
    its siblings' budget.
    """

    max_steps_per_branch: int | None = None


@dataclass
class SplitNode:
    """A node in the prefix-sharing trie.

    Represents the set of environments whose conversations are identical so
    far.  Children are keyed by the canonicalized tool-result group key that
    caused the split.  Will grow to carry conversation state, pending tool
    calls, and content hashes for incremental re-runs.
    """

    env_ids: list[str]
    children: dict[str, SplitNode] = field(default_factory=dict)


@dataclass
class EnvClass:
    """A final equivalence class of environments."""

    id: str
    env_ids: list[str]


@dataclass
class BatchResult:
    """Output of a completed ``run_batch`` invocation."""

    tree: SplitNode
    """The split trie; the persisted form that makes re-runs cheap."""

    classes: list[EnvClass]
    """Final partition of environments by full tool-call trace."""

    results: dict[str, str]
    """Final assistant message per environment (identical within a class)."""


def run_batch(
    *,
    prompt: str,
    env_ids: Sequence[str],
    tools: Callable[[str], Sequence[Tool]],
    group_by: Callable[[ToolResult], str] | None = None,
    scheduler: Literal["dfs"] = "dfs",
    budget: Budget | None = None,
) -> BatchResult:
    """Run one prompt across many environments in lockstep.

    Args:
        prompt: The single prompt shared by every environment.
        env_ids: Identifiers of the environments to run against.
        tools: Factory producing each environment's tool implementations.
            Tool specs (name/description/parameters) must be identical across
            environments; only behavior may differ.
        group_by: Canonicalizes a tool result into a group key.  Environments
            whose results share a key stay in one conversation; distinct keys
            split it.  ``None`` means "use the default canonicalizer"
            (planned): volatile fields must be normalized away here, or
            near-identical environments split on noise.
        scheduler: Branch visit order.  Only ``"dfs"`` is planned: depth-first
            keeps parent prefixes warm in provider prompt caches.
        budget: Per-branch limits.

    Returns:
        The split trie, the final equivalence classes, and the final message
        per environment.

    Invariants:
        * No token is generated twice for environments that remain in the
          same equivalence class.
        * Classes only refine (split), never merge.
    """
    raise NotImplementedError("run_batch is a skeleton; the loop is not implemented yet")
