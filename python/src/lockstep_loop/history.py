"""Immutable AgentSnapshot snapshots, contextual Asset views, and shallow snapshot records.

Payloads, schemas and arguments are borrowed immutable values. NodeRef is a
storage address, never a public conversation/session identity. No core I/O.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Literal, NoReturn, TypeAlias, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from .core import Message, ToolCall, ToolSpec

P = TypeVar("P")

# Execution cursors, not persisted references or public conversation identities.
_HistoryPath: TypeAlias = tuple[tuple[Literal["observation", "user"], int], ...]


@dataclass(frozen=True)
class NodeRef:
    namespace: str
    index: int


@dataclass(frozen=True)
class OutputRecord(Generic[P]):
    tool_output: P


@dataclass(frozen=True)
class EnvObservation(Generic[P]):
    env_id: str
    outputs: tuple[OutputRecord[P], ...]


@dataclass(frozen=True)
class Observation(Generic[P]):
    representative: tuple[OutputRecord[P], ...]
    evidence: tuple[EnvObservation[P], ...]
    continuation: NodeRef


@dataclass(frozen=True)
class UserContinuation(Generic[P]):
    """Shared-path continuation: every environment reaching this endpoint follows it."""

    message: Message[P]
    continuation: NodeRef
    resolved_env_id: str | None = None
    """Resolution provenance only, NOT an inheritance/selection predicate."""


@dataclass(frozen=True)
class SubagentContract:
    """Structural subagent configuration, without caller version keys."""

    spec: ToolSpec
    max_depth: int
    # None preserves bounded self-recursion; a tuple is an explicit child allowlist.
    subagents: tuple[SubagentContract, ...] | None = None


@dataclass(frozen=True)
class AgentQuery(Generic[P]):
    """Child structural configuration, independent of parent/invocation identity.

    External provider/grouping policy isolation is caller-owned.
    """

    prompt: P
    tools: tuple[ToolSpec, ...]
    subagents: tuple[SubagentContract, ...] = ()


@dataclass(frozen=True)
class AgentDescriptor(Generic[P]):
    """Persistence envelope. Save this AND the referenced shallow records.

    Attach a RecordStore to reconstruct an AgentSnapshot without eager loading.
    """

    prompt: P
    tools: tuple[ToolSpec, ...]
    root: NodeRef
    subagents: tuple[SubagentContract, ...] = ()

    def query(self) -> AgentQuery[P]:
        return AgentQuery(self.prompt, self.tools, self.subagents)


AgentResolver: TypeAlias = Callable[[AgentQuery[P]], AgentDescriptor[P] | None]


@dataclass(frozen=True)
class ChildAgentRun(Generic[P]):
    """ChildAgentRun: one invocation's environments, base and resulting snapshot revision."""

    env_ids: tuple[str, ...]
    base: AgentDescriptor[P] | None
    snapshot: AgentDescriptor[P]

    @property
    def agent_descriptor(self) -> AgentDescriptor[P]:
        return self.snapshot


@dataclass(frozen=True)
class SubagentRecord(Generic[P]):
    """SubagentRecord: parent-call provenance; use AgentSnapshot.child to inspect its snapshot."""

    call_id: str
    snapshot: AgentDescriptor[P]
    runs: tuple[ChildAgentRun[P], ...] = ()

    @property
    def agent_descriptor(self) -> AgentDescriptor[P]:
        return self.snapshot


@dataclass(frozen=True)
class ChildAgentUpdate(Generic[P]):
    """Completed invocation, not a merged cache entry or public conversation identity."""

    request_prefix: str
    branch_id: str
    depth: int
    tool_call: ToolCall
    env_ids: tuple[str, ...]
    base: AgentDescriptor[P] | None
    snapshot: AgentSnapshot[P]


@dataclass(frozen=True)
class HistoryNode(Generic[P]):
    """A shared prefix or historical endpoint; an Asset also needs its snapshot/path context."""

    assistant: Message[P] | None = None
    env_ids: tuple[str, ...] = ()
    observations: tuple[Observation[P], ...] = ()
    subagents: tuple[SubagentRecord[P], ...] = ()
    user_continuations: tuple[UserContinuation[P], ...] = ()


class RecordStore(Generic[P]):
    """Append-only local writes plus an optional lazy caller loader.

    Use a fresh globally unique namespace for EACH writer if persisting/sharing
    records. The loader returns immutable records by exact address, or raises
    on missing/corrupt records. It must not re-enter the store. Writes never
    invoke the loader. Persist records() before publishing a descriptor. The
    store retains writes and caches loads; it is not a storage backend or GC.
    """

    def __init__(
        self,
        namespace: str | None = None,
        load: Callable[[NodeRef], HistoryNode[P]] | None = None,
    ) -> None:
        self.namespace = namespace if namespace is not None else str(uuid4())
        if not self.namespace:
            raise ValueError("record namespace must be non-empty")
        self._loader = load
        self._records: dict[NodeRef, HistoryNode[P]] = {}
        self._cache: dict[NodeRef, HistoryNode[P]] = {}
        self._lock = threading.Lock()

    def put(self, node: HistoryNode[P]) -> NodeRef:
        with self._lock:
            ref = NodeRef(self.namespace, len(self._records))
            # A reused writer namespace is a protocol error, not deduplication.
            if ref in self._cache:
                raise ValueError("writer namespace collides with loaded records")
            self._records[ref] = node
            return ref

    def load(self, ref: NodeRef) -> HistoryNode[P]:
        with self._lock:
            if ref in self._records:
                return self._records[ref]
            if ref not in self._cache:
                if ref.namespace == self.namespace:
                    raise ValueError("writer namespace reused for existing or missing records")
                if self._loader is None:
                    raise KeyError(ref)
                self._cache[ref] = self._loader(ref)
            return self._cache[ref]

    def records(self) -> Mapping[NodeRef, HistoryNode[P]]:
        """Snapshot of local writes only; old/loaded records need not be rewritten."""
        with self._lock:
            return dict(self._records)


@dataclass(frozen=True)
class AgentSnapshot(Generic[P]):
    """Immutable prompt, structural configuration and accumulated conversation trie.

    The descriptor selects a root; the store resolves immutable records lazily.
    Persist the descriptor and records, not mutable Agent/AgentRun handles or views.
    Payloads, arguments and schemas are immutable by caller contract.
    """

    descriptor: AgentDescriptor[P]
    store: RecordStore[P] = field(compare=False, repr=False)

    @property
    def root(self) -> HistoryNode[P]:
        return self.load(self.descriptor.root)

    @property
    def prompt(self) -> P:
        return self.descriptor.prompt

    def load(self, ref: NodeRef) -> HistoryNode[P]:
        return self.store.load(ref)

    def child(self, entry: SubagentRecord[P]) -> AgentSnapshot[P]:
        """Attach this store to the recorded child snapshot, without resolving an environment."""
        return AgentSnapshot(entry.agent_descriptor, self.store)

    def assets(self) -> Iterator[Asset[P]]:
        """Lazily inspect this snapshot's structural leaves, not a per-run partition.

        Child agents are separate tries: inspect them with child(entry).assets().
        This does not execute tools, select feedback targets, or create records.
        """
        stack: list[tuple[NodeRef, int, Observation[P] | UserContinuation[P] | None]] = [
            (self.descriptor.root, 0, None)
        ]
        path: list[_AssetStep[P]] = []
        while stack:
            reference, depth, via = stack.pop()
            del path[depth:]
            node = self.load(reference)
            path.append(_AssetStep(node, via))
            if not node.observations and not node.user_continuations:
                yield Asset._view(self, tuple(path))
            else:
                for user in reversed(node.user_continuations):
                    stack.append((user.continuation, depth + 1, user))
                for observation in reversed(node.observations):
                    stack.append((observation.continuation, depth + 1, observation))

    def _node(self, path: _HistoryPath) -> HistoryNode[P]:
        node = self.root
        for kind, index in path:
            edge = (
                node.observations[index]
                if kind == "observation"
                else node.user_continuations[index]
            )
            node = self.load(edge.continuation)
        return node

    def _replace_node(self, path: _HistoryPath, node: HistoryNode[P]) -> AgentSnapshot[P]:
        """Storage-level path copying. Paths are not conversation identities."""
        ancestors = []
        parent = self.root
        for kind, index in path:
            ancestors.append((parent, kind, index))
            edge = (
                parent.observations[index]
                if kind == "observation"
                else parent.user_continuations[index]
            )
            parent = self.load(edge.continuation)
        ref = self.store.put(node)
        for ancestor, kind, index in reversed(ancestors):
            if kind == "observation":
                updated = replace(ancestor.observations[index], continuation=ref)
                parent = replace(
                    ancestor,
                    observations=(
                        *ancestor.observations[:index],
                        updated,
                        *ancestor.observations[index + 1 :],
                    ),
                )
            else:
                updated_user = replace(ancestor.user_continuations[index], continuation=ref)
                parent = replace(
                    ancestor,
                    user_continuations=(
                        *ancestor.user_continuations[:index],
                        updated_user,
                        *ancestor.user_continuations[index + 1 :],
                    ),
                )
            ref = self.store.put(parent)
        return replace(self, descriptor=replace(self.descriptor, root=ref))


@dataclass(frozen=True, eq=False)
class _AssetStep(Generic[P]):
    node: HistoryNode[P]
    via: Observation[P] | UserContinuation[P] | None


@dataclass(frozen=True, eq=False, init=False)
class Asset(Generic[P]):
    """Read-only leaf investigation in one AgentSnapshot snapshot, backed by its path.

    Values do not identify assets. An open continuation may retain an earlier
    published message, or have no value yet. No ID, persistence or resume API is
    provided: feedback still requires snapshot snapshot + environment replay.
    """

    snapshot: AgentSnapshot[P] = field(repr=False)
    _path: tuple[_AssetStep[P], ...] = field(repr=False)

    def __init__(self) -> None:
        raise TypeError(
            "Asset views come from AgentSnapshot.assets(); persist the snapshot, not a leaf"
        )

    @classmethod
    def _view(cls, snapshot: AgentSnapshot[P], path: tuple[_AssetStep[P], ...]) -> Asset[P]:
        view = object.__new__(cls)
        object.__setattr__(view, "snapshot", snapshot)
        object.__setattr__(view, "_path", path)
        return view

    def __reduce__(self) -> NoReturn:
        raise TypeError("persist the snapshot descriptor and records; assets are contextual views")

    @property
    def node(self) -> HistoryNode[P]:
        return self._path[-1].node

    @property
    def nodes(self) -> tuple[HistoryNode[P], ...]:
        """Shared prefix through leaf, including child-snapshot provenance records."""
        return tuple(step.node for step in self._path)

    @property
    def observations(self) -> tuple[Observation[P], ...]:
        """Selected checkpoints, retaining representative AND exact per-env evidence."""
        return tuple(step.via for step in self._path if isinstance(step.via, Observation))

    @property
    def messages(self) -> tuple[Message[P], ...]:
        """The model-visible conversation, using historical representatives."""
        from .core import Message

        messages = [Message(role="user", content=self.snapshot.descriptor.prompt)]
        for index, step in enumerate(self._path):
            if isinstance(step.via, Observation):
                assistant = self._path[index - 1].node.assistant
                assert assistant is not None
                for call, output in zip(assistant.tool_calls, step.via.representative, strict=True):
                    messages.append(
                        Message(role="tool", content=output.tool_output, tool_call_id=call.id)
                    )
            elif isinstance(step.via, UserContinuation):
                messages.append(step.via.message)
            if step.node.assistant is not None:
                messages.append(step.node.assistant)
        return tuple(messages)

    @property
    def is_complete(self) -> bool:
        message = self.node.assistant
        return message is not None and message.role == "assistant" and not message.tool_calls

    @property
    def published_message(self) -> Message[P] | None:
        """Latest terminal assistant reply on this path, even before pending feedback work."""
        for step in reversed(self._path):
            message = step.node.assistant
            if message is not None and message.role == "assistant" and not message.tool_calls:
                return message
        return None

    @property
    def has_value(self) -> bool:
        return self.published_message is not None

    @property
    def value(self) -> P:
        """Latest published payload. Raises LookupError if none; a real None remains a value."""
        message = self.published_message
        if message is None:
            raise LookupError("this open asset has no published value yet")
        return message.content

    @property
    def env_ids(self) -> tuple[str, ...]:
        """Known path memberships, not a global/latest environment partition.

        The latest observation bounds membership; later node visits add evidence.
        Feedback provenance is not an environment gate. Exact repeated visits
        remain in nodes/observations. In-flight partial tool results are not recorded.
        """
        visits = []
        evidence: tuple[str, ...] = ()
        for step in reversed(self._path):
            visits.append(step.node.env_ids)
            if isinstance(step.via, Observation):
                evidence = tuple(e.env_id for e in step.via.evidence)
                break
        return tuple(dict.fromkeys((*evidence, *(e for group in reversed(visits) for e in group))))
