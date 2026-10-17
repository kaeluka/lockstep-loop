"""Immutable shallow records and a storage-neutral lazy loading boundary.

Payloads, schemas and arguments are borrowed immutable values. NodeRef is a
storage address, never a public conversation/session identity. No core I/O.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from .core import Message, ToolSpec

P = TypeVar("P")


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
    """Represented, not automatically inherited or selected during replay."""

    message: Message[P]
    continuation: NodeRef
    resolved_env_id: str | None = None
    """Resolution provenance only, NOT an inheritance/selection predicate."""


@dataclass(frozen=True)
class SubagentContract:
    spec: ToolSpec
    max_depth: int


@dataclass(frozen=True)
class HistoryDescriptor(Generic[P]):
    """Small persistence envelope. Serialize this AND each new shallow record."""

    contract: str
    prompt: P
    tools: tuple[ToolSpec, ...]
    custom_grouping: bool
    root: NodeRef
    subagent: SubagentContract | None = None
    requires_user_policy: bool = False


@dataclass(frozen=True)
class SubagentHistory(Generic[P]):
    call_id: str
    history: HistoryDescriptor[P]


@dataclass(frozen=True)
class HistoryNode(Generic[P]):
    assistant: Message[P] | None = None
    env_ids: tuple[str, ...] = ()
    observations: tuple[Observation[P], ...] = ()
    subagents: tuple[SubagentHistory[P], ...] = ()
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
class History(Generic[P]):
    """Snapshot plus its resolver; descriptor and records are the persistence API."""

    descriptor: HistoryDescriptor[P]
    store: RecordStore[P] = field(compare=False, repr=False)

    @property
    def root(self) -> HistoryNode[P]:
        return self.load(self.descriptor.root)

    @property
    def contract(self) -> str:
        return self.descriptor.contract

    def load(self, ref: NodeRef) -> HistoryNode[P]:
        return self.store.load(ref)

    def _node(self, path: tuple[int, ...]) -> HistoryNode[P]:
        node = self.root
        for index in path:
            node = self.load(node.observations[index].continuation)
        return node

    def _replace_node(self, path: tuple[int, ...], node: HistoryNode[P]) -> History[P]:
        """Storage-level path copying. Paths are not conversation identities."""
        ancestors = []
        parent = self.root
        for index in path:
            ancestors.append((parent, index))
            parent = self.load(parent.observations[index].continuation)
        ref = self.store.put(node)
        for ancestor, index in reversed(ancestors):
            edge = replace(ancestor.observations[index], continuation=ref)
            ref = self.store.put(
                replace(
                    ancestor,
                    observations=(
                        *ancestor.observations[:index],
                        edge,
                        *ancestor.observations[index + 1 :],
                    ),
                )
            )
        return replace(self, descriptor=replace(self.descriptor, root=ref))
