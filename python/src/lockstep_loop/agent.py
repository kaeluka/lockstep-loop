"""Mutable entry points over immutable, structurally shared snapshots. No core I/O."""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Sequence
from typing import Generic, Literal, overload

from .core import (
    BatchLoop,
    BatchResult,
    Budget,
    Completion,
    Feedback,
    LlmRequest,
    P,
    Request,
    SubagentTool,
    ToolRequest,
    ToolSpec,
    _ExecutionError,
    _registration,
)
from .history import AgentDescriptor, AgentResolver, AgentSnapshot, HistoryNode, RecordStore


class Agent(Generic[P]):
    """Mutable owner of a current immutable snapshot and caller-owned runtime callbacks.

    Recording is unconditional. Payloads/schemas/arguments must remain immutable.
    Construct an independent handle with from_snapshot(agent.snapshot()); neither
    handles nor callbacks are persistence values. Provider policy and child-cache
    selection/update policy remain caller-owned.
    """

    def __init__(
        self,
        *,
        prompt: P,
        tool_specs: Sequence[ToolSpec],
        subagents: Sequence[SubagentTool[P]] = (),
        agent_resolver: AgentResolver[P] | None = None,
        comparison_key: Callable[[P], Hashable] | None = None,
        record_store: RecordStore[P] | None = None,
    ) -> None:
        definitions = tuple(subagents)
        specs, _, contracts = _registration(tool_specs, definitions)
        store = record_store if record_store is not None else RecordStore[P]()
        self._head = AgentSnapshot(
            AgentDescriptor(prompt, specs, store.put(HistoryNode()), contracts), store
        )
        self._subagents = definitions
        self._resolver = agent_resolver
        self._comparison_key = comparison_key
        self._lock = threading.Lock()

    @classmethod
    def from_snapshot(
        cls,
        snapshot: AgentSnapshot[P],
        *,
        subagents: Sequence[SubagentTool[P]] = (),
        agent_resolver: AgentResolver[P] | None = None,
        comparison_key: Callable[[P], Hashable] | None = None,
    ) -> Agent[P]:
        """Attach runtime callbacks to a snapshot, without loading any records.

        Prompt and ordered tool specs come from the snapshot. Supply matching
        subagent definitions when configured; executable callbacks are not stored.
        A new handle is independent, though immutable records/store may be shared.
        """
        if not isinstance(snapshot, AgentSnapshot):
            raise TypeError("snapshot must be an AgentSnapshot")
        definitions = tuple(subagents)
        specs, _, contracts = _registration(snapshot.descriptor.tools, definitions)
        if specs != snapshot.descriptor.tools or contracts != snapshot.descriptor.subagents:
            raise ValueError("incompatible snapshot subagent configuration")
        agent = cls.__new__(cls)
        agent._head = snapshot
        agent._subagents = definitions
        agent._resolver = agent_resolver
        agent._comparison_key = comparison_key
        agent._lock = threading.Lock()
        return agent

    def snapshot(self) -> AgentSnapshot[P]:
        """Capture the current immutable root, without traversing records."""
        with self._lock:
            return self._head

    def start(
        self,
        environments: Sequence[str],
        *,
        feedback: Feedback[P] | None = None,
        budget: Budget | None = None,
    ) -> AgentRun[P]:
        """Start from the current root; successful completion publishes exactly once.

        Warning:
            Overlapping runs start from the agent state available when each run
            begins. Their histories are not merged. The last run to complete
            replaces the agent's current state, potentially excluding experience
            from other runs. Retain each run's snapshot to preserve independent
            results, or finish each run before starting the next to accumulate
            experience sequentially.

        Invalid completions are correctable. Terminal failures and abandoned runs
        do not publish. Budget completion publishes real history, not its fallback.
        """
        snapshot = self.snapshot()
        engine = BatchLoop(
            prompt=snapshot.prompt,
            env_ids=environments,
            tool_specs=snapshot.descriptor.tools,
            subagents=self._subagents,
            agent_resolver=self._resolver,
            comparison_key=self._comparison_key,
            snapshot=snapshot,
            feedback=feedback,
            budget=budget,
            _settle_on_completion=True,
        )
        return AgentRun._start(self, engine)

    def _publish(self, snapshot: AgentSnapshot[P]) -> None:
        with self._lock:
            self._head = snapshot


class AgentRun(Generic[P]):
    """Mutable request/completion state with independently captured immutable history.

    Obtain runs from Agent.start(). Successful completion automatically publishes;
    inspection never republishes an older completed run. Snapshots contain only
    validated historical facts, not pending requests or a crash recovery protocol.
    Request IDs are local to a run: route completions to their originating run.
    Public advancement is synchronized, as in the underlying execution engine.
    """

    _owner: Agent[P]
    _engine: BatchLoop[P]
    _published: bool

    def __init__(self) -> None:
        raise TypeError("AgentRun instances come from Agent.start()")

    @classmethod
    def _start(cls, owner: Agent[P], engine: BatchLoop[P]) -> AgentRun[P]:
        run = cls.__new__(cls)
        run._owner = owner
        run._engine = engine
        run._published = False
        with engine._lock:
            run._settle()
        return run

    def _settle(self) -> bool:
        done = self._engine.done  # settle cached children, including final-only starts
        if done and self._engine.error is None and not self._published:
            self._owner._publish(self._engine.snapshot)
            self._published = True
        return done

    @property
    def done(self) -> bool:
        with self._engine._lock:
            return self._settle()

    @overload
    def next_request(self, kind: Literal["llm"]) -> LlmRequest[P] | None: ...

    @overload
    def next_request(self, kind: Literal["tool"]) -> ToolRequest | None: ...

    @overload
    def next_request(self, kind: Literal["any"] = "any") -> Request[P] | None: ...

    def next_request(self, kind: Literal["any", "llm", "tool"] = "any") -> Request[P] | None:
        with self._engine._lock:
            request = self._engine.next_request(kind)
            self._settle()
            return request

    def complete(self, completion: Completion[P]) -> None:
        with self._engine._lock:
            self._engine.complete(completion)
            self._settle()

    def snapshot(self) -> AgentSnapshot[P]:
        """Capture validated history without advancing execution or publishing."""
        return self._engine.snapshot

    def result(self) -> BatchResult[P]:
        with self._engine._lock:
            self._settle()
            return self._engine.result()

    @property
    def error(self) -> _ExecutionError | None:
        return self._engine.error

    @property
    def outstanding_llm(self) -> int:
        return self._engine.outstanding_llm

    @property
    def outstanding_tool(self) -> int:
        return self._engine.outstanding_tool

    @property
    def outstanding_count(self) -> int:
        return self._engine.outstanding_count
