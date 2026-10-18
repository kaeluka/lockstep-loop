"""Explicit full-history inspection without serialization or implicit repr loads."""

import pickle
from pprint import pformat

import pytest

from lockstep_loop import (
    Agent,
    AgentDescriptor,
    AgentSnapshot,
    ChildAgentRun,
    EnvObservation,
    HistoryNode,
    Message,
    NodeRef,
    Observation,
    OutputRecord,
    RecordStore,
    SubagentRecord,
    ToolCall,
    ToolSpec,
    UserContinuation,
)


def test_format_empty_snapshot_width_and_pickle_roundtrip(capsys):
    snapshot = Agent(
        prompt="Inspect the greeting", tool_specs=[ToolSpec("read", "", {})]
    ).snapshot()
    expected = {
        "descriptor": snapshot.descriptor,
        "records": {snapshot.descriptor.root: snapshot.root},
    }
    before = snapshot.store.records()
    for width in (40, 100, 160):
        assert snapshot.format(width=width) == pformat(expected, width=width, sort_dicts=False)
    assert snapshot.format() == snapshot.format(width=100)
    restored = pickle.loads(pickle.dumps(snapshot))
    assert restored.format() == snapshot.format()
    assert snapshot.store.records() == before
    assert capsys.readouterr().out == ""


def test_format_loads_every_reachable_record_once_not_unrelated_store_records():
    store = RecordStore()

    def final(value):
        return store.put(HistoryNode(assistant=Message("assistant", value)))

    observation = final("observed-answer")
    feedback = final("feedback-answer")
    base = final("child-base-answer")
    prior = final("child-prior-answer")
    current = final("child-current-answer")
    orphan = final("UNRELATED-RECORD")

    def descriptor(ref):
        return AgentDescriptor("child-prompt", (), ref)

    representative = (OutputRecord("representative-output"),)
    root = store.put(
        HistoryNode(
            assistant=Message(
                "assistant", "inspect", (ToolCall("read", "read", {"path": "/entry"}),)
            ),
            env_ids=("first-env", "second-env"),
            observations=(
                Observation(
                    representative,
                    (
                        EnvObservation("first-env", representative),
                        EnvObservation("second-env", (OutputRecord("unequal-actual-output"),)),
                    ),
                    observation,
                ),
            ),
            user_continuations=(UserContinuation(Message("user", "feedback-message"), feedback),),
            subagents=(
                SubagentRecord(
                    "child-call",
                    descriptor(current),
                    (
                        ChildAgentRun(("first-env",), descriptor(base), descriptor(prior)),
                        ChildAgentRun(("second-env",), descriptor(prior), descriptor(current)),
                    ),
                ),
            ),
        )
    )
    backend = store.records()
    loaded = []

    def load(ref):
        loaded.append(ref)
        return backend[ref]

    snapshot = AgentSnapshot(AgentDescriptor("root-prompt", (), root), RecordStore(load=load))
    assert "AgentDescriptor" in repr(snapshot)
    assert not loaded  # lightweight default representation
    formatted = snapshot.format()
    assert len(loaded) == len(set(loaded)) == 6
    assert set(loaded) == {root, observation, feedback, base, prior, current}
    assert orphan not in loaded
    for text in (
        "root-prompt",
        "representative-output",
        "unequal-actual-output",
        "second-env",
        "observed-answer",
        "feedback-answer",
        "feedback-message",
        "child-base-answer",
        "child-prior-answer",
        "child-current-answer",
        "child-call",
        "/entry",
    ):
        assert text in formatted
    assert "UNRELATED-RECORD" not in formatted
    assert formatted.count("assistant=Message(role='assistant'") == 6
    assert snapshot.store.records() == {}
    assert snapshot.descriptor.root == root and store.records() == backend
    assert snapshot.format() == formatted
    assert len(loaded) == 6


class Opaque:
    def __repr__(self):
        return "<opaque-evidence>"

    def __reduce__(self):
        raise TypeError("Not pickleable")


def test_format_uses_payload_repr_without_requiring_pickle_or_truncating_strings():
    payload = Opaque()
    store = RecordStore()
    root = store.put(HistoryNode(assistant=Message("assistant", payload)))
    long_prompt = "x" * 4000
    snapshot = AgentSnapshot(AgentDescriptor(long_prompt, (), root), store)
    assert long_prompt in snapshot.format()
    assert "<opaque-evidence>" in snapshot.format()
    with pytest.raises(TypeError, match="Not pickleable"):
        pickle.dumps(snapshot)
    assert snapshot.root.assistant.content is payload


def test_deep_history_formatting_is_iterative_and_preserves_shared_addresses():
    store = RecordStore()
    root = store.put(HistoryNode(assistant=Message("assistant", "end-marker")))
    for _ in range(1200):
        root = store.put(
            HistoryNode(
                observations=(
                    Observation((), (), root),
                    Observation((), (), root),
                )
            )
        )
    snapshot = AgentSnapshot(AgentDescriptor("deep", (), root), store)
    before = store.records()
    formatted = snapshot.format()
    assert formatted.count("HistoryNode(") == 1201
    assert "end-marker" in formatted
    assert store.records() == before


@pytest.mark.parametrize("corrupt", [False, True])
def test_format_surfaces_missing_or_corrupt_records_without_changing_root(corrupt):
    root = NodeRef("missing", 0)
    store = RecordStore(load=(lambda _: None) if corrupt else None)
    snapshot = AgentSnapshot(AgentDescriptor("p", (), root), store)
    with pytest.raises(TypeError if corrupt else KeyError):
        snapshot.format()
    assert snapshot.descriptor.root == root and store.records() == {}
