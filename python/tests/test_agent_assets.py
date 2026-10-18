import pickle
import pickletools
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import lockstep_loop as api
from lockstep_loop import (
    Agent,
    AgentDescriptor,
    AgentQuery,
    Asset,
    BatchLoop,
    EnvObservation,
    Feedback,
    History,
    HistoryNode,
    LlmCompletion,
    LlmRequest,
    Message,
    Observation,
    OutputRecord,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

READ = ToolSpec("read", "read-only", {})


def record():
    loop = BatchLoop(
        prompt="inspect", env_ids=["A", "B", "D"], tool_specs=[READ], recording=True
    )
    empty = loop.agent
    initial = next(empty.assets())
    assert not initial.is_complete and not initial.has_value and initial.env_ids == ()
    assert initial.messages[0].content == "inspect"
    with pytest.raises(LookupError):
        _ = initial.value
    request = loop.next_request()
    loop.complete(
        LlmCompletion(request.request_id, "read", (ToolCall("r", "read", {"path": "/entry"}),))
    )
    pending = next(loop.agent.assets())
    assert not pending.is_complete and not pending.has_value
    assert pending.env_ids == ("A", "B", "D")
    assert pending.messages[-1].tool_calls[0].args == {"path": "/entry"}
    request = loop.next_request()
    loop.complete(
        ToolCompletion(request.request_id, (("D", "different"), ("B", "shared"), ("A", "shared")))
    )
    open_agent = loop.agent
    assert [a.env_ids for a in open_agent.assets()] == [("A", "B"), ("D",)]
    assert all(not a.is_complete and not a.has_value for a in open_agent.assets())
    while not loop.done:
        r = loop.next_request()
        loop.complete(LlmCompletion(r.request_id, "no issues"))
    result = loop.result()
    assert result.agent is loop.agent is loop.history
    assert result.child_agents is result.child_histories
    return loop.agent, empty, open_agent


def test_assets_are_path_backed_leaves_not_values_or_environment_classes():
    agent, empty, opened = record()
    assert isinstance(agent, Agent) and Agent is History
    assert agent.prompt == "inspect"
    assets = list(agent.assets())
    assert len(assets) == 2
    assert [a.env_ids for a in assets] == [("A", "B"), ("D",)]
    assert all(a.is_complete and a.value == "no issues" for a in assets)
    assert assets[0].messages != assets[1].messages  # same answer, different investigations
    assert assets[0].agent is agent
    assert [m.role for m in assets[0].messages] == ["user", "assistant", "tool", "assistant"]
    assert assets[0].observations[0].evidence[1].env_id == "B"
    assert assets[0].nodes[-1] is assets[0].node
    assert not hasattr(assets[0], "id") and not hasattr(assets[0], "resume")
    with pytest.raises(TypeError, match="Agent.assets"):
        Asset()
    with pytest.raises(TypeError, match="persist the agent"):
        pickle.dumps(assets[0])
    with pytest.raises(FrozenInstanceError):
        assets[0].agent = empty
    assert not next(empty.assets()).has_value
    assert all(not a.is_complete for a in opened.assets())
    with pytest.raises(TypeError, match="not an asset"):
        BatchLoop(
            prompt="inspect",
            env_ids=["A"],
            tool_specs=[READ],
            agent=assets[0],
            feedback=Feedback("check"),
        )


def test_feedback_moves_the_leaf_but_preserves_old_assets_and_publication():
    old, _, _ = record()
    old_asset = next(old.assets())
    loop = BatchLoop(
        prompt=old.prompt,
        env_ids=["A"],
        tool_specs=[READ],
        agent=old,
        feedback=Feedback("check /foo"),
    )
    tool = loop.next_request()
    assert not isinstance(tool, LlmRequest)  # assets do not provide a replay shortcut
    loop.complete(ToolCompletion(tool.request_id, (("A", "shared"),)))
    pending_agent = loop.agent
    pending, unrelated = list(pending_agent.assets())
    assert pending.env_ids == ("A", "B")  # shared-path inheritance, not resolved_env_id filtering
    assert not pending.is_complete and pending.has_value
    assert pending.value == "no issues"  # old publication, not a fabricated new completed value
    assert pending.messages[-1].content == "check /foo"
    assert unrelated.is_complete and unrelated.messages == list(old.assets())[1].messages
    assert old_asset.is_complete and old_asset.messages[-1].content == "no issues"
    assert old_asset.node.user_continuations == ()
    request = loop.next_request()
    assert request.messages == pending.messages
    loop.complete(LlmCompletion(request.request_id, "no issues"))
    current = loop.result().agent
    revised = next(current.assets())
    assert revised.is_complete and revised.value == old_asset.value
    assert len(revised.messages) == len(old_asset.messages) + 2
    assert revised.nodes[-2].assistant == old_asset.node.assistant
    assert revised.nodes[-2].user_continuations  # historical endpoint, not another asset
    assert len(list(current.assets())) == 2
    assert not next(pending_agent.assets()).is_complete
    replay = BatchLoop(
        prompt=current.prompt,
        env_ids=["C"],
        tool_specs=[READ],
        agent=current,
    )
    tool = replay.next_request()
    replay.complete(ToolCompletion(tool.request_id, (("C", "shared"),)))
    assert replay.done and replay.result().results == {"C": "no issues"}


def test_asset_views_keep_lossy_evidence_and_real_null_values():
    loop = BatchLoop(
        prompt=b"inspect",
        env_ids=["A", "B"],
        tool_specs=[READ],
        recording=True,
        group_by=bytes.strip,
    )
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, b"read", (ToolCall("r", "read", {}),)))
    r = loop.next_request()
    loop.complete(ToolCompletion(r.request_id, (("A", b"bytes "), ("B", b"bytes"))))
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, None))
    asset = next(loop.agent.assets())
    assert asset.is_complete and asset.has_value and asset.value is None
    assert asset.published_message.content is None
    assert asset.messages[-2].content == b"bytes "
    assert [e.outputs[0].tool_output for e in asset.observations[0].evidence] == [
        b"bytes ",
        b"bytes",
    ]
    assert asset.env_ids == ("A", "B")


def test_asset_enumeration_is_lazy_read_only_and_does_not_visit_child_agents():
    agent, _, _ = record()
    records = {ref: pickle.dumps(node) for ref, node in agent.store.records().items()}
    loaded = []

    def load(reference):
        loaded.append(reference)
        return pickle.loads(records[reference])

    restored = Agent(pickle.loads(pickle.dumps(agent.descriptor)), RecordStore(load=load))
    views = restored.assets()
    assert loaded == []
    first = next(views)
    assert first.env_ids == ("A", "B")
    assert len(loaded) == 2
    assert agent.root.observations[1].continuation not in loaded
    assert restored.store.records() == {}
    assert first.messages == next(agent.assets()).messages
    assert len(list(views)) == 1
    assert restored.store.records() == {}


def test_new_names_are_aliases_and_legacy_wire_classes_are_retained():
    pairs = (
        ("Agent", "History"),
        ("AgentDescriptor", "HistoryDescriptor"),
        ("AgentQuery", "HistoryQuery"),
        ("SubagentRecord", "SubagentHistory"),
        ("ChildAgentRun", "ChildHistoryRun"),
        ("ChildAgentUpdate", "ChildHistoryUpdate"),
    )
    for new, old in pairs:
        assert getattr(api, new) is getattr(api, old)
        assert new in api.__all__ and old in api.__all__
    assert api.AgentResolver == api.HistoryResolver
    agent, _, _ = record()
    assert type(agent).__name__ == "Agent"
    assert type(agent.descriptor).__name__ == "HistoryDescriptor"  # pickle wire global unchanged
    assert AgentDescriptor is api.HistoryDescriptor
    assert AgentQuery is api.HistoryQuery
    encoded = pickle.dumps(agent.descriptor, protocol=4)
    assert b"HistoryDescriptor" in encoded and b"AgentDescriptor" not in encoded
    assert isinstance(pickle.loads(encoded), AgentDescriptor)
    for reference, node in agent.store.records().items():
        assert pickle.loads(pickle.dumps(node)) == node
        assert reference in agent.store.records()
    for options in (
        {"agent": agent, "history": agent},
        {"agent_resolver": lambda _: None, "history_resolver": lambda _: None},
    ):
        with pytest.raises(ValueError, match="not both"):
            BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], **options)


def test_pre_rename_pickle_fixture_loads_replays_and_preserves_wire_names():
    encoded = (Path(__file__).parent / "fixtures" / "agent-naming-2c6d901.pickle").read_bytes()
    wire = pickle.loads(encoded)  # trusted fixture generated by unmodified 2c6d901
    reencoded = pickle.dumps(wire, protocol=4)
    assert pickle.loads(reencoded) == wire

    # Pickle memoization depends on string identity/interning; require stable
    # record globals/evidence values, excluding labels, not memo-table bytes.
    def wire_strings(data):
        return {arg for op, arg, _ in pickletools.genops(data) if op.name.endswith("UNICODE")}

    assert wire_strings(encoded) - wire_strings(reencoded) == {
        "contract", "history_contract", "fixture/v1", "child/v1"
    }
    assert isinstance(wire["descriptor"], AgentDescriptor)
    assert isinstance(wire["query"], AgentQuery)
    records = wire["records"]
    loaded = []

    def load(reference):
        loaded.append(reference)
        return records[reference]

    agent = Agent(wire["descriptor"], RecordStore(load=load))
    asset = next(agent.assets())
    assert asset.value == b"same answer" and asset.env_ids == ("A", "B")
    assert [m.role for m in asset.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert len(loaded) == 3  # do not walk child-agent provenance as parent assets
    assert asset.observations[0].evidence[1].outputs[0].tool_output == b"same"
    entry = agent.root.subagents[0]
    assert entry.agent_descriptor is entry.history
    assert entry.runs[0].agent_descriptor is entry.runs[0].history
    assert next(agent.child(entry).assets()).value == b"child answer"
    sub = SubagentTool(
        "delegate", "child investigation", {}, lambda _: b"child", lambda _: (), subagents=()
    )
    loop = BatchLoop(
        prompt=agent.prompt,
        env_ids=["C"],
        tool_specs=agent.descriptor.tools,
        agent=agent,
        subagents=[sub],
        group_by=bytes.strip,
    )
    tool = loop.next_request()
    assert not isinstance(tool, LlmRequest)
    loop.complete(ToolCompletion(tool.request_id, (("C", b"same "),)))
    assert loop.done and loop.result().results == {"C": b"same answer"}
    assert pickle.dumps(wire, protocol=4) == reencoded


def test_deep_asset_inspection_is_iterative_and_read_only():
    store = RecordStore("deep-assets")
    root = store.put(HistoryNode(assistant=Message("assistant", "done"), env_ids=("A",)))
    for step in reversed(range(1500)):
        outputs = (OutputRecord(str(step)),)
        root = store.put(
            HistoryNode(
                assistant=Message("assistant", "read", (ToolCall("r", "read", {"step": step}),)),
                env_ids=("A",),
                observations=(Observation(outputs, (EnvObservation("A", outputs),), root),),
            )
        )
    agent = Agent(AgentDescriptor("inspect", (READ,), False, root), store)
    before = len(store.records())
    asset = next(agent.assets())
    assert len(asset.observations) == 1500
    assert len(asset.messages) == 3002
    assert asset.value == "done" and asset.env_ids == ("A",)
    assert len(store.records()) == before


def test_child_agent_aliases_preserve_lookup_provenance_and_persistence():
    tool = SubagentTool("inspect", "", {}, lambda _: "child", lambda _: ())
    assert not hasattr(tool, "agent_contract") and not hasattr(tool, "history_contract")
    loop = BatchLoop(
        prompt="parent",
        env_ids=["A"],
        tool_specs=[],
        subagents=[tool],
        recording=True,
        agent_resolver=lambda _: None,
    )
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, "delegate", (ToolCall("c", "inspect", {}),)))
    child = loop.next_request()
    loop.complete(LlmCompletion(child.request_id, "child value"))
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, "parent value"))
    result = loop.result()
    update = result.child_agents[0]
    assert update.agent is update.history
    assert len(list(result.agent.assets())) == 1  # child leaves are not parent assets
    entry = result.agent.root.subagents[0]
    assert entry.agent_descriptor is entry.history
    assert entry.runs[0].agent_descriptor is entry.runs[0].history
    assert [a.value for a in result.agent.child(entry).assets()] == ["child value"]
    assert update.agent.descriptor.query() == update.history.descriptor.query()
    stored = pickle.dumps(result.agent.root)
    assert b"SubagentHistory" in stored and b"ChildHistoryRun" in stored
    assert b"SubagentRecord" not in stored and b"ChildAgentRun" not in stored
    assert "history" in vars(entry) and "agent_descriptor" not in vars(entry)
    # Preferred aliases drive the same child cache and still rerun/resolve its conversation.
    replay = BatchLoop(
        prompt="changed parent",
        env_ids=["B"],
        tool_specs=[],
        subagents=[tool],
        agent_resolver=lambda _: update.agent.descriptor,
        record_store=update.agent.store,
    )
    r = replay.next_request()
    replay.complete(LlmCompletion(r.request_id, "delegate", (ToolCall("new", "inspect", {}),)))
    r = replay.next_request()
    assert r.messages[0].content == "changed parent"  # child final was reused
    replay.complete(LlmCompletion(r.request_id, "parent value"))
    assert replay.result().child_agents[0].base == update.agent.descriptor
