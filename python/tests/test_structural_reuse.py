"""Reuse depends on observations/configuration, never implementation/version labels."""

import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from lockstep_loop import (
    Agent,
    BatchLoop,
    Budget,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

READ = ToolSpec("read", "read-only", {})
OTHER = ToolSpec("other", "read-only", {})


def drive(loop, implementation, provider=lambda _: "answer"):
    llms = 0
    while not loop.done:
        request = loop.next_request()
        if isinstance(request, LlmRequest):
            llms += 1
            if len(request.messages) == 1:
                reply = LlmCompletion(request.request_id, "read", (ToolCall("r", "read", {}),))
            else:
                reply = LlmCompletion(request.request_id, provider(request))
        else:
            reply = ToolCompletion(
                request.request_id,
                tuple((env, implementation(request.tool_call.args)) for env in request.env_ids),
            )
        loop.complete(reply)
    return loop.result(), llms


def config(**options):
    return BatchLoop(prompt="inspect", env_ids=["e"], tool_specs=[READ, OTHER], **options)


def test_changed_tool_implementation_same_output_reuses_different_output_branches():
    original, count = drive(config(recording=True), lambda _: "same")
    assert count == 2
    old = original.agent
    # Different executor implementation, identical tool specs and observations.
    same, count = drive(config(agent=old), lambda _: "".join(["sa", "me"]))
    assert count == 0 and same.results == original.results
    changed, count = drive(config(agent=same.agent), lambda _: "different")
    assert count == 1 and len(changed.agent.root.observations) == 2
    assert len(old.root.observations) == 1
    for value in ("same", "different"):
        _, count = drive(config(agent=changed.agent), lambda _, value=value: value)
        assert count == 0
    with pytest.raises(ValueError, match="incompatible"):
        BatchLoop(prompt="inspect", env_ids=["e"], tool_specs=[OTHER, READ], agent=old)


def test_provider_and_custom_grouping_policy_are_external_not_inferred():
    old, _ = drive(config(recording=True, group_by=str.strip), lambda _: "same ")

    def changed_provider(_):
        raise AssertionError("external provider change does not invalidate recorded turns")

    replay, count = drive(
        config(agent=old.agent, group_by=lambda value: value.strip().lower()),
        lambda _: "SAME",
        changed_provider,
    )
    assert count == 0 and replay.results == old.results
    evidence = replay.agent.root.observations[0].evidence
    assert [e.outputs[0].tool_output for e in evidence] == ["same ", "SAME"]
    # Only default/custom mode is checked. Selecting an isolated cache/snapshot
    # when changing provider or lossy grouping semantics belongs to the caller.
    with pytest.raises(ValueError, match="incompatible"):
        config(agent=old.agent)


def test_legacy_pickle_labels_are_discarded_even_in_child_cache_identity():
    path = Path(__file__).parent / "fixtures" / "agent-naming-2c6d901.pickle"
    wire = pickle.loads(path.read_bytes())
    root = wire["descriptor"]
    entry = wire["records"][root.root].subagents[0]
    selected = entry.history
    # Simulate separately imported historical labels without touching the fixture.
    legacy = pickle.loads(pickle.dumps(selected))
    object.__setattr__(legacy, "contract", "unrelated old label")
    imported = pickle.loads(pickle.dumps(legacy))
    assert imported == selected and imported.query() == selected.query()
    assert "contract" not in vars(imported)
    nested = root.subagents[0]
    object.__setattr__(nested, "history_contract", "another deprecated label")
    restored_nested = pickle.loads(pickle.dumps(nested))
    assert restored_nested == nested and "history_contract" not in vars(restored_nested)
    store = RecordStore(load=wire["records"].__getitem__)
    queries = []

    def resolve(query):
        queries.append(query)
        return imported if query == selected.query() else None

    tool = SubagentTool(
        "delegate", "child investigation", {}, lambda _: b"child", lambda _: (), subagents=()
    )
    for parent in (b"parent one", b"changed parent"):
        loop = BatchLoop(
            prompt=parent, env_ids=["new"], tool_specs=[], subagents=[tool],
            agent_resolver=resolve, record_store=store, group_by=bytes.strip,
        )
        r = loop.next_request()
        loop.complete(LlmCompletion(r.request_id, b"delegate", (ToolCall("c", "delegate", {}),)))
        r = loop.next_request()
        assert r.messages[0].content == parent  # no child LLM request
        assert r.messages[-1].content == b"child answer"
        loop.complete(LlmCompletion(r.request_id, b"done"))
        assert loop.agent is None  # resolver does not opt the root into recording
        assert loop.result().child_agents[0].base == selected
    assert queries[0] == queries[1]
    assert Agent(root, store).prompt == root.prompt


def test_recording_switch_not_structural_identity_and_legacy_budget_stays_available():
    tool = SubagentTool("child", "", {}, lambda _: "child", lambda _: (), subagents=())
    loop = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool], recording=True)
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, "done"))
    # Toggling child recording does not change the replay compatibility header.
    replay = BatchLoop(
        prompt="p", env_ids=["e"], tool_specs=[],
        subagents=[replace(tool, recording=True)], agent=loop.agent,
    )
    assert replay.done
    legacy = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool], budget=Budget(1))
    assert legacy.agent is None
    with pytest.raises(ValueError, match="budgets"):
        BatchLoop(
            prompt="p", env_ids=["e"], tool_specs=[],
            subagents=[replace(tool, recording=True)], budget=Budget(1),
        )
