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
    # Neither mode nor function identity is checked: today's function compares
    # raw outputs to historical representatives. Provider policy remains external.
    exact, count = drive(config(agent=old.agent), lambda _: "same ", changed_provider)
    assert count == 0 and exact.results == old.results


def test_legacy_pickle_labels_are_discarded_even_in_child_cache_identity():
    path = Path(__file__).parent / "fixtures" / "agent-naming-2c6d901.pickle"
    wire = pickle.loads(path.read_bytes())
    root = wire["descriptor"]
    entry = wire["records"][root.root].subagents[0]
    selected = entry.history
    # Simulate separately imported historical labels without touching the fixture.
    legacy = pickle.loads(pickle.dumps(selected))
    object.__setattr__(legacy, "contract", "unrelated old label")
    object.__setattr__(legacy, "custom_grouping", False)
    imported = pickle.loads(pickle.dumps(legacy))
    assert imported == selected and imported.query() == selected.query()
    assert "contract" not in vars(imported) and "custom_grouping" not in vars(imported)
    legacy_query = selected.query()
    object.__setattr__(legacy_query, "custom_grouping", True)
    assert pickle.loads(pickle.dumps(legacy_query)) == selected.query()
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
    for parent, grouping in ((b"parent one", bytes.strip), (b"changed parent", None)):
        loop = BatchLoop(
            prompt=parent,
            env_ids=["new"],
            tool_specs=[],
            subagents=[tool],
            agent_resolver=resolve,
            record_store=store,
            group_by=grouping,
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


@pytest.mark.parametrize("initial_custom", [False, True])
def test_default_and_custom_identity_reuse_each_other(initial_custom):
    identity = lambda value: value  # noqa: E731
    old, _ = drive(
        config(recording=True, group_by=identity if initial_custom else None), lambda _: "same"
    )
    before = old.agent.root
    replay, count = drive(
        config(agent=old.agent, group_by=None if initial_custom else identity), lambda _: "same"
    )
    assert count == 0 and replay.results == old.results
    assert old.agent.root == before and len(before.observations[0].evidence) == 1
    assert len(replay.agent.root.observations[0].evidence) == 2
    assert not hasattr(replay.agent.descriptor.query(), "custom_grouping")


@pytest.mark.parametrize("strict", [None, lambda value: value])
def test_stricter_keys_match_representatives_not_old_evidence(strict):
    old, _ = drive(config(recording=True, group_by=str.strip), lambda _: "same ")
    grouped, count = drive(config(agent=old.agent, group_by=str.strip), lambda _: "same")
    assert count == 0
    before = grouped.agent.root
    # "same" is in historical evidence but differs from the representative "same ".
    changed, count = drive(
        config(agent=grouped.agent, group_by=strict),
        lambda _: "same",
        lambda r: "strict:" + r.messages[-1].content,
    )
    assert count == 1 and changed.results == {"e": "strict:same"}
    assert len(changed.agent.root.observations) == 2
    assert grouped.agent.root == before and len(old.agent.root.observations) == 1
    assert [e.outputs[0].tool_output for e in before.observations[0].evidence] == ["same ", "same"]
    assert changed.agent.root.observations[1].representative[0].tool_output == "same"
    assert next(grouped.agent.assets()).messages[-2].content == "same "


def test_coarser_keys_choose_first_existing_edge_without_merging_history():
    first, _ = drive(config(recording=True), lambda _: "A", lambda _: "first")
    old, count = drive(config(agent=first.agent), lambda _: "a", lambda _: "second")
    assert count == 1
    before = old.agent.root
    replay, count = drive(
        config(agent=old.agent, group_by=lambda v: v.strip().lower()), lambda _: "a "
    )
    assert count == 0 and replay.results == {"e": "first"}
    edges = replay.agent.root.observations
    assert len(edges) == 2 and edges[1] == before.observations[1]
    assert [e.outputs[0].tool_output for e in edges[0].evidence] == ["A", "a "]
    assert edges[0].representative[0].tool_output == "A"
    assert next(replay.agent.assets()).messages[-2].content == "A"
    assert old.agent.root == before and len(first.agent.root.observations) == 1
    exact, count = drive(config(agent=replay.agent), lambda _: "a")
    assert count == 0 and exact.results == {"e": "second"}


def test_child_query_and_execution_do_not_distinguish_default_from_custom_identity():
    tool = SubagentTool("child", "", {}, lambda _: "child prompt", lambda _: (READ,), subagents=())
    selected = None
    store = RecordStore()
    queries = []

    def resolve(query):
        queries.append(query)
        return selected if selected is not None and query == selected.query() else None

    for grouping in (None, lambda v: v, None):
        loop = BatchLoop(
            prompt="parent",
            env_ids=["e"],
            tool_specs=[],
            subagents=[tool],
            agent_resolver=resolve,
            record_store=store,
            group_by=grouping,
        )
        r = loop.next_request()
        loop.complete(LlmCompletion(r.request_id, "delegate", (ToolCall("c", "child", {}),)))
        child_llms = 0
        while not loop.done:
            r = loop.next_request()
            if isinstance(r, LlmRequest):
                if r.messages[0].content == "parent":
                    loop.complete(LlmCompletion(r.request_id, r.messages[-1].content))
                else:
                    child_llms += 1
                    calls = (ToolCall("r", "read", {}),) if len(r.messages) == 1 else ()
                    loop.complete(LlmCompletion(r.request_id, "child answer", calls))
            else:
                loop.complete(ToolCompletion(r.request_id, (("e", "same"),)))
        assert child_llms == (2 if selected is None else 0)
        update = loop.result().child_agents[0]
        assert update.base == selected
        selected = update.agent.descriptor
    assert queries[0] == queries[1] == queries[2]


def test_recording_switch_not_structural_identity_and_legacy_budget_stays_available():
    tool = SubagentTool("child", "", {}, lambda _: "child", lambda _: (), subagents=())
    loop = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool], recording=True)
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, "done"))
    # Toggling child recording does not change the replay compatibility header.
    replay = BatchLoop(
        prompt="p",
        env_ids=["e"],
        tool_specs=[],
        subagents=[replace(tool, recording=True)],
        agent=loop.agent,
    )
    assert replay.done
    legacy = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool], budget=Budget(1))
    assert legacy.agent is None
    with pytest.raises(ValueError, match="budgets"):
        BatchLoop(
            prompt="p",
            env_ids=["e"],
            tool_specs=[],
            subagents=[replace(tool, recording=True)],
            budget=Budget(1),
        )
