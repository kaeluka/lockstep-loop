
"""Reuse depends on observations/configuration, never implementation/version labels."""


import pytest

from lockstep_loop import (
    Budget,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)
from lockstep_loop.core import BatchLoop

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
    original, count = drive(config(), lambda _: "same")
    assert count == 2
    old = original.snapshot
    # Different executor implementation, identical tool specs and observations.
    same, count = drive(config(snapshot=old), lambda _: "".join(["sa", "me"]))
    assert count == 0 and same.results == original.results
    changed, count = drive(config(snapshot=same.snapshot), lambda _: "different")
    assert count == 1 and len(changed.snapshot.root.observations) == 2
    assert len(old.root.observations) == 1
    for value in ("same", "different"):
        _, count = drive(config(snapshot=changed.snapshot), lambda _, value=value: value)
        assert count == 0
    with pytest.raises(ValueError, match="incompatible"):
        BatchLoop(prompt="inspect", env_ids=["e"], tool_specs=[OTHER, READ], snapshot=old)


def test_provider_and_custom_grouping_policy_are_external_not_inferred():
    old, _ = drive(config(comparison_key=str.strip), lambda _: "same ")

    def changed_provider(_):
        raise AssertionError("external provider change does not invalidate recorded turns")

    replay, count = drive(
        config(snapshot=old.snapshot, comparison_key=lambda value: value.strip().lower()),
        lambda _: "SAME",
        changed_provider,
    )
    assert count == 0 and replay.results == old.results
    evidence = replay.snapshot.root.observations[0].evidence
    assert [e.outputs[0].tool_output for e in evidence] == ["same ", "SAME"]
    # Neither mode nor function identity is checked: today's function compares
    # raw outputs to historical representatives. Provider policy remains external.
    exact, count = drive(config(snapshot=old.snapshot), lambda _: "same ", changed_provider)
    assert count == 0 and exact.results == old.results




@pytest.mark.parametrize("initial_custom", [False, True])
def test_default_and_custom_identity_reuse_each_other(initial_custom):
    identity = lambda value: value  # noqa: E731
    old, _ = drive(
        config(comparison_key=identity if initial_custom else None),
        lambda _: "same",
    )
    before = old.snapshot.root
    replay, count = drive(
        config(snapshot=old.snapshot, comparison_key=None if initial_custom else identity),
        lambda _: "same",
    )
    assert count == 0 and replay.results == old.results
    assert old.snapshot.root == before and len(before.observations[0].evidence) == 1
    assert len(replay.snapshot.root.observations[0].evidence) == 2
    assert not hasattr(replay.snapshot.descriptor.query(), "custom_grouping")


@pytest.mark.parametrize("strict", [None, lambda value: value])
def test_stricter_keys_match_representatives_not_old_evidence(strict):
    old, _ = drive(config(comparison_key=str.strip), lambda _: "same ")
    grouped, count = drive(
        config(snapshot=old.snapshot, comparison_key=str.strip), lambda _: "same"
    )
    assert count == 0
    before = grouped.snapshot.root
    # "same" is in historical evidence but differs from the representative "same ".
    changed, count = drive(
        config(snapshot=grouped.snapshot, comparison_key=strict),
        lambda _: "same",
        lambda r: "strict:" + r.messages[-1].content,
    )
    assert count == 1 and changed.results == {"e": "strict:same"}
    assert len(changed.snapshot.root.observations) == 2
    assert grouped.snapshot.root == before and len(old.snapshot.root.observations) == 1
    assert [e.outputs[0].tool_output for e in before.observations[0].evidence] == ["same ", "same"]
    assert changed.snapshot.root.observations[1].representative[0].tool_output == "same"
    assert next(grouped.snapshot.assets()).messages[-2].content == "same "


def test_coarser_keys_choose_first_existing_edge_without_merging_history():
    first, _ = drive(config(), lambda _: "A", lambda _: "first")
    old, count = drive(config(snapshot=first.snapshot), lambda _: "a", lambda _: "second")
    assert count == 1
    before = old.snapshot.root
    replay, count = drive(
        config(snapshot=old.snapshot, comparison_key=lambda v: v.strip().lower()), lambda _: "a "
    )
    assert count == 0 and replay.results == {"e": "first"}
    edges = replay.snapshot.root.observations
    assert len(edges) == 2 and edges[1] == before.observations[1]
    assert [e.outputs[0].tool_output for e in edges[0].evidence] == ["A", "a "]
    assert edges[0].representative[0].tool_output == "A"
    assert next(replay.snapshot.assets()).messages[-2].content == "A"
    assert old.snapshot.root == before and len(first.snapshot.root.observations) == 1
    exact, count = drive(config(snapshot=replay.snapshot), lambda _: "a")
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
            comparison_key=grouping,
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
        update = loop.result().child_updates[0]
        assert update.base == selected
        selected = update.snapshot.descriptor
    assert queries[0] == queries[1] == queries[2]


def test_child_structural_identity_and_fresh_budget():
    tool = SubagentTool("child", "", {}, lambda _: "child", lambda _: (), subagents=())
    loop = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool])
    r = loop.next_request()
    loop.complete(LlmCompletion(r.request_id, "done"))
    replay = BatchLoop(
        prompt="p",
        env_ids=["e"],
        tool_specs=[],
        subagents=[tool],
        snapshot=loop.snapshot,
    )
    assert replay.done
    fresh = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagents=[tool], budget=Budget(1))
    assert fresh.snapshot.root.assistant is None
    request = fresh.next_request()
    fresh.complete(LlmCompletion(request.request_id, "done"))
    assert fresh.result().snapshot.root.assistant.content == "done"
