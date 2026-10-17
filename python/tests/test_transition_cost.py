"""Structural guards, not wall-clock thresholds, for the serving hot path."""

import pytest

import lockstep_loop.core as core
from lockstep_loop import (
    BatchLoop,
    LlmCompletion,
    LlmRequest,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)


class NoTraversalList(list):
    def __iter__(self):
        raise AssertionError("unrelated conversation/list was traversed or copied")

    def __reversed__(self):
        raise AssertionError("unrelated conversation/list was traversed or copied")

    def copy(self):
        raise AssertionError("unrelated conversation/list was copied")

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise AssertionError("unrelated conversation/list was sliced")
        return super().__getitem__(index)


class LookupOnlyDict(dict):
    def __iter__(self):
        raise AssertionError("all sibling builders were traversed")

    def items(self):
        raise AssertionError("all sibling builders were traversed")

    def values(self):
        raise AssertionError("all sibling builders were traversed")

    def keys(self):
        raise AssertionError("all sibling builders were traversed")

    def copy(self):
        raise AssertionError("all sibling builders were copied")


def unguard(values):
    return list(list.__iter__(values))


@pytest.mark.parametrize("recording", [False, True])
def test_one_leaf_completion_and_queue_pull_do_not_walk_siblings_or_past_conversations(
    recording, monkeypatch
):
    n = 1000
    loop = BatchLoop(
        prompt="p",
        env_ids=[str(i) for i in range(n)],
        tool_specs=[],
        history_contract="cost/v1" if recording else None,
    )
    root = loop.next_request()
    loop.complete(LlmCompletion(root.request_id, "", (ToolCall("x", "read", {}),)))
    tool = loop.next_request()
    loop.complete(ToolCompletion(tool.request_id, tuple((e, e) for e in tool.env_ids)))
    # Include already finalized siblings as well as the retired pre-split parent.
    for _ in range(20):
        request = loop.next_request()
        loop.complete(LlmCompletion(request.request_id, "done"))
    active = dict(loop._branches)
    for branch in loop._partition:
        branch.messages = NoTraversalList(branch.messages)
    loop._partition = NoTraversalList(loop._partition)
    loop._finals = NoTraversalList(loop._finals)
    loop._branches = LookupOnlyDict(loop._branches)
    loop._outstanding = LookupOnlyDict(loop._outstanding)
    containers = (
        loop._partition,
        loop._finals,
        loop._branches,
        loop._outstanding,
        loop._ready_llm,
        loop._ready_tool,
    )
    watched = []
    watch = core._Transition.watch_branch

    def watch_branch(journal, branch):
        watched.append(branch)
        watch(journal, branch)

    monkeypatch.setattr(core._Transition, "watch_branch", watch_branch)
    assert not loop.done
    request = loop.next_request()
    assert not watched  # queue pull and no-rollup done check do not touch any branch
    selected = active[request.branch_id]
    loop.complete(LlmCompletion(request.request_id, "done"))
    assert {id(branch) for branch in watched} == {id(selected)}
    assert loop._finals[-1] is selected
    assert all(
        before is after
        for before, after in zip(
            containers,
            (
                loop._partition,
                loop._finals,
                loop._branches,
                loop._outstanding,
                loop._ready_llm,
                loop._ready_tool,
            ),
            strict=True,
        )
    )
    assert not loop.done and loop.outstanding_count == 0
    for key, branch in active.items():
        if key != request.branch_id:
            assert loop._branches[key] is branch
    # Building the result is intentionally O(n); serving each branch is not.
    loop._partition = unguard(loop._partition)
    loop._finals = unguard(loop._finals)
    while not loop.done:
        request = loop.next_request()
        loop.complete(LlmCompletion(request.request_id, "done"))
    assert len(loop.result().results) == n


@pytest.mark.parametrize("recording", [False, True])
def test_child_completion_and_pull_do_not_copy_unrelated_children(monkeypatch, recording):
    sub = SubagentTool("child", "", {}, lambda a: a["prompt"], lambda _: [])
    loop = BatchLoop(
        prompt="parent",
        env_ids=["e"],
        tool_specs=[],
        subagent=sub,
        history_contract="child-cost/v1" if recording else None,
    )
    root = loop.next_request()
    loop.complete(
        LlmCompletion(
            root.request_id,
            "delegate",
            tuple(ToolCall(str(i), "child", {"prompt": f"child-{i}"}) for i in range(100)),
        )
    )
    first = loop.next_request()
    first_link = loop._child_ids[first.request_id]
    others = [link.child for link in loop._children.values() if link is not first_link]
    for child in others:
        for branch in child._partition:
            branch.messages = NoTraversalList(branch.messages)
        child._partition = NoTraversalList(child._partition)
        child._finals = NoTraversalList(child._finals)
        child._branches = LookupOnlyDict(child._branches)
    watched = []
    watch = core._Transition.watch_loop

    def watch_loop(journal, changed):
        watched.append(changed)
        watch(journal, changed)

    monkeypatch.setattr(core._Transition, "watch_loop", watch_loop)
    loop.complete(LlmCompletion(first.request_id, "answer"))
    assert {id(changed) for changed in watched} <= {id(loop), id(first_link.child)}
    assert not loop.done
    watched.clear()
    second = loop.next_request()
    assert second.messages[0].content == "child-1"
    assert {id(changed) for changed in watched} == {id(loop), id(others[0])}
    for child in others:
        child._partition = unguard(child._partition)
        child._finals = unguard(child._finals)
    loop.complete(LlmCompletion(second.request_id, "answer"))
    while not loop.done:
        request = loop.next_request()
        loop.complete(LlmCompletion(request.request_id, "answer"))
    assert loop.result().results == {"e": "answer"}


def test_callback_history_reads_never_publish_a_root_that_later_rolls_back():
    current = None
    broken = False
    inspected = []

    def build(args):
        if current is not None:
            published = (
                current.history
            )  # reentrant read; must not deadlock or publish working state
            inspected.append(published)
            assert current._recorded.descriptor.root != published.descriptor.root
            assert len(published.root.observations[0].evidence) == 1
        if broken:
            raise KeyError("callback unavailable")
        return args["prompt"]

    specs = (ToolSpec("read", "", {}), ToolSpec("child", "", {}))
    sub = SubagentTool("child", "", {}, build, lambda _: specs)

    def make(history=None, env="old"):
        return BatchLoop(
            prompt="parent",
            env_ids=[env],
            tool_specs=specs,
            subagent=sub,
            history_contract="published/v1",
            history=history,
        )

    original = make()
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "read", (ToolCall("r", "read", {}),)))
    tool = original.next_request()
    original.complete(ToolCompletion(tool.request_id, (("old", "same"),)))
    request = original.next_request()
    original.complete(
        LlmCompletion(
            request.request_id, "delegate", (ToolCall("c", "child", {"prompt": "child"}),)
        )
    )
    child = original.next_request()
    original.complete(LlmCompletion(child.request_id, "answer"))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "done"))
    current = make(original.history, "new")
    tool = current.next_request()
    before = current.history
    broken = True
    with pytest.raises(KeyError):
        current.complete(ToolCompletion(tool.request_id, (("new", "same"),)))
    assert current.history is before
    assert inspected == [before]
    assert current.outstanding_tool == 1 and current.error is None
    broken = False
    current.complete(ToolCompletion(tool.request_id, (("new", "same"),)))
    assert current.done and current.result().results == {"new": "done"}
    assert inspected == [before, before]
    assert len(before.root.observations[0].evidence) == 1
    assert len(current.history.root.observations[0].evidence) == 2


def test_callback_cannot_reenter_serving_and_issue_uncommitted_requests():
    current = None

    def build(_):
        assert current.history.root.assistant is None
        with pytest.raises(RuntimeError, match="reenter"):
            current.next_request()
        raise KeyError("reject this completion")

    sub = SubagentTool("child", "", {}, build, lambda _: [])
    current = BatchLoop(
        prompt="p", env_ids=["e"], tool_specs=[], subagent=sub, history_contract="reentry/v1"
    )
    root = current.next_request()
    before = current.history
    with pytest.raises(KeyError):
        current.complete(LlmCompletion(root.request_id, "bad", (ToolCall("c", "child", {}),)))
    assert current.history is before and current.outstanding_llm == 1
    current.complete(LlmCompletion(root.request_id, "corrected"))
    assert current.done and current.result().results == {"e": "corrected"}


def test_rollback_restores_child_serving_order_without_cloning_siblings():
    broken = False
    calls = []

    def build(args):
        calls.append(args["prompt"])
        if broken and args["prompt"] == "second-alpha":
            raise KeyError("second child unavailable")
        return args["prompt"]

    specs = (ToolSpec("read", "", {}), ToolSpec("child", "", {}))
    sub = SubagentTool("child", "", {}, build, lambda _: specs)

    def make(history=None):
        return BatchLoop(
            prompt="parent",
            env_ids=["a", "b"],
            tool_specs=specs,
            subagent=sub,
            history_contract="child-order/v1",
            history=history,
        )

    def tool_reply(request):
        return ToolCompletion(
            request.request_id,
            tuple(
                (env, ("alpha" if env == "a" else "beta") if not request.tool_call.args else "ok")
                for env in request.env_ids
            ),
        )

    original = make()
    while not original.done:
        request = original.next_request()
        if not isinstance(request, LlmRequest):
            original.complete(tool_reply(request))
            continue
        prompt = request.messages[0].content
        observed = [m for m in request.messages if m.role == "tool"]
        calls_for_turn = ()
        if prompt == "parent":
            if not observed:
                calls_for_turn = (ToolCall("r", "read", {}),)
            elif len(observed) < 3:
                prefix = "first" if len(observed) == 1 else "second"
                calls_for_turn = (
                    ToolCall("c", "child", {"prompt": f"{prefix}-{observed[0].content}"}),
                )
        elif prompt.startswith("second") and not observed:
            calls_for_turn = (ToolCall("r", "read", {"scope": prompt}),)
        original.complete(LlmCompletion(request.request_id, "done", calls_for_turn))
    replay = make(original.history)
    tool = replay.next_request()
    replay.complete(tool_reply(tool))
    order = tuple(replay._children)
    before = replay.history
    broken = True
    with pytest.raises(KeyError):
        _ = replay.done
    assert tuple(replay._children) == order
    assert replay.history is before and replay.error is None
    broken = False
    calls.clear()
    assert not replay.done
    assert calls == ["second-alpha", "second-beta"]
    tools = [replay.next_request("tool"), replay.next_request("tool")]
    assert [r.tool_call.args["scope"] for r in tools] == calls
    for tool in reversed(tools):
        replay.complete(tool_reply(tool))
    assert replay.done and replay.result().results == {"a": "done", "b": "done"}
