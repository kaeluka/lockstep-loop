
"""Routing addresses must not be assembled from opaque provider tool-call IDs."""

import pytest

from lockstep_loop import (
    LlmCompletion,
    LlmRequest,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)
from lockstep_loop.core import BatchLoop

OPAQUE_IDS = ("a/turn-1/sub:root/b", "\0/sub:1/turn-1/llm-1/☃:b")
PROBE_ID = "probe/turn-1/sub:root/b/llm-1:\0☃"
EXPECTED = "child1:grandchild:observed:grandchild|child2:observed:child2"
SPECS = (ToolSpec("investigate", "", {}), ToolSpec("probe", "", {}))


def make_loop(env="old", snapshot=None):
    return BatchLoop(
        prompt="parent",
        env_ids=[env],
        tool_specs=SPECS,
        subagent=SubagentTool(
            "investigate", "", {}, lambda args: args["prompt"], lambda _: SPECS, max_depth=2
        ),
        snapshot=snapshot,
    )


def answer(request, opaque_id):
    scope = request.messages[0].content
    observed = [m for m in request.messages if m.role == "tool"]
    if scope == "parent":
        # Identical tool-call IDs are legal again on the parent's second turn.
        assert len(observed) in (0, 2, 4)
        for first, second in zip(observed[::2], observed[1::2], strict=True):
            assert (first.tool_call_id, first.content) == (
                "a",
                "child1:grandchild:observed:grandchild",
            )
            assert (second.tool_call_id, second.content) == (opaque_id, "child2:observed:child2")
        if len(observed) < 4:
            return LlmCompletion(
                request.request_id,
                "delegate",
                (
                    ToolCall("a", "investigate", {"prompt": "child1"}),
                    ToolCall(opaque_id, "investigate", {"prompt": "child2"}),
                ),
            )
        return LlmCompletion(request.request_id, EXPECTED)
    if not observed:
        if scope == "child1":
            call = ToolCall("b", "investigate", {"prompt": "grandchild"})
        else:
            assert scope in ("grandchild", "child2")
            call = ToolCall(PROBE_ID, "probe", {"scope": scope})
        return LlmCompletion(request.request_id, "investigate", (call,))
    assert len(observed) == 1
    if scope == "child1":
        assert (observed[0].tool_call_id, observed[0].content) == (
            "b",
            "grandchild:observed:grandchild",
        )
    else:
        assert (observed[0].tool_call_id, observed[0].content) == (PROBE_ID, f"observed:{scope}")
    return LlmCompletion(request.request_id, f"{scope}:{observed[0].content}")


def pull(loop, seen, kind="any"):
    request = loop.next_request(kind)
    if request is not None:
        assert request.request_id not in seen
        seen.add(request.request_id)
    return request


def finish(loop, seen, opaque_id, reverse, *, replay=False):
    llms = tools = 0
    while not loop.done:
        pending = []
        while (request := pull(loop, seen)) is not None:
            pending.append(request)
        assert pending, "outstanding work was stranded"
        assert loop.outstanding_count == len(pending)
        for request in reversed(pending) if reverse else pending:
            if isinstance(request, LlmRequest):
                assert not replay, "unchanged nested replay requested fresh reasoning"
                llms += 1
                loop.complete(answer(request, opaque_id))
            else:
                tools += 1
                assert request.tool_call.id == PROBE_ID
                output = f"observed:{request.tool_call.args['scope']}"
                loop.complete(
                    ToolCompletion(
                        request.request_id, tuple((env, output) for env in request.env_ids)
                    )
                )
        assert loop.outstanding_count == 0
    assert loop.outstanding_llm == loop.outstanding_tool == loop.outstanding_count == 0
    assert loop.next_request() is None
    return llms, tools


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("opaque_id", OPAQUE_IDS)
def test_opaque_nested_ids_route_concurrent_completions_without_aliasing(
    reverse, opaque_id
):
    loop = make_loop()
    seen = set()
    parent = pull(loop, seen, "llm")
    loop.complete(answer(parent, opaque_id))
    child1 = pull(loop, seen, "llm")
    assert child1.messages[0].content == "child1"
    loop.complete(answer(child1, opaque_id))
    grandchild = pull(loop, seen, "llm")
    child2 = pull(loop, seen, "llm")
    # Exact reproduction: previously these both received
    # sub:root/a/turn-1/sub:root/b/turn-1/llm-1 and one route overwrote the other.
    assert grandchild.messages[0].content == "grandchild"
    assert child2.messages[0].content == "child2"
    assert grandchild.request_id != child2.request_id
    assert loop.outstanding_llm == 2
    pending = [grandchild, child2]
    for request in reversed(pending) if reverse else pending:
        loop.complete(answer(request, opaque_id))
    assert loop.outstanding_count == 0
    _, tools = finish(loop, seen, opaque_id, reverse)
    assert tools == 4
    assert loop.result().results == {"old": EXPECTED}
    with pytest.raises(ValueError, match="no outstanding"):
        loop.complete(LlmCompletion(grandchild.request_id, "stale"))
    assert loop.outstanding_count == 0

    snapshot = loop.snapshot
    assert snapshot.root.assistant.tool_calls[1].id == opaque_id
    assert {entry.call_id for entry in snapshot.root.subagents} == {"a", opaque_id}
    for replay_reverse in (False, True):
        replay = make_loop("new", snapshot)
        assert finish(replay, set(), opaque_id, replay_reverse, replay=True) == (0, 4)
        assert replay.result().results == {"new": EXPECTED}
    assert snapshot.root.env_ids == ("old",)
