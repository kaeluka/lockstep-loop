"""Inherited subagent sessions, sibling isolation, and joins."""

import pytest

from lockstep_loop import (
    BatchLoop,
    LlmCompletion,
    LlmRequest,
    Message,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)


@pytest.mark.parametrize("inherit_session", [False, True])
@pytest.mark.parametrize("ordinary_last", [False, True])
def test_multiple_subagents_inherit_prefix_and_join_out_of_order(inherit_session, ordinary_last):
    envs = ("a", "b", "c")
    read = ToolSpec("read", "Read the environment", {})
    probe = ToolSpec("probe", "Child-only tool", {})
    subagent = SubagentTool(
        name="fork",
        description="Investigate with the current context",
        parameters={},
        build_prompt=lambda args: args["task"],
        sub_tool_specs=lambda env: [probe],
        inherit_session=inherit_session,
    )
    loop = BatchLoop(
        prompt="main",
        env_ids=envs,
        tool_specs=[read, ToolSpec(subagent.name, subagent.description, subagent.parameters)],
        subagent=subagent,
    )

    # Establish a prefix containing an earlier, fully answered tool turn.
    first = loop.next_request("llm")
    assert first is not None
    earlier_call = ToolCall("earlier", "read", {})
    loop.complete(LlmCompletion(first.request_id, "look first", (earlier_call,)))
    earlier = loop.next_request("tool")
    assert earlier is not None
    loop.complete(ToolCompletion(earlier.request_id, tuple((e, "context") for e in envs)))
    parent = loop.next_request("llm")
    assert parent is not None
    prefix = parent.messages
    assert prefix == (
        Message("user", "main"),
        Message("assistant", "look first", (earlier_call,)),
        Message("tool", "context", tool_call_id="earlier"),
    )

    calls = (
        ToolCall("x", "fork", {"task": "do x"}),
        ToolCall("y", "fork", {"task": "do y"}),
        ToolCall("regular", "read", {}),
    )
    loop.complete(LlmCompletion(parent.request_id, "fork now", calls))
    x = loop.next_request("llm")
    y = loop.next_request("llm")
    regular = loop.next_request("tool")
    assert x is not None and y is not None and regular is not None
    assert x.env_ids == y.env_ids == regular.env_ids == envs
    expected_prefix = prefix if inherit_session else ()
    assert x.messages == (*expected_prefix, Message("user", "do x"))
    assert y.messages == (*expected_prefix, Message("user", "do y"))
    assert x.tools == y.tools == (probe,)
    assert len({x.request_id, y.request_id, regular.request_id}) == 3

    # Y investigates and splits while X and the parent's ordinary tool wait.
    probe_call = ToolCall("p", "probe", {})
    loop.complete(LlmCompletion(y.request_id, "", (probe_call,)))
    probe_request = loop.next_request("tool")
    assert probe_request is not None
    assert probe_request.tool_call == probe_call
    loop.complete(
        ToolCompletion(probe_request.request_id, (("a", "same"), ("b", "same"), ("c", "other")))
    )
    y_same = loop.next_request("llm")
    y_other = loop.next_request("llm")
    assert y_same is not None and y_other is not None
    assert y_same.env_ids == ("a", "b")
    assert y_other.env_ids == ("c",)
    loop.complete(LlmCompletion(y_other.request_id, "Z"))
    loop.complete(LlmCompletion(y_same.request_id, "Y"))
    assert loop.next_request() is None
    assert not loop.done
    assert loop.outstanding_count == 2

    final_completions = [
        ToolCompletion(regular.request_id, tuple((e, "ordinary") for e in envs)),
        LlmCompletion(x.request_id, "X"),
    ]
    if ordinary_last:
        final_completions.reverse()
    loop.complete(final_completions[0])
    assert loop.next_request() is None  # Parent must wait for every call.
    loop.complete(final_completions[1])
    parents = [loop.next_request("llm"), loop.next_request("llm")]
    assert all(isinstance(request, LlmRequest) for request in parents)
    assert {request.env_ids for request in parents} == {("a", "b"), ("c",)}
    for request in reversed(parents):
        assert request is not None
        answer_y = "Y" if request.env_ids == ("a", "b") else "Z"
        assert request.messages == (
            *prefix,
            Message("assistant", "fork now", calls),
            Message("tool", "X", tool_call_id="x"),
            Message("tool", answer_y, tool_call_id="y"),
            Message("tool", "ordinary", tool_call_id="regular"),
        )
        loop.complete(LlmCompletion(request.request_id, f"X/{answer_y}"))
    assert loop.done
    assert loop.result().results == {"a": "X/Y", "b": "X/Y", "c": "X/Z"}


def test_inheritance_uses_each_parent_branch_history():
    fork_spec = ToolSpec("fork", "Investigate", {})
    subagent = SubagentTool(
        name=fork_spec.name,
        description=fork_spec.description,
        parameters=fork_spec.parameters,
        build_prompt=lambda args: "child",
        sub_tool_specs=lambda env: [],
        inherit_session=True,
    )
    loop = BatchLoop(
        prompt="main",
        env_ids=["a", "b"],
        tool_specs=[ToolSpec("read", "Read", {}), fork_spec],
        subagent=subagent,
    )
    root = loop.next_request("llm")
    assert root is not None
    loop.complete(LlmCompletion(root.request_id, "", (ToolCall("read", "read", {}),)))
    read = loop.next_request("tool")
    assert read is not None
    loop.complete(ToolCompletion(read.request_id, (("a", "left"), ("b", "right"))))
    parents = [loop.next_request("llm"), loop.next_request("llm")]
    prefixes = {}
    for parent in parents:
        assert parent is not None
        prefixes[parent.env_ids] = parent.messages
        loop.complete(LlmCompletion(parent.request_id, "", (ToolCall("fork", "fork", {}),)))
    children = [loop.next_request("llm"), loop.next_request("llm")]
    for child in reversed(children):
        assert child is not None
        assert child.messages == (*prefixes[child.env_ids], Message("user", "child"))
        loop.complete(LlmCompletion(child.request_id, "child answer"))
    for _ in parents:
        resumed = loop.next_request("llm")
        assert resumed is not None
        assert resumed.messages[:-2] == prefixes[resumed.env_ids]
        assert resumed.messages[-1] == Message("tool", "child answer", tool_call_id="fork")
        loop.complete(LlmCompletion(resumed.request_id, "done"))
    assert loop.result().results == {"a": "done", "b": "done"}


def test_nested_subagents_preserve_inheritance_setting():
    subagent = SubagentTool(
        name="fork",
        description="Nested investigation",
        parameters={},
        build_prompt=lambda args: args["task"],
        sub_tool_specs=lambda env: [ToolSpec("fork", "Nested investigation", {})],
        max_depth=2,
        inherit_session=True,
    )
    loop = BatchLoop(
        prompt="main",
        env_ids=["a"],
        tool_specs=[ToolSpec(subagent.name, subagent.description, subagent.parameters)],
        subagent=subagent,
    )
    parent = loop.next_request("llm")
    assert parent is not None
    loop.complete(
        LlmCompletion(parent.request_id, "", (ToolCall("x", "fork", {"task": "child"}),))
    )
    child = loop.next_request("llm")
    assert child is not None
    assert child.messages == (Message("user", "main"), Message("user", "child"))
    loop.complete(
        LlmCompletion(child.request_id, "", (ToolCall("y", "fork", {"task": "grandchild"}),))
    )
    grandchild = loop.next_request("llm")
    assert grandchild is not None
    assert grandchild.messages == (*child.messages, Message("user", "grandchild"))
    loop.complete(LlmCompletion(grandchild.request_id, "deep answer"))
    resumed_child = loop.next_request("llm")
    assert resumed_child is not None
    assert resumed_child.messages[-1] == Message("tool", "deep answer", tool_call_id="y")
    loop.complete(LlmCompletion(resumed_child.request_id, "child answer"))
    resumed_parent = loop.next_request("llm")
    assert resumed_parent is not None
    assert resumed_parent.messages[-1] == Message("tool", "child answer", tool_call_id="x")
    loop.complete(LlmCompletion(resumed_parent.request_id, "done"))
    assert loop.result().results == {"a": "done"}
