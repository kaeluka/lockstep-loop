"""Tests for the prototype lockstep loop."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from lockstep_loop import (
    BatchLoop,
    Budget,
    LlmCompletion,
    LlmRequest,
    Message,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolRequest,
    ToolSpec,
    run_batch,
    run_to_completion,
)


@dataclass(frozen=True)
class FakeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    impl: Callable[[Mapping[str, Any]], Any]

    def __call__(self, args: Mapping[str, Any]) -> Any:
        return self.impl(args)


def test_single_env_no_tool_calls():
    def llm(request: LlmRequest) -> LlmCompletion:
        return LlmCompletion(request_id=request.request_id, content="done")

    result = run_to_completion(prompt="hi", env_ids=["a"], tools=lambda env: [], complete_llm=llm)

    assert result.results == {"a": "done"}
    assert result.tree.env_ids == ("a",)
    assert result.tree.children == {}
    assert [c.env_ids for c in result.classes] == [("a",)]


def test_split_on_divergent_results():
    files = {"a": "hello", "b": "hello", "c": "goodbye"}

    def make_tools(env: str) -> list[FakeTool]:
        return [
            FakeTool("read_file", "read a file", {"type": "object"}, lambda args: files[env])
        ]

    requests: list[LlmRequest] = []

    def llm(request: LlmRequest) -> LlmCompletion:
        requests.append(request)
        if len(request.messages) == 1:  # first turn: call the tool
            return LlmCompletion(
                request_id=request.request_id,
                content="",
                tool_calls=(ToolCall(id="t1", name="read_file", args={}),),
            )
        tool_message = request.messages[-1]
        return LlmCompletion(
            request_id=request.request_id, content=f"saw:{tool_message.content}"
        )

    result = run_to_completion(
        prompt="read it", env_ids=["a", "b", "c"], tools=make_tools, complete_llm=llm
    )

    assert result.results == {"a": "saw:hello", "b": "saw:hello", "c": "saw:goodbye"}
    assert sorted(c.env_ids for c in result.classes) == [("a", "b"), ("c",)]
    assert len(result.tree.children) == 2

    # the whole point: the shared prefix cost one LLM call for three envs
    first_turns = [r for r in requests if len(r.messages) == 1]
    assert len(first_turns) == 1
    assert first_turns[0].stats.env_count == 3
    second_turns = [r for r in requests if len(r.messages) == 3]
    assert sorted(r.stats.env_count for r in second_turns) == [1, 2]


def test_multi_tool_turn_groups_by_result_tuple():
    # envs a and b agree on both calls; c agrees on the first but not the second
    data = {"a": ("1", "x"), "b": ("1", "x"), "c": ("1", "y")}

    def make_tools(env: str) -> list[FakeTool]:
        return [
            FakeTool("t1", "", {}, lambda args: data[env][0]),
            FakeTool("t2", "", {}, lambda args: data[env][1]),
        ]

    def llm(request: LlmRequest) -> LlmCompletion:
        if len(request.messages) == 1:
            return LlmCompletion(
                request_id=request.request_id,
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="t1", args={}),
                    ToolCall(id="c2", name="t2", args={}),
                ),
            )
        return LlmCompletion(
            request_id=request.request_id, content=request.messages[-1].content
        )

    result = run_to_completion(
        prompt="x", env_ids=["a", "b", "c"], tools=make_tools, complete_llm=llm
    )

    assert result.results == {"a": "x", "b": "x", "c": "y"}
    assert sorted(c.env_ids for c in result.classes) == [("a", "b"), ("c",)]


def test_bytes_payloads_group_by_value():
    blobs = {"a": b"\x89PNG-a", "b": b"\x89PNG-a", "c": b"\x89PNG-b"}

    def make_tools(env: str) -> list[FakeTool]:
        return [FakeTool("snap", "take a snapshot", {}, lambda args: blobs[env])]

    def llm(request: LlmRequest) -> LlmCompletion:
        if len(request.messages) == 1:
            return LlmCompletion(
                request_id=request.request_id,
                content=b"",
                tool_calls=(ToolCall(id="s", name="snap", args={}),),
            )
        return LlmCompletion(
            request_id=request.request_id, content=request.messages[-1].content
        )

    result = run_to_completion(
        prompt=b"go", env_ids=["a", "b", "c"], tools=make_tools, complete_llm=llm
    )

    assert result.results == {"a": b"\x89PNG-a", "b": b"\x89PNG-a", "c": b"\x89PNG-b"}
    assert sorted(c.env_ids for c in result.classes) == [("a", "b"), ("c",)]


def test_custom_group_by_merges_unequal_values():
    data = {"a": "result v1\n", "b": "result v1"}  # differ by whitespace only

    def make_tools(env: str) -> list[FakeTool]:
        return [FakeTool("t", "", {}, lambda args: data[env])]

    def llm(request: LlmRequest) -> LlmCompletion:
        if len(request.messages) == 1:
            return LlmCompletion(
                request_id=request.request_id,
                content="",
                tool_calls=(ToolCall(id="t1", name="t", args={}),),
            )
        return LlmCompletion(request_id=request.request_id, content="fin")

    result = run_to_completion(
        prompt="x",
        env_ids=["a", "b"],
        tools=make_tools,
        complete_llm=llm,
        group_by=lambda s: s.strip(),
    )

    assert [c.env_ids for c in result.classes] == [("a", "b")]
    assert result.results == {"a": "fin", "b": "fin"}


def test_subagent_merges_by_answer():
    subanswers = {"a": "yes", "b": "yes", "c": "no"}
    subagent = SubagentTool(
        name="investigate",
        description="Run a mini-investigation.",
        parameters={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        build_prompt=lambda args: f"Answer: {args['question']}",
        sub_tool_specs=lambda env: [
            ToolSpec(name="probe", description="", parameters={"type": "object", "properties": {}})
        ],
        max_depth=1,
    )

    def make_tools(env: str):
        return [
            FakeTool("probe", "probe the env", {}, lambda args: subanswers[env]),
            subagent,
        ]

    def llm(request: LlmRequest) -> LlmCompletion:
        if request.messages[0].content == "main":
            if len(request.messages) == 1:
                return LlmCompletion(
                    request_id=request.request_id,
                    content="",
                    tool_calls=(
                        ToolCall(id="inv", name="investigate", args={"question": "q?"}),
                    ),
                )
            return LlmCompletion(
                request_id=request.request_id,
                content=f"done:{request.messages[-1].content}",
            )
        # child loop: probes once, returns the tool result as its answer
        if len(request.messages) == 1:
            return LlmCompletion(
                request_id=request.request_id,
                content="",
                tool_calls=(ToolCall(id="pr", name="probe", args={}),),
            )
        return LlmCompletion(
            request_id=request.request_id, content=request.messages[-1].content
        )

    result = run_to_completion(
        prompt="main",
        env_ids=["a", "b", "c"],
        tools=make_tools,
        complete_llm=llm,
    )

    assert sorted(c.env_ids for c in result.classes) == [("a", "b"), ("c",)]
    assert result.results == {"a": "done:yes", "b": "done:yes", "c": "done:no"}


def test_subagent_max_depth_zero_rejected():
    subagent = SubagentTool(
        name="s",
        description="",
        parameters={},
        build_prompt=lambda args: "p",
        sub_tool_specs=lambda env: [],
        max_depth=0,
    )
    with pytest.raises(ValueError, match="max_depth"):
        BatchLoop(prompt="x", env_ids=["e"], tool_specs=[], subagent=subagent)


def test_budget_stops_runaway_branch():
    def llm(request: LlmRequest) -> LlmCompletion:
        return LlmCompletion(
            request_id=request.request_id,
            content="",
            tool_calls=(ToolCall(id=f"c{len(request.messages)}", name="noop", args={}),),
        )

    def make_tools(env: str) -> list[FakeTool]:
        return [FakeTool("noop", "does nothing", {}, lambda args: "ok")]

    result = run_to_completion(
        prompt="x",
        env_ids=["a"],
        tools=make_tools,
        complete_llm=llm,
        budget=Budget(max_steps_per_branch=3),
    )

    assert result.results["a"].startswith("(budget exhausted")


def test_kind_filtered_pulls_and_any_merge_order():
    loop = BatchLoop(
        prompt="x", env_ids=["a", "b"], tool_specs=[ToolSpec("t", "", {})]
    )

    first = loop.next_request("llm")
    assert isinstance(first, LlmRequest)
    assert loop.next_request("tool") is None

    loop.complete(
        LlmCompletion(
            request_id=first.request_id,
            content="",
            tool_calls=(ToolCall(id="c1", name="t", args={}),),
        )
    )

    tool_req = loop.next_request("tool")
    assert isinstance(tool_req, ToolRequest)
    assert tool_req.env_ids == ("a", "b")
    assert loop.next_request("llm") is None  # per-kind None: llm work is not done, just empty

    # divergent results split the branch into two llm requests
    loop.complete(
        ToolCompletion(request_id=tool_req.request_id, results=(("a", "1"), ("b", "2")))
    )

    llm_a = loop.next_request("llm")
    assert isinstance(llm_a, LlmRequest)
    assert loop.outstanding_llm == 1
    # branch a issues another tool call while branch b's llm request sits queued
    loop.complete(
        LlmCompletion(
            request_id=llm_a.request_id,
            content="",
            tool_calls=(ToolCall(id="c2", name="t", args={}),),
        )
    )

    # both kinds queued now; "any" must return the older one first
    older = loop.next_request("any")
    assert isinstance(older, LlmRequest)
    newer = loop.next_request("any")
    assert isinstance(newer, ToolRequest)
    assert not loop.done


def test_result_before_done_raises():
    loop = BatchLoop(prompt="x", env_ids=["a"], tool_specs=[])
    with pytest.raises(RuntimeError):
        loop.result()
    req = loop.next_request()
    assert isinstance(req, LlmRequest)
    loop.complete(LlmCompletion(request_id=req.request_id, content="done"))
    assert loop.done
    assert loop.result().results == {"a": "done"}


def test_wrong_completion_type_leaves_loop_intact():
    loop = BatchLoop(prompt="x", env_ids=["a"], tool_specs=[])
    req = loop.next_request()
    assert isinstance(req, LlmRequest)
    with pytest.raises(TypeError):
        loop.complete(ToolCompletion(request_id=req.request_id, results=(("a", "x"),)))
    assert loop.outstanding_count == 1
    loop.complete(LlmCompletion(request_id=req.request_id, content="done"))
    assert loop.done


def test_generator_adapter_protocol():
    gen = run_batch(prompt="hi", env_ids=["a"], tool_specs=[])
    request = next(gen)
    assert isinstance(request, LlmRequest)
    assert request.messages == (Message(role="user", content="hi"),)

    with pytest.raises(StopIteration) as exc_info:
        gen.send(LlmCompletion(request_id=request.request_id, content="done"))
    result = exc_info.value.value
    assert result.results == {"a": "done"}


def test_completion_for_unknown_request_rejected():
    loop = BatchLoop(prompt="hi", env_ids=["a"], tool_specs=[])
    loop.next_request()
    with pytest.raises(ValueError, match="no outstanding request"):
        loop.complete(LlmCompletion(request_id="bogus", content="done"))


def test_empty_env_ids_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        BatchLoop(prompt="x", env_ids=[], tool_specs=[])


def test_duplicate_env_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        BatchLoop(prompt="x", env_ids=["a", "a"], tool_specs=[])
