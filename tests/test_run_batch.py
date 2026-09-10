"""Tests for the prototype lockstep loop."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from lockstep_loop import (
    Budget,
    LlmCompletion,
    LlmRequest,
    Message,
    ToolCall,
    ToolResult,
    run_batch,
    run_to_completion,
)


@dataclass(frozen=True)
class FakeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    impl: Callable[[Mapping[str, Any]], ToolResult]

    def __call__(self, args: Mapping[str, Any]) -> ToolResult:
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


def test_generator_protocol_directly():
    gen = run_batch(prompt="hi", env_ids=["a"], tools=lambda env: [])
    request = next(gen)
    assert isinstance(request, LlmRequest)
    assert request.messages == (Message(role="user", content="hi"),)

    with pytest.raises(StopIteration) as exc_info:
        gen.send(LlmCompletion(request_id=request.request_id, content="done"))
    result = exc_info.value.value
    assert result.results == {"a": "done"}


def test_completion_for_unknown_request_rejected():
    gen = run_batch(prompt="hi", env_ids=["a"], tools=lambda env: [])
    next(gen)
    with pytest.raises(ValueError, match="no outstanding request"):
        gen.send(LlmCompletion(request_id="bogus", content="done"))


def test_empty_env_ids_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        run_batch(prompt="x", env_ids=[], tools=lambda env: [])


def test_duplicate_env_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        run_batch(prompt="x", env_ids=["a", "a"], tools=lambda env: [])


def test_tool_spec_mismatch_rejected():
    def factory(env: str) -> list[FakeTool]:
        return [FakeTool("t", f"docs for {env}", {}, lambda args: "")]

    with pytest.raises(ValueError, match="identical"):
        run_batch(prompt="x", env_ids=["a", "b"], tools=factory)
