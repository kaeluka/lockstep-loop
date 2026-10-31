"""Runtime and strict-mypy coverage for null and mapping callback arguments.

Check with ``mypy src tests/test_tool_argument_typing.py`` as well as pytest.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lockstep_loop import (
    LlmCompletion,
    LlmRequest,
    MessageEnvelope,
    SubagentTool,
    Tool,
    ToolCall,
    run_to_completion,
)


@dataclass(frozen=True)
class CallbackTool:
    name: str
    fn: Callable[[Any], str]
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, args: Any) -> str:
        return self.fn(args)


def test_null_and_mapping_ordinary_tool_callbacks_keep_their_argument_types() -> None:
    seen: list[object] = []

    def null_input(args: None) -> str:
        assert args is None
        seen.append(args)
        return "null result"

    def mapping_input(args: Mapping[str, Any]) -> str:
        seen.append(args)
        return str(args["value"])

    null_tool: Tool[str] = CallbackTool("null", null_input, parameters={"type": "null"})
    mapping_tool: Tool[str] = CallbackTool("mapping", mapping_input)

    def tools(_: str) -> Sequence[Tool[str] | SubagentTool[str]]:
        return (null_tool, mapping_tool)

    def model(request: LlmRequest[str]) -> LlmCompletion[str]:
        if len(request.messages) == 1:
            return LlmCompletion(
                request.request_id,
                "call",
                (
                    ToolCall("null-call", "null", None),
                    ToolCall("mapping-call", "mapping", {"value": "mapped result"}),
                ),
            )
        assert [m.content for m in request.messages if m.role == "tool"] == [
            "null result",
            "mapped result",
        ]
        return LlmCompletion(request.request_id, "done")

    result = run_to_completion(
        initial_messages=(MessageEnvelope("user", "root"),),
        env_ids=("one", "two"),
        tools=tools,
        complete_llm=model,
    )
    assert seen == [None, None, {"value": "mapped result"}, {"value": "mapped result"}]
    assert result.results == {"one": "done", "two": "done"}
    conversation = next(result.snapshot.conversations())
    assert conversation.get_final_tool_input_asset("null") is None
    assert conversation.get_final_tool_input_asset("mapping") == {"value": "mapped result"}


def test_child_builders_explicitly_accept_null_or_mapping_arguments() -> None:
    seen: list[object] = []

    def null_child(args: None) -> Sequence[MessageEnvelope[str]]:
        assert args is None
        seen.append(args)
        return (MessageEnvelope("system", "child rules"), MessageEnvelope("user", "null child"))

    def mapping_child(args: Mapping[str, Any]) -> Sequence[MessageEnvelope[str]]:
        seen.append(args)
        return (MessageEnvelope("user", str(args["question"])),)

    null_definition: SubagentTool[str] = SubagentTool(
        "null_child", "", {"type": "null"}, null_child, lambda _: (), subagents=()
    )
    mapping_definition: SubagentTool[str] = SubagentTool(
        "mapping_child", "", {"type": "object"}, mapping_child, lambda _: (), subagents=()
    )

    def tools(_: str) -> Sequence[Tool[str] | SubagentTool[str]]:
        return (null_definition, mapping_definition)

    def model(request: LlmRequest[str]) -> LlmCompletion[str]:
        if request.messages[0].content != "root":
            return LlmCompletion(request.request_id, request.messages[-1].content)
        if len(request.messages) == 1:
            return LlmCompletion(
                request.request_id,
                "delegate",
                (
                    ToolCall("null", "null_child", None),
                    ToolCall("mapping", "mapping_child", {"question": "mapping child"}),
                ),
            )
        assert [m.content for m in request.messages if m.role == "tool"] == [
            "null child",
            "mapping child",
        ]
        return LlmCompletion(request.request_id, "done")

    result = run_to_completion(
        initial_messages=(MessageEnvelope("user", "root"),),
        env_ids=("one", "two"),
        tools=tools,
        complete_llm=model,
    )
    assert seen == [None, {"question": "mapping child"}]
    assert result.results == {"one": "done", "two": "done"}
    assert len(result.child_updates) == 2
    conversation = next(result.snapshot.conversations())
    assert conversation.get_final_tool_input_asset("null_child") is None
    assert conversation.get_final_tool_input_asset("mapping_child") == {"question": "mapping child"}
