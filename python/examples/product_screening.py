"""Product-specification screening: shared work, then replay on new evidence.

Run from python/: PYTHONPATH=src python examples/product_screening.py
No provider or storage service required. Scripted model replies make the protocol
and call counts reproducible; the toy criteria are not certification advice.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from lockstep_loop import (
    Agent,
    BatchLoop,
    BatchResult,
    LlmCompletion,
    LlmRequest,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

PROMPT = (
    "Check this sensor against our toy outdoor requirements: enclosure rating "
    "IP65 or IP67, and an operating range covering -10 to 40 degrees Celsius. "
    "Read the ratings before concluding."
)
RATINGS = ToolSpec(
    "read_ratings",
    "Read the sensor's enclosure rating and minimum/maximum operating temperatures in Celsius.",
    {"type": "object", "properties": {}, "additionalProperties": False},
)


def ratings(enclosure: str, min_c: int, max_c: int) -> str:
    # Return full relevant evidence in a stable representation, not an output hash.
    return json.dumps({"enclosure": enclosure, "min_c": min_c, "max_c": max_c}, sort_keys=True)


def scripted_model(request: LlmRequest[str]) -> LlmCompletion[str]:
    """Replace with an LLM adapter. Subject IDs are not part of the conversation."""
    if len(request.messages) == 1:
        return LlmCompletion(
            request.request_id,
            "Read the product ratings.",
            (ToolCall("ratings", "read_ratings", {}),),
        )
    observation = request.messages[-1]
    assert observation.role == "tool"
    evidence = json.loads(observation.content)
    meets_requirements = (
        evidence["enclosure"] in ("IP65", "IP67")
        and evidence["min_c"] <= -10
        and evidence["max_c"] >= 40
    )
    return LlmCompletion(
        request.request_id,
        "meets toy requirements" if meets_requirements else "does not meet toy requirements",
    )


def screen(
    evidence: Mapping[str, str], agent: Agent[str] | None = None
) -> tuple[BatchResult[str], int, int]:
    loop = BatchLoop(
        prompt=PROMPT,
        env_ids=tuple(evidence),
        tool_specs=(RATINGS,),
        recording=True,
        agent=agent,
    )
    llm_calls = tool_executions = 0
    while not loop.done:
        request = loop.next_request()
        assert request is not None  # this serial driver completes every request immediately
        if isinstance(request, LlmRequest):
            llm_calls += 1
            loop.complete(scripted_model(request))
        else:
            assert request.tool_call.name == "read_ratings" and not request.tool_call.args
            tool_executions += len(request.env_ids)
            loop.complete(
                ToolCompletion(
                    request.request_id,
                    tuple((env, evidence[env]) for env in request.env_ids),
                )
            )
    return loop.result(), llm_calls, tool_executions


def main() -> None:
    outdoor = ratings("IP65", -20, 50)
    initial, llms, reads = screen(
        {"product-a/v1": outdoor, "product-b/v1": outdoor, "product-c/v1": ratings("IP20", 0, 40)}
    )
    assert (llms, reads, len(initial.classes)) == (3, 3, 2)
    old = initial.agent
    assert old is not None
    print(f"Initial: {llms} LLM calls, {reads} tool executions, {len(initial.classes)} classes")

    matching, llms, reads = screen({"product-d/v1": outdoor}, old)
    assert (llms, reads) == (0, 1)
    assert matching.results["product-d/v1"] == "meets toy requirements"
    print(f"Matching new product: {llms} LLM calls, {reads} tool execution")

    changed, llms, reads = screen({"product-a/v2": ratings("IP65", 0, 50)}, matching.agent)
    assert (llms, reads) == (1, 1)
    assert changed.results["product-a/v2"] == "does not meet toy requirements"
    assert changed.agent is not None
    assert len(list(old.assets())) == 2 and len(list(changed.agent.assets())) == 3
    print(f"Revised specifications: {llms} LLM call, {reads} tool execution")
    for asset in changed.agent.assets():
        print(asset.env_ids, "->", asset.value)
    # Persist descriptors and shallow records with caller-owned codecs as shown
    # in history_replay.py. No product-specific types or storage live in the core.


if __name__ == "__main__":
    main()
