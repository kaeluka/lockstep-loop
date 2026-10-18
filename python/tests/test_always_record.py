
"""Run results and immutable model evidence have different budget semantics."""

import pytest

from lockstep_loop import (
    AgentSnapshot,
    Budget,
    Feedback,
    FeedbackResolutionError,
    LlmCompletion,
    LlmRequest,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
    run_batch,
    run_to_completion,
)
from lockstep_loop.core import BatchLoop

READ = ToolSpec("read", "", {})


def test_constructor_generator_and_sync_driver_publish_agents():
    loop = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[])
    empty = loop.snapshot
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "answer"))
    assert isinstance(loop.result().snapshot, AgentSnapshot)
    assert empty.root.assistant is None
    assert loop.snapshot.root.assistant.content == "answer"
    lower = BatchLoop(
        prompt="p", env_ids=["new"], tool_specs=[], snapshot=loop.snapshot, budget=Budget(0)
    )
    assert lower.done and lower.snapshot.root.env_ids == ("e",)
    assert lower.result().results == {"new": "(budget exhausted after 0 steps)"}
    with pytest.raises(ValueError, match="assistant answer"):
        lower.append_user_message("feedback")

    gen = run_batch(prompt="p", env_ids=["e"], tool_specs=[])
    request = next(gen)
    with pytest.raises(StopIteration) as done:
        gen.send(LlmCompletion(request.request_id, "answer"))
    assert done.value.value.snapshot.root.assistant.content == "answer"

    result = run_to_completion(
        prompt="p", env_ids=["e"], tools=lambda _: [],
        complete_llm=lambda r: LlmCompletion(r.request_id, "answer"),
    )
    assert result.snapshot.root.assistant.content == "answer"


@pytest.mark.parametrize("driver", ["constructor", "generator", "sync"])
def test_zero_budget_is_an_open_agent_not_an_assistant_reply(driver):
    if driver == "constructor":
        loop = BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], budget=Budget(0))
        assert loop.done and loop.next_request() is None
        with pytest.raises(ValueError, match="assistant answer"):
            loop.append_user_message("feedback")
        result = loop.result()
    elif driver == "generator":
        gen = run_batch(prompt="p", env_ids=["e"], tool_specs=[], budget=Budget(0))
        with pytest.raises(StopIteration) as done:
            next(gen)
        result = done.value.value
    else:
        result = run_to_completion(
            prompt="p", env_ids=["e"], tools=lambda _: [], budget=Budget(0),
            complete_llm=lambda _: pytest.fail("zero budget must not request model work"),
        )
    assert result.results == {"e": "(budget exhausted after 0 steps)"}
    assert result.snapshot.root.assistant is None
    assert result.snapshot.root.env_ids == ()
    resumed = BatchLoop(prompt="p", env_ids=["new"], tool_specs=[], snapshot=result.snapshot)
    request = resumed.next_request()
    assert [m.content for m in request.messages] == ["p"]
    resumed.complete(LlmCompletion(request.request_id, "real answer"))
    assert resumed.result().snapshot.root.assistant.content == "real answer"
    assert result.snapshot.root.assistant is None


def make(snapshot=None, budget=None, env="e", subagents=()):
    return BatchLoop(
        prompt="p", env_ids=[env], tool_specs=[READ], snapshot=snapshot,
        budget=budget, subagents=subagents,
    )


def observations(loop):
    """Complete independent calls backwards; preserve their model-visible order."""
    requests = [loop.next_request(), loop.next_request()]
    for request in reversed(requests):
        loop.complete(ToolCompletion(
            request.request_id, tuple((env, request.tool_call.id) for env in request.env_ids)
        ))


def test_budget_checkpoint_resumes_and_replay_obeys_lower_limit():
    loop = make(budget=Budget(1))
    empty = loop.snapshot
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "inspect", (
        ToolCall("one", "read", {}), ToolCall("two", "read", {}),
    )))
    pending = loop.snapshot
    observations(loop)
    exhausted = loop.result()
    old = exhausted.snapshot
    edge = old.root.observations[0]
    assert exhausted.results == {"e": "(budget exhausted after 1 steps)"}
    assert old.load(edge.continuation).assistant is None
    assert [o.tool_output for o in edge.evidence[0].outputs] == ["one", "two"]
    assert empty.root.assistant is None
    assert not pending.root.observations

    resumed = make(old, Budget(2), "new")
    observations(resumed)
    request = resumed.next_request()
    assert isinstance(request, LlmRequest) and request.stats.depth == 1
    assert [m.content for m in request.messages] == ["p", "inspect", "one", "two"]
    resumed.complete(LlmCompletion(request.request_id, "real answer"))
    full = resumed.result().snapshot
    assert old.load(edge.continuation).assistant is None

    lower = make(full, Budget(1), "lower")
    observations(lower)
    assert lower.done and lower.next_request() is None
    assert lower.result().results == {"lower": "(budget exhausted after 1 steps)"}
    endpoint = lower.snapshot.load(lower.snapshot.root.observations[0].continuation)
    assert endpoint.assistant.content == "real answer"  # never truncate old snapshot
    assert endpoint.env_ids == ("new",)  # the stored answer was NOT accepted
    for limit in (None, Budget(2)):
        replay = make(lower.snapshot, limit, "again")
        observations(replay)
        assert replay.done and replay.result().results == {"again": "real answer"}
    assert full.load(full.root.observations[0].continuation).env_ids == ("new",)

    # Feedback must resolve the actual answer, not attach to a budget fallback.
    feedback = BatchLoop(
        prompt="p", env_ids=["feedback"], tool_specs=[READ], snapshot=full,
        budget=Budget(1), feedback=Feedback("revise"),
    )
    with pytest.raises(FeedbackResolutionError):
        observations(feedback)
    assert feedback.snapshot == full
    assert not full.load(full.root.observations[0].continuation).user_continuations


def test_budget_after_feedback_preserves_open_continuation_and_counts_replayed_answers():
    original = make()
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "old answer"))
    bounded = BatchLoop(
        prompt="p", env_ids=["e"], tool_specs=[READ], snapshot=original.snapshot,
        feedback=Feedback("revise"), budget=Budget(1),
    )
    assert bounded.done
    assert bounded.result().results == {"e": "(budget exhausted after 1 steps)"}
    opened = bounded.snapshot
    continuation = opened.root.user_continuations[0]
    assert continuation.message.content == "revise"
    assert opened.load(continuation.continuation).assistant is None
    assert not original.snapshot.root.user_continuations
    resumed = make(opened)
    request = resumed.next_request()
    assert [m.content for m in request.messages] == ["p", "old answer", "revise"]
    assert request.stats.depth == 1
    resumed.complete(LlmCompletion(request.request_id, "new answer"))
    full = resumed.snapshot
    lower = make(full, Budget(1), "lower")
    assert lower.done and lower.result().results == {"lower": "(budget exhausted after 1 steps)"}
    endpoint = lower.snapshot.load(lower.snapshot.root.user_continuations[0].continuation)
    assert endpoint.assistant.content == "new answer" and endpoint.env_ids == ("e",)
    assert make(full, Budget(2)).result().results == {"e": "new answer"}
    assert opened.load(continuation.continuation).assistant is None


def test_child_budget_keeps_child_open_and_parent_records_actual_tool_result():
    sub = SubagentTool("child", "", {}, lambda _: "child", lambda _: [READ], subagents=())
    loop = make(budget=Budget(1), subagents=(sub,))
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "delegate", (ToolCall("c", "child", {}),)))
    child_request = loop.next_request()
    loop.complete(LlmCompletion(
        child_request.request_id, "inspect", (ToolCall("r", "read", {}),)
    ))
    request = loop.next_request()
    loop.complete(ToolCompletion(request.request_id, (("e", "observed"),)))
    result = loop.result()
    child = result.child_updates[0].snapshot
    assert child.load(child.root.observations[0].continuation).assistant is None
    assert result.snapshot.root.observations[0].representative[0].tool_output == result.results["e"]
    assert result.snapshot.child(result.snapshot.root.subagents[0]) == child

    resumed = make(result.snapshot, Budget(2), "new", (sub,))
    request = resumed.next_request()
    resumed.complete(ToolCompletion(request.request_id, (("new", "observed"),)))
    request = resumed.next_request()
    assert request.messages[0].content == "child"
    resumed.complete(LlmCompletion(request.request_id, "child answer"))
    request = resumed.next_request()
    assert request.messages[-1].content == "child answer"
    resumed.complete(LlmCompletion(request.request_id, "parent answer"))
    assert resumed.result().results == {"new": "parent answer"}
    assert len(resumed.snapshot.root.observations) == 2
    assert len(result.snapshot.root.observations) == 1
    assert child.load(child.root.observations[0].continuation).assistant is None
