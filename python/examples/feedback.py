"""Run from python/: PYTHONPATH=src python examples/feedback.py

The fake model/tool executors are caller code. Persistence uses shallow records
and descriptors, not sessions or endpoint handles. Pickle is for trusted data only.
"""

from history_replay import persist, restore

from lockstep_loop import (
    BatchLoop,
    Feedback,
    FeedbackResolutionError,
    LlmCompletion,
    LlmRequest,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

PROMPT = "Inspect the repository"
NOTE = "Also inspect /foo and revise if needed"
SPECS = (ToolSpec("read", "Read entry", {}), ToolSpec("foo", "Inspect /foo", {}))


def investigate(envs, agent=None, feedback=None):
    loop = BatchLoop(
        prompt=PROMPT,
        env_ids=envs,
        tool_specs=SPECS,
        agent_contract="feedback-example/v1",
        agent=agent,
        feedback=feedback,
    )
    llms, tools = [], []
    while not loop.done:
        request = loop.next_request()
        assert request is not None  # synchronous caller; parallel drivers may wait here
        if isinstance(request, LlmRequest):
            llms.append(request)
            if len(request.messages) == 1:
                reply = LlmCompletion(
                    request.request_id, "read entry", (ToolCall("read", "read", {}),)
                )
            elif request.messages[-1].role == "user":
                reply = LlmCompletion(
                    request.request_id, "inspect /foo", (ToolCall("foo", "foo", {"path": "/foo"}),)
                )
            else:
                answer = "found" if request.messages[-1].content == "vulnerability" else "no issues"
                reply = LlmCompletion(request.request_id, answer)
            loop.complete(reply)
        else:
            tools.append(request)
            results = []
            for env in request.env_ids:
                if request.tool_call.name == "read":
                    value = "unrelated" if env == "D" else "shared"
                else:
                    value = "vulnerability" if env == "B" else "safe"
                results.append((env, value))
            loop.complete(ToolCompletion(request.request_id, tuple(results)))
    return loop.result(), llms, tools


if __name__ == "__main__":
    old, _, _ = investigate(("A", "B", "D"))
    revised, llms, _ = investigate(("A",), old.agent, Feedback(NOTE))
    assert len(llms) == 2  # none while locating the old answer; two AFTER feedback
    assert [m.content for m in llms[0].messages[-2:]] == ["no issues", NOTE]
    assert revised.results == {"A": "no issues"}  # same answer is valid, no reinjection
    assert not old.agent.load(old.agent.root.observations[0].continuation).user_continuations

    backend = {}
    old_descriptor = persist(old.agent, backend)
    descriptor = persist(revised.agent, backend)
    inherited, llms, tools = investigate(("A", "B", "C", "D"), restore(descriptor, backend))
    assert inherited.results == {"A": "no issues", "B": "found", "C": "no issues", "D": "no issues"}
    assert len(llms) == 1 and llms[0].env_ids == ("B",)
    assert [r.env_ids for r in tools if r.tool_call.name == "foo"] == [("A", "B", "C")]

    # A NEW explicit operation appends the same feedback again after the latest answer.
    descriptor = persist(inherited.agent, backend)
    repeated, llms, _ = investigate(("B",), restore(descriptor, backend), Feedback(NOTE))
    assert [m.content for m in llms[0].messages if m.role == "user"] == [PROMPT, NOTE, NOTE]
    assert llms[0].messages[-2].content == "found"
    assert repeated.results == {"B": "found"}
    old_again, llms, tools = investigate(("B",), restore(old_descriptor, backend))
    assert old_again.results == {"B": "no issues"} and not llms
    assert all(r.tool_call.name == "read" for r in tools)

    # A valid but unseen result is a terminal resolution failure, not an LLM fallback.
    failed = BatchLoop(
        prompt=PROMPT,
        env_ids=["changed"],
        tool_specs=SPECS,
        agent_contract="feedback-example/v1",
        agent=old.agent,
        feedback=Feedback(NOTE),
    )
    request = failed.next_request()
    try:
        failed.complete(ToolCompletion(request.request_id, (("changed", "unseen"),)))
    except FeedbackResolutionError:
        assert failed.done and failed.error is not None and failed.outstanding_count == 0
        assert failed.agent.descriptor == old.agent.descriptor
    else:
        raise AssertionError("feedback must not manufacture an unseen target")
    print(
        "shared feedback: A/B/C inherit; D unchanged; old snapshot reusable; replay miss explicit"
    )
