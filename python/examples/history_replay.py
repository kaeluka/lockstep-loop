"""Run: PYTHONPATH=src python examples/history_replay.py (from python/)."""

import pickle

from lockstep_loop import (
    BatchLoop,
    History,
    LlmCompletion,
    LlmRequest,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)


def investigate(env: str, observation: str, history: History[str] | None = None):
    loop = BatchLoop(
        prompt="Read the revision's status.",
        env_ids=[env],
        tool_specs=[ToolSpec("status", "Read status without changing it.", {})],
        history_contract="status-investigation/v1;exact-grouping;provider-policy-v1",
        history=history,
    )
    llm_calls = 0
    while not loop.done:
        request = loop.next_request()
        assert request is not None  # this example completes work synchronously
        if isinstance(request, LlmRequest):
            llm_calls += 1
            # Replace this deterministic stand-in with a provider adapter.
            if len(request.messages) == 1:
                reply = LlmCompletion(
                    request.request_id, "Check status.", (ToolCall("s", "status", {}),)
                )
            else:
                reply = LlmCompletion(request.request_id, f"Status: {request.messages[-1].content}")
            loop.complete(reply)
        else:
            # Caller executes the pure tool in each requested environment.
            loop.complete(
                ToolCompletion(
                    request.request_id, tuple((env_id, observation) for env_id in request.env_ids)
                )
            )
    result = loop.result()
    assert result.history is not None
    return result.history, llm_calls, result.results[env]


if __name__ == "__main__":
    old, calls, _ = investigate("yesterday", "clean")
    assert calls == 2
    # Caller-side codec. Persist these bytes wherever you choose.
    # Never unpickle untrusted data. No stable cross-version pickle format is promised.
    restored = pickle.loads(pickle.dumps(old))
    _, calls, answer = investigate("today", "clean", restored)
    assert (calls, answer) == (0, "Status: clean")
    extended, calls, _ = investigate("tomorrow", "dirty", restored)
    assert calls == 1
    assert len(old.root.observations) == 1
    assert len(extended.root.observations) == 2
    for status in ("clean", "dirty"):
        _, calls, _ = investigate("later", status, extended)
        assert calls == 0
    print("unchanged: 0 LLM calls; changed: 1; both paths reusable")
