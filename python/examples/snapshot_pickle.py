"""Easy Python persistence: pickle the snapshot itself, not a descriptor/record bundle.

Run from python/: PYTHONPATH=src python examples/snapshot_pickle.py
Mock executors only. BytesIO stands in for caller-owned files; only load trusted data.
For advanced lazy/incremental storage, see history_replay.py.
"""

import io
import pickle

from lockstep_loop import Agent, LlmCompletion, LlmRequest, ToolCall, ToolCompletion, ToolSpec


def investigate(agent, env, observation):
    run = agent.start([env])
    llms = 0
    while not run.done:
        request = run.next_request()
        assert request is not None
        if isinstance(request, LlmRequest):
            llms += 1
            if len(request.messages) == 1:
                reply = LlmCompletion(
                    request.request_id, "Read status", (ToolCall("s", "status", {}),)
                )
            else:
                reply = LlmCompletion(request.request_id, request.messages[-1].content)
            run.complete(reply)
        else:
            run.complete(ToolCompletion(request.request_id, ((env, observation),)))
    assert run.result().results == {env: observation}
    return llms


def save(agent):
    file = io.BytesIO()
    pickle.dump(agent.snapshot(), file)
    file.seek(0)
    return file


def main():
    agent = Agent(prompt="Read the status.", tool_specs=[ToolSpec("status", "Read status", {})])
    assert investigate(agent, "yesterday", "clean") == 2
    old = agent.snapshot()
    saved = save(agent)

    # No repeated prompt/specs, backend, record merging, or persisted callbacks.
    agent = Agent.from_snapshot(pickle.load(saved))
    assert investigate(agent, "today", "clean") == 0
    assert investigate(agent, "tomorrow", "dirty") == 1
    saved = save(agent)
    agent = Agent.from_snapshot(pickle.load(saved))
    for value in ("clean", "dirty"):
        assert investigate(agent, "later", value) == 0
    assert len(old.root.observations) == 1
    assert len(agent.snapshot().root.observations) == 2
    print("Direct snapshot pickle: fresh=2, reload=0, changed=1; old/new paths reusable")


if __name__ == "__main__":
    main()
