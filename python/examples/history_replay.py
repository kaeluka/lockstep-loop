"""Run: PYTHONPATH=src python examples/history_replay.py (from python/)."""

import pickle

from lockstep_loop import (
    BatchLoop,
    History,
    LlmCompletion,
    LlmRequest,
    RecordStore,
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


def persist(history, backend):
    # Each node is SHALLOW: continuation/subagent references are addresses.
    # Store immutable records first; only then atomically publish the descriptor.
    for reference, node in history.store.records().items():
        encoded = pickle.dumps(node)
        if reference in backend:
            assert backend[reference] == encoded  # never overwrite a different record
        else:
            backend[reference] = encoded
    return pickle.dumps(history.descriptor)


def restore(descriptor, backend):
    # No traversal here: core loads just the selected path, including children.
    # Each RecordStore gets a fresh writer namespace (UUID by default in Python).
    store = RecordStore(load=lambda reference: pickle.loads(backend[reference]))
    return History(pickle.loads(descriptor), store)


if __name__ == "__main__":
    old, calls, _ = investigate("yesterday", "clean")
    assert calls == 2
    # Caller-owned backend and codecs. Never unpickle untrusted bytes.
    backend = {}
    old_descriptor = persist(old, backend)
    restored = restore(old_descriptor, backend)
    _, calls, answer = investigate("today", "clean", restored)
    assert (calls, answer) == (0, "Status: clean")
    extended, calls, _ = investigate("tomorrow", "dirty", restored)
    assert calls == 1
    assert len(old.root.observations) == 1
    assert len(extended.root.observations) == 2
    extended_descriptor = persist(extended, backend)
    for status in ("clean", "dirty"):
        _, calls, _ = investigate("later", status, restore(extended_descriptor, backend))
        assert calls == 0
    _, calls, _ = investigate("old snapshot", "clean", restore(old_descriptor, backend))
    assert calls == 0
    print("unchanged: 0 LLM calls; changed: 1; both paths reusable")
