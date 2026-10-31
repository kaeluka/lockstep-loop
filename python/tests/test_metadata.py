"""Runtime routing metadata belongs to the originating agent, never history."""

import pickle
from dataclasses import replace

from lockstep_loop import (
    Agent,
    LlmCompletion,
    LlmRequest,
    MessageEnvelope,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolRequest,
    ToolSpec,
    run_batch,
    run_to_completion,
)

READ = ToolSpec("read", "", {})


class Opaque:
    # The engine must not compare, copy, serialize, or inspect runtime metadata.
    def __eq__(self, other):
        raise AssertionError("metadata compared")

    def __repr__(self):
        raise AssertionError("metadata inspected")

    def __reduce__(self):
        raise AssertionError("metadata persisted")


def user(text):
    return MessageEnvelope(role="user", content=text)


def test_parent_split_and_restore_use_current_metadata_not_history():
    metadata = Opaque()
    agent = Agent(initial_messages=[user("parent")], tool_specs=[READ], metadata=metadata)
    run = agent.start(["a", "b"])
    req = run.next_request()
    assert isinstance(req, LlmRequest) and req.metadata is metadata
    run.complete(LlmCompletion(req.request_id, "inspect", (ToolCall("r", "read", {}),)))
    req = run.next_request()
    assert isinstance(req, ToolRequest) and req.metadata is metadata
    run.complete(ToolCompletion(req.request_id, (("a", "one"), ("b", "two"))))
    for _ in range(2):
        req = run.next_request()
        assert isinstance(req, LlmRequest) and req.metadata is metadata
        run.complete(LlmCompletion(req.request_id, "answer"))
    assert run.done
    snapshot = pickle.loads(pickle.dumps(run.snapshot()))

    current = Opaque()
    restored = Agent.from_snapshot(snapshot, metadata=current)
    replay = restored.start(["c"])
    req = replay.next_request()
    assert isinstance(req, ToolRequest) and req.metadata is current
    replay.complete(ToolCompletion(req.request_id, (("c", "one"),)))
    assert replay.done and replay.next_request() is None

    # A new observation requests fresh reasoning using reattached metadata.
    new_path = restored.start(["d"])
    req = new_path.next_request()
    assert req.metadata is current
    new_path.complete(ToolCompletion(req.request_id, (("d", "new"),)))
    req = new_path.next_request()
    assert isinstance(req, LlmRequest) and req.metadata is current

    default = Agent.from_snapshot(snapshot).start(["e"])
    assert default.next_request().metadata is None


def test_nested_siblings_defaults_and_replayed_children_keep_own_metadata():
    parent_metadata, child_metadata, leaf_metadata = Opaque(), Opaque(), Opaque()
    leaf = SubagentTool(
        "leaf", "", {}, lambda _: [user("leaf")], lambda _: [READ],
        metadata=leaf_metadata, subagents=(),
    )
    child = SubagentTool(
        "child", "", {}, lambda _: [user("child")], lambda _: [],
        max_depth=2, subagents=(leaf,), metadata=child_metadata,
    )
    sibling = SubagentTool(
        "sibling", "", {}, lambda _: [user("sibling")], lambda _: [],
        subagents=(),
    )
    agent = Agent(
        initial_messages=[user("parent")], tool_specs=[],
        subagents=[child, sibling], metadata=parent_metadata,
    )
    run = agent.start(["a"])
    req = run.next_request()
    assert req.metadata is parent_metadata
    run.complete(LlmCompletion(req.request_id, "delegate", (
        ToolCall("c", "child", {}), ToolCall("s", "sibling", {}),
    )))
    req = run.next_request()
    assert req.metadata is child_metadata
    run.complete(LlmCompletion(req.request_id, "delegate", (ToolCall("l", "leaf", {}),)))
    req = run.next_request()
    assert req.metadata is leaf_metadata
    run.complete(LlmCompletion(req.request_id, "inspect", (ToolCall("r", "read", {}),)))
    req = run.next_request("tool")
    assert req.metadata is leaf_metadata
    run.complete(ToolCompletion(req.request_id, (("a", "seen"),)))
    for expected in [leaf_metadata, child_metadata, None, parent_metadata]:
        req = run.next_request()
        assert isinstance(req, LlmRequest) and req.metadata is expected
        run.complete(LlmCompletion(req.request_id, "answer"))
    assert run.done

    # No metadata is included in the pickle, including child provenance.
    snapshot = pickle.loads(pickle.dumps(run.snapshot()))
    replacement = Opaque()
    leaf_now = replace(leaf, metadata=replacement)
    child_now = replace(child, subagents=(leaf_now,), metadata=Opaque())
    restored = Agent.from_snapshot(
        snapshot, subagents=[child_now, sibling], metadata=Opaque()
    )
    replay = restored.start(["b"])
    req = replay.next_request()
    assert isinstance(req, ToolRequest) and req.metadata is replacement
    replay.complete(ToolCompletion(req.request_id, (("b", "seen"),)))
    assert replay.done and replay.next_request() is None


def test_convenience_drivers_forward_root_metadata():
    metadata = Opaque()
    gen = run_batch(
        initial_messages=[user("question")], env_ids=["a"], tool_specs=[],
        metadata=metadata,
    )
    req = next(gen)
    assert req.metadata is metadata
    gen.close()

    def model(req):
        assert req.metadata is metadata
        return LlmCompletion(req.request_id, "answer")

    result = run_to_completion(
        initial_messages=[user("question")], env_ids=["a"], tools=lambda _: [],
        complete_llm=model, metadata=metadata,
    )
    assert result.results == {"a": "answer"}
