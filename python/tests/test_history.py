"""Recorded turns are the replay engine's source of truth, not a split summary."""

import pickle
from dataclasses import FrozenInstanceError, dataclass

import pytest

from lockstep_loop import (
    BatchLoop,
    Budget,
    History,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolRequest,
    ToolSpec,
)

SPECS = (ToolSpec("read", "read-only", {"type": "object"}),)


def make_loop(envs=("old",), history=None, **kwargs):
    return BatchLoop(
        prompt="inspect",
        env_ids=envs,
        tool_specs=SPECS,
        history=history,
        **kwargs,
    )


def drive(loop, values, turns=2):
    """Each turn has two independent tool calls, completed in reverse order."""
    llms = []
    while not loop.done:
        request = loop.next_request()
        assert request is not None
        if isinstance(request, LlmRequest):
            llms.append(request)
            if request.stats.depth < turns:
                loop.complete(
                    LlmCompletion(
                        request.request_id,
                        f"think {request.stats.depth}",
                        (
                            ToolCall("a", "read", {"step": request.stats.depth, "slot": 0}),
                            ToolCall("b", "read", {"step": request.stats.depth, "slot": 1}),
                        ),
                    )
                )
            else:
                loop.complete(LlmCompletion(request.request_id, request.messages[-1].content))
        else:
            other = loop.next_request("tool")
            assert other is not None
            for tool in (other, request):
                step, slot = tool.tool_call.args["step"], tool.tool_call.args["slot"]
                loop.complete(
                    ToolCompletion(
                        tool.request_id,
                        tuple((env, values[env][step][slot]) for env in reversed(tool.env_ids)),
                    )
                )
    return loop.result(), llms


def test_non_splitting_checkpoints_replay_branch_and_preserve_snapshots():
    old_values = (("same", "one"), ("same", "two"))
    loop = make_loop()
    empty = loop.history
    result, llms = drive(loop, {"old": old_values})
    assert len(llms) == 3
    old = result.history
    assert empty.root.assistant is None
    with pytest.raises(FrozenInstanceError):
        empty.root.assistant = old.root.assistant
    assert old.root.assistant.content == "think 0"
    first = old.root.observations[0]
    first_next = old.load(first.continuation)
    second = first_next.observations[0]
    assert old.load(second.continuation).assistant.content == "two"
    assert result.tree.children == {}  # both checkpoints used to be invisible

    replay, llms = drive(make_loop(("new",), old), {"new": old_values})
    assert llms == []
    assert replay.results == {"new": "two"}
    assert first.evidence[0].env_id == "old"
    assert len(first.evidence) == 1  # replay didn't mutate old
    assert [e.env_id for e in replay.history.root.observations[0].evidence] == ["old", "new"]

    changed_values = (("same", "one"), ("same", "CHANGED"))
    changed, llms = drive(make_loop(("tomorrow",), old), {"tomorrow": changed_values})
    assert len(llms) == 1
    assert llms[0].stats.depth == 2
    assert llms[0].messages[-1].content == "CHANGED"
    edges = changed.history.load(changed.history.root.observations[0].continuation).observations
    assert len(edges) == 2
    assert edges[0] is second  # untouched old path is structurally shared
    assert len(first_next.observations) == 1
    for value in (old_values, changed_values):
        _, llms = drive(make_loop(("again",), changed.history), {"again": value})
        assert llms == []

    # Divergence at the FIRST formerly non-splitting checkpoint requests work there.
    _, llms = drive(
        make_loop(("early",), old), {"early": (("different", "first"), ("later", "second"))}
    )
    assert [r.stats.depth for r in llms] == [1, 2]


def test_parallel_branches_never_merge_and_do_not_clobber_history():
    old, _ = drive(make_loop(), {"old": (("x", "y"),)}, turns=1)
    loop = make_loop(("same", "a", "b"), old.history)
    result, llms = drive(
        loop, {"same": (("x", "y"),), "a": (("x", "a"),), "b": (("x", "b"),)}, turns=1
    )
    assert len(llms) == 2
    assert len(result.classes) == 3
    assert len(result.history.root.observations) == 3
    assert len(old.history.root.observations) == 1
    for value in ("y", "a", "b"):
        _, llms = drive(make_loop(("next",), result.history), {"next": (("x", value),)}, turns=1)
        assert not llms


def test_comparison_key_applies_to_new_outputs_and_stored_representatives():
    # Tuple keys cannot equal raw bytes: both sides must pass through the callback.
    loop = make_loop(("a", "b"), comparison_key=lambda value: (value.strip(),))
    result, llms = drive(loop, {"a": ((b"one ", b"two "),), "b": ((b"one", b"two"),)}, turns=1)
    assert len(result.classes) == 1
    assert [m.content for m in llms[-1].messages[-2:]] == [b"one ", b"two "]
    edge = result.history.root.observations[0]
    assert edge.representative[1].tool_output == b"two "
    assert edge.evidence[1].outputs[1].tool_output == b"two"
    encoded = {ref: pickle.dumps(node) for ref, node in result.history.store.records().items()}
    restored = History(
        pickle.loads(pickle.dumps(result.history.descriptor)),
        RecordStore(load=lambda ref: pickle.loads(encoded[ref])),
    )
    replay, requests = drive(
        # A fresh lambda with the same semantics must replay; never compare lambda identity.
        make_loop(("c",), restored, comparison_key=lambda value: (value.strip(),)),
        {"c": ((b"one\n", b"two\n"),)},
        turns=1,
    )
    assert not requests
    assert replay.results == {"c": b"two "}
    assert len(replay.history.root.observations) == 1
    assert len(edge.evidence) == 2  # replay leaves the old snapshot unchanged
    assert replay.history.root.observations[0].evidence[-1].outputs[1].tool_output == b"two\n"
    assert restored == result.history


@dataclass(frozen=True)
class Collision:
    value: str

    def __hash__(self):
        return 0

    def __repr__(self):
        return "identical-debug-string"


def test_hash_and_repr_collisions_do_not_conflate_records_or_branches():
    a, b = Collision("a"), Collision("b")
    result, _ = drive(make_loop(("a", "b")), {"a": ((a, a),), "b": ((b, b),)}, turns=1)
    assert result.results == {"a": a, "b": b}
    assert len({c.id for c in result.classes}) == 2
    assert len(result.tree.children) == 2
    for value in (a, b):
        replay, requests = drive(
            make_loop(("next",), result.history), {"next": ((value, value),)}, turns=1
        )
        assert not requests
        assert replay.results["next"] == value


def test_validation_is_transactional_including_partial_turn_and_grouping_errors():
    loop = make_loop(("a", "b"))
    llm = loop.next_request()
    snapshot = loop.history
    duplicate = ToolCall("dup", "read", {})
    with pytest.raises(ValueError, match="duplicate"):
        loop.complete(LlmCompletion(llm.request_id, "bad", (duplicate, duplicate)))
    assert loop.history is snapshot
    assert loop.outstanding_llm == 1
    loop.complete(
        LlmCompletion(llm.request_id, "ok", (ToolCall("a", "read", {}), ToolCall("b", "read", {})))
    )
    first, second = loop.next_request(), loop.next_request()
    snapshot = loop.history
    for bad in ((("a", "x"),), (("a", "x"), ("a", "x")), (("a", "x"), ("unknown", "x"))):
        with pytest.raises(ValueError, match="exactly one"):
            loop.complete(ToolCompletion(second.request_id, bad))
        assert loop.outstanding_tool == 2
        assert loop.history is snapshot
    with pytest.raises(TypeError):
        loop.complete(ToolCompletion(second.request_id, (("a", []), ("b", []))))
    assert loop.outstanding_tool == 2
    loop.complete(ToolCompletion(second.request_id, (("a", "2"), ("b", "2"))))
    assert loop.history is snapshot  # partial turn remains a builder
    assert loop.next_request() is None
    loop.complete(ToolCompletion(first.request_id, (("a", "1"), ("b", "1"))))
    with pytest.raises(ValueError, match="no outstanding"):
        loop.complete(ToolCompletion(first.request_id, (("a", "overwrite"), ("b", "overwrite"))))
    request = loop.next_request()
    assert [m.content for m in request.messages[-2:]] == ["1", "2"]
    loop.complete(LlmCompletion(request.request_id, "done"))
    assert loop.done


def test_final_only_replay_and_structural_compatibility():
    loop = make_loop()
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "done"))
    history = loop.history
    replay = make_loop(("new",), history)
    assert replay.done
    assert replay.next_request() is None
    assert replay.result().results == {"new": "done"}
    bounded = make_loop(("new",), history, budget=Budget(1))
    assert bounded.done and bounded.result().results == {"new": "done"}
    for kwargs in (
        {"prompt": "changed prompt"},
        {"tool_specs": []},
        {"subagent": SubagentTool("s", "", {}, lambda _: "", lambda _: [])},
    ):
        options = dict(
            prompt="inspect",
            env_ids=["new"],
            tool_specs=SPECS,
            history=history,
        )
        options.update(kwargs)
        with pytest.raises(ValueError):
            BatchLoop(**options)


def test_reverse_branch_completions_keep_siblings_and_never_merge():
    loop = make_loop(("a", "b"))
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "look", (ToolCall("r", "read", {}),)))
    tool = loop.next_request()
    loop.complete(ToolCompletion(tool.request_id, (("a", "one"), ("b", "two"))))
    before = loop.history
    first, second = loop.next_request(), loop.next_request()
    for request in (second, first):
        loop.complete(LlmCompletion(request.request_id, "same final"))
    result = loop.result()
    assert len(result.classes) == 2  # identical final answers cannot merge classes
    assert all(before.load(e.continuation).assistant is None for e in before.root.observations)
    assert all(
        result.history.load(e.continuation).assistant.content == "same final"
        for e in result.history.root.observations
    )


def test_grouping_callback_failure_is_recoverable_after_partial_tool_completion():
    def comparison_key(value):
        if value == "invalid":
            raise ValueError("cannot group")
        return value

    loop = make_loop(comparison_key=comparison_key)
    request = loop.next_request()
    loop.complete(
        LlmCompletion(
            request.request_id, "look", (ToolCall("a", "read", {}), ToolCall("b", "read", {}))
        )
    )
    first, second = loop.next_request(), loop.next_request()
    loop.complete(ToolCompletion(second.request_id, (("old", "second"),)))
    snapshot = loop.history
    with pytest.raises(ValueError, match="cannot group"):
        loop.complete(ToolCompletion(first.request_id, (("old", "invalid"),)))
    assert loop.history is snapshot
    assert loop.outstanding_tool == 1
    loop.complete(ToolCompletion(first.request_id, (("old", "first"),)))
    request = loop.next_request()
    assert [m.content for m in request.messages[-2:]] == ["first", "second"]
    loop.complete(LlmCompletion(request.request_id, "done"))
    assert loop.done


def test_fulfilled_assistant_snapshot_can_restart_tools_not_resume_requests():
    loop = make_loop()
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "look", (ToolCall("r", "read", {}),)))
    snapshot = loop.history
    replay = make_loop(("new",), snapshot)
    tool = replay.next_request()
    assert isinstance(tool, ToolRequest)
    replay.complete(ToolCompletion(tool.request_id, (("new", "observation"),)))
    assert isinstance(replay.next_request(), LlmRequest)
    assert snapshot.root.observations == ()
