import pickle
from dataclasses import replace

import pytest

from lockstep_loop import (
    BatchLoop,
    History,
    HistoryNode,
    LlmCompletion,
    LlmRequest,
    Message,
    Observation,
    OutputRecord,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
    UserContinuation,
)

SPECS = (ToolSpec("probe", "", {}), ToolSpec("status", "", {}), ToolSpec("investigate", "", {}))


def nested_loop(envs, history=None, depth=2):
    return BatchLoop(
        prompt="main",
        env_ids=envs,
        tool_specs=SPECS,
        recording=True,
        history=history,
        subagent=SubagentTool(
            "investigate", "", {}, lambda args: args["question"], lambda _: SPECS, depth
        ),
    )


def nested_drive(loop, values):
    llms, snapshots, tools = [], [], []
    while not loop.done:
        pending = []
        while (request := loop.next_request()) is not None:
            pending.append(request)
        assert pending
        # Mixed ordinary tools, concurrent child calls, and independent branches.
        for request in reversed(pending):
            if isinstance(request, LlmRequest):
                scope = request.messages[0].content
                llms.append((scope, request.env_ids))
                calls = ()
                if len(request.messages) == 1:
                    if scope == "main":
                        calls = (
                            ToolCall("local", "status", {}),
                            ToolCall("child", "investigate", {"question": "child"}),
                            ToolCall("sibling", "investigate", {"question": "sibling"}),
                        )
                    elif scope == "child":
                        calls = (
                            ToolCall("nested", "investigate", {"question": "grand"}),
                            ToolCall("probe", "probe", {"scope": scope}),
                        )
                    else:
                        calls = (ToolCall("probe", "probe", {"scope": scope}),)
                loop.complete(
                    LlmCompletion(request.request_id, "thinking" if calls else "same answer", calls)
                )
            else:
                tools.append(request)
                assert request.tool_call.name != "investigate"  # handled only by core
                loop.complete(
                    ToolCompletion(
                        request.request_id,
                        tuple(
                            (
                                env,
                                "ok"
                                if request.tool_call.name == "status"
                                else values[env] + ":" + request.tool_call.args["scope"],
                            )
                            for env in reversed(request.env_ids)
                        ),
                    )
                )
            snapshots.append(loop.history)
    return loop.result(), llms, snapshots, tools


def child_history(history, name):
    entry = next(e for e in history.root.subagents if e.call_id == name)
    return History(entry.history, history.store)


def restore(history, loaded=None):
    # Caller codec/backend only. Each serialized record is shallow.
    records = {ref: pickle.dumps(node) for ref, node in history.store.records().items()}

    def load(ref):
        if loaded is not None:
            loaded.append(ref)
        return pickle.loads(records[ref])

    return History(pickle.loads(pickle.dumps(history.descriptor)), RecordStore(load=load))


def test_nested_histories_mixed_calls_depth_and_equal_parent_answers():
    loop = nested_loop(("a", "b"))
    result, llms, snapshots, _ = nested_drive(loop, {"a": "A", "b": "B"})
    old = result.history
    assert len(result.classes) == 1  # child traces split, parent answers do not
    assert result.classes[0].env_ids == ("a", "b")
    assert [s for s, _ in llms].count("main") == 2
    child = child_history(old, "child")
    grand = child_history(child, "nested")
    assert grand.descriptor.subagent is None  # configured depth was exhausted
    assert "investigate" not in [spec.name for spec in grand.descriptor.tools]
    assert len(grand.root.observations) == 2
    assert {e.evidence[0].outputs[0].tool_output for e in grand.root.observations} == {
        "A:grand",
        "B:grand",
    }
    assert len(old.root.observations) == 1
    assert [o.tool_output for o in old.root.observations[0].representative] == [
        "ok",
        "same answer",
        "same answer",
    ]
    # A snapshot while children are open remains open after the run completes.
    early = snapshots[0]
    assert early.root.subagents
    assert all(early.load(s.history.root).assistant is None for s in early.root.subagents)

    replay, llms, _, tools = nested_drive(nested_loop(("next",), restore(old)), {"next": "A"})
    assert not llms
    assert len(tools) == 4  # local + child probe + grand probe + sibling probe
    assert replay.results == {"next": "same answer"}

    changed, llms, _, _ = nested_drive(nested_loop(("same", "new"), old), {"same": "A", "new": "C"})
    assert llms and all(scope != "main" for scope, _ in llms)
    assert len(changed.classes) == 1
    extended_grand = child_history(child_history(changed.history, "child"), "nested")
    assert len(extended_grand.root.observations) == 3
    assert len(grand.root.observations) == 2  # old roots/evidence retained
    assert len(old.root.observations[0].evidence) == 2
    for value in ("A", "B", "C"):
        _, llms, _, _ = nested_drive(nested_loop(("again",), changed.history), {"again": value})
        assert not llms
    with pytest.raises(ValueError, match="incompatible"):
        nested_loop(("bad",), old, depth=1)


def test_repeated_subagent_call_ids_cannot_accept_stale_completions():
    loop = nested_loop(("a",))
    old_child_id = None
    for _ in range(2):
        request = loop.next_request()
        loop.complete(
            LlmCompletion(
                request.request_id,
                "delegate",
                (ToolCall("same", "investigate", {"question": "child"}),),
            )
        )
        child = loop.next_request()
        if old_child_id is not None:
            assert child.request_id != old_child_id
            with pytest.raises(ValueError, match="no outstanding"):
                loop.complete(LlmCompletion(old_child_id, "stale"))
            assert loop.outstanding_llm == 1
        loop.complete(LlmCompletion(child.request_id, "same answer"))
        old_child_id = child.request_id
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "done"))
    old = loop.history
    assert loop.done
    # Child-only recorded turns settle to completion without any new work.
    replay = nested_loop(("next",), old)
    assert replay.done
    assert replay.next_request() is None
    assert replay.result().results == {"next": "done"}
    assert child_history(old, "same").root.env_ids == ("a",)


def test_invalid_nested_completion_preserves_routes_and_snapshot():
    loop = nested_loop(("a", "b"))
    request = loop.next_request()
    loop.complete(
        LlmCompletion(
            request.request_id,
            "delegate",
            (ToolCall("child", "investigate", {"question": "child"}),),
        )
    )
    child = loop.next_request()
    before = loop.history
    duplicate = ToolCall("p", "probe", {})
    with pytest.raises(ValueError, match="duplicate"):
        loop.complete(LlmCompletion(child.request_id, "bad", (duplicate, duplicate)))
    assert loop.history.descriptor == before.descriptor
    assert loop.outstanding_llm == 1
    loop.complete(LlmCompletion(child.request_id, "probe", (duplicate,)))
    tool = loop.next_request()
    before = loop.history
    with pytest.raises(ValueError, match="exactly one"):
        loop.complete(ToolCompletion(tool.request_id, (("a", "A"),)))
    assert loop.outstanding_tool == 1
    assert loop.history.descriptor == before.descriptor
    loop.complete(ToolCompletion(tool.request_id, (("a", "A"), ("b", "A"))))
    final = loop.next_request()
    before_final = loop.history
    with pytest.raises(TypeError):
        loop.complete(LlmCompletion(final.request_id, []))
    assert loop.outstanding_llm == 1
    assert loop.history.descriptor == before_final.descriptor
    loop.complete(LlmCompletion(final.request_id, "same"))
    parent = loop.next_request()
    loop.complete(LlmCompletion(parent.request_id, "done"))
    assert loop.done
    assert not child_history(before, "child").root.observations
    assert child_history(loop.history, "child").root.observations[0].evidence[1].env_id == "b"


def simple_loop(envs, history=None, store=None):
    return BatchLoop(
        prompt="inspect",
        env_ids=envs,
        tool_specs=SPECS[:1],
        recording=True,
        history=history,
        record_store=store,
    )


def sequential(loop, values, turns):
    llms = 0
    while not loop.done:
        request = loop.next_request()
        assert request is not None
        if isinstance(request, LlmRequest):
            llms += 1
            calls = (
                (ToolCall("p", "probe", {"turn": request.stats.depth}),)
                if request.stats.depth < turns
                else ()
            )
            loop.complete(LlmCompletion(request.request_id, "same final", calls))
        else:
            loop.complete(
                ToolCompletion(
                    request.request_id, tuple((env, values[env]) for env in request.env_ids)
                )
            )
    return loop.result(), llms


def test_deep_shallow_persistence_and_lazy_replay_do_not_load_sibling_paths():
    loop = simple_loop(("left", "right"), store=RecordStore("initial-writer"))
    result, _ = sequential(loop, {"left": "L", "right": "R"}, turns=256)
    old = result.history
    # The records' child fields are addresses, never recursively encoded nodes.
    loaded = []
    backend = {}

    def publish(history):
        for ref, node in history.store.records().items():
            encoded = pickle.dumps(node)
            assert ref not in backend or backend[ref] == encoded
            backend[ref] = encoded
        return pickle.dumps(history.descriptor)

    def reopen(descriptor):
        def load(ref):
            loaded.append(ref)
            return pickle.loads(backend[ref])

        return History(pickle.loads(descriptor), RecordStore(load=load))

    original_descriptor = publish(old)
    restored = reopen(original_descriptor)
    assert loaded == []  # attaching a loader does not traverse any nodes
    right = old.root.observations[1].continuation
    replay, llms = sequential(simple_loop(("today",), restored), {"today": "L"}, turns=256)
    assert llms == 0
    assert replay.results == {"today": "same final"}
    assert right not in loaded
    assert len(loaded) == 257  # selected root/path only, not old ancestor versions
    assert old.root.observations[0].evidence[0].env_id == "left"
    assert len(old.root.observations[0].evidence) == 1
    replay_descriptor = publish(replay.history)
    changed, calls = sequential(simple_loop(("new",), reopen(replay_descriptor)), {"new": "NEW"}, 1)
    assert calls == 1
    extended_descriptor = publish(changed.history)
    for value in ("L", "R", "NEW"):
        _, calls = sequential(
            simple_loop(("again",), reopen(extended_descriptor)), {"again": value}, 256
        )
        assert calls == 0
    assert sequential(simple_loop(("old",), reopen(original_descriptor)), {"old": "R"}, 256)[1] == 0


def test_user_continuations_are_scoped_by_resolved_path_not_answer_value():
    old, _ = sequential(simple_loop(("a", "b")), {"a": "A", "b": "B"}, 1)
    # Resolve the investigation anew using snapshot + stable environment. No
    # public session/endpoint ID, nor lookup by the shared "same final" answer.
    resolving = simple_loop(("a",), old.history)
    _, calls = sequential(resolving, {"a": "A"}, 1)
    assert calls == 0
    extended = resolving.append_user_message("Please check again.")
    answered = extended.load(extended.root.observations[0].continuation)
    untouched = extended.load(extended.root.observations[1].continuation)
    assert answered.assistant.content == untouched.assistant.content == "same final"
    assert not untouched.user_continuations
    edge = answered.user_continuations[0]
    assert edge.resolved_env_id == "a"
    assert edge.message == Message(role="user", content="Please check again.")
    assert extended.load(edge.continuation).assistant is None
    assert not old.history.load(old.history.root.observations[0].continuation).user_continuations
    assert not resolving.history.load(
        resolving.history.root.observations[0].continuation
    ).user_continuations
    resumed, calls = sequential(simple_loop(("a",), extended), {"a": "A"}, 1)
    assert calls == 1
    assert resumed.results == {"a": "same final"}
    # Representability of another investigation yielding the SAME answer, then
    # another arbitrary user turn. These are storage records, not routing policy.
    store = extended.store
    open_ref = store.put(HistoryNode())
    same_answer = store.put(
        HistoryNode(
            assistant=Message(role="assistant", content="same final"),
            user_continuations=(
                UserContinuation(Message(role="user", content="One more check"), open_ref),
            ),
        )
    )
    investigate = store.put(
        HistoryNode(
            assistant=Message(
                role="assistant", content="checking", tool_calls=(ToolCall("p", "probe", {}),)
            ),
            observations=(Observation((OutputRecord("A"),), (), same_answer),),
        )
    )
    # Construct a different immutable snapshot; the prior continuation stays open.
    new_answer = store.put(
        replace(answered, user_continuations=(replace(edge, continuation=investigate),))
    )
    root = extended.root
    new_root = store.put(
        replace(
            root,
            observations=(
                replace(root.observations[0], continuation=new_answer),
                *root.observations[1:],
            ),
        )
    )
    populated = replace(extended, descriptor=replace(extended.descriptor, root=new_root))
    restored = restore(populated)
    user_edge = restored.load(restored.root.observations[0].continuation).user_continuations[0]
    checked = restored.load(user_edge.continuation)
    final = restored.load(checked.observations[0].continuation)
    assert final.assistant.content == answered.assistant.content
    assert final.user_continuations[0].message.content == "One more check"
    assert extended.load(edge.continuation).assistant is None
    assert sequential(simple_loop(("a",), old.history), {"a": "A"}, 1)[1] == 0


def test_storage_addresses_are_not_content_hashes_and_writer_reuse_is_rejected():
    store = RecordStore("reserved-writer")
    first, second = store.put(HistoryNode()), store.put(HistoryNode())
    assert first != second  # even equal records have distinct allocated addresses
    assert store.load(first) == store.load(second)
    invalid_writer = RecordStore("reserved-writer", load=store.load)
    with pytest.raises(ValueError, match="namespace reused"):
        invalid_writer.load(second)


def test_non_recording_legacy_labels_and_collisions_are_preserved():
    class SameRepr:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            return 0

        def __eq__(self, other):
            return self.value == other.value

        def __repr__(self):
            return "identical"

    for outputs, expected in (
        ({"a": "A", "b": "B"}, {"('A',)", "('B',)"}),
        ({"a": SameRepr(1), "b": SameRepr(2)}, {"(identical,)", "(identical,) [collision 2]"}),
    ):
        loop = BatchLoop(prompt="inspect", env_ids=("a", "b"), tool_specs=SPECS[:1])
        result, _ = sequential(loop, outputs, 1)
        assert set(result.tree.children) == expected
        assert len({c.id for c in result.classes}) == 2
        assert result.history is None
