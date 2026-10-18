import pickle
import random
from dataclasses import replace

import pytest

from lockstep_loop import (
    BatchLoop,
    Budget,
    Feedback,
    FeedbackResolutionError,
    History,
    InvalidHistoryError,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolRequest,
    ToolSpec,
)

SPECS = tuple(ToolSpec(name, "read-only", {}) for name in ("read", "detail", "extra"))
PROMPT = "Inspect the repository"
NOTE = "Also inspect /foo"
OLD = "no issues"


def loop_for(envs, history=None, feedback=None, **options):
    return BatchLoop(
        prompt=PROMPT,
        env_ids=envs,
        tool_specs=SPECS,
        history=history,
        feedback=feedback,
        **options,
    )


def output_for(request, data):
    return ToolCompletion(
        request.request_id,
        tuple((env, data[env][request.tool_call.name]) for env in reversed(request.env_ids)),
    )


def model(request):
    if len(request.messages) == 1:
        calls = (ToolCall("r", "read", {"path": "/entry"}),)
    elif request.messages[-1].role == "user":
        calls = (
            ToolCall("d", "detail", {"path": "/foo"}),
            ToolCall("e", "extra", {"path": "/metadata"}),
        )
    else:
        calls = ()
    detail = [m.content for m in request.messages if m.role == "tool" and m.tool_call_id == "d"]
    answer = "found" if detail and detail[-1] == "vulnerability" else OLD
    return LlmCompletion(request.request_id, "investigating" if calls else answer, calls)


def drive(loop, data, *, allow_llm=True):
    llms, tools = [], []
    while not loop.done:
        pending = []
        while (request := loop.next_request()) is not None:
            pending.append(request)
        assert pending, "stranded work"
        for request in reversed(pending):
            if isinstance(request, LlmRequest):
                assert allow_llm, "unexpected fresh reasoning during replay"
                llms.append(request)
                loop.complete(model(request))
            else:
                tools.append(request)
                loop.complete(output_for(request, data))
    assert loop.error is None
    assert loop.outstanding_count == 0
    return loop.result(), llms, tools


def initial():
    result, _, _ = drive(
        loop_for(("A", "B", "D")),
        {
            "A": {"read": "shared"},
            "B": {"read": "shared"},
            "D": {"read": "unrelated"},
        },
    )
    return result.history


def shared_node(history):
    edge = next(
        e for e in history.root.observations if e.representative[0].tool_output.strip() == "shared"
    )
    return history.load(edge.continuation)


def publish(history, backend):
    for reference, node in history.store.records().items():
        encoded = pickle.dumps(node)
        assert reference not in backend or backend[reference] == encoded
        backend[reference] = encoded
    return pickle.dumps(history.descriptor)


def restore(descriptor, backend):
    return History(
        pickle.loads(descriptor), RecordStore(load=lambda ref: pickle.loads(backend[ref]))
    )


def test_resume_once_shared_path_inheritance_splits_and_immutable_snapshots():
    old = initial()
    old_shared = shared_node(old)
    old_unrelated = old.root.observations[1].continuation
    loop = loop_for(("A",), old, Feedback(NOTE))
    assert loop.next_request("llm") is None
    read = loop.next_request("tool")
    assert read.tool_call.name == "read"
    loop.complete(output_for(read, {"A": {"read": "shared"}}))
    request = loop.next_request("llm")
    assert [m.role for m in request.messages] == ["user", "assistant", "tool", "assistant", "user"]
    assert request.messages[0].content == PROMPT
    assert [m.content for m in request.messages[-2:]] == [OLD, NOTE]
    assert request.messages[1].tool_calls[0].args == {"path": "/entry"}
    opened = loop.history
    edge = shared_node(opened).user_continuations[0]
    assert edge.message.content == NOTE
    assert opened.load(edge.continuation).assistant is None
    assert not old_shared.user_continuations
    assert opened.root.observations[1].continuation == old_unrelated

    # Invalid fulfillments leave both history and pending request recoverable.
    before = loop.history
    with pytest.raises(TypeError):
        loop.complete(ToolCompletion(request.request_id, (("A", "bad"),)))
    duplicate = ToolCall("dup", "detail", {})
    with pytest.raises(ValueError, match="duplicate"):
        loop.complete(LlmCompletion(request.request_id, "bad", (duplicate, duplicate)))
    assert loop.history is before
    assert loop.outstanding_llm == 1
    loop.complete(model(request))
    detail, extra = loop.next_request("tool"), loop.next_request("tool")
    before = loop.history
    with pytest.raises(ValueError, match="exactly one"):
        loop.complete(ToolCompletion(extra.request_id, (("wrong", "bad"),)))
    assert loop.history is before
    assert loop.outstanding_tool == 2
    loop.complete(output_for(extra, {"A": {"extra": "metadata"}}))
    assert loop.next_request("llm") is None
    loop.complete(output_for(detail, {"A": {"detail": "safe"}}))
    with pytest.raises(ValueError, match="no outstanding"):
        loop.complete(output_for(detail, {"A": {"detail": "overwrite"}}))
    final = loop.next_request("llm")
    assert [m.tool_call_id for m in final.messages[-2:]] == ["d", "e"]
    loop.complete(LlmCompletion(final.request_id, OLD))  # same answer is valid
    assert loop.done
    assert loop.next_request() is None
    revised = loop.result().history
    revised_shared = shared_node(revised)
    assert revised_shared.assistant.content == OLD
    after_user = revised.load(revised_shared.user_continuations[0].continuation)
    assert after_user.assistant.tool_calls[0].args == {"path": "/foo"}
    assert [o.tool_output for o in after_user.observations[0].evidence[0].outputs] == [
        "safe",
        "metadata",
    ]
    assert revised.load(after_user.observations[0].continuation).assistant.content == OLD
    assert opened.load(edge.continuation).assistant is None

    data = {
        "A": {"read": "shared", "detail": "safe", "extra": "metadata"},
        "B": {"read": "shared", "detail": "vulnerability", "extra": "metadata"},
        "C": {"read": "shared", "detail": "safe", "extra": "metadata"},
        "D": {"read": "unrelated"},
    }
    result, llms, tools = drive(loop_for(tuple(data), revised), data)
    assert result.results == {"A": OLD, "B": "found", "C": OLD, "D": OLD}
    assert len(llms) == 1 and llms[0].env_ids == ("B",)
    assert [m.content for m in llms[0].messages if m.role == "user"] == [PROMPT, NOTE]
    assert {r.env_ids for r in tools if r.tool_call.name != "read"} == {("A", "B", "C")}
    assert sorted(c.env_ids for c in result.classes) == [("A", "C"), ("B",), ("D",)]
    assert result.history.load(old_unrelated).user_continuations == ()
    assert not old_shared.user_continuations
    assert old.root.env_ids == ("A", "B", "D")
    assert len(after_user.observations) == 1
    _, _, tools = drive(loop_for(("B",), old), {"B": {"read": "shared"}}, allow_llm=False)
    assert [r.tool_call.name for r in tools] == ["read"]
    for env in data:
        drive(loop_for((env,), result.history), {env: data[env]}, allow_llm=False)


def test_interleaved_feedback_paths_preserve_each_others_updates():
    data = {
        "A": {"read": "shared", "detail": "safe", "extra": "metadata"},
        "B": {"read": "shared", "detail": "vulnerability", "extra": "metadata"},
        "C": {"read": "shared", "detail": "safe", "extra": "metadata"},
        "D": {"read": "unrelated", "detail": "safe", "extra": "metadata"},
    }
    first, _, _ = drive(loop_for(("A",), initial(), Feedback(NOTE)), data)
    second, _, _ = drive(loop_for(("D",), first.history, Feedback("Check another path")), data)
    assert not first.history.load(
        first.history.root.observations[1].continuation
    ).user_continuations
    for seed in range(8):
        rng = random.Random(seed)
        loop = loop_for(tuple(data), second.history)
        pending, llms = [], []
        while not loop.done:
            for _ in range(rng.randrange(1, 5)):
                request = loop.next_request(rng.choice(("any", "llm", "tool")))
                if request is None:
                    request = loop.next_request()
                if request is None:
                    break
                pending.append(request)
            assert pending
            request = pending.pop(rng.randrange(len(pending)))
            if isinstance(request, LlmRequest):
                llms.append(request)
                assert [m.content for m in request.messages if m.role == "user"] == [PROMPT, NOTE]
                loop.complete(model(request))
            else:
                loop.complete(output_for(request, data))
            assert loop.outstanding_count == len(pending)
        assert not pending
        assert len(llms) == 1 and llms[0].env_ids == ("B",)
        result = loop.result()
        assert result.results == {"A": OLD, "B": "found", "C": OLD, "D": OLD}
        assert len(result.classes) == 3
        assert len(shared_node(result.history).user_continuations) == 1


def test_repeated_identical_feedback_after_storage_roundtrip_targets_revised_endpoint():
    old = initial()
    data = {"A": {"read": "shared", "detail": "vulnerability", "extra": "metadata"}}
    first, _, _ = drive(loop_for(("A",), old, Feedback(NOTE)), data)
    assert first.results == {"A": "found"}
    backend = {}
    descriptor = publish(first.history, backend)
    restored = restore(descriptor, backend)
    loop = loop_for(("A",), restored, Feedback(NOTE))
    requests = []
    while (request := loop.next_request()) is not None:
        if isinstance(request, LlmRequest):
            break
        requests.append(request)
        loop.complete(output_for(request, data))
    assert isinstance(request, LlmRequest)
    assert len(requests) == 3  # old investigation AND previous feedback tools replayed
    assert [m.content for m in request.messages if m.role == "user"] == [PROMPT, NOTE, NOTE]
    assert request.messages[-2].content == "found"
    assert request.messages[-1].content == NOTE
    loop.complete(LlmCompletion(request.request_id, "found"))
    assert loop.done
    second = loop.result().history
    second_descriptor = publish(second, backend)
    reopened = restore(second_descriptor, backend)
    result, _, tools = drive(loop_for(("NEW",), reopened), {"NEW": data["A"]}, allow_llm=False)
    assert result.results == {"NEW": "found"} and len(tools) == 3
    first_end = restored.load(
        restored.load(shared_node(restored).user_continuations[0].continuation)
        .observations[0]
        .continuation
    )
    assert first_end.assistant.content == "found" and not first_end.user_continuations


def assert_failed(loop, old):
    assert loop.done
    assert isinstance(loop.error, FeedbackResolutionError)
    assert loop.next_request() is None and loop.outstanding_count == 0
    assert loop.history.descriptor == old.descriptor
    with pytest.raises(FeedbackResolutionError):
        loop.result()


def test_feedback_miss_is_terminal_not_correctable_validation_or_fresh_reasoning():
    old = initial()
    loop = loop_for(("A",), old, Feedback(NOTE))
    request = loop.next_request()
    assert isinstance(request, ToolRequest)
    with pytest.raises(ValueError, match="exactly one"):
        loop.complete(ToolCompletion(request.request_id, ()))
    assert loop.error is None and loop.outstanding_count == 1
    with pytest.raises(FeedbackResolutionError):
        loop.complete(ToolCompletion(request.request_id, (("A", "never seen"),)))
    assert_failed(loop, old)
    with pytest.raises(FeedbackResolutionError):
        loop.complete(ToolCompletion(request.request_id, (("A", "shared"),)))
    assert not shared_node(old).user_continuations


def test_partial_paths_and_unfinished_feedback_are_not_targets():
    pending = loop_for(("A",))
    empty = pending.history
    with pytest.raises(FeedbackResolutionError):
        loop_for(("A",), empty, Feedback(NOTE))
    request = pending.next_request()
    pending.complete(model(request))
    request = pending.next_request()
    pending.complete(ToolCompletion(request.request_id, (("A", "shared"),)))
    partial = pending.history  # observation known, assistant answer not yet fulfilled
    loop = loop_for(("A",), partial, Feedback(NOTE))
    tool = loop.next_request()
    with pytest.raises(FeedbackResolutionError):
        loop.complete(ToolCompletion(tool.request_id, (("A", "shared"),)))
    assert_failed(loop, partial)
    old = initial()
    first = loop_for(("A",), old, Feedback(NOTE))
    first.complete(output_for(first.next_request(), {"A": {"read": "shared"}}))
    open_feedback = first.history
    second = loop_for(("A",), open_feedback, Feedback("another intervention"))
    with pytest.raises(FeedbackResolutionError):
        second.complete(output_for(second.next_request(), {"A": {"read": "shared"}}))
    assert_failed(second, open_feedback)


def test_opaque_null_feedback_presence_and_constructor_limits():
    initial_loop = BatchLoop(
        prompt=b"prompt", env_ids=["A"], tool_specs=[]
    )
    request = initial_loop.next_request()
    initial_loop.complete(LlmCompletion(request.request_id, b"same"))
    old = initial_loop.history
    updated = old
    for _ in range(3):
        loop = BatchLoop(
            prompt=b"prompt",
            env_ids=["A"],
            tool_specs=[],
            history=updated,
            feedback=Feedback(None),
        )
        request = loop.next_request()
        assert request.messages[-1].role == "user" and request.messages[-1].content is None
        assert request.messages[-2].content == b"same"
        loop.complete(LlmCompletion(request.request_id, b"same"))
        assert loop.done
        updated = loop.history
    replay = BatchLoop(
        prompt=b"prompt", env_ids=["C"], tool_specs=[], history=updated
    )
    assert replay.done and replay.result().results == {"C": b"same"}
    assert not old.root.user_continuations
    bounded = BatchLoop(
        prompt=b"prompt", env_ids=["A"], tool_specs=[], history=old,
        feedback=Feedback(None), budget=Budget(100),
    )
    assert isinstance(bounded.next_request(), LlmRequest)
    for options in (
        {"history": None},
        {"env_ids": ["A", "B"]},
        {"prompt": b"changed"},
    ):
        config = dict(
            prompt=b"prompt",
            env_ids=["A"],
            tool_specs=[],
            history=old,
            feedback=Feedback(None),
        )
        config.update(options)
        with pytest.raises(ValueError):
            BatchLoop(**config)


def test_lossy_grouping_keeps_actual_evidence_and_shared_feedback_context():
    result, _, _ = drive(
        loop_for(("A", "B"), comparison_key=str.strip),
        {
            "A": {"read": "shared "},
            "B": {"read": "shared"},
        },
    )
    loop = loop_for(("B",), result.history, Feedback(NOTE), comparison_key=str.strip)
    loop.complete(output_for(loop.next_request(), {"B": {"read": "shared\n"}}))
    request = loop.next_request()
    assert request.messages[2].content == "shared "  # historical representative
    loop.complete(LlmCompletion(request.request_id, OLD))
    history = loop.history
    assert history.root.observations[0].evidence[-1].outputs[0].tool_output == "shared\n"
    replay = loop_for(("C",), history, comparison_key=str.strip)
    replay.complete(output_for(replay.next_request(), {"C": {"read": "shared\t"}}))
    assert replay.done and replay.result().results == {"C": OLD}
    assert history.root.observations[0].evidence[-1].outputs[0].tool_output == "shared\n"


def test_ambiguous_user_edges_are_rejected_not_implicitly_merged():
    old = initial()
    loop = loop_for(("A",), old, Feedback(NOTE))
    loop.complete(output_for(loop.next_request(), {"A": {"read": "shared"}}))
    history = loop.history
    node = shared_node(history)
    root = history.root
    bad_node = history.store.put(replace(node, user_continuations=node.user_continuations * 2))
    bad_root = history.store.put(
        replace(
            root,
            observations=(
                replace(root.observations[0], continuation=bad_node),
                *root.observations[1:],
            ),
        )
    )
    malformed = replace(history, descriptor=replace(history.descriptor, root=bad_root))
    loop = loop_for(("A",), malformed)
    with pytest.raises(InvalidHistoryError):
        loop.complete(output_for(loop.next_request(), {"A": {"read": "shared"}}))
    assert loop.done and isinstance(loop.error, InvalidHistoryError)
    # An unrelated path is still lazy and unaffected.
    drive(loop_for(("D",), malformed), {"D": {"read": "unrelated"}}, allow_llm=False)


def nested_loop(history=None, feedback=None):
    specs = (ToolSpec("investigate", "", {}), ToolSpec("probe", "", {}))
    return BatchLoop(
        prompt="parent",
        env_ids=["A"],
        tool_specs=specs,
        subagent=SubagentTool("investigate", "", {}, lambda a: a["prompt"], lambda _: specs, 1),
        history=history,
        feedback=feedback,
    )


def serve_nested(request):
    scope = request.messages[0].content
    calls = ()
    if scope == "parent" and (len(request.messages) == 1 or request.messages[-1].role == "user"):
        child = "old-child" if len(request.messages) == 1 else "new-child"
        calls = (
            ToolCall("child", "investigate", {"prompt": child}),
            ToolCall("local", "probe", {"scope": "local"}),
        )
    elif scope != "parent" and len(request.messages) == 1:
        calls = (ToolCall("p", "probe", {"scope": scope}),)
    return LlmCompletion(request.request_id, "same" if not calls else "checking", calls)


def nested_drive(loop, *, replay=False):
    llms = []
    while not loop.done:
        pending = []
        while (request := loop.next_request()) is not None:
            pending.append(request)
        assert pending
        for request in reversed(pending):
            if isinstance(request, LlmRequest):
                assert not replay
                llms.append(request)
                loop.complete(serve_nested(request))
            else:
                loop.complete(ToolCompletion(request.request_id, (("A", "known"),)))
    return loop.history, llms


def test_resolution_miss_in_nested_child_cancels_other_work_then_new_children_can_resume():
    pending = nested_loop()
    pending.complete(serve_nested(pending.next_request()))
    # The child exists but has no fulfilled assistant turn: do not generate one
    # just to manufacture a parent feedback target.
    with pytest.raises(FeedbackResolutionError):
        nested_loop(pending.history, Feedback(NOTE))
    old, _ = nested_drive(nested_loop())
    loop = nested_loop(old, Feedback(NOTE))
    child_tool, local_tool = loop.next_request(), loop.next_request()
    assert child_tool.tool_call.args["scope"] == "old-child"
    assert local_tool.tool_call.args["scope"] == "local"
    assert loop.outstanding_count == 2
    with pytest.raises(FeedbackResolutionError):
        loop.complete(ToolCompletion(child_tool.request_id, (("A", "unseen child evidence"),)))
    assert_failed(loop, old)
    with pytest.raises(FeedbackResolutionError):
        loop.complete(ToolCompletion(local_tool.request_id, (("A", "known"),)))
    revised, requests = nested_drive(nested_loop(old, Feedback(NOTE)))
    assert requests[0].messages[-1].content == NOTE  # no LLM in child resolution
    assert [r.messages[0].content for r in requests] == [
        "parent",
        "new-child",
        "new-child",
        "parent",
    ]
    _, requests = nested_drive(nested_loop(revised), replay=True)
    assert not requests
    assert not old.load(old.root.observations[0].continuation).user_continuations
