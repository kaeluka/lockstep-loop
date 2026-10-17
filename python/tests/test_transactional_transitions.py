"""Fallible child preparation must never fulfill a pending assistant turn."""

import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lockstep_loop import (
    BatchLoop,
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
    ToolSpec,
)

SPECS = (ToolSpec("probe", "", {}), ToolSpec("investigate", "", {}))


def make_loop(sub, *, recording=True, history=None, env="old"):
    return BatchLoop(
        prompt="parent",
        env_ids=[env],
        tool_specs=SPECS,
        subagent=sub,
        history_contract="prepare/v1" if recording else None,
        history=history,
    )


def state(loop):
    # Freeze builders rather than retaining references to their mutable lists.
    return (
        loop.history,
        loop._seq,
        loop._next_branch,
        loop._next_child,
        tuple(loop._ready_llm),
        tuple(loop._ready_tool),
        dict(loop._outstanding),
        tuple(loop._branches),
        tuple(b.branch_id for b in loop._finals),
        tuple(
            (
                b.branch_id,
                b.steps,
                tuple(b.messages),
                b.history_path,
                b.turn_calls,
                tuple((call, tuple(values.items())) for call, values in b.turn_results.items()),
            )
            for b in loop._partition
        ),
        tuple((key, state(link.child)) for key, link in loop._children.items()),
        tuple((key, link.child._id_prefix) for key, link in loop._child_ids.items()),
        loop.outstanding_llm,
        loop.outstanding_tool,
        loop.error,
    )


def mixed(bad_args):
    return (
        ToolCall("ordinary", "probe", {}),
        ToolCall("good", "investigate", {"required": "good"}),
        ToolCall("bad", "investigate", bad_args),
    )


def serve(loop, *, replay=False):
    while not loop.done:
        pending = []
        while (request := loop.next_request()) is not None:
            pending.append(request)
        assert pending, "stranded work"
        for request in reversed(pending):
            if isinstance(request, LlmRequest):
                assert not replay, "persisted conversation disagrees with successful live execution"
                if request.messages[0].content == "parent":
                    assert request.messages[1].content == "corrected"
                    values = [m.content for m in request.messages if m.role == "tool"]
                    reply = "|".join(values)
                else:
                    reply = request.messages[0].content
                loop.complete(LlmCompletion(request.request_id, reply))
            else:
                loop.complete(
                    ToolCompletion(
                        request.request_id, tuple((env, "ordinary") for env in request.env_ids)
                    )
                )
    assert loop.outstanding_count == 0
    return loop.result()


def persisted(history):
    records = {ref: pickle.dumps(node) for ref, node in history.store.records().items()}
    return History(
        pickle.loads(pickle.dumps(history.descriptor)),
        RecordStore(load=lambda ref: pickle.loads(records[ref])),
    )


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("failure", ["prompt", "specs"])
@pytest.mark.parametrize("corrected_tools", [False, True])
def test_failed_mixed_turn_is_unfulfilled_and_retry_agrees_with_replay(
    recording, failure, corrected_tools
):
    prompts, specs = [], []

    def build(args):
        prompts.append(dict(args))
        return "child-" + args["required"]

    def tool_specs(env):
        specs.append(env)
        yield SPECS[0]
        if failure == "specs" and len(specs) == 2:
            raise RuntimeError("spec iterator failed after yielding")

    sub = SubagentTool("investigate", "", {}, build, tool_specs)
    loop = make_loop(sub, recording=recording)
    request = loop.next_request()
    before = state(loop)
    bad_args = {} if failure == "prompt" else {"required": "bad"}
    with pytest.raises(KeyError if failure == "prompt" else RuntimeError):
        loop.complete(LlmCompletion(request.request_id, "bad", mixed(bad_args)))
    assert state(loop) == before
    if recording:
        # Preparation may allocate orphan OPEN child nodes, but must not even
        # create a recorded assistant fact for the unvalidated "bad" turn.
        assert all(node.assistant is None for node in loop.history.store.records().values())
    assert loop.next_request() is None
    assert not loop.done and loop.outstanding_llm == 1
    assert prompts == [{"required": "good"}, bad_args]
    assert len(specs) == (1 if failure == "prompt" else 2)

    calls = mixed({"required": "fixed"}) if corrected_tools else ()
    loop.complete(LlmCompletion(request.request_id, "corrected", calls))
    # Successful preparation is used directly, not rerun after commit.
    assert len(prompts) == (4 if corrected_tools else 2)
    expected = "ordinary|child-good|child-fixed" if corrected_tools else "corrected"
    result = serve(loop)
    assert result.results == {"old": expected}
    assert loop._partition[0].steps == (2 if corrected_tools else 1)
    if recording:
        assert before[0].root.assistant is None
        assert result.history.root.assistant.content == "corrected"
        assert result.history.root.assistant.tool_calls == calls
        replay = make_loop(sub, history=persisted(result.history), env="new")
        assert serve(replay, replay=True).results == {"new": expected}


def test_failed_child_preparation_after_feedback_does_not_reinject_or_poison_history():
    sub = SubagentTool("investigate", "", {}, lambda a: a["required"], lambda _: SPECS)
    original = make_loop(sub)
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "old answer"))
    loop = BatchLoop(
        prompt="parent",
        env_ids=["old"],
        tool_specs=SPECS,
        subagent=sub,
        history_contract="prepare/v1",
        history=original.history,
        feedback=Feedback("check again"),
    )
    request = loop.next_request()
    assert [m.content for m in request.messages[-2:]] == ["old answer", "check again"]
    before = state(loop)
    with pytest.raises(KeyError):
        loop.complete(LlmCompletion(request.request_id, "bad", mixed({})))
    assert state(loop) == before
    loop.complete(LlmCompletion(request.request_id, "corrected"))
    assert loop.done and loop.result().results == {"old": "corrected"}
    history = loop.history
    assert len(history.root.user_continuations) == 1
    assert (
        history.load(history.root.user_continuations[0].continuation).assistant.content
        == "corrected"
    )
    replay = make_loop(sub, history=persisted(history), env="new")
    assert replay.done and replay.result().results == {"new": "corrected"}
    assert not original.history.root.user_continuations


def test_staged_commit_preserves_the_serving_lock_and_concurrent_routes():
    sub = SubagentTool("investigate", "", {}, lambda a: "child-" + a["required"], lambda _: SPECS)
    loop = make_loop(sub)
    lock = loop._lock
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "corrected", mixed({"required": "fixed"})))
    pending = []
    while (request := loop.next_request()) is not None:
        pending.append(request)
    assert len(pending) == 3
    barrier = threading.Barrier(3)

    def complete(request):
        barrier.wait(timeout=5)
        if isinstance(request, LlmRequest):
            loop.complete(LlmCompletion(request.request_id, request.messages[0].content))
        else:
            loop.complete(ToolCompletion(request.request_id, (("old", "ordinary"),)))

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(complete, pending))
    assert loop._lock is lock
    assert loop.outstanding_count == 0
    result = serve(loop)
    assert result.results == {"old": "ordinary|child-good|child-fixed"}
    assert serve(make_loop(sub, history=result.history, env="new"), replay=True).results == {
        "new": "ordinary|child-good|child-fixed"
    }


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("error_type", [FeedbackResolutionError, InvalidHistoryError])
def test_callback_exception_type_alone_does_not_commit_a_terminal_failure(recording, error_type):
    def build(_):
        raise error_type("caller callback, not a detected loop failure")

    sub = SubagentTool("investigate", "", {}, build, lambda _: SPECS)
    loop = make_loop(sub, recording=recording)
    request = loop.next_request()
    before = state(loop)
    with pytest.raises(error_type):
        loop.complete(LlmCompletion(request.request_id, "bad", mixed({})))
    assert state(loop) == before
    assert loop.error is None and loop.outstanding_llm == 1
    loop.complete(LlmCompletion(request.request_id, "corrected"))
    assert loop.done and loop.result().results == {"old": "corrected"}


@pytest.mark.parametrize("recording", [False, True])
def test_callback_failure_in_nested_completion_preserves_parent_routes(recording):
    sub = SubagentTool("investigate", "", {}, lambda a: a["required"], lambda _: SPECS, 2)
    loop = make_loop(sub, recording=recording)
    parent = loop.next_request()
    loop.complete(
        LlmCompletion(
            parent.request_id,
            "corrected",
            (ToolCall("outer", "investigate", {"required": "child"}),),
        )
    )
    child = loop.next_request()
    before = state(loop)
    with pytest.raises(KeyError):
        loop.complete(LlmCompletion(child.request_id, "bad child", mixed({})))
    assert state(loop) == before
    assert loop.outstanding_llm == 1
    loop.complete(LlmCompletion(child.request_id, "corrected child"))
    result = serve(loop)
    assert result.results == {"old": "corrected child"}
    if recording:
        entry = result.history.root.subagents[0]
        assert result.history.load(entry.history.root).assistant.content == "corrected child"
        replay = make_loop(sub, history=result.history, env="new")
        assert serve(replay, replay=True).results == {"new": "corrected child"}


def test_parent_callback_failure_after_child_completion_restores_child_pending_route():
    broken = False

    def build(args):
        if broken and args["required"] == "second":
            raise KeyError("second child unavailable")
        return args["required"]

    sub = SubagentTool("investigate", "", {}, build, lambda _: SPECS)
    original = make_loop(sub)
    request = original.next_request()
    original.complete(
        LlmCompletion(
            request.request_id, "first", (ToolCall("first", "investigate", {"required": "first"}),)
        )
    )
    child = original.next_request()
    original.complete(LlmCompletion(child.request_id, "read", (ToolCall("p", "probe", {}),)))
    tool = original.next_request()
    original.complete(ToolCompletion(tool.request_id, (("old", "value"),)))
    child = original.next_request()
    original.complete(LlmCompletion(child.request_id, "first answer"))
    request = original.next_request()
    original.complete(
        LlmCompletion(
            request.request_id,
            "second",
            (ToolCall("second", "investigate", {"required": "second"}),),
        )
    )
    child = original.next_request()
    original.complete(LlmCompletion(child.request_id, "second answer"))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "done"))
    replay = make_loop(sub, history=original.history, env="new")
    tool = replay.next_request()
    before = state(replay)
    broken = True
    with pytest.raises(KeyError):
        replay.complete(ToolCompletion(tool.request_id, (("new", "value"),)))
    assert state(replay) == before
    assert replay.outstanding_tool == 1 and replay.error is None
    broken = False
    replay.complete(ToolCompletion(tool.request_id, (("new", "value"),)))
    assert replay.done and replay.result().results == {"new": "done"}
    history = replay.history
    first = history.load(history.root.subagents[0].history.root)
    assert [e.env_id for e in first.observations[0].evidence] == ["old", "new"]


@pytest.mark.parametrize("poll", ["done", "next_request", "result"])
def test_callback_failure_while_polling_cached_child_rollup_is_atomic(poll):
    broken = False
    calls = []

    def build(args):
        calls.append(args["required"])
        if broken and args["required"] == "second":
            raise KeyError("unavailable second prompt")
        return args["required"]

    sub = SubagentTool("investigate", "", {}, build, lambda _: SPECS)
    original = make_loop(sub)
    for name in ("first", "second"):
        request = original.next_request()
        original.complete(
            LlmCompletion(
                request.request_id, name, (ToolCall(name, "investigate", {"required": name}),)
            )
        )
        child = original.next_request()
        original.complete(LlmCompletion(child.request_id, name + " answer"))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "done"))
    replay = make_loop(sub, history=original.history, env="new")
    before = state(replay)
    broken = True
    with pytest.raises(KeyError):
        if poll == "done":
            _ = replay.done
        else:
            getattr(replay, poll)()
    assert state(replay) == before
    assert replay.error is None
    broken = False
    count = len(calls)
    assert replay.done and replay.result().results == {"new": "done"}
    assert calls[count:] == ["second"]


@pytest.mark.parametrize("failure", ["prompt", "specs", "compatibility"])
def test_last_tool_completion_prepares_replayed_children_before_consumption(failure):
    broken = False
    callbacks = []

    def build(args):
        callbacks.append("prompt")
        if broken and failure == "prompt":
            raise KeyError("temporarily unavailable prompt input")
        return args["required"]

    def specs(_):
        callbacks.append("specs")
        yield ToolSpec("probe", "changed" if broken and failure == "compatibility" else "", {})
        if broken and failure == "specs":
            raise RuntimeError("temporarily unavailable specs")

    sub = SubagentTool("investigate", "", {}, build, specs)
    original = make_loop(sub)
    first = original.next_request()
    original.complete(LlmCompletion(first.request_id, "read first", (ToolCall("p", "probe", {}),)))
    tool = original.next_request()
    original.complete(ToolCompletion(tool.request_id, (("old", "ordinary"),)))
    second = original.next_request()
    original.complete(
        LlmCompletion(
            second.request_id,
            "delegate",
            (ToolCall("child", "investigate", {"required": "child"}),),
        )
    )
    child = original.next_request()
    original.complete(LlmCompletion(child.request_id, "answer"))
    final = original.next_request()
    original.complete(LlmCompletion(final.request_id, "done"))
    old = original.history

    replay = make_loop(sub, history=old, env="new")
    tool = replay.next_request()
    before = state(replay)
    broken = True
    with pytest.raises(
        {"prompt": KeyError, "specs": RuntimeError, "compatibility": ValueError}[failure]
    ):
        replay.complete(ToolCompletion(tool.request_id, (("new", "ordinary"),)))
    assert state(replay) == before
    assert replay.next_request() is None and replay.outstanding_tool == 1
    broken = False
    calls_before_retry = len(callbacks)
    replay.complete(ToolCompletion(tool.request_id, (("new", "ordinary"),)))
    assert replay.done and replay.result().results == {"new": "done"}
    assert len(callbacks) == calls_before_retry + 2  # prompt and specs once each
    assert old.root.env_ids == ("old",)
    again = make_loop(sub, history=persisted(replay.history), env="again")
    again.complete(ToolCompletion(again.next_request().request_id, (("again", "ordinary"),)))
    assert again.done and again.result().results == {"again": "done"}
