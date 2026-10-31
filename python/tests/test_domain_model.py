"""Domain refactor: one history trie, opaque initial messages, and asset accessors."""

import pickle
from dataclasses import replace

import pytest

import lockstep_loop
from lockstep_loop import (
    Agent,
    AgentDescriptor,
    AgentQuery,
    AgentSnapshot,
    Budget,
    Feedback,
    HistoryNode,
    LlmCompletion,
    LlmRequest,
    MessageEnvelope,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)
from lockstep_loop.core import BatchLoop

READ = ToolSpec("read", "", {})
FINAL = ToolSpec("submitFinalAnswer", "", {})
INITIAL = (
    MessageEnvelope("system", "system"),
    MessageEnvelope("user", "one"),
    MessageEnvelope("user", "two"),
)


def test_initial_sequence_is_frozen_ordered_opaque_and_pickleable():
    supplied = [MessageEnvelope("system", b"\x00\xff"), MessageEnvelope("user", b"\x80\x00")]
    initial = tuple(supplied)
    agent = Agent(initial_messages=supplied, tool_specs=[])
    supplied.clear()
    assert agent.snapshot().initial_messages == initial
    run = agent.start(["old"], budget=Budget(1))
    request = run.next_request()
    assert request.messages == initial
    assert request.stats.depth == request.stats.split_depth == 0
    answer = b"\x00\x80answer"
    run.complete(LlmCompletion(request.request_id, answer))
    old = agent.snapshot()
    conversation = next(old.conversations())
    assert conversation.messages[:2] == initial
    assert conversation.get_last_answer_asset() == answer
    restored = pickle.loads(pickle.dumps(old))
    assert restored.initial_messages == initial
    assert restored.store.namespace != old.store.namespace
    fresh = Agent.from_snapshot(restored)
    replay = fresh.start(["new"])
    assert replay.done and replay.next_request() is None
    assert replay.result().results == {"new": answer}
    assert old.root.env_ids == ("old",)
    assert conversation.get_last_answer_asset() == answer


@pytest.mark.parametrize(
    "changed",
    [
        (*INITIAL[:2], MessageEnvelope("user", "changed")),
        (replace(INITIAL[0], role="user"), *INITIAL[1:]),
        (INITIAL[0], INITIAL[2], INITIAL[1]),
        INITIAL[:2],
        (*INITIAL, MessageEnvelope("user", "extra")),
        (INITIAL[0], replace(INITIAL[1], tool_calls=(ToolCall("x", "read", {}),)), INITIAL[2]),
        (*INITIAL[:2], replace(INITIAL[2], tool_call_id="other")),
    ],
)
def test_compatibility_checks_all_initial_content_order_and_metadata(changed):
    snapshot = Agent(initial_messages=INITIAL, tool_specs=[]).snapshot()
    with pytest.raises(ValueError, match="incompatible snapshot initial_messages"):
        BatchLoop(initial_messages=changed, env_ids=["e"], tool_specs=[], snapshot=snapshot)
    matching = BatchLoop(initial_messages=INITIAL, env_ids=["e"], tool_specs=[], snapshot=snapshot)
    assert matching.next_request().messages == INITIAL


def test_empty_initial_sequences_reject_at_entry_without_loading_records():
    for constructor, options in ((Agent, {}), (BatchLoop, {"env_ids": ["e"]})):
        with pytest.raises(ValueError, match="at least one message"):
            constructor(initial_messages=[], tool_specs=[], **options)
        with pytest.raises(TypeError, match="MessageEnvelope"):
            constructor(initial_messages="not an envelope sequence", tool_specs=[], **options)
    original = Agent(initial_messages=INITIAL, tool_specs=[]).snapshot()
    lazy = AgentSnapshot(
        replace(original.descriptor, initial_messages=()),
        RecordStore(load=lambda _: pytest.fail("restoring empty input must not load history")),
    )
    with pytest.raises(ValueError, match="at least one message"):
        Agent.from_snapshot(lazy)
    source = list(INITIAL)
    descriptor = AgentDescriptor(source, (), original.descriptor.root)
    query = AgentQuery(source, ())
    source.clear()
    assert descriptor.initial_messages == query.initial_messages == INITIAL


def test_child_lookup_uses_full_initial_sequence_and_is_parent_independent():
    child_initial = (MessageEnvelope("system", "child rules"), *INITIAL[1:])
    child = Agent(initial_messages=child_initial, tool_specs=[])
    seed = child.start(["old"])
    request = seed.next_request()
    seed.complete(LlmCompletion(request.request_id, "child answer"))
    base = child.snapshot()
    queries = []

    def resolver(query):
        queries.append(query)
        return base.descriptor

    definition = SubagentTool("child", "", {}, lambda _: child_initial, lambda _: (), subagents=())
    parent = Agent(
        initial_messages=(MessageEnvelope("user", "unrelated parent"),),
        tool_specs=[],
        subagents=[definition],
        agent_resolver=resolver,
        record_store=base.store,
    )
    run = parent.start(["new"])
    request = run.next_request()
    run.complete(LlmCompletion(request.request_id, "delegate", (ToolCall("c", "child", {}),)))
    request = run.next_request()
    assert queries[0].initial_messages == child_initial
    assert request.messages[0].content == "unrelated parent"
    assert request.messages[-1].content == "child answer"
    run.complete(LlmCompletion(request.request_id, "done"))
    assert len(run.result().child_updates) == 1

    changed = replace(
        definition,
        build_initial_messages=lambda _: (
            replace(child_initial[0], content="different child rules"),
            *child_initial[1:],
        ),
    )
    other = Agent(
        initial_messages=(MessageEnvelope("user", "other parent"),),
        tool_specs=[],
        subagents=[changed],
        agent_resolver=resolver,
        record_store=base.store,
    )
    run = other.start(["new"])
    request = run.next_request()
    old = other.snapshot()
    with pytest.raises(ValueError, match="incompatible snapshot initial_messages"):
        run.complete(LlmCompletion(request.request_id, "delegate", (ToolCall("c", "child", {}),)))
    assert run.outstanding_llm == 1
    assert other.snapshot() is old
    assert run.snapshot().root.assistant is None
    run.complete(LlmCompletion(request.request_id, "corrected"))
    assert run.done


def test_empty_child_initial_sequence_does_not_leak_prepared_siblings_or_requests():
    def build(args):
        return INITIAL if args["valid"] else ()

    definition = SubagentTool("child", "", {}, build, lambda _: (), subagents=())
    parent = Agent(initial_messages=INITIAL, tool_specs=[READ], subagents=[definition])
    run = parent.start(["e"])
    request = run.next_request()
    before = run.snapshot()
    calls = (
        ToolCall("ordinary", "read", {}),
        ToolCall("good", "child", {"valid": True}),
        ToolCall("bad", "child", {"valid": False}),
    )
    with pytest.raises(ValueError, match="at least one message"):
        run.complete(LlmCompletion(request.request_id, "delegate", calls))
    assert run.outstanding_llm == 1 and run.outstanding_tool == 0
    assert run.next_request() is None
    assert run.snapshot() is before and parent.snapshot() is before
    run.complete(LlmCompletion(request.request_id, "corrected"))
    assert run.done
    assert run.result().results == {"e": "corrected"}


def test_empty_child_on_replay_keeps_owner_and_pending_tool_completion_correctable():
    valid = True

    def build(_):
        return INITIAL if valid else ()

    definition = SubagentTool("child", "", {}, build, lambda _: (), subagents=())
    parent = Agent(initial_messages=INITIAL, tool_specs=[READ], subagents=[definition])
    original = parent.start(["old"])
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "read", (ToolCall("r", "read", {}),)))
    request = original.next_request()
    original.complete(ToolCompletion(request.request_id, (("old", "same"),)))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "delegate", (ToolCall("c", "child", {}),)))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "child answer"))
    request = original.next_request()
    original.complete(LlmCompletion(request.request_id, "parent answer"))
    base = parent.snapshot()
    valid = False
    replay = parent.start(["new"])
    request = replay.next_request()
    pending = replay.snapshot()
    completion = ToolCompletion(request.request_id, (("new", "same"),))
    with pytest.raises(ValueError, match="at least one message"):
        replay.complete(completion)
    assert parent.snapshot() is base and replay.snapshot() is pending
    assert replay.outstanding_tool == 1 and not replay.done
    valid = True
    replay.complete(completion)
    assert replay.done and replay.next_request() is None
    assert replay.result().results == {"new": "parent answer"}
    assert base.root.env_ids == ("old",)


def test_asset_accessors_keep_missing_null_pending_and_latest_named_call_distinct():
    agent = Agent(initial_messages=(MessageEnvelope("user", None),), tool_specs=[FINAL, READ])
    empty = agent.snapshot()
    conversation = next(empty.conversations())
    assert not conversation.has_answer
    for accessor in (
        conversation.get_last_answer_asset,
        lambda: conversation.get_final_tool_input_asset("submitFinalAnswer"),
    ):
        with pytest.raises(LookupError):
            accessor()
    run = agent.start(["a", "b"])
    request = run.next_request()
    run.complete(
        LlmCompletion(
            request.request_id,
            "submit",
            (
                ToolCall("first", "submitFinalAnswer", {"answer": 1}),
                ToolCall("second", "submitFinalAnswer", {"answer": 2}),
                ToolCall("third", "read", {"answer": 99}),
            ),
        )
    )
    pending = run.snapshot()
    conversation = next(pending.conversations())
    assert conversation.get_final_tool_input_asset("submitFinalAnswer") == {"answer": 2}
    assert not conversation.is_complete and not conversation.has_answer
    requests = [run.next_request("tool") for _ in range(3)]
    for request in reversed(requests):
        run.complete(ToolCompletion(request.request_id, tuple((e, "ack") for e in request.env_ids)))
    request = run.next_request()
    run.complete(
        LlmCompletion(
            request.request_id, "submit again", (ToolCall("fourth", "submitFinalAnswer", None),)
        )
    )
    null_pending = next(run.snapshot().conversations())
    assert null_pending.get_final_tool_input_asset("submitFinalAnswer") is None
    request = run.next_request("tool")
    run.complete(ToolCompletion(request.request_id, tuple((e, "ack") for e in request.env_ids)))
    request = run.next_request()
    run.complete(LlmCompletion(request.request_id, None))
    result = run.result()
    conversation = next(result.snapshot.conversations())
    assert len(result.classes) == 1 and result.classes[0].env_ids == ("a", "b")
    assert len(conversation.observations) == 2  # Both non-splitting checkpoints persisted.
    assert conversation.has_answer and conversation.get_last_answer_asset() is None
    assert conversation.get_final_tool_input_asset("submitFinalAnswer") is None
    with pytest.raises(LookupError):
        conversation.get_final_tool_input_asset("missing")
    assert next(pending.conversations()).get_final_tool_input_asset("submitFinalAnswer") == {
        "answer": 2
    }
    imported = pickle.loads(pickle.dumps(result.snapshot))
    restored = Agent.from_snapshot(imported)
    replay = restored.start(["c"])
    while not replay.done:
        request = replay.next_request()
        assert not isinstance(request, LlmRequest)
        replay.complete(
            ToolCompletion(request.request_id, tuple((e, "ack") for e in request.env_ids))
        )
    assert replay.result().classes[0].env_ids == ("c",)
    assert result.classes[0].env_ids == ("a", "b")
    feedback = restored.start(["c"], feedback=Feedback("revise"))
    while (request := feedback.next_request("tool")) is not None:
        feedback.complete(ToolCompletion(request.request_id, (("c", "ack"),)))
    opened = next(feedback.snapshot().conversations())
    assert not opened.is_complete
    assert opened.get_last_answer_asset() is None
    assert opened.get_final_tool_input_asset("submitFinalAnswer") is None


def test_initial_assistant_context_does_not_make_budget_fallback_a_completed_answer():
    initial = (MessageEnvelope("system", "context"), MessageEnvelope("assistant", "initial answer"))
    loop = BatchLoop(initial_messages=initial, env_ids=["e"], tool_specs=[], budget=Budget(0))
    assert loop.done and loop.next_request() is None
    result = loop.result()
    assert result.results == {"e": "(budget exhausted after 0 steps)"}
    old = result.snapshot
    assert old.root.assistant is None and not old.root.env_ids
    conversation = next(old.conversations())
    assert conversation.get_last_answer_asset() == "initial answer"
    assert not conversation.is_complete
    with pytest.raises(ValueError, match="completed assistant answer"):
        loop.append_user_message("feedback")
    assert loop.snapshot is old and not old.root.user_continuations
    resumed = BatchLoop(
        initial_messages=initial, env_ids=["new"], tool_specs=[], snapshot=old, budget=Budget(1)
    )
    request = resumed.next_request()
    assert request.stats.depth == 0
    resumed.complete(LlmCompletion(request.request_id, "real answer"))
    extended = resumed.append_user_message("feedback")
    feedback = BatchLoop(
        initial_messages=initial, env_ids=["new"], tool_specs=[], snapshot=extended
    )
    request = feedback.next_request()
    feedback.complete(LlmCompletion(request.request_id, "revised"))
    assert feedback.result().results == {"new": "revised"}
    assert conversation.get_last_answer_asset() == "initial answer"
    assert old.root.assistant is None


def test_context_accessors_borrow_opaque_content_and_do_not_execute_calls():
    class Opaque:
        def __copy__(self):
            pytest.fail("must not clone content for inspection")

        def __deepcopy__(self, memo):
            pytest.fail("must not deep-copy content")

    answer = Opaque()
    args = {"typed": [1, 2]}
    initial = (
        MessageEnvelope("system", Opaque()),
        MessageEnvelope("assistant", answer),
        MessageEnvelope("assistant", Opaque(), (ToolCall("f", "submitFinalAnswer", args),)),
        MessageEnvelope("tool", Opaque(), tool_call_id="f"),
        MessageEnvelope("user", Opaque()),
    )
    snapshot = Agent(initial_messages=initial, tool_specs=[]).snapshot()
    conversation = next(snapshot.conversations())
    assert conversation.get_last_answer_asset() is answer
    assert conversation.get_final_tool_input_asset("submitFinalAnswer") is args
    assert conversation.messages == initial
    assert not conversation.is_complete
    assert snapshot.root == HistoryNode()


def test_canonical_public_types_do_not_keep_old_aliases_or_partition_trees():
    for obsolete in ("Asset", "Message", "SplitNode", "SplitEdge"):
        assert not hasattr(lockstep_loop, obsolete)
    run = Agent(initial_messages=INITIAL, tool_specs=[]).start(["e"])
    request = run.next_request()
    run.complete(LlmCompletion(request.request_id, "done"))
    result = run.result()
    assert not hasattr(result, "tree") and not hasattr(run._engine, "_partition")
    conversation = next(result.snapshot.conversations())
    assert isinstance(conversation, lockstep_loop.Conversation)
    assert not hasattr(conversation, "value") and not hasattr(conversation, "published_message")
