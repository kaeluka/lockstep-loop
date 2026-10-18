"""Public-handle lifecycle and mock-only startup/load/run/save acceptance tests."""

import pickle
from dataclasses import FrozenInstanceError

import pytest

from lockstep_loop import (
    Agent,
    AgentRun,
    AgentSnapshot,
    Budget,
    Feedback,
    FeedbackResolutionError,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

READ = ToolSpec("read", "Read the greeting", {})


def drive(run, values, *, answer=None):
    llms = 0
    while not run.done:
        requests = []
        while (r := run.next_request()) is not None:
            requests.append(r)
        assert requests
        for r in reversed(requests):
            if isinstance(r, LlmRequest):
                llms += 1
                calls = (
                    (ToolCall("a", "read", {}), ToolCall("b", "read", {}))
                    if len(r.messages) == 1
                    else ()
                )
                run.complete(
                    LlmCompletion(
                        r.request_id,
                        "inspect" if calls else (answer or r.messages[-1].content),
                        calls,
                    )
                )
            else:
                run.complete(
                    ToolCompletion(
                        r.request_id, tuple((env, values[env]) for env in reversed(r.env_ids))
                    )
                )
    return llms


def test_fresh_sequential_branch_and_immutable_captures():
    agent = Agent(prompt="greetings", tool_specs=[READ])
    empty = agent.snapshot()
    run = agent.start(["alice", "bob"])
    assert isinstance(run, AgentRun) and run.snapshot() is empty
    assert drive(run, {"alice": "hello", "bob": "hello"}) == 2
    # No result() needed to advance the owner.
    old = agent.snapshot()
    assert old is run.snapshot() and old is not empty
    assert run.result().snapshot is old
    assert run.result().results == {"alice": "hello", "bob": "hello"}
    assert empty.root.assistant is None
    with pytest.raises(FrozenInstanceError):
        old.descriptor = empty.descriptor
    asset = next(old.assets())
    assert asset.snapshot is old
    assert [o.tool_output for o in asset.observations[0].representative] == ["hello", "hello"]
    same = agent.start(["carol"])
    assert drive(same, {"carol": "".join(["hel", "lo"])}) == 0
    changed = agent.start(["dave"])
    assert drive(changed, {"dave": "hola"}) == 1
    assert len(agent.snapshot().root.observations) == 2
    assert len(old.root.observations) == 1 and asset.value == "hello"
    assert len(asset.observations[0].evidence) == 2
    for value in ("hello", "hola"):
        assert drive(agent.start(["again"]), {"again": value}) == 0
    fork = Agent.from_snapshot(old)
    assert drive(fork.start(["again"]), {"again": "hello"}) == 0
    assert len(fork.snapshot().root.observations) == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_overlapping_runs_capture_roots_last_completion_wins_without_merge(reverse):
    agent = Agent(prompt="greetings", tool_specs=[READ])
    drive(agent.start(["seed"]), {"seed": "known"})
    base = agent.snapshot()
    left, right = agent.start(["left"]), agent.start(["right"])
    assert agent.snapshot() is base
    runs = [(left, "left"), (right, "right")]
    if reverse:
        runs.reverse()
    first, first_env = runs[0]
    last, last_env = runs[1]
    assert drive(first, {first_env: first_env}) == 1
    first_snapshot = first.snapshot()
    assert agent.snapshot() is first_snapshot
    assert drive(last, {last_env: last_env}) == 1
    last_snapshot = last.snapshot()
    assert agent.snapshot() is last_snapshot
    assert {e.representative[0].tool_output for e in last_snapshot.root.observations} == {
        "known",
        last_env,
    }
    assert {e.representative[0].tool_output for e in first_snapshot.root.observations} == {
        "known",
        first_env,
    }
    first.result()
    first.snapshot()
    assert first.done and first.next_request() is None
    with pytest.raises(ValueError):
        first.complete(LlmCompletion("bogus", "stale"))
    assert agent.snapshot() is last_snapshot
    for run, env in runs:
        independent = Agent.from_snapshot(run.snapshot())
        assert drive(independent.start(["again"]), {"again": env}) == 0
    # A later start captures the newly published head, not the original base.
    assert drive(agent.start(["new"]), {"new": last_env}) == 0
    assert len(base.root.observations) == 1


def test_storage_roundtrip_restores_prompt_specs_lazily_and_retains_opaque_evidence():
    agent = Agent(prompt=b"greetings", tool_specs=[READ], comparison_key=bytes.strip)
    assert (
        drive(
            agent.start(["a", "b", "other"]),
            {
                "a": b"hello ",
                "b": b"hello",
                "other": b"different",
            },
        )
        == 3
    )
    snapshot = agent.snapshot()
    # Caller-owned codec/backend, standing in for startup/shutdown persistence.
    backend = {ref: pickle.dumps(node) for ref, node in snapshot.store.records().items()}
    descriptor = pickle.loads(pickle.dumps(snapshot.descriptor))
    loaded = []

    def load(ref):
        loaded.append(ref)
        return pickle.loads(backend[ref])

    restored = AgentSnapshot(descriptor, RecordStore(load=load))
    handle = Agent.from_snapshot(restored, comparison_key=bytes.strip)
    assert not loaded
    run = handle.start(["today"])
    assert drive(run, {"today": b"hello\n"}) == 0
    assert run.result().results == {"today": b"hello "}
    assert snapshot.root.observations[1].continuation not in loaded
    evidence = handle.snapshot().root.observations[0].evidence
    assert [e.outputs[0].tool_output for e in evidence] == [b"hello ", b"hello", b"hello\n"]
    # Persist only new writes while retaining all old records; save/reopen again.
    backend.update(
        {ref: pickle.dumps(node) for ref, node in handle.snapshot().store.records().items()}
    )
    reopened = Agent.from_snapshot(
        AgentSnapshot(
            pickle.loads(pickle.dumps(handle.snapshot().descriptor)), RecordStore(load=load)
        ),
        comparison_key=bytes.strip,
    )
    assert drive(reopened.start(["again"]), {"again": b"hello"}) == 0
    assert len(snapshot.root.observations[0].evidence) == 2


def test_final_only_immediate_completion_publication_and_invalid_completion():
    agent = Agent(prompt="p", tool_specs=[])
    empty = agent.snapshot()
    run = agent.start(["a"])
    request = run.next_request()
    with pytest.raises(TypeError):
        run.complete(ToolCompletion(request.request_id, (("a", "wrong"),)))
    assert run.outstanding_count == 1 and agent.snapshot() is empty
    assert run.snapshot() is empty
    run.complete(LlmCompletion(request.request_id, "answer"))
    first = agent.snapshot()
    replay = agent.start(["b"])  # already completed AND published before any inspection
    assert agent.snapshot().root.env_ids == ("a", "b")
    latest = agent.snapshot()
    assert replay.done and replay.snapshot() is latest
    run.result()
    run.snapshot()
    assert agent.snapshot() is latest and first.root.env_ids == ("a",)
    # Independent handles share immutable data, not publications.
    fork = Agent.from_snapshot(first)
    fork.start(["c"])
    assert fork.snapshot().root.env_ids == ("a", "c")
    assert agent.snapshot() is latest


def test_abandonment_failure_partial_snapshots_and_budget_are_not_assistant_facts():
    agent = Agent(prompt="p", tool_specs=[READ])
    empty = agent.snapshot()
    abandoned = agent.start(["a"])
    r = abandoned.next_request()
    abandoned.complete(LlmCompletion(r.request_id, "inspect", (ToolCall("r", "read", {}),)))
    partial = abandoned.snapshot()
    assert partial.root.assistant.content == "inspect"
    del abandoned
    assert agent.snapshot() is empty
    bounded = agent.start(["a"], budget=Budget(0))
    assert bounded.done and agent.snapshot() is bounded.snapshot()
    assert agent.snapshot().root.assistant is None
    assert bounded.result().results == {"a": "(budget exhausted after 0 steps)"}
    with pytest.raises(FeedbackResolutionError):
        agent.start(["a"], feedback=Feedback("no target"))
    assert agent.snapshot() is empty
    resumed = Agent.from_snapshot(partial)
    run = resumed.start(["a"], budget=Budget(1))
    r = run.next_request()
    before = run.snapshot()
    with pytest.raises(ValueError):
        run.complete(ToolCompletion(r.request_id, ()))
    assert resumed.snapshot() is partial and run.snapshot() is before
    run.complete(ToolCompletion(r.request_id, (("a", "seen"),)))
    checkpoint = resumed.snapshot()
    assert run.done and checkpoint is run.snapshot()
    assert checkpoint.load(checkpoint.root.observations[0].continuation).assistant is None
    assert drive(resumed.start(["a"]), {"a": "seen"}) == 1
    known = resumed.snapshot()
    failure = resumed.start(["a"], feedback=Feedback("revise"))
    r = failure.next_request()
    with pytest.raises(FeedbackResolutionError):
        failure.complete(ToolCompletion(r.request_id, (("a", "unseen"),)))
    assert failure.done and failure.error is not None
    with pytest.raises(FeedbackResolutionError):
        failure.result()
    assert resumed.snapshot() is known
    assert checkpoint.load(checkpoint.root.observations[0].continuation).assistant is None


def test_feedback_budget_publishes_real_open_continuation_immediately():
    agent = Agent(prompt="p", tool_specs=[])
    run = agent.start(["a"])
    run.complete(LlmCompletion(run.next_request().request_id, "old"))
    old = agent.snapshot()
    asset = next(old.assets())
    bounded = agent.start(["a"], feedback=Feedback("revise"), budget=Budget(1))
    opened = agent.snapshot()
    assert opened is bounded.snapshot() and opened is not old
    assert not next(opened.assets()).is_complete
    assert next(opened.assets()).value == "old" and asset.is_complete
    followup = agent.start(["b"])
    r = followup.next_request()
    assert [m.content for m in r.messages] == ["p", "old", "revise"]
    followup.complete(LlmCompletion(r.request_id, "new"))
    newest = agent.snapshot()
    bounded.result()
    assert agent.snapshot() is newest
    assert not old.root.user_continuations and asset.value == "old"


def test_specialized_nested_children_cache_updates_and_cached_start_publication():
    leaf = SubagentTool("leaf", "", {}, lambda _: "leaf prompt", lambda _: (), subagents=())
    outer = SubagentTool("outer", "", {}, lambda _: "outer prompt", lambda _: (), 2, (leaf,))
    agent = Agent(prompt="parent", tool_specs=[], subagents=[outer])
    run = agent.start(["a"])
    for name in ("outer", "leaf"):
        r = run.next_request()
        run.complete(LlmCompletion(r.request_id, "delegate", (ToolCall("opaque/sub:1", name, {}),)))
    for answer in ("leaf value", "outer value", "parent value"):
        run.complete(LlmCompletion(run.next_request().request_id, answer))
    old = agent.snapshot()
    updates = run.result().child_updates
    assert all(isinstance(u.snapshot, AgentSnapshot) for u in updates)
    child = next(u.snapshot for u in updates if u.tool_call.name == "outer")
    assert old.child(old.root.subagents[0]) == child
    with pytest.raises(ValueError, match="subagent"):
        Agent.from_snapshot(old)  # executable factories must be reattached
    backend = {ref: pickle.dumps(node) for ref, node in old.store.records().items()}
    loaded = []

    def load(ref):
        loaded.append(ref)
        return pickle.loads(backend[ref])

    imported = AgentSnapshot(pickle.loads(pickle.dumps(old.descriptor)), RecordStore(load=load))
    restored = Agent.from_snapshot(imported, subagents=[outer])
    assert not loaded
    assert next(imported.assets()).value == "parent value"
    assert len(loaded) == 2 and child.descriptor.root not in loaded
    imported_child = imported.child(imported.root.subagents[0])
    assert next(imported_child.assets()).value == "outer value"
    assert len(loaded) == 4
    assert child.root.subagents[0].snapshot.root not in loaded
    cached = restored.start(["b"])
    assert restored.snapshot().root.env_ids == ("a", "b")
    assert cached.done and cached.next_request() is None
    assert len(cached.result().child_updates) == 2
    changed_parent = Agent(
        prompt="changed parent",
        tool_specs=[],
        subagents=[outer],
        record_store=child.store,
        agent_resolver=lambda q: child.descriptor if q == child.descriptor.query() else None,
    )
    other = changed_parent.start(["c"])
    r = other.next_request()
    other.complete(LlmCompletion(r.request_id, "delegate", (ToolCall("new", "outer", {}),)))
    r = other.next_request()
    assert r.messages[0].content == "changed parent" and r.messages[-1].content == "outer value"
    other.complete(LlmCompletion(r.request_id, "new parent value"))
    assert changed_parent.snapshot() is other.snapshot()
    assert old.root.env_ids == ("a",) and child.root.env_ids == ("a",)


def test_last_tool_completion_publishes_without_polling_and_failure_cannot_revert_new_head():
    agent = Agent(prompt="p", tool_specs=[READ])
    drive(agent.start(["old"]), {"old": "known"})
    old = agent.snapshot()
    failed = agent.start(["bad"], feedback=Feedback("revise"))
    replay = agent.start(["new"])
    requests = [replay.next_request(), replay.next_request()]
    for r in reversed(requests):
        replay.complete(ToolCompletion(r.request_id, (("new", "known"),)))
    # No done/result/snapshot call on replay after the last completion.
    newer = agent.snapshot()
    assert newer.root.env_ids == ("old", "new")
    assert newer.load(newer.root.observations[0].continuation).env_ids == ("old", "new")
    requests = [failed.next_request(), failed.next_request()]
    failed.complete(ToolCompletion(requests[0].request_id, (("bad", "unseen"),)))
    with pytest.raises(FeedbackResolutionError):
        failed.complete(ToolCompletion(requests[1].request_id, (("bad", "unseen"),)))
    assert agent.snapshot() is newer
    assert failed.snapshot() is old  # resolution rollback never publishes this older root


def test_cached_child_rollup_callback_failure_rolls_back_completion_before_publication():
    broken = False

    def prompt(args):
        if broken and args["name"] == "second":
            raise ValueError("factory unavailable")
        return args["name"]

    sub = SubagentTool("child", "", {}, prompt, lambda _: ())
    agent = Agent(prompt="parent", tool_specs=[READ], subagents=[sub])
    original = agent.start(["old"])
    r = original.next_request()
    original.complete(LlmCompletion(r.request_id, "read", (ToolCall("r", "read", {}),)))
    original.complete(ToolCompletion(original.next_request().request_id, (("old", "known"),)))
    for name in ("first", "second"):
        r = original.next_request()
        original.complete(
            LlmCompletion(r.request_id, "delegate", (ToolCall(name, "child", {"name": name}),))
        )
        original.complete(LlmCompletion(original.next_request().request_id, name + " answer"))
    original.complete(LlmCompletion(original.next_request().request_id, "done"))
    old = agent.snapshot()
    replay = agent.start(["new"])
    r = replay.next_request()
    before = replay.snapshot()
    broken = True
    with pytest.raises(ValueError, match="unavailable"):
        replay.complete(ToolCompletion(r.request_id, (("new", "known"),)))
    assert replay.snapshot() is before and replay.outstanding_count == 1
    assert agent.snapshot() is old and replay.error is None
    broken = False
    replay.complete(ToolCompletion(r.request_id, (("new", "known"),)))
    assert agent.snapshot().root.env_ids == ("old", "new")
    assert replay.done and replay.result().results == {"new": "done"}
