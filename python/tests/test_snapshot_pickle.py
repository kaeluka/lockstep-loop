"""Self-contained pickle transport of reachable history, never runtime machinery."""

import io
import pickle
from dataclasses import dataclass

import pytest

from lockstep_loop import (
    Agent,
    AgentDescriptor,
    AgentSnapshot,
    ChildAgentRun,
    EnvObservation,
    Feedback,
    HistoryNode,
    LlmCompletion,
    LlmRequest,
    Message,
    NodeRef,
    Observation,
    OutputRecord,
    RecordStore,
    SubagentRecord,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
    UserContinuation,
)

READ = ToolSpec("read", "Read stable evidence", {"type": "object"})


@pytest.fixture(params=[None, 4, pickle.HIGHEST_PROTOCOL], ids=["default", "protocol4", "highest"])
def protocol(request):
    return {} if request.param is None else {"protocol": request.param}


def roundtrip(snapshot, protocol):
    stream = io.BytesIO()
    pickle.dump(snapshot, stream, **protocol)
    stream.seek(0)
    return pickle.load(stream)


def drive(run, values):
    """Mock ordinary/nested investigations; complete independent requests backwards."""
    llms = 0
    while not run.done:
        requests = []
        while (r := run.next_request()) is not None:
            requests.append(r)
        assert requests
        for r in reversed(requests):
            if isinstance(r, LlmRequest):
                llms += 1
                calls = ()
                if len(r.messages) == 1:
                    scope = r.messages[0].content
                    calls = (ToolCall("read", "read", {"path": "/entry", "raw": b"\xff"}),)
                    if scope == b"parent":
                        calls += (ToolCall("outer", "outer", {}),)
                    elif scope == b"outer":
                        calls = (ToolCall("leaf", "leaf", {}),)
                content = (
                    b"inspect"
                    if calls
                    else b"revised"
                    if r.messages[-1].role == "user"
                    else b"answer:" + r.messages[-1].content
                )
                run.complete(LlmCompletion(r.request_id, content, calls))
            else:
                run.complete(
                    ToolCompletion(
                        r.request_id, tuple((env, values[env]) for env in reversed(r.env_ids))
                    )
                )
    return llms, run.result()


@pytest.mark.parametrize("completed", [False, True])
def test_empty_and_completed_snapshots_support_dump_load_and_from_snapshot(protocol, completed):
    agent = Agent(prompt=b"prompt", tool_specs=[READ])
    original = agent.snapshot()
    if completed:
        run = agent.start(["old"])
        r = run.next_request()
        run.complete(LlmCompletion(r.request_id, b"done"))
        original = agent.snapshot()
    restored = roundtrip(original, protocol)
    assert restored == original and restored.root == original.root
    assert restored.descriptor.tools == (READ,) and restored.prompt == b"prompt"
    assert restored.store.namespace != original.store.namespace
    assert restored.store.records() == {}  # bundled records are not new local writes
    agent = Agent.from_snapshot(restored)
    run = agent.start(["new"])
    if not completed:
        r = run.next_request()
        assert r.messages == (Message("user", b"prompt"),) and r.tools == (READ,)
        run.complete(LlmCompletion(r.request_id, b"done"))
        assert original.root.assistant is None
    assert run.done and run.next_request() is None
    assert run.result().results == {"new": b"done"}
    assert roundtrip(agent.snapshot(), protocol).root == agent.snapshot().root


def test_opaque_provider_payloads_and_unequal_evidence_survive_without_callbacks(protocol):
    key = lambda payload: payload["text"].strip()  # noqa: E731
    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(key)
    prompt = {"text": "prompt", "content_blocks": (b"\x00\xffimage",)}
    agent = Agent(prompt=prompt, tool_specs=[READ], comparison_key=key)
    run = agent.start(["a", "b"])
    assistant = {"text": "reasoning", "provider": {"encrypted": b"\xfe\x00", "usage": 42}}
    calls = (ToolCall("x", "read", {"opaque": b"\xff"}), ToolCall("y", "read", {}))
    run.complete(LlmCompletion(run.next_request().request_id, assistant, calls))
    tools = [run.next_request(), run.next_request()]
    outputs = ({"text": "same ", "bytes": b"a\x00"}, {"text": "same", "bytes": b"b\xff"})
    for r in reversed(tools):
        run.complete(ToolCompletion(r.request_id, (("b", outputs[1]), ("a", outputs[0]))))
    answer = {"text": "answer", "raw_response": (b"opaque", {"extra": [1, 2]})}
    run.complete(LlmCompletion(run.next_request().request_id, answer))
    original = agent.snapshot()
    restored = roundtrip(original, protocol)
    assert restored.root.assistant == Message("assistant", assistant, calls)
    asset = next(restored.assets())
    assert asset.value == answer and asset.messages[0].content == prompt
    assert asset.messages[2].content == outputs[0]  # historical representative
    evidence = asset.observations[0].evidence
    assert [e.outputs[0].tool_output for e in evidence] == list(outputs)
    assert evidence[0].outputs[0].tool_output is evidence[0].outputs[1].tool_output
    # Grouping is runtime policy: restoring a snapshot does not restore this lambda.
    default_run = Agent.from_snapshot(restored).start(["new"])
    tool = default_run.next_request()
    with pytest.raises(TypeError, match="unhashable"):
        default_run.complete(ToolCompletion(tool.request_id, (("new", outputs[0]),)))
    replay = Agent.from_snapshot(restored, comparison_key=key).start(["new"])
    new_output = {"text": "same\n", "bytes": b"new"}
    for _ in range(2):
        r = replay.next_request()
        assert not isinstance(r, LlmRequest)
        replay.complete(ToolCompletion(r.request_id, (("new", new_output),)))
    assert replay.done and replay.result().results == {"new": answer}
    assert next(replay.snapshot().assets()).messages[2].content == outputs[0]
    assert replay.snapshot().root.observations[0].evidence[-1].outputs[0].tool_output == new_output
    assert len(original.root.observations[0].evidence) == 2


def test_nested_children_feedback_and_all_invocation_base_result_roots_are_bundled(protocol):
    store = RecordStore()
    seed = Agent(prompt=b"leaf", tool_specs=[READ], record_store=store)
    drive(seed.start(["seed"]), {"seed": b"known"})
    base = seed.snapshot()
    leaf = SubagentTool("leaf", "", {}, lambda _: b"leaf", lambda _: [READ], subagents=())
    outer = SubagentTool("outer", "", {}, lambda _: b"outer", lambda _: [], 2, (leaf,))
    parent = Agent(
        prompt=b"parent",
        tool_specs=[READ],
        subagents=[outer],
        record_store=store,
        agent_resolver=lambda q, selected=base.descriptor: (
            selected if q == selected.query() else None
        ),
    )
    _, first = drive(parent.start(["a", "other"]), {"a": b"known", "other": b"different"})
    old_bytes = pickle.dumps(first.snapshot, **protocol)
    count, second = drive(parent.start(["again"]), {"again": b"known"})
    assert count == 0
    _, revised = drive(parent.start(["a"], feedback=Feedback(b"check again")), {"a": b"known"})
    expected_roots = {base.descriptor.root: base.root}
    for result in (first, second, revised):
        for update in result.child_updates:
            expected_roots[update.snapshot.descriptor.root] = update.snapshot.root
            if update.base is not None:
                expected_roots[update.base.root] = store.load(update.base.root)
    encoded = pickle.dumps(parent.snapshot(), **protocol)
    # No external backing store or runtime callback is available to loaded snapshots.
    del store, seed, base, parent, first, second, revised, result, update
    restored = pickle.loads(encoded)
    for reference, node in expected_roots.items():
        assert restored.load(reference) == node
    assert len(restored.root.subagents[0].runs) == 3
    child = restored.child(restored.root.subagents[0])
    assert len(child.root.subagents[0].runs) == 3
    assert next(restored.assets()).messages[-2] == Message("user", b"check again")
    with pytest.raises(ValueError, match="subagent"):
        Agent.from_snapshot(restored)  # factories must be supplied again
    current = Agent.from_snapshot(restored, subagents=[outer])
    count, result = drive(current.start(["a", "other"]), {"a": b"known", "other": b"different"})
    assert count == 0 and result.results["a"] == b"revised"
    old = Agent.from_snapshot(pickle.loads(old_bytes), subagents=[outer])
    count, old_result = drive(old.start(["a", "other"]), {"a": b"known", "other": b"different"})
    assert count == 0 and old_result.results["a"] != b"revised"
    assert old_result.results["other"] == result.results["other"]
    assert roundtrip(current.snapshot(), protocol).root == current.snapshot().root


def test_lazy_export_loads_all_reachable_branches_but_omits_other_runs_and_orphans(protocol):
    agent = Agent(prompt=b"prompt", tool_specs=[READ])
    drive(agent.start(["left", "right"]), {"left": b"L", "right": b"R"})
    drive(agent.start(["last"]), {"last": b"L"})
    selected = agent.snapshot()
    expected = {selected.descriptor.root, *(e.continuation for e in selected.root.observations)}
    unrelated = agent.start(["unrelated"])
    drive(unrelated, {"unrelated": b"not selected"})
    orphan = selected.store.put(HistoryNode(assistant=Message("assistant", lambda: None)))
    backend = dict(selected.store.records())
    loaded = []

    def load(ref):  # local closure cannot be pickled
        loaded.append(ref)
        return backend[ref]

    lazy = AgentSnapshot(selected.descriptor, RecordStore(load=load))
    assert not loaded
    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(load)
    restored = roundtrip(lazy, protocol)
    assert set(loaded) == expected and len(loaded) == len(expected)
    assert orphan not in loaded and unrelated.snapshot().descriptor.root not in loaded
    assert lazy.descriptor == selected.descriptor and lazy.store.records() == {}
    backend.clear()  # the restored loader must NOT consult the original backend
    for value in (b"L", b"R"):
        count, _ = drive(Agent.from_snapshot(restored).start(["new"]), {"new": value})
        assert count == 0
    with pytest.raises(KeyError):
        restored.load(orphan)
    with pytest.raises(KeyError):
        restored.load(unrelated.snapshot().descriptor.root)


@dataclass(frozen=True)
class Collision:
    value: int

    def __hash__(self):
        return 0

    def __repr__(self):
        return "same-debug-text"


def test_deep_graph_walk_is_iterative_address_deduplicated_and_preserves_sharing(protocol):
    store = RecordStore()
    tail = store.put(HistoryNode(assistant=Message("assistant", b"done")))
    equal_but_distinct = store.put(HistoryNode(assistant=Message("assistant", b"done")))
    root = tail
    left, right = OutputRecord(Collision(1)), OutputRecord(Collision(2))
    for _ in range(1800):
        root = store.put(
            HistoryNode(
                assistant=Message("assistant", b"read", (ToolCall("r", "read", {}),)),
                observations=(Observation((left,), (), root), Observation((right,), (), root)),
            )
        )
    # Two equal records at distinct addresses must not be collapsed by value/hash/debug text.
    node = store.load(root)
    root = store.put(
        HistoryNode(
            assistant=node.assistant,
            observations=(*node.observations, Observation((left,), (), equal_but_distinct)),
        )
    )
    source = store.records()
    loaded = []

    def load(ref):
        loaded.append(ref)
        return source[ref]

    original = AgentSnapshot(AgentDescriptor(b"prompt", (READ,), root), RecordStore(load=load))
    restored = roundtrip(original, protocol)
    assert len(loaded) == len(set(loaded)) == 1802
    assert original.store.records() == {} and original.descriptor.root == root
    assert restored.load(tail) == restored.load(equal_but_distinct)
    assert restored.load(tail) is not restored.load(equal_but_distinct)
    node = restored.root
    for _ in range(1800):
        first, second = node.observations[:2]
        assert first.representative[0].tool_output != second.representative[0].tool_output
        assert hash(first.representative[0].tool_output) == hash(
            second.representative[0].tool_output
        )
        assert first.continuation == second.continuation
        assert restored.load(first.continuation) is restored.load(second.continuation)
        node = restored.load(first.continuation)
    assert node.assistant.content == b"done"
    assert roundtrip(restored, protocol).descriptor == restored.descriptor


def test_independent_loads_allocate_unique_writers_and_repickle_without_external_records(protocol):
    agent = Agent(prompt=b"prompt", tool_specs=[READ])
    drive(agent.start(["old"]), {"old": b"known"})
    old = agent.snapshot()
    encoded = pickle.dumps(old, **protocol)
    first, second = pickle.loads(encoded), pickle.loads(encoded)
    assert len({old.store.namespace, first.store.namespace, second.store.namespace}) == 3
    a, b = Agent.from_snapshot(first), Agent.from_snapshot(second)
    drive(a.start(["a"]), {"a": b"left"})
    first_extended = a.snapshot()
    drive(b.start(["b"]), {"b": b"right"})
    second_extended = b.snapshot()
    writes_a, writes_b = first_extended.store.records(), second_extended.store.records()
    assert writes_a and writes_b and not (writes_a.keys() & writes_b.keys())
    assert not (old.store.records().keys() & (writes_a.keys() | writes_b.keys()))
    assert (
        len(old.root.observations)
        == len(first.root.observations)
        == len(second.root.observations)
        == 1
    )
    # Both independently written revisions can coexist in one incremental backend.
    backend = {**old.store.records(), **writes_a, **writes_b}
    for snapshot, output in ((first_extended, b"left"), (second_extended, b"right")):
        stored = AgentSnapshot(snapshot.descriptor, RecordStore(load=backend.__getitem__))
        count, _ = drive(Agent.from_snapshot(stored).start(["new"]), {"new": output})
        assert count == 0
    # Direct save/load/extend/save/load needs no manual record collection.
    a = Agent.from_snapshot(roundtrip(first_extended, protocol))
    drive(a.start(["later"]), {"later": b"extra"})
    final = roundtrip(a.snapshot(), protocol)
    for output in (b"known", b"left", b"extra"):
        count, _ = drive(Agent.from_snapshot(final).start(["new"]), {"new": output})
        assert count == 0
    assert len(first_extended.root.observations) == len(second_extended.root.observations) == 2


@pytest.mark.parametrize("field", ["prompt", "schema", "assistant", "output"])
def test_unpickleable_payload_errors_are_not_lossily_hidden_or_mutating(field):
    bad = lambda: None  # noqa: E731
    store = RecordStore()
    end = store.put(HistoryNode(assistant=Message("assistant", b"done")))
    outputs = (OutputRecord(bad if field == "output" else b"ok"),)
    root = store.put(
        HistoryNode(
            assistant=Message(
                "assistant", bad if field == "assistant" else b"read", (ToolCall("r", "read", {}),)
            ),
            observations=(Observation(outputs, (EnvObservation("e", outputs),), end),),
        )
    )
    specs = (ToolSpec("read", "", {"bad": bad}),) if field == "schema" else (READ,)
    snapshot = AgentSnapshot(
        AgentDescriptor(bad if field == "prompt" else b"prompt", specs, root), store
    )
    before = store.records()
    with pytest.raises((AttributeError, pickle.PicklingError)):
        pickle.dumps(snapshot)
    assert snapshot.descriptor.root == root and store.records() == before
    assert snapshot.root is before[root]


@pytest.mark.parametrize("edge", ["observation", "feedback", "child", "base", "result"])
def test_missing_reachable_records_fail_including_provenance_only_references(edge):
    store = RecordStore()
    missing = NodeRef("missing", 0)
    present = store.put(HistoryNode())
    descriptor = lambda ref: AgentDescriptor(b"child", (), ref)  # noqa: E731
    node = HistoryNode(
        observations=(Observation((), (), missing),) if edge == "observation" else (),
        user_continuations=(UserContinuation(Message("user", b"note"), missing),)
        if edge == "feedback"
        else (),
        subagents=(
            SubagentRecord(
                "c",
                descriptor(missing if edge == "child" else present),
                (
                    ChildAgentRun(
                        ("e",),
                        descriptor(missing if edge == "base" else present),
                        descriptor(missing if edge == "result" else present),
                    ),
                ),
            ),
        ),
    )
    root = store.put(node)
    snapshot = AgentSnapshot(AgentDescriptor(b"prompt", (), root), store)
    before = store.records()
    with pytest.raises(KeyError):
        pickle.dumps(snapshot)
    assert snapshot.root is node and store.records() == before


def test_corrupt_loader_result_surfaces_without_changing_the_snapshot():
    root = NodeRef("source", 0)
    snapshot = AgentSnapshot(AgentDescriptor(b"prompt", (), root), RecordStore(load=lambda _: None))
    with pytest.raises(TypeError, match="expected HistoryNode"):
        pickle.dumps(snapshot)
    assert snapshot.descriptor.root == root and snapshot.store.records() == {}
