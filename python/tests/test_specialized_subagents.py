import pickle
from dataclasses import dataclass, replace

import pytest

from lockstep_loop import (
    BatchLoop,
    Budget,
    Feedback,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
    run_to_completion,
)

READ = ToolSpec("read_file", "Read without modifying", {"type": "object"})
STATUS = ToolSpec("status", "Read status", {})
SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}},
    "required": ["file_path"],
    "additionalProperties": False,
}


def definition(name="list_imports", kind="imports", contract="imports/v1"):
    def prompt(args):
        if set(args) != {"file_path"} or not isinstance(args["file_path"], str):
            raise ValueError("expected exactly one string file_path")
        return f"{kind}:{args['file_path']}"  # no path normalization or model-written prompt

    return SubagentTool(
        name,
        f"Run {kind}",
        SCHEMA,
        prompt,
        lambda _: (READ,),
        history_contract=contract,
        subagents=(),
    )


def definitions():
    return (definition(), definition("count_lines", "metrics", "metrics/v1"))


def calls(path="foo.py"):
    return (
        ToolCall("imports", "list_imports", {"file_path": path}),
        ToolCall("metrics", "count_lines", {"file_path": path}),
        ToolCall("local", "status", {}),
    )


def make(
    prompt="parent-A",
    envs=("old",),
    *,
    resolver=None,
    store=None,
    history=None,
    registered=None,
    recording=True,
    **options,
):
    return BatchLoop(
        prompt=prompt,
        env_ids=envs,
        tool_specs=(STATUS,),
        subagents=definitions() if registered is None else registered,
        history_contract=f"{prompt}/v1" if recording else None,
        history=history,
        history_resolver=resolver,
        record_store=store,
        **options,
    )


def model(request, parent_calls):
    prompt = request.messages[0].content
    if prompt.startswith("parent"):
        if len(request.messages) == 1:
            return LlmCompletion(request.request_id, "delegate", parent_calls)
        observations = [m for m in request.messages if m.role == "tool"]
        assert [m.tool_call_id for m in observations] == [c.id for c in parent_calls]
        return LlmCompletion(request.request_id, "|".join(m.content for m in observations))
    kind, path = prompt.split(":", 1)
    if request.stats.depth < 2:
        return LlmCompletion(
            request.request_id,
            "inspect",
            (ToolCall("read", "read_file", {"path": path, "checkpoint": request.stats.depth}),),
        )
    return LlmCompletion(request.request_id, f"{kind}:done")


def drive(loop, values, parent_calls=None, *, expect_child_llms=None):
    parent_calls = calls() if parent_calls is None else parent_calls
    llms, tools, ids = [], [], set()
    while not loop.done:
        pending = []
        while (request := loop.next_request()) is not None:
            assert request.request_id not in ids
            ids.add(request.request_id)
            pending.append(request)
        assert pending
        for request in reversed(pending):
            if isinstance(request, LlmRequest):
                llms.append(request)
                loop.complete(model(request, parent_calls))
            else:
                tools.append(request)
                assert request.tool_call.name in ("status", "read_file")
                if request.tool_call.name == "status":
                    results = tuple((env, "ok") for env in reversed(request.env_ids))
                else:
                    args = request.tool_call.args
                    results = tuple(
                        (
                            env,
                            f"{args['path']}:"
                            f"{'prefix' if args['checkpoint'] == 0 else values[env]}",
                        )
                        for env in reversed(request.env_ids)
                    )
                loop.complete(ToolCompletion(request.request_id, results))
    child_llms = [r for r in llms if not r.messages[0].content.startswith("parent")]
    if expect_child_llms is not None:
        assert len(child_llms) == expect_child_llms
    assert loop.error is None and loop.outstanding_count == 0
    return loop.result(), llms, tools


def publish(history, backend):
    for reference, node in history.store.records().items():
        encoded = pickle.dumps(node)
        assert reference not in backend or backend[reference] == encoded
        backend[reference] = encoded
    return pickle.loads(pickle.dumps(history.descriptor))


def loader(backend):
    return RecordStore(load=lambda reference: pickle.loads(backend[reference]))


def test_multiple_tools_changed_parent_reuses_independent_children_and_persists_updates():
    first = make()
    request = first.next_request()
    assert [s.name for s in request.tools] == ["status", "list_imports", "count_lines"]
    assert request.tools[1].parameters == SCHEMA
    first.complete(model(request, calls()))
    old, _, _ = drive(first, {"old": "same"}, expect_child_llms=6)
    assert old.results == {"old": "imports:done|metrics:done|ok"}
    backend, selected = {}, {}
    for update in old.child_histories:
        selected[(update.history.contract, update.history.descriptor.prompt)] = publish(
            update.history, backend
        )
        assert update.base is None and update.env_ids == ("old",)
    assert len(selected) == 2
    lookups = []

    def resolve(query):
        lookups.append(query)
        assert not hasattr(query, "env_ids") and not hasattr(query, "branch_id")
        return selected.get((query.contract, query.prompt))

    loaded = []

    def load_node(reference):
        loaded.append(reference)
        return pickle.loads(backend[reference])

    changed_ids = tuple(replace(c, id=f"opaque/turn-1/sub:root/{c.id}") for c in calls())
    second, requests, tools = drive(
        make("parent-B", ("tomorrow",), resolver=resolve, store=RecordStore(load=load_node)),
        {"tomorrow": "same"},
        changed_ids,
        expect_child_llms=0,
    )
    assert len(requests) == 2 and all(r.messages[0].content == "parent-B" for r in requests)
    assert old.history.descriptor.root not in loaded  # no parent history dependency
    assert len([r for r in tools if r.tool_call.name == "read_file"]) == 4
    assert {q.prompt for q in lookups} == {"imports:foo.py", "metrics:foo.py"}
    assert second.results["tomorrow"] == old.results["old"]
    assert len(second.child_histories) == 2
    for update in second.child_histories:
        key = (update.history.contract, update.history.descriptor.prompt)
        assert update.base == selected[key]
        assert update.history.descriptor.root != update.base.root
        assert update.tool_call.id.startswith("opaque/")
        entry = next(s for s in second.history.root.subagents if s.call_id == update.tool_call.id)
        assert entry.runs[-1].base == update.base
        assert entry.runs[-1].history == update.history.descriptor
        assert entry.runs[-1].env_ids == ("tomorrow",)
        publish(update.history, backend)
    assert all(u.history.root.env_ids == ("old",) for u in old.child_histories)

    # Replaying the parent uses its pinned revision, not today's external cache pointer.
    def must_not_resolve(_):
        raise AssertionError("recorded parent invocation must remain pinned")

    replay, requests, _ = drive(
        make("parent-B", ("later",), resolver=must_not_resolve, history=second.history),
        {"later": "same"},
        changed_ids,
        expect_child_llms=0,
    )
    assert not requests
    for entry in replay.history.root.subagents:
        assert len(entry.runs) == 2
        assert entry.runs[0].history == next(
            s.history for s in second.history.root.subagents if s.call_id == entry.call_id
        )


def test_changed_file_and_changed_observation_branch_without_parent_splits():
    old, _, _ = drive(make(), {"old": "same"})
    base = next(u.history for u in old.child_histories if u.tool_call.name == "list_imports")
    backend = {}
    descriptor = publish(base, backend)
    queries = []

    def resolve(query):
        queries.append(query)
        return (
            descriptor
            if (query.prompt, query.contract) == (descriptor.prompt, descriptor.contract)
            else None
        )

    selected_calls = (calls()[0], calls()[2])
    changed, llms, _ = drive(
        make("parent-changed", ("A", "B"), resolver=resolve, store=loader(backend)),
        {"A": "same", "B": "different"},
        selected_calls,
        expect_child_llms=1,
    )
    child = changed.child_histories[0].history
    checkpoint = child.load(child.root.observations[0].continuation)
    assert len(checkpoint.observations) == 2
    assert {e.evidence[-1].outputs[0].tool_output for e in checkpoint.observations} == {
        "foo.py:same",
        "foo.py:different",
    }
    assert len(changed.classes) == 1 and changed.classes[0].env_ids == ("A", "B")
    assert [r.stats.depth for r in llms if r.messages[0].content.startswith("imports:")] == [2]
    assert len(base.load(base.root.observations[0].continuation).observations) == 1
    for path in ("bar.py", "./foo.py"):
        result, _, tools = drive(
            make("parent-file", resolver=resolve, store=loader(backend)),
            {"old": "same"},
            (calls(path)[0],),
            expect_child_llms=3,
        )
        update = result.child_histories[0]
        assert update.base is None
        assert update.history.descriptor.prompt == f"imports:{path}"
        assert all(r.tool_call.args["path"] == path for r in tools)
    new_definition = replace(definition(), history_contract="imports/v2")
    versioned, _, _ = drive(
        make(
            "parent-new-contract",
            registered=(new_definition,),
            resolver=resolve,
            store=loader(backend),
        ),
        {"old": "same"},
        (calls()[0],),
        expect_child_llms=3,
    )
    assert versioned.child_histories[0].base is None
    assert versioned.child_histories[0].history.contract == "imports/v2"


def test_concurrent_cache_hits_are_independent_roots_not_single_flight_or_last_writer():
    old, _, _ = drive(make(), {"old": "same"})
    base = old.child_histories[0].history
    # Completion ordering puts metrics first in this driver's update list.
    name = "count_lines" if base.contract == "metrics/v1" else "list_imports"
    backend = {}
    descriptor = publish(base, backend)
    invocations = (
        ToolCall("a", name, {"file_path": "foo.py"}),
        ToolCall("a/turn-1/sub:root/b", name, {"file_path": "foo.py"}),
    )
    lookups = []

    def resolve(query):
        lookups.append(query)
        return descriptor

    result, _, _ = drive(
        make("parent-concurrent", ("new",), resolver=resolve, store=loader(backend)),
        {"new": "changed"},
        invocations,
        expect_child_llms=2,
    )
    assert len(lookups) == 2 and lookups[0] == lookups[1]
    updates = result.child_histories
    assert len(updates) == 2
    assert {u.tool_call.id for u in updates} == {c.id for c in invocations}
    assert len({u.request_prefix for u in updates}) == 2
    assert len({u.history.descriptor.root for u in updates}) == 2
    assert all(u.base == descriptor for u in updates)
    assert len(result.history.root.subagents) == 2
    assert len(base.load(base.root.observations[0].continuation).observations) == 1
    for update in updates:
        assert (
            len(update.history.load(update.history.root.observations[0].continuation).observations)
            == 2
        )
        publish(update.history, backend)  # both remain publishable; caller chooses future pointer


def test_concurrent_parent_branches_publish_distinct_child_forks():
    old, _, _ = drive(make(), {"old": "same"})
    base = next(u.history for u in old.child_histories if u.tool_call.name == "list_imports")
    backend = {}
    descriptor = publish(base, backend)
    loop = make("parent-split", ("A", "B"), resolver=lambda _: descriptor, store=loader(backend))
    root = loop.next_request()
    split_call = ToolCall("split", "status", {})
    child_call = ToolCall("child", "list_imports", {"file_path": "foo.py"})
    loop.complete(LlmCompletion(root.request_id, "split first", (split_call,)))
    tool = loop.next_request()
    loop.complete(ToolCompletion(tool.request_id, (("A", "alpha"), ("B", "beta"))))
    branches = [loop.next_request("llm"), loop.next_request("llm")]
    for branch in reversed(branches):
        loop.complete(LlmCompletion(branch.request_id, "delegate", (child_call,)))
    result, _, _ = drive(
        loop,
        {"A": "first variant", "B": "second variant"},
        (split_call, child_call),
        expect_child_llms=2,
    )
    assert len(result.classes) == 2  # classes never merge despite equal child answers
    assert len(result.child_histories) == 2
    forks = {}
    for update in result.child_histories:
        assert update.base == descriptor
        child = update.history
        checkpoint = child.load(child.root.observations[0].continuation)
        forks[update.env_ids] = {e.representative[0].tool_output for e in checkpoint.observations}
        publish(child, backend)
    assert forks == {
        ("A",): {"foo.py:same", "foo.py:first variant"},
        ("B",): {"foo.py:same", "foo.py:second variant"},
    }
    assert len(base.load(base.root.observations[0].continuation).observations) == 1


def test_resolver_compatibility_failures_and_argument_errors_are_transactional():
    old, _, _ = drive(make(), {"old": "same"})
    base = next(u.history for u in old.child_histories if u.tool_call.name == "list_imports")
    backend = {}
    descriptor = publish(base, backend)
    legacy = pickle.loads(pickle.dumps(descriptor))
    legacy.__dict__["subagent"] = legacy.__dict__.pop("subagents")
    legacy_loop = make(resolver=lambda _: legacy, store=loader(backend))
    legacy_request = legacy_loop.next_request()
    with pytest.raises(ValueError, match="migrate legacy"):
        legacy_loop.complete(LlmCompletion(legacy_request.request_id, "bad", (calls()[0],)))
    assert legacy_loop.outstanding_llm == 1 and legacy_loop.history.root.assistant is None
    for bad in (
        replace(descriptor, prompt="other prompt"),
        replace(descriptor, contract="v2"),
        replace(descriptor, tools=()),
        replace(descriptor, custom_grouping=True),
    ):
        loop = make(resolver=lambda _, bad=bad: bad, store=loader(backend))
        request = loop.next_request()
        before = loop.history
        with pytest.raises(ValueError, match="incompatible"):
            loop.complete(LlmCompletion(request.request_id, "bad", (calls()[0],)))
        assert loop.history is before and loop.outstanding_llm == 1 and loop.error is None
        assert loop.next_request() is None
        loop.complete(LlmCompletion(request.request_id, "corrected"))
        assert loop.done and loop.result().results == {"old": "corrected"}
        assert not loop.result().child_histories
    loop = make(resolver=lambda _: descriptor, store=loader(backend))
    request = loop.next_request()
    before = loop.history
    with pytest.raises(ValueError, match="file_path"):
        loop.complete(
            LlmCompletion(
                request.request_id,
                "bad",
                (
                    calls()[2],
                    calls()[0],
                    ToolCall("broken", "count_lines", {}),
                ),
            )
        )
    assert loop.history is before and loop.outstanding_count == 1
    assert loop.next_request() is None
    with pytest.raises(ValueError, match="duplicate"):
        loop.complete(LlmCompletion(request.request_id, "bad", (calls()[0], calls()[0])))
    loop.complete(LlmCompletion(request.request_id, "corrected"))
    assert loop.result().history.root.assistant.content == "corrected"
    with pytest.raises(ValueError, match="no outstanding"):
        loop.complete(LlmCompletion(request.request_id, "overwrite"))


def test_resolver_failure_after_a_prepared_hit_keeps_the_whole_turn_pending():
    old, _, _ = drive(make(), {"old": "same"})
    backend = {}
    base = next(u.history for u in old.child_histories if u.tool_call.name == "list_imports")
    descriptor = publish(base, backend)
    broken = True
    queried = []

    def resolve(query):
        queried.append(query.prompt)
        if query.contract == "imports/v1":
            return descriptor
        if broken:
            raise OSError("cache unavailable")
        return None

    loop = make("parent-retry", ("new",), resolver=resolve, store=loader(backend))
    request = loop.next_request()
    before = loop.history
    with pytest.raises(OSError, match="unavailable"):
        loop.complete(model(request, calls()))
    assert queried == ["imports:foo.py", "metrics:foo.py"]
    assert loop.history is before and loop.outstanding_count == 1 and loop.error is None
    assert not loop._children and not loop._child_histories
    assert loop.next_request() is None
    broken = False
    loop.complete(model(request, calls()))
    result, _, _ = drive(loop, {"new": "same"}, expect_child_llms=3)
    assert result.results == {"new": "imports:done|metrics:done|ok"}
    assert len(result.child_histories) == 2


def test_registration_rules_legacy_advertisements_and_serial_driver():
    imports, metrics = definitions()
    with pytest.raises(ValueError, match="not both"):
        BatchLoop(prompt="p", env_ids=["e"], tool_specs=[], subagent=imports, subagents=[])
    with pytest.raises(ValueError, match="duplicate subagent"):
        make(registered=(imports, imports))
    with pytest.raises(ValueError, match="conflicting"):
        BatchLoop(
            prompt="p",
            env_ids=["e"],
            tool_specs=[ToolSpec(imports.name, "wrong", {})],
            subagents=(imports,),
        )
    with pytest.raises(ValueError, match="duplicate tool"):
        BatchLoop(prompt="p", env_ids=["e"], tool_specs=[READ, READ])
    with pytest.raises(ValueError, match="history_contract"):
        make(registered=(replace(imports, history_contract=None),), resolver=lambda _: None)
    with pytest.raises(ValueError, match="budget"):
        make(resolver=lambda _: None, budget=Budget(100))
    legacy = BatchLoop(
        prompt="parent",
        env_ids=["e"],
        tool_specs=[ToolSpec(imports.name, imports.description, imports.parameters)],
        subagent=imports,
    )
    assert [s.name for s in legacy.next_request().tools] == [imports.name]

    @dataclass
    class Local:
        name: str = "read_file"
        description: str = "Read without modifying"
        parameters = {"type": "object"}

        def __call__(self, args):
            return "value"

    result = run_to_completion(
        prompt="parent-driver",
        env_ids=["e"],
        tools=lambda _: [imports, metrics, Local()],
        complete_llm=lambda r: model(r, calls()[:2]),
    )
    assert result.results == {"e": "imports:done|metrics:done"}
    assert len(result.child_histories) == 2 and result.history is None
    assert all(not u.history.descriptor.custom_grouping for u in result.child_histories)
    with pytest.raises(ValueError, match="duplicate tool implementation"):
        run_to_completion(
            prompt="parent-driver",
            env_ids=["a", "b"],
            tools=lambda env: [imports, Local()] + ([Local()] if env == "b" else []),
            complete_llm=lambda r: model(r, calls()[:1]),
        )
    ordinary = Local(imports.name, imports.description)
    ordinary.parameters = imports.parameters
    with pytest.raises(ValueError, match="duplicate tool implementation"):
        run_to_completion(
            prompt="p",
            env_ids=["e"],
            tools=lambda _: [imports, ordinary],
            complete_llm=lambda r: LlmCompletion(r.request_id, "done"),
        )
    with pytest.raises(ValueError, match="duplicate tool implementation"):
        run_to_completion(
            prompt="p",
            env_ids=["e"],
            tools=lambda _: [imports, imports],
            complete_llm=lambda r: LlmCompletion(r.request_id, "done"),
        )


def test_nested_allowlists_and_ancestor_depth_cap_do_not_leak_reserved_tools():
    leaf = replace(definition("leaf", "leaf", "leaf/v1"), max_depth=99)
    middle = replace(
        definition("middle", "middle", "middle/v1"),
        max_depth=99,
        subagents=(leaf,),
        sub_tool_specs=lambda _: (READ, ToolSpec("outer", "", {})),
    )
    outer = SubagentTool(
        "outer",
        "",
        SCHEMA,
        lambda a: "outer:" + a["file_path"],
        lambda _: (READ, ToolSpec("sibling", "", {})),
        max_depth=2,
        history_contract="outer/v1",
        subagents=(middle,),
    )
    sibling = definition("sibling", "sibling", "sibling/v1")
    loop = make(registered=(outer, sibling))
    root = loop.next_request()
    loop.complete(
        LlmCompletion(root.request_id, "", (ToolCall("o", "outer", {"file_path": "foo.py"}),))
    )
    child = loop.next_request()
    assert [s.name for s in child.tools] == ["read_file", "middle"]
    before = loop.history
    with pytest.raises(ValueError, match="unavailable"):
        loop.complete(
            LlmCompletion(
                child.request_id, "bad", (ToolCall("s", "sibling", {"file_path": "foo.py"}),)
            )
        )
    assert loop.history is before and loop.outstanding_llm == 1
    loop.complete(
        LlmCompletion(child.request_id, "", (ToolCall("m", "middle", {"file_path": "foo.py"}),))
    )
    grand = loop.next_request()
    assert [s.name for s in grand.tools] == ["read_file"]  # no leaf, no outer
    with pytest.raises(ValueError, match="unavailable"):
        loop.complete(
            LlmCompletion(
                grand.request_id, "bad", (ToolCall("l", "leaf", {"file_path": "foo.py"}),)
            )
        )
    loop.complete(LlmCompletion(grand.request_id, "nested answer"))
    for _ in range(2):
        r = loop.next_request()
        loop.complete(LlmCompletion(r.request_id, "nested answer"))
    assert loop.done
    updates = loop.result().child_histories
    assert [u.tool_call.name for u in updates] == ["middle", "outer"]
    assert updates[0].history.descriptor.subagents == ()
    assert updates[1].history.descriptor.subagents[0].max_depth == 1


def test_caller_selected_child_feedback_revision_does_not_rewrite_old_parent():
    old, _, _ = drive(make(), {"old": "same"}, (calls()[0],))
    child = old.child_histories[0].history
    feedback = BatchLoop(
        prompt=child.descriptor.prompt,
        env_ids=["old"],
        tool_specs=child.descriptor.tools,
        history_contract=child.contract,
        history=child,
        feedback=Feedback("Also inspect the header"),
    )
    while True:
        r = feedback.next_request()
        if isinstance(r, LlmRequest):
            assert r.messages[-1].content == "Also inspect the header"
            feedback.complete(LlmCompletion(r.request_id, "improved imports"))
            break
        a = r.tool_call.args
        feedback.complete(
            ToolCompletion(
                r.request_id,
                (("old", f"{a['path']}:{'prefix' if a['checkpoint'] == 0 else 'same'}"),),
            )
        )
    improved = feedback.result().history
    assert not child.load(
        child.load(child.root.observations[0].continuation).observations[0].continuation
    ).user_continuations
    backend = {}
    selected = publish(improved, backend)
    queries = []

    def resolve(query):
        queries.append(query)
        return selected

    new, _, _ = drive(
        make("parent-revised", ("new",), resolver=resolve, store=loader(backend)),
        {"new": "same"},
        (calls()[0],),
        expect_child_llms=0,
    )
    assert new.results == {"new": "improved imports"}
    assert new.child_histories[0].base == selected
    count = len(queries)
    pinned, llms, _ = drive(
        make(history=old.history, resolver=resolve, envs=("new",)),
        {"new": "same"},
        (calls()[0],),
        expect_child_llms=0,
    )
    assert not llms and pinned.results == {"new": "imports:done"}
    assert len(queries) == count  # completed parent pins old child, no automatic propagation
    assert len(pinned.history.root.subagents[0].runs) == 2
    assert pinned.history.root.subagents[0].runs[0].history == old.history.root.subagents[0].history


def test_opaque_prompts_and_hash_collisions_are_verified_by_value():
    @dataclass(frozen=True)
    class Blob:
        value: bytes

        def __hash__(self):
            return 0

        def __repr__(self):
            return "same-debug"

    tool = SubagentTool(
        "opaque",
        "",
        {},
        lambda a: Blob(a["value"]),
        lambda _: (),
        history_contract="opaque/v1",
        subagents=(),
    )

    def make_opaque(parent, resolver=None, store=None):
        return BatchLoop(
            prompt=Blob(parent),
            env_ids=["e"],
            tool_specs=[],
            subagents=(tool,),
            history_resolver=resolver,
            record_store=store,
        )

    first = make_opaque(b"parent-A")
    r = first.next_request()
    first.complete(
        LlmCompletion(r.request_id, Blob(b"call"), (ToolCall("c", "opaque", {"value": b"child"}),))
    )
    r = first.next_request()
    first.complete(LlmCompletion(r.request_id, Blob(b"answer")))
    r = first.next_request()
    first.complete(LlmCompletion(r.request_id, Blob(b"parent answer")))
    update = first.result().child_histories[0]
    second = make_opaque(b"parent-B", lambda _: update.history.descriptor, update.history.store)
    r = second.next_request()
    before_count = second.outstanding_count
    with pytest.raises(ValueError, match="incompatible"):
        second.complete(
            LlmCompletion(
                r.request_id, Blob(b"bad"), (ToolCall("c", "opaque", {"value": b"different"}),)
            )
        )
    assert second.outstanding_count == before_count
    second.complete(
        LlmCompletion(r.request_id, Blob(b"good"), (ToolCall("c", "opaque", {"value": b"child"}),))
    )
    r = second.next_request()
    assert r.messages[0].content == Blob(b"parent-B")  # no fresh child LLM
    assert r.messages[-1].content == Blob(b"answer")
    second.complete(LlmCompletion(r.request_id, Blob(b"parent answer")))
    assert second.done
