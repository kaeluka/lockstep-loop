"""From python/: PYTHONPATH=src python examples/specialized_subagents.py

Metadata selection and per-record loading are caller code. No parent snapshot is
needed to reuse a child. The constructed prompt is an initial USER message.
"""

import pickle

from history_replay import persist

from lockstep_loop import (
    BatchLoop,
    LlmCompletion,
    LlmRequest,
    RecordStore,
    SubagentTool,
    ToolCall,
    ToolCompletion,
    ToolSpec,
)

SCHEMA = {
    "type": "object",
    "properties": {"file_path": {"type": "string"}},
    "required": ["file_path"],
    "additionalProperties": False,
}
READ = ToolSpec("read_file", "Read a repository file without modifying it.", SCHEMA)


def imports_prompt(args):
    # A schema advertised to the model is not validation of its reply.
    if set(args) != {"file_path"} or not isinstance(args["file_path"], str):
        raise ValueError("expected exactly one string file_path")
    # No global path normalization: different strings deliberately get different tries.
    return f"List imports in {args['file_path']!r}."


IMPORTS = SubagentTool(
    name="list_imports",
    description="List the packages imported by a file.",
    parameters=SCHEMA,
    build_prompt=imports_prompt,
    sub_tool_specs=lambda _: (READ,),
    subagents=(),
).with_agent_contract("imports/template-v1/read-semantics-v1/exact-grouping")


def investigate(parent_prompt, env, cache, backend):
    store = RecordStore(load=lambda reference: pickle.loads(backend[reference]))

    def resolve(query):
        # These configurations are fixed in this example. A more general index
        # may include all query fields; core verifies FULL compatibility either way.
        return cache.get((query.contract, query.prompt))

    loop = BatchLoop(
        prompt=parent_prompt,
        env_ids=[env],
        tool_specs=[],
        subagents=[IMPORTS],
        agent_contract=f"parent-policy:{parent_prompt}",
        record_store=store,
        agent_resolver=resolve,
    )
    parent_llms = child_llms = child_tools = 0
    while not loop.done:
        request = loop.next_request()
        assert request is not None
        if isinstance(request, LlmRequest):
            if request.messages[0].content == parent_prompt:
                parent_llms += 1
                if len(request.messages) == 1:
                    assert [s.name for s in request.tools] == ["list_imports"]
                    reply = LlmCompletion(
                        request.request_id,
                        "Inspect imports.",
                        (ToolCall("imports", "list_imports", {"file_path": "foo.py"}),),
                    )
                else:
                    reply = LlmCompletion(request.request_id, request.messages[-1].content)
            else:
                child_llms += 1
                assert request.messages[0].content == imports_prompt({"file_path": "foo.py"})
                if len(request.messages) == 1:
                    reply = LlmCompletion(
                        request.request_id,
                        "Read the file.",
                        (ToolCall("read", "read_file", {"file_path": "foo.py"}),),
                    )
                else:
                    reply = LlmCompletion(request.request_id, "os, sys")
            loop.complete(reply)  # replace fake model with any provider adapter
        else:
            child_tools += 1
            assert request.tool_call.name == "read_file"  # never list_imports
            assert request.env_ids == (env,)  # inherited repo/commit, no file environment IDs
            loop.complete(ToolCompletion(request.request_id, ((env, "import os\nimport sys\n"),)))
    return loop.result(), (parent_llms, child_llms, child_tools)


if __name__ == "__main__":
    cache, backend = {}, {}
    first, counts = investigate("Parent A: analyze dependencies", "repo@commit-1", cache, backend)
    assert counts == (2, 2, 1)
    original = first.child_agents[0].agent
    # Persist shallow records BEFORE publishing the selected child descriptor.
    # Trusted pickle only; choose your own codecs and backend in production.
    selected = pickle.loads(persist(original, backend))
    cache[(selected.contract, selected.prompt)] = selected

    second, counts = investigate(
        "Parent B: a completely different parent prompt", "repo@commit-2", cache, backend
    )
    assert counts == (2, 0, 1)  # parent regenerated, child reasoning reused, child tool rerun
    assert second.results == {"repo@commit-2": "os, sys"}
    update = second.child_agents[0]
    assert update.base == selected
    assert update.agent.descriptor.root != selected.root
    assert original.root.env_ids == ("repo@commit-1",)
    provenance = second.agent.root.subagents[0].runs[-1]
    assert provenance.base == selected and provenance.agent_descriptor == update.agent.descriptor
    # There is one update for this query here. Concurrent invocations would yield
    # separate entries; selecting/merging/publishing them is a caller decision.
    updated = pickle.loads(persist(update.agent, backend))
    cache[(updated.contract, updated.prompt)] = updated
    print("Changed parent: 2 fresh parent LLM calls, 0 child LLM calls, 1 child tool rerun")
