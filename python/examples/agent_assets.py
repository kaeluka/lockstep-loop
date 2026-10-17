"""From python/: PYTHONPATH=src python examples/agent_assets.py

Agent = prompt + compatible contract + accumulated trie.
Asset = a read-only leaf investigation, backed by its root-to-leaf conversation.
"""

from feedback import NOTE, PROMPT, SPECS, investigate
from history_replay import persist, restore

from lockstep_loop import BatchLoop, Feedback, LlmCompletion, LlmRequest, ToolCall, ToolCompletion

if __name__ == "__main__":
    recorded, _, _ = investigate(("A", "B", "D"))
    backend = {}
    original_descriptor = persist(recorded.agent, backend)
    original = restore(original_descriptor, backend)  # load an Agent, including its prompt
    original_assets = list(original.assets())
    assert [a.env_ids for a in original_assets] == [("A", "B"), ("D",)]
    assert original_assets[0].value == original_assets[1].value == "no issues"
    assert original_assets[0].messages != original_assets[1].messages

    replayed, llms, tools = investigate(("C",), agent=original)
    assert not llms and len(tools) == 1  # resolution still executes the recorded tool
    before = replayed.agent
    old_asset = next(before.assets())
    assert old_asset.env_ids == ("A", "B", "C")
    assert old_asset.is_complete and old_asset.value == "no issues"
    print("Agent prompt:", before.prompt)
    print("Asset known environments:", old_asset.env_ids)
    print("Asset conversation:", [(m.role, m.content) for m in old_asset.messages])

    # The Asset is not an endpoint handle. Supply the agent + environment again;
    # feedback must resolve the conversation through normal tool replay.
    loop = BatchLoop(
        prompt=PROMPT,
        env_ids=["C"],
        tool_specs=SPECS,
        agent=before,
        agent_contract=before.contract,
        feedback=Feedback(NOTE),
    )
    read = loop.next_request()
    assert not isinstance(read, LlmRequest)
    loop.complete(ToolCompletion(read.request_id, (("C", "shared"),)))
    opened = loop.agent
    pending_asset = next(opened.assets())
    assert not pending_asset.is_complete
    assert pending_asset.value == "no issues"  # previous publication, not a new completed value
    assert pending_asset.messages[-1].content == NOTE
    request = loop.next_request()
    assert request.messages == pending_asset.messages
    loop.complete(
        LlmCompletion(request.request_id, "Check /foo", (ToolCall("foo", "foo", {"path": "/foo"}),))
    )
    tool = loop.next_request()
    loop.complete(ToolCompletion(tool.request_id, (("C", "safe"),)))
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "no issues"))  # same answer is valid
    revised = loop.result().agent
    revised_asset = next(revised.assets())
    assert revised_asset.is_complete and revised_asset.value == old_asset.value
    assert len(revised_asset.messages) > len(old_asset.messages)
    assert old_asset.node.user_continuations == ()  # still a leaf in its old snapshot
    assert revised_asset.nodes[-3].user_continuations  # former leaf is now an ancestor
    assert not pending_asset.is_complete  # captured open snapshot also stayed unchanged

    descriptor = persist(revised, backend)
    loaded = restore(descriptor, backend)
    inherited, _, _ = investigate(("A", "B", "C", "D"), agent=loaded)
    assets = list(inherited.agent.assets())
    assert len(assets) == 3
    assert {tuple(sorted(a.env_ids)) for a in assets} == {("A", "C"), ("B",), ("D",)}
    assert inherited.results == {"A": "no issues", "B": "found", "C": "no issues", "D": "no issues"}
    persist(inherited.agent, backend)  # caller publishes the updated agent snapshot
    old_again, llms, _ = investigate(("B",), agent=restore(original_descriptor, backend))
    assert not llms and old_again.results == {"B": "no issues"}
    print("Updated assets:", [(a.env_ids, a.value) for a in assets])
    print("Old, open, and updated agent snapshots remain independent; no asset IDs required")
