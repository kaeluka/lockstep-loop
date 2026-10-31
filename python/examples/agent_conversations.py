"""From python/: PYTHONPATH=src python examples/agent_conversations.py

AgentSnapshot = initial messages + structural configuration + accumulated trie.
Conversation = a read-only path in that snapshot; an asset is an extracted value.
"""

from feedback import NOTE, investigate
from history_replay import persist, restore

from lockstep_loop import Agent, Feedback, LlmCompletion, LlmRequest, ToolCall, ToolCompletion

if __name__ == "__main__":
    recorded, _, _ = investigate(("A", "B", "D"))
    backend = {}
    original_descriptor = persist(recorded.snapshot, backend)
    original = restore(original_descriptor, backend)
    conversations = list(original.conversations())
    assert [c.env_ids for c in conversations] == [("A", "B"), ("D",)]
    assert all(c.get_last_answer_asset() == "no issues" for c in conversations)
    assert conversations[0].messages != conversations[1].messages

    replayed, llms, tools = investigate(("C",), snapshot=original)
    assert not llms and len(tools) == 1  # replay still executes the recorded tool
    before = replayed.snapshot
    old = next(before.conversations())
    assert old.env_ids == ("A", "B", "C")
    assert old.is_complete and old.get_last_answer_asset() == "no issues"
    print("AgentSnapshot initial messages:", before.initial_messages)
    print("Conversation known environments:", old.env_ids)
    print("Conversation:", [(m.role, m.content) for m in old.messages])

    # A conversation view is not an endpoint handle. Supply the agent + environment;
    # feedback resolves the conversation through normal tool replay.
    agent = Agent.from_snapshot(before)
    loop = agent.start(["C"], feedback=Feedback(NOTE))
    read = loop.next_request()
    assert not isinstance(read, LlmRequest)
    loop.complete(ToolCompletion(read.request_id, (("C", "shared"),)))
    opened = loop.snapshot()
    pending = next(opened.conversations())
    assert not pending.is_complete
    assert pending.get_last_answer_asset() == "no issues"  # previous answer, not a new one
    assert pending.messages[-1].content == NOTE
    request = loop.next_request()
    assert request.messages == pending.messages
    loop.complete(
        LlmCompletion(request.request_id, "Check /foo", (ToolCall("foo", "foo", {"path": "/foo"}),))
    )
    tool = loop.next_request()
    loop.complete(ToolCompletion(tool.request_id, (("C", "safe"),)))
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "no issues"))  # same answer is valid
    revised = agent.snapshot()  # automatically published by the final completion
    updated = next(revised.conversations())
    assert updated.is_complete
    assert updated.get_last_answer_asset() == old.get_last_answer_asset()
    assert len(updated.messages) > len(old.messages)
    assert old.node.user_continuations == ()  # still a leaf in its old snapshot
    assert updated.nodes[-3].user_continuations  # former leaf is now an ancestor
    assert not pending.is_complete  # captured open snapshot also stayed unchanged

    descriptor = persist(revised, backend)
    loaded = restore(descriptor, backend)
    inherited, _, _ = investigate(("A", "B", "C", "D"), snapshot=loaded)
    conversations = list(inherited.snapshot.conversations())
    assert len(conversations) == 3
    assert {tuple(sorted(c.env_ids)) for c in conversations} == {("A", "C"), ("B",), ("D",)}
    assert inherited.results == {"A": "no issues", "B": "found", "C": "no issues", "D": "no issues"}
    persist(inherited.snapshot, backend)
    old_again, llms, _ = investigate(("B",), snapshot=restore(original_descriptor, backend))
    assert not llms and old_again.results == {"B": "no issues"}
    print("Updated assets:", [(c.env_ids, c.get_last_answer_asset()) for c in conversations])
    print("Old, open, and updated agent snapshots remain independent; no conversation IDs required")
