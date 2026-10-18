"""Reviewer workload: one fan-out splits N envs, then each branch immediately ends.

From python/: PYTHONPATH=src python benchmarks/split_completion.py --repeats 3
Timing includes construction, fan-out, and the plain done/pull/complete loop;
result validation is outside the timed region. Includes AgentSnapshot snapshot; no subagents.
"""

import argparse
from statistics import median
from time import perf_counter

from lockstep_loop import Agent, LlmCompletion, ToolCall, ToolCompletion


def run(n):
    start = perf_counter()
    agent = Agent(prompt="p", tool_specs=[])
    loop = agent.start([str(i) for i in range(n)])
    request = loop.next_request()
    loop.complete(LlmCompletion(request.request_id, "", (ToolCall("x", "read", {}),)))
    request = loop.next_request()
    loop.complete(ToolCompletion(request.request_id, tuple((e, e) for e in request.env_ids)))
    while not loop.done:
        request = loop.next_request()
        loop.complete(LlmCompletion(request.request_id, "done"))
    elapsed = perf_counter() - start
    assert len(loop.result().results) == n
    assert loop.outstanding_count == 0
    return elapsed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for n in (100, 400, 1000):
        times = [run(n) for _ in range(args.repeats)]
        print(f"n={n}: median={median(times):.6f}s min={min(times):.6f}s repeats={args.repeats}")
