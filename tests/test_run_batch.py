import pytest

from lockstep_loop import run_batch


def test_run_batch_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        run_batch(prompt="summarize src/pkgA", env_ids=["env-a", "env-b"], tools=lambda env: [])
