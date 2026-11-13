import importlib.util
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "schedule", Path(__file__).resolve().parents[1] / "scripts/try_teacher_schedule.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_boundaries_and_stream():
    for name in ("A", "B"):
        values = [module.probability(name, u, 302) for u in range(15100)]
        assert all(0 <= p <= 1 for p in values)
        assert all(a >= b for a, b in pairwise(values))
    assert module.probability("B", 0, 302) == 1
    assert module.probability("B", 9060, 302) == 0
    assert module.probability("B", 15099, 302) == 0
    from traffic.dcrnn import DCRNN

    for enabled in (True, False):
        torch.manual_seed(42)
        model = DCRNN(np.ones((3, 3)), enabled)
        assert sum(p.numel() for p in model.parameters()) == 32385
        x = torch.randn(2, 12, 3, 2)
        y = torch.randn(2, 12, 3)
        a = torch.Generator().manual_seed(1042)
        b = torch.Generator().manual_seed(1042)
        assert torch.equal(model(x), model(x, y, 0, a))
        torch.rand(11, generator=b)
        assert torch.equal(a.get_state(), b.get_state())
        model.eval()
        assert torch.equal(model(x), model(x, y, 1, a))
        assert torch.equal(a.get_state(), b.get_state())
