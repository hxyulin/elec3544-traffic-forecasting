"""Capacity controls and numerical compatibility with the frozen implementation."""

import subprocess

import pytest

torch = pytest.importorskip("torch")
from traffic.multihop import MultihopMLP  # noqa: E402


def test_default_matches_frozen_nonzero_head():
    source = subprocess.check_output(
        ["git", "show", "caf18f3:src/traffic/multihop.py"], text=True
    )
    namespace = {"__name__": "traffic.frozen_multihop", "__package__": "traffic"}
    exec(compile(source, "frozen_multihop.py", "exec"), namespace)
    torch.manual_seed(42)
    old = namespace["MultihopMLP"](207, "two").eval()
    torch.manual_seed(42)
    new = MultihopMLP(207, "two").eval()
    for k, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[k])
    head = torch.randn_like(old.head.weight)
    old.head.weight.data.copy_(head)
    new.head.weight.data.copy_(head)
    inputs = (
        torch.randn(5, 12, 31),
        torch.arange(5),
        torch.zeros(5, 3, dtype=torch.long),
        torch.rand(5),
    )
    assert torch.equal(old(*inputs), new(*inputs))


@pytest.mark.parametrize("size,width", [("current", 128), ("wide", 224)])
def test_capacity_and_mask(size, width):
    models = []
    for variant in ("one", "two"):
        torch.manual_seed(43)
        model = MultihopMLP(207, variant, size).eval()
        assert model.head.in_features == width
        models.append(model)
    for (name, a), (other, b) in zip(
        models[0].named_parameters(), models[1].named_parameters()
    ):
        assert name == other and torch.equal(a, b)
    model = models[0]
    torch.nn.init.normal_(model.head.weight)
    x = torch.randn(4, 12, 31)
    args = (torch.arange(4), torch.zeros(4, 3, dtype=torch.long), torch.rand(4))
    changed = x.clone()
    changed[:, :, 15:] += 20
    assert torch.equal(model(x, *args), model(changed, *args))
