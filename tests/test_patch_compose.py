import torch
import torch.nn as nn

from cogmem.patches.compose import _get_logical_projection_shape, compose_patches
from cogmem.patches.patch import CognitivePatch


class FakePackedProj(nn.Module):
    def __init__(self, in_features: int, out_features: int, packed_weight_shape: tuple[int, int]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(packed_weight_shape))

    def forward(self, x):
        return torch.zeros(*x.shape[:-1], self.out_features, dtype=x.dtype, device=x.device)


class FakeSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = FakePackedProj(4, 6, (24, 1))


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeSelfAttn()


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([FakeLayer()])


def test_get_logical_projection_shape_prefers_feature_dims_over_packed_weight():
    module = FakePackedProj(4, 6, (24, 1))
    assert _get_logical_projection_shape(module) == (6, 4)


def test_get_logical_projection_shape_returns_none_for_unknown_packed_module():
    class UnknownPackedProj(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(24, 1))

    assert _get_logical_projection_shape(UnknownPackedProj()) is None


def test_compose_patches_uses_logical_projection_shape_for_packed_weights():
    model = FakeModel()
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    patch = CognitivePatch(
        patch_id="patch_test",
        embedding=[],
        lora_weights={
            "model.layers.0.self_attn.q_proj.weight": {"A": a, "B": b}
        },
        rank=1,
        q_value=1.0,
    )

    hooks = compose_patches(model, [patch], scaling_factor=1.0)
    try:
        assert len(hooks) == 1

        x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        out = model.model.layers[0].self_attn.q_proj(x)
        expected = torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]])
        assert torch.allclose(out, expected)
    finally:
        for hook in hooks:
            hook.remove()
