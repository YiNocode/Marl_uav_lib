"""Tests for critic/value output head shapes and numerical stability."""

from __future__ import annotations

import torch
import pytest

from marl_uav.modules.heads.categorical_policy_head import CategoricalPolicyHead


@pytest.mark.parametrize("input_dim,n_actions", [(6, 4), (13, 7), (32, 9)])
@pytest.mark.parametrize(
    "prefix_shape",
    [
        (1,),          # (B=1, F)
        (4,),          # (B, F)
        (2, 3),        # (B, N, F)
        (2, 5, 3),     # (B, T, N, F)
    ],
)
def test_value_output_shape_matches_features_prefix(input_dim: int, n_actions: int, prefix_shape: tuple[int, ...]):
    torch.manual_seed(0)
    head = CategoricalPolicyHead(input_dim=input_dim, n_actions=n_actions)

    features = torch.randn(*prefix_shape, input_dim, dtype=torch.float32)
    out = head(features, deterministic=False)

    assert "values" in out
    assert out["values"].shape == prefix_shape

    # actor outputs should also match prefix dims
    assert out["actions"].shape == prefix_shape
    assert out["log_probs"].shape == prefix_shape
    assert out["entropy"].shape == prefix_shape
    assert out["logits"].shape == (*prefix_shape, n_actions)


@pytest.mark.parametrize("input_dim,n_actions", [(5, 3), (17, 11)])
def test_value_output_is_finite_no_nan_inf(input_dim: int, n_actions: int):
    torch.manual_seed(123)
    head = CategoricalPolicyHead(input_dim=input_dim, n_actions=n_actions)

    # Use a slightly larger tensor to stress distribution sampling / entropy
    B, T, N = 3, 7, 4
    features = torch.randn(B, T, N, input_dim, dtype=torch.float32)
    out = head(features, deterministic=False)

    # values / logits / log_probs / entropy should be finite
    for k in ("values", "logits", "log_probs", "entropy"):
        x = out[k]
        assert torch.isfinite(x).all(), f"{k} contains NaN/Inf"


@pytest.mark.parametrize("input_dim1,input_dim2", [(6, 12), (8, 9)])
def test_independent_instances_support_different_input_dims(input_dim1: int, input_dim2: int):
    """Actor/critic heads can be instantiated independently with different input dims."""
    torch.manual_seed(7)
    head1 = CategoricalPolicyHead(input_dim=input_dim1, n_actions=5)
    head2 = CategoricalPolicyHead(input_dim=input_dim2, n_actions=5)

    x1 = torch.randn(2, 3, input_dim1)
    x2 = torch.randn(2, 3, input_dim2)

    out1 = head1(x1, deterministic=True)
    out2 = head2(x2, deterministic=True)

    assert out1["values"].shape == (2, 3)
    assert out2["values"].shape == (2, 3)
    assert torch.isfinite(out1["values"]).all()
    assert torch.isfinite(out2["values"]).all()

