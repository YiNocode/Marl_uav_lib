"""Structure-preserving cooperative encirclement (SCE) framework components."""

from marl_uav.framework.role_allocation import (
    entropic_ot_assignment,
    sinkhorn_transport_plan,
)

__all__ = [
    "entropic_ot_assignment",
    "sinkhorn_transport_plan",
]
