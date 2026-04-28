"""Simple registries for policies / learners / envs, etc."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from marl_uav.learners.on_policy import IPPOLearner, MAPPOLearner, SCMAPPOLearner
from marl_uav.policies.actor_critic_policy import ActorCriticPolicy
from marl_uav.policies.centralized_critic_policy import CentralizedCriticPolicy

T = TypeVar("T")


class Registry(dict[str, Any]):
    """Register and get callables by name."""

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            self[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Any:  # type: ignore[override]
        return super().get(name)


# 全局 registry：后续可扩展 env / model 等
POLICY_REGISTRY: Registry = Registry()
LEARNER_REGISTRY: Registry = Registry()


@POLICY_REGISTRY.register("actor_critic")
def build_actor_critic_policy(**kwargs: Any) -> ActorCriticPolicy:
    """默认基于局部 obs 的共享 ActorCriticPolicy。"""
    return ActorCriticPolicy(**kwargs)


@POLICY_REGISTRY.register("centralized_critic")
def build_centralized_critic_policy(**kwargs: Any) -> CentralizedCriticPolicy:
    """CentralizedCriticPolicy：actor 用 obs，critic 用 state。"""
    return CentralizedCriticPolicy(**kwargs)


@LEARNER_REGISTRY.register("ippo")
def build_ippo_learner(policy: ActorCriticPolicy, **kwargs: Any) -> IPPOLearner:
    """IPPO learner，使用共享 ActorCriticPolicy。"""
    return IPPOLearner(policy=policy, **kwargs)


@LEARNER_REGISTRY.register("mappo")
def build_mappo_learner(policy: Any, **kwargs: Any) -> MAPPOLearner:
    """MAPPO learner，policy 可为 CentralizedCriticPolicy 或其它兼容 centralized critic 的策略。"""
    return MAPPOLearner(policy=policy, **kwargs)


@LEARNER_REGISTRY.register("sc_mappo")
def build_sc_mappo_learner(policy: Any, **kwargs: Any) -> SCMAPPOLearner:
    """SC-MAPPO：MAPPO + 空间分散度 advantage shaping。"""
    return SCMAPPOLearner(policy=policy, **kwargs)
