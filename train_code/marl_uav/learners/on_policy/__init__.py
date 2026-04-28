"""On-policy learners (MAPPO, SC-MAPPO, IPPO)."""

from marl_uav.learners.on_policy.ippo_learner import IPPOLearner
from marl_uav.learners.on_policy.mappo_learner import MAPPOLearner
from marl_uav.learners.on_policy.sc_mappo_learner import SCMAPPOLearner

__all__ = ["MAPPOLearner", "SCMAPPOLearner", "IPPOLearner"]
