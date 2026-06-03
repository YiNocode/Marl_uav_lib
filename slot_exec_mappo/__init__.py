"""Standalone MAPPO execution layer for slot navigation in obstacles."""

from slot_exec_mappo.adapter import SlotExecEnvWrapper, make_slot_exec_get_actions_fn

__all__ = ["SlotExecEnvWrapper", "make_slot_exec_get_actions_fn"]
