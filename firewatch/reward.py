"""
firewatch/reward.py
===================
Dense per-step reward computation.

No sys.path hacks — clean imports only.
"""

from typing import List


# Per-episode tracking to prevent duplicate bonuses
_diagnosis_given: set = set()
_fix_given: set = set()


def reset_episode_tracking(episode_id: str) -> None:
    """Reset per-episode bonus tracking."""
    _diagnosis_given.discard(episode_id)
    _fix_given.discard(episode_id)


def compute_reward(
    episode_id: str,
    prev_health: float,
    curr_health: float,
    action_tool: str,
    action_target: str,
    action_result: dict,
    root_cause_services: List[str],
) -> float:
    """
    Compute per-step reward value.

    r(t) = health_delta × 2.0
         + correct_fix_bonus (1.0 one-time)
         + diagnosis_bonus (0.3 one-time)
         - wrong_fix_penalty (-0.3)
         - step_cost (-0.02, except get_topology)

    Returns:
        float reward value
    """
    health_delta = round(curr_health - prev_health, 4)
    health_reward = round(health_delta * 2.0, 4)

    correct_fix_bonus = 0.0
    if action_result.get("correct_fix") is True and episode_id not in _fix_given:
        correct_fix_bonus = 1.0
        _fix_given.add(episode_id)

    diagnosis_bonus = 0.0
    if (
        action_tool in ("get_metrics", "get_logs")
        and action_target in root_cause_services
        and episode_id not in _diagnosis_given
    ):
        diagnosis_bonus = 0.3
        _diagnosis_given.add(episode_id)

    wrong_fix_penalty = -0.3 if action_result.get("wasted_action") else 0.0
    step_cost = 0.0 if action_tool == "get_topology" else -0.02

    reward_value = round(
        health_reward + correct_fix_bonus + diagnosis_bonus
        + wrong_fix_penalty + step_cost,
        4,
    )

    return reward_value