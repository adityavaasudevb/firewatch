"""
firewatch/reward.py
===================
Dense per-step reward computation.

r(t) = health_delta * 2.0       — health improvement/degradation
     + correct_fix_bonus        — +1.0 one-time for applying the correct fix
     + diagnosis_bonus          — +0.3 one-time for investigating root cause
     - wrong_fix_penalty        — -0.3 for restarting a healthy service
     - step_cost                — -0.02 per step (prevents stalling)
"""

from typing import List


_diagnosis_given: set = set()
_fix_given: set = set()


def reset_episode_tracking(episode_id: str) -> None:
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
    Returns raw reward — can be negative, that's intentional and correct.
    Negative rewards teach the agent what NOT to do.
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

    raw_reward = round(
        health_reward + correct_fix_bonus + diagnosis_bonus
        + wrong_fix_penalty + step_cost,
        4,
    )

    return raw_reward