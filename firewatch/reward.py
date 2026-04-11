"""
firewatch/reward.py
===================
Dense per-step reward computation.
"""

from typing import List


_diagnosis_given: set = set()
_fix_given: set = set()


def reset_episode_tracking(episode_id: str) -> None:
    _diagnosis_given.discard(episode_id)
    _fix_given.discard(episode_id)


def clamp_reward(raw: float) -> float:
    """Clamp reward to strictly within (0, 1) — never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, raw)), 4)


def compute_reward(
    episode_id: str,
    prev_health: float,
    curr_health: float,
    action_tool: str,
    action_target: str,
    action_result: dict,
    root_cause_services: List[str],
) -> float:
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

    raw_reward = (
        health_reward + correct_fix_bonus + diagnosis_bonus
        + wrong_fix_penalty + step_cost
    )

    # Normalize from roughly [-0.5, 1.5] into (0, 1)
    normalized = (raw_reward + 0.5) / 2.0

    return clamp_reward(normalized)