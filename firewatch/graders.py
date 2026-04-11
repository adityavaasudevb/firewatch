"""
firewatch/graders.py
====================
Deterministic grading functions for all 4 tasks.
Each returns {"score": 0.0-1.0, "breakdown": {...}, "reason": "..."}.
"""

from typing import Dict, List


def strict_score(raw: float) -> float:
    """Clamp any score into the strict open interval (0, 1)."""
    return round(max(0.001, min(0.999, float(raw))), 4)


# =============================================================================
# Task 1 Grader: Single Service Failure
# =============================================================================

def grade_task1(
    action_history: List[dict],
    final_system_health: float,
    scenario_config: dict,
    steps_taken: int,
    max_steps: int,
) -> Dict:
    """Grade Task 1: Database OOM — restart_service(database)."""
    ROOT_CAUSE = "database"
    CORRECT_TOOL = "restart_service"

    correct_fix = any(
        a["tool"] == CORRECT_TOOL and a["target"] == ROOT_CAUSE
        for a in action_history
    )
    investigated = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == ROOT_CAUSE
        for a in action_history
    )

    fix_score = 0.50 if correct_fix else 0.0
    if correct_fix and not investigated:
        fix_score *= 0.4

    efficiency_score = 0.0
    fix_step = None
    if correct_fix:
        fix_step = next(
            a["step"] for a in action_history
            if a["tool"] == CORRECT_TOOL and a["target"] == ROOT_CAUSE
        )
        if investigated:
            if fix_step <= 4:
                efficiency_score = 0.15
            elif fix_step <= 6:
                efficiency_score = 0.10
            elif fix_step <= 8:
                efficiency_score = 0.05
            else:
                efficiency_score = 0.02

    diagnosis_score = 0.10 if investigated else 0.0
    if any(a["tool"] == "get_metrics" for a in action_history):
        diagnosis_score += 0.025
    if any(a["tool"] == "get_logs" for a in action_history):
        diagnosis_score += 0.025

    health_score = final_system_health * 0.10

    useful_tools = {"get_metrics", "get_logs", "restart_service", "mark_resolved", "get_topology"}
    useful_count = sum(1 for a in action_history if a["tool"] in useful_tools)
    useless_count = sum(1 for a in action_history if a["tool"] not in useful_tools)
    behavior_score = 0.05 if useful_count > 0 else 0.0
    behavior_score -= min(0.12, useless_count * 0.03)

    healthy_svcs = {"auth-service", "cache", "notification-service"}
    wrong_restarts = sum(
        1 for a in action_history
        if a["tool"] == "restart_service" and a["target"] in healthy_svcs
    )
    penalty = min(0.25, wrong_restarts * 0.10)

    raw = fix_score + efficiency_score + diagnosis_score + health_score + behavior_score - penalty
    # Clamp to strictly between 0 and 1 (not 0.0 or 1.0)
    score = strict_score(raw)

    if not correct_fix:
        reason = "Agent failed to restart the database (OOM root cause)."
    elif fix_step is not None and fix_step <= 4:
        reason = f"Excellent — resolved in {fix_step} steps."
    elif fix_step is not None and fix_step <= 6:
        reason = f"Good — resolved in {fix_step} steps."
    else:
        reason = f"Slow — resolved in {fix_step} steps."
    if not investigated:
        reason += " No proper investigation."
    if wrong_restarts:
        reason += f" {wrong_restarts} wrong restart(s)."

    return {
        "score": score,
        "breakdown": {
            "correct_fix": fix_score,
            "efficiency": efficiency_score,
            "diagnosis": diagnosis_score,
            "health": health_score,
            "behavior": behavior_score,
            "penalty": -penalty,
        },
        "reason": reason,
    }


# =============================================================================
# Task 2 Grader: Cascading Failure with Red Herring
# =============================================================================

def grade_task2(
    action_history: List[dict],
    final_system_health: float,
    scenario_config: dict,
    steps_taken: int,
    max_steps: int,
) -> Dict:
    """Grade Task 2: Database connection pool + auth-service red herring."""
    ROOT_CAUSE = "database"
    CORRECT_FIX = "clear_connections"
    RED_HERRING = "auth-service"

    correct_fix = any(
        a["tool"] == CORRECT_FIX and a["target"] == ROOT_CAUSE
        for a in action_history
    )
    investigated_db = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == ROOT_CAUSE
        for a in action_history
    )

    fix_score = 0.45 if correct_fix else 0.0
    if correct_fix and not investigated_db:
        fix_score *= 0.3

    efficiency_score = 0.0
    if correct_fix:
        fix_step = next(
            a["step"] for a in action_history
            if a["tool"] == CORRECT_FIX and a["target"] == ROOT_CAUSE
        )
        if fix_step <= 6:
            efficiency_score = 0.08
        elif fix_step <= 10:
            efficiency_score = 0.05
        else:
            efficiency_score = 0.02
        if not investigated_db:
            efficiency_score *= 0.5

    touched_rh = any(
        a["tool"] in (
            "restart_service", "rollback_config", "clear_connections",
            "reset_ratelimit", "sync_replica",
        )
        and a["target"] == RED_HERRING
        for a in action_history
    )
    rh_score = 0.15 if not touched_rh else 0.0

    investigated = [
        a["target"] for a in action_history
        if a["tool"] in ("get_metrics", "get_logs")
    ]
    cascade_score = sum(
        0.05 for svc in ("api-gateway", "payment-service", "database")
        if svc in investigated
    )
    cascade_score = min(0.15, cascade_score)

    health_score = final_system_health * 0.10

    symptom_penalty = sum(
        0.08 for a in action_history
        if a["tool"] in ("restart_service", "rollback_config")
        and a["target"] in ("api-gateway", "payment-service")
    )
    symptom_penalty = min(0.20, symptom_penalty)

    useless = sum(
        1 for a in action_history
        if a["tool"] not in {
            "get_metrics", "get_logs", "clear_connections",
            "mark_resolved", "get_topology",
        }
    )
    behavior_penalty = min(0.15, useless * 0.03)
    if touched_rh:
        behavior_penalty += 0.10

    raw = (
        fix_score + rh_score + cascade_score
        + efficiency_score + health_score
        - symptom_penalty - behavior_penalty
    )
    score = strict_score(raw)

    return {
        "score": score,
        "breakdown": {
            "correct_fix": fix_score,
            "red_herring": rh_score,
            "cascade_trace": cascade_score,
            "efficiency": efficiency_score,
            "health": health_score,
            "symptom_penalty": -symptom_penalty,
            "behavior_penalty": -behavior_penalty,
        },
        "reason": "Graded successfully",
    }


# =============================================================================
# Task 3 Grader: Multi-vector Ordered Incident
# =============================================================================

def grade_task3(
    action_history: List[dict],
    final_system_health: float,
    scenario_config: dict,
    steps_taken: int,
    max_steps: int,
) -> Dict:
    """Grade Task 3: Three fixes in correct order."""
    REQUIRED = [
        ("rollback_config", "api-gateway"),
        ("reset_ratelimit", "api-gateway"),
        ("sync_replica", "database"),
    ]

    fix_actions = [
        a for a in action_history
        if a["tool"] in (
            "rollback_config", "reset_ratelimit", "sync_replica",
            "clear_connections", "restart_service",
        )
    ]

    fix1_idx = fix2_idx = fix3_idx = None
    for idx, a in enumerate(fix_actions):
        key = (a["tool"], a["target"])
        if key == REQUIRED[0] and fix1_idx is None:
            fix1_idx = idx
        elif key == REQUIRED[1] and fix2_idx is None:
            fix2_idx = idx
        elif key == REQUIRED[2] and fix3_idx is None:
            fix3_idx = idx

    fix1_score = 0.25 if fix1_idx is not None else 0.0
    fix3_score = 0.20 if fix3_idx is not None else 0.0

    fix2_score = 0.0
    if fix2_idx is not None:
        if fix1_idx is not None and fix1_idx < fix2_idx:
            fix2_score = 0.25
        else:
            fix2_score = 0.10

    order_bonus = 0.0
    if fix1_idx is not None and fix2_idx is not None and fix3_idx is not None:
        if fix1_idx < fix2_idx < fix3_idx:
            order_bonus = 0.10

    investigated = {
        a["target"] for a in action_history
        if a["tool"] in ("get_metrics", "get_logs")
    }
    important = {"api-gateway", "database", "payment-service"}
    inv_score = (
        0.10 if len(investigated & important) >= 2
        else 0.05 if investigated & important
        else 0.0
    )

    useful_tools = {
        "get_metrics", "get_logs", "rollback_config", "reset_ratelimit",
        "sync_replica", "mark_resolved", "get_topology",
    }
    useless = sum(1 for a in action_history if a["tool"] not in useful_tools)
    bpenalty = min(0.20, useless * 0.03)

    wrong_order_penalty = 0.0
    if fix2_idx is not None and fix1_idx is not None:
        if fix2_idx < fix1_idx:
            wrong_order_penalty = 0.10

    raw = (
        fix1_score + fix2_score + fix3_score
        + order_bonus + inv_score - bpenalty - wrong_order_penalty
    )
    if inv_score == 0:
        raw *= 0.75
    score = strict_score(raw)

    if not (fix1_idx or fix2_idx or fix3_idx):
        reason = "No meaningful fixes applied."
    elif (
        fix1_idx is not None
        and fix2_idx is not None
        and fix3_idx is not None
        and order_bonus > 0
    ):
        reason = (
            "All fixes in correct order."
            if inv_score > 0
            else "Correct but no investigation."
        )
    elif fix1_idx is not None and fix2_idx is not None and fix3_idx is not None:
        reason = "All fixes applied, wrong order reduced effectiveness."
    else:
        reason = "Partial remediation."
    if inv_score == 0:
        reason += " No investigation."

    return {
        "score": score,
        "breakdown": {
            "fix1": fix1_score,
            "fix2": fix2_score,
            "fix3": fix3_score,
            "order_bonus": order_bonus,
            "investigation": inv_score,
            "behavior_penalty": -bpenalty,
            "wrong_order_penalty": -wrong_order_penalty,
        },
        "reason": reason,
    }


# =============================================================================
# Task 4 Grader: Non-stationary Adaptive Incident
# =============================================================================

def grade_task4(
    action_history: List[dict],
    final_system_health: float,
    scenario_config: dict,
    steps_taken: int,
    max_steps: int,
) -> Dict:
    """Grade Task 4: Primary fix + secondary response + efficiency."""
    primary_fix = any(
        a["tool"] == "clear_connections" and a["target"] == "database"
        for a in action_history
    )
    investigated_db = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == "database"
        for a in action_history
    )

    primary_score = 0.65 if primary_fix else (0.20 if investigated_db else 0.0)

    actions_after_5 = [a for a in action_history if a["step"] >= 5]
    actions_after_8 = [a for a in action_history if a["step"] >= 8]

    fixed_cache = any(
        a["tool"] == "restart_service" and a["target"] == "cache"
        for a in actions_after_5
    )
    fixed_notification = any(
        a["tool"] in ("restart_service", "scale_service")
        and a["target"] == "notification-service"
        for a in actions_after_8
    )
    investigated_cache = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == "cache"
        for a in actions_after_5
    )
    investigated_notification = any(
        a["tool"] in ("get_metrics", "get_logs")
        and a["target"] == "notification-service"
        for a in actions_after_8
    )

    secondary_score = 0.0
    if fixed_cache:
        secondary_score += 0.12
    elif investigated_cache:
        secondary_score += 0.04
    if fixed_notification:
        secondary_score += 0.08
    elif investigated_notification:
        secondary_score += 0.03
    secondary_score = min(0.20, secondary_score)

    efficiency_score = 0.0
    if primary_fix:
        primary_step = next(
            a["step"] for a in action_history
            if a["tool"] == "clear_connections" and a["target"] == "database"
        )
        if primary_step <= 5:
            efficiency_score = 0.15
        elif primary_step <= 8:
            efficiency_score = 0.10
        elif primary_step <= 12:
            efficiency_score = 0.06
        else:
            efficiency_score = 0.02

    restart_counts: Dict[str, int] = {}
    for a in action_history:
        if a["tool"] in ("restart_service", "clear_connections"):
            restart_counts[a["target"]] = restart_counts.get(a["target"], 0) + 1
    redundant_restarts = sum(max(0, v - 1) for v in restart_counts.values())
    penalty = min(0.10, redundant_restarts * 0.04)

    raw_score = primary_score + secondary_score + efficiency_score - penalty
    if primary_fix and not investigated_db:
        raw_score *= 0.75

    final_score = strict_score(raw_score)

    parts = []
    if primary_fix:
        if investigated_db:
            parts.append("Primary objective achieved with proper diagnosis.")
        else:
            parts.append("Primary fixed but without investigation (likely guess).")
    elif investigated_db:
        parts.append("Diagnosed primary issue but did not fix it.")
    else:
        parts.append("Primary objective missed.")

    if fixed_cache and fixed_notification:
        parts.append("Handled both secondary failures.")
    elif fixed_cache:
        parts.append("Handled cache failure only.")
    elif fixed_notification:
        parts.append("Handled notification-service only.")
    else:
        parts.append("Secondary failures not handled (acceptable).")

    if redundant_restarts:
        parts.append(f"{redundant_restarts} redundant restart(s).")

    return {
        "score": final_score,
        "breakdown": {
            "primary_objective": primary_score,
            "secondary_response": secondary_score,
            "efficiency": efficiency_score,
            "penalty": -penalty,
        },
        "reason": " ".join(parts),
    }


# =============================================================================
# Grader registry
# =============================================================================

GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
    "task4": grade_task4,
}
