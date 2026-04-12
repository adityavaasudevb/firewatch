"""
firewatch/graders.py
====================
Deterministic grading functions for all 4 tasks.
Each returns {"score": 0.0-1.0, "breakdown": {...}, "reason": "..."}.
"""

from typing import Dict, List

FIX_TOOLS_SET = {
    "restart_service", "rollback_config", "reset_ratelimit",
    "sync_replica", "clear_connections", "scale_service",
}


def strict_score(raw: float) -> float:
    """Clamp score to strictly within (0, 1) — never 0.0 or 1.0."""
    return round(max(0.01, min(0.99, raw)), 4)


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

    useful_tools = {
        "get_metrics", "get_logs", "restart_service",
        "mark_resolved", "get_topology",
    }
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

    # Step penalty — Easy task should be solved in ≤6 steps
    step_penalty = 0.0
    if steps_taken > 6:
        step_penalty = min(0.10, (steps_taken - 6) * 0.02)

    raw = (
        fix_score + efficiency_score + diagnosis_score
        + health_score + behavior_score
        - penalty - step_penalty
    )
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
    if steps_taken > 6:
        reason += f" Step penalty: {steps_taken} total steps."

    return {
        "score": score,
        "breakdown": {
            "correct_fix": fix_score,
            "efficiency": efficiency_score,
            "diagnosis": diagnosis_score,
            "health": health_score,
            "behavior": behavior_score,
            "penalty": -penalty,
            "step_penalty": -step_penalty,
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
    investigated_db_logs = any(
        a["tool"] == "get_logs" and a["target"] == ROOT_CAUSE
        for a in action_history
    )
    investigated_db_any = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == ROOT_CAUSE
        for a in action_history
    )

    # Fix score — reduced from 0.40 to 0.35
    fix_score = 0.35 if correct_fix else 0.0
    if correct_fix and not investigated_db_any:
        fix_score *= 0.3

    # Efficiency — tighter windows
    efficiency_score = 0.0
    if correct_fix:
        fix_step = next(
            a["step"] for a in action_history
            if a["tool"] == CORRECT_FIX and a["target"] == ROOT_CAUSE
        )
        if fix_step <= 4:
            efficiency_score = 0.10
        elif fix_step <= 6:
            efficiency_score = 0.07
        elif fix_step <= 10:
            efficiency_score = 0.04
        else:
            efficiency_score = 0.01
        if not investigated_db_any:
            efficiency_score *= 0.5

    # Red herring avoidance
    touched_rh = any(
        a["tool"] in (
            "restart_service", "rollback_config", "clear_connections",
            "reset_ratelimit", "sync_replica",
        )
        and a["target"] == RED_HERRING
        for a in action_history
    )
    rh_score = 0.10 if not touched_rh else 0.04

    # Investigation
    investigated_services = {
        a["target"] for a in action_history
        if a["tool"] in ("get_metrics", "get_logs")
    }
    investigated_with_logs = {
        a["target"] for a in action_history
        if a["tool"] == "get_logs"
    }
    cascade_services = {"api-gateway", "payment-service", "database"}
    logs_on_cascade = investigated_with_logs & cascade_services
    any_on_cascade = investigated_services & cascade_services

    diagnosis_score = 0.0
    if investigated_db_logs:
        diagnosis_score = 0.12
    elif investigated_db_any:
        diagnosis_score = 0.06
    if len(logs_on_cascade) >= 3:
        diagnosis_score += 0.08
    elif len(logs_on_cascade) >= 2:
        diagnosis_score += 0.04
    elif len(any_on_cascade) >= 2:
        diagnosis_score += 0.02
    diagnosis_score = min(0.20, diagnosis_score)

    health_score = final_system_health * 0.05

    # Wrong fix penalty — penalize any fix that isn't the correct one
    wrong_fixes = sum(
        1 for a in action_history
        if a["tool"] in FIX_TOOLS_SET
        and a["tool"] != "scale_service" 
        and not (a["tool"] == CORRECT_FIX and a["target"] == ROOT_CAUSE)
    )
    wrong_fix_penalty = min(0.15, wrong_fixes * 0.03)

    # Step penalty — medium task should be solved in ≤8 steps
    step_penalty = 0.0
    if steps_taken > 10:
        step_penalty = min(0.10, (steps_taken - 10) * 0.02)

    useless = sum(
        1 for a in action_history
        if a["tool"] not in {
            "get_metrics", "get_logs", "clear_connections",
            "mark_resolved", "get_topology",
        }
    )
    behavior_penalty = min(0.10, useless * 0.03)
    if touched_rh:
        behavior_penalty += 0.04

    raw = (
        fix_score + rh_score + diagnosis_score
        + efficiency_score + health_score
        - wrong_fix_penalty - step_penalty - behavior_penalty
    )
    score = strict_score(raw)

    if correct_fix:
        reason = "Correctly identified and fixed database connection pool issue."
    elif investigated_db_logs:
        reason = "Read database logs and diagnosed root cause but did not apply fix."
    elif investigated_db_any:
        reason = "Investigated database but did not read logs or apply fix."
    else:
        reason = "Did not investigate the root cause service."
    if touched_rh:
        reason += " Fell for red herring (auth-service)."
    elif RED_HERRING in investigated_services:
        reason += " Investigated auth-service but correctly avoided fixing it."
    if wrong_fixes:
        reason += f" {wrong_fixes} wrong fix attempt(s)."
    if steps_taken > 8:
        reason += f" Step penalty: {steps_taken} steps."

    return {
        "score": score,
        "breakdown": {
            "correct_fix": fix_score,
            "red_herring_avoidance": rh_score,
            "diagnosis": diagnosis_score,
            "efficiency": efficiency_score,
            "health": health_score,
            "wrong_fix_penalty": -wrong_fix_penalty,
            "step_penalty": -step_penalty,
            "behavior_penalty": -behavior_penalty,
        },
        "reason": reason,
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
    REQUIRED_SET = set(REQUIRED)

    fix_actions = [
        a for a in action_history
        if a["tool"] in FIX_TOOLS_SET
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

    # Fix scores — reduced total possible to 0.35
    fix1_score = 0.12 if fix1_idx is not None else 0.0
    fix3_score = 0.08 if fix3_idx is not None else 0.0

    fix2_score = 0.0
    if fix2_idx is not None:
        if fix1_idx is not None and fix1_idx < fix2_idx:
            fix2_score = 0.15
        else:
            fix2_score = 0.03  # applied out of order — minimal credit

    # Order bonus
    order_bonus = 0.0
    if fix1_idx is not None and fix2_idx is not None and fix3_idx is not None:
        if fix1_idx < fix2_idx < fix3_idx:
            order_bonus = 0.10

    # Investigation score — reduced max to 0.20
    investigated_with_logs = {
        a["target"] for a in action_history
        if a["tool"] == "get_logs"
    }
    investigated_any = {
        a["target"] for a in action_history
        if a["tool"] in ("get_metrics", "get_logs")
    }
    root_causes = {"api-gateway", "database"}
    log_on_root = investigated_with_logs & root_causes

    inv_score = 0.0
    if len(log_on_root) >= 2:
        inv_score = 0.15
    elif len(log_on_root) == 1:
        inv_score = 0.08
    elif investigated_any & root_causes:
        inv_score = 0.04
    if "payment-service" in investigated_with_logs:
        inv_score += 0.05
    inv_score = min(0.20, inv_score)

    # Wrong fix penalty — penalize fixes that aren't in the required set
    wrong_fixes = [
        a for a in fix_actions
        if (a["tool"], a["target"]) not in REQUIRED_SET
    ]
    wrong_fix_penalty = min(0.20, len(wrong_fixes) * 0.06)

    # Wrong order penalty
    wrong_order_penalty = 0.0
    if fix2_idx is not None and fix1_idx is not None and fix2_idx < fix1_idx:
        wrong_order_penalty = 0.12

    # Step penalty — hard task should be solved in ≤10 steps
    step_penalty = 0.0
    if steps_taken > 12:
        step_penalty = min(0.10, (steps_taken - 12) * 0.02)

    health_score = final_system_health * 0.05

    raw = (
        fix1_score + fix2_score + fix3_score
        + order_bonus + inv_score + health_score
        - wrong_fix_penalty - wrong_order_penalty - step_penalty
    )
    score = strict_score(raw)

    if not (fix1_idx or fix2_idx or fix3_idx):
        if inv_score >= 0.08:
            reason = "Diagnosed root causes but did not apply any fixes."
        else:
            reason = "No meaningful fixes or investigation."
    elif (
        fix1_idx is not None
        and fix2_idx is not None
        and fix3_idx is not None
        and order_bonus > 0
    ):
        reason = "All fixes in correct order."
    elif fix1_idx is not None and fix2_idx is not None and fix3_idx is not None:
        reason = "All fixes applied, wrong order reduced effectiveness."
    else:
        applied = sum(1 for x in [fix1_idx, fix2_idx, fix3_idx] if x is not None)
        reason = f"Partial remediation — {applied}/3 fixes applied."
    if wrong_fixes:
        reason += f" {len(wrong_fixes)} wrong fix attempt(s)."
    if steps_taken > 10:
        reason += f" Step penalty: {steps_taken} steps."

    return {
        "score": score,
        "breakdown": {
            "fix1_rollback": fix1_score,
            "fix2_ratelimit": fix2_score,
            "fix3_sync": fix3_score,
            "order_bonus": order_bonus,
            "investigation": inv_score,
            "health": health_score,
            "wrong_fix_penalty": -wrong_fix_penalty,
            "wrong_order_penalty": -wrong_order_penalty,
            "step_penalty": -step_penalty,
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
    """Grade Task 4: Non-stationary adaptive incident (EXPERT)."""
    primary_fix = any(
        a["tool"] == "clear_connections" and a["target"] == "database"
        for a in action_history
    )
    investigated_db_logs = any(
        a["tool"] == "get_logs" and a["target"] == "database"
        for a in action_history
    )
    investigated_db_any = any(
        a["tool"] in ("get_metrics", "get_logs") and a["target"] == "database"
        for a in action_history
    )

    # Primary score — reduced max to 0.35
    if primary_fix:
        primary_score = 0.35
    elif investigated_db_logs:
        primary_score = 0.10
    elif investigated_db_any:
        primary_score = 0.05
    else:
        primary_score = 0.02

    # Secondary objectives
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
    if primary_fix:
        if fixed_cache:
            secondary_score += 0.10
        elif investigated_cache:
            secondary_score += 0.02
        if fixed_notification:
            secondary_score += 0.08
        elif investigated_notification:
            secondary_score += 0.01
    else:
        # Without primary fix, secondary credit is minimal
        if investigated_cache:
            secondary_score += 0.01
        if investigated_notification:
            secondary_score += 0.01
    secondary_score = min(0.18, secondary_score)

    # Efficiency — expert must be fast on primary fix
    efficiency_score = 0.0
    if primary_fix:
        primary_step = next(
            a["step"] for a in action_history
            if a["tool"] == "clear_connections" and a["target"] == "database"
        )
        if primary_step <= 3:
            efficiency_score = 0.12
        elif primary_step <= 5:
            efficiency_score = 0.08
        elif primary_step <= 8:
            efficiency_score = 0.04
        else:
            efficiency_score = 0.01

    health_score = final_system_health * 0.05

    # Redundancy penalty
    restart_counts: Dict[str, int] = {}
    for a in action_history:
        if a["tool"] in ("restart_service", "clear_connections"):
            restart_counts[a["target"]] = restart_counts.get(a["target"], 0) + 1
    redundant_restarts = sum(max(0, v - 1) for v in restart_counts.values())
    redundancy_penalty = min(0.08, redundant_restarts * 0.03)

    # Step penalty — expert task penalizes wasted steps heavily (≤12 steps)
    step_penalty = 0.0
    if steps_taken > 16:
        step_penalty = min(0.15, (steps_taken - 16) * 0.02)

    # No fix penalty
    no_fix_penalty = 0.0
    any_fix = any(a["tool"] in FIX_TOOLS_SET for a in action_history)
    if not any_fix and steps_taken >= 5:
        no_fix_penalty = 0.05

    raw_score = (
        primary_score + secondary_score
        + efficiency_score + health_score
        - redundancy_penalty - step_penalty - no_fix_penalty
    )

    final_score = strict_score(raw_score)

    parts = []
    if primary_fix:
        parts.append("Primary objective achieved (database fixed).")
    elif investigated_db_logs:
        parts.append("Read database logs but did not apply fix.")
    elif investigated_db_any:
        parts.append("Checked database but did not read logs or fix.")
    else:
        parts.append("Primary objective missed entirely.")

    if primary_fix:
        if fixed_cache and fixed_notification:
            parts.append("Handled both secondary failures.")
        elif fixed_cache:
            parts.append("Handled cache failure only.")
        elif fixed_notification:
            parts.append("Handled notification-service only.")
        else:
            parts.append("Secondary failures not addressed.")
    else:
        if investigated_cache or investigated_notification:
            parts.append("Investigated secondary services but primary unfixed.")
        else:
            parts.append("No secondary response.")

    if not any_fix:
        parts.append("No fixes attempted — investigation only.")
    if steps_taken > 12:
        parts.append(f"Step penalty: {steps_taken} steps.")

    return {
        "score": final_score,
        "breakdown": {
            "primary_objective": primary_score,
            "secondary_response": secondary_score,
            "efficiency": efficiency_score,
            "health": health_score,
            "redundancy_penalty": -redundancy_penalty,
            "step_penalty": -step_penalty,
            "no_fix_penalty": -no_fix_penalty,
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