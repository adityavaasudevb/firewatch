"""
firewatch/tasks.py
==================
All 4 task configurations in one file.
"""

TASK_CONFIGS = {
    "task1": {
        "task_id": "task1",
        "name": "Single Service Failure",
        "difficulty": "easy",
        "scenario_id": "task1",
        "step_budget": 10,
        "hide_step_budget": False,
        "root_causes": ["database"],
        "description": (
            "INCIDENT ALERT — P1 INCIDENT OPENED\n\n"
            "Your distributed payment system is experiencing a critical failure.\n"
            "Customers are unable to complete transactions.\n\n"
            "Your job: Diagnose the root cause and restore the system.\n\n"
            "Available tools: get_metrics, get_logs, get_topology, restart_service,\n"
            "clear_connections, rollback_config, sync_replica, reset_ratelimit,\n"
            "scale_service, mark_resolved\n\n"
            "Services: api-gateway, auth-service, payment-service, database, "
            "cache, notification-service"
        ),
    },
    "task2": {
        "task_id": "task2",
        "name": "Cascading Failure with Red Herring",
        "difficulty": "medium",
        "scenario_id": "task2",
        "step_budget": 15,
        "hide_step_budget": False,
        "root_causes": ["database"],
        "description": (
            "INCIDENT ALERT — P1 INCIDENT OPENED\n\n"
            "Multiple services are degraded. The source is unclear.\n"
            "At least one alert may be a red herring unrelated to the main incident.\n\n"
            "Your job: Find the TRUE root cause, fix it efficiently, and avoid\n"
            "wasting actions on symptoms or unrelated issues.\n\n"
            "Hint: Check dependencies. A failing downstream service often points\n"
            "upstream to the real cause."
        ),
    },
    "task3": {
        "task_id": "task3",
        "name": "Multi-vector Ordered Incident",
        "difficulty": "hard",
        "scenario_id": "task3",
        "step_budget": 20,
        "hide_step_budget": False,
        "root_causes": ["api-gateway", "database"],
        "description": (
            "INCIDENT ALERT — P1 INCIDENT OPENED (MULTI-SYSTEM)\n\n"
            "Three simultaneous failure modes detected across the system.\n"
            "Order of remediation matters — some fixes depend on others being "
            "applied first.\n\n"
            "Your job: Identify all three root causes, determine the correct\n"
            "remediation order, and restore full system health.\n\n"
            "Warning: Applying fixes in the wrong order will have reduced effect.\n"
            "Plan before you act."
        ),
    },
    "task4": {
        "task_id": "task4",
        "name": "Non-stationary Adaptive Incident",
        "difficulty": "expert",
        "scenario_id": "task4",
        "step_budget": 25,
        "hide_step_budget": True,
        "root_causes": ["database"],
        "description": (
            "INCIDENT ALERT — P1 INCIDENT OPENED (EVOLVING)\n\n"
            "A critical incident is in progress. The situation is evolving —\n"
            "new failures may emerge as you work. Stay adaptive.\n\n"
            "Your job: Address the initial root cause first, then respond to\n"
            "any new failures that appear. The system is live and degrading.\n\n"
            "Note: Step budget is not shown. Manage your actions carefully.\n"
            "Prioritize the most critical issues first."
        ),
    },
}