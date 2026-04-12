"""
firewatch/simulation.py
========================
The core distributed system simulator.

Pure Python — no frameworks. Simulates 6 interconnected services
that can fail in various ways. The tick() mechanism degrades
unhealthy services autonomously every step.
"""

import copy
import random
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Service dependency graph
# ---------------------------------------------------------------------------

SERVICE_DEPENDENCIES = {
    "api-gateway": ["auth-service", "payment-service", "notification-service"],
    "auth-service": [],
    "payment-service": ["database", "cache"],
    "database": [],
    "cache": [],
    "notification-service": [],
}

ALL_SERVICES = list(SERVICE_DEPENDENCIES.keys())

DEGRADATION_RATES = {
    "api-gateway": 0.04,
    "auth-service": 0.03,
    "payment-service": 0.05,
    "database": 0.06,
    "cache": 0.04,
    "notification-service": 0.02,
}

LOG_TEMPLATES = {
    "oom": [
        "ERROR OOMKiller: process killed due to out-of-memory condition",
        "WARN  memory usage at 97% of limit (3.88GB / 4GB)",
        "ERROR failed to allocate memory for query execution",
        "INFO  heap dump written to /var/log/heapdump.hprof",
    ],
    "connection_pool": [
        "ERROR HikariPool-1 - Connection is not available, request timed out after 30000ms",
        "ERROR Cannot acquire connection from pool (max=20, active=20, idle=0)",
        "WARN  connection pool exhausted — queuing requests",
        "ERROR SQLSTATE[08006]: Connection failure: server closed the connection unexpectedly",
    ],
    "config_leak": [
        "WARN  memory usage climbing: 1.2GB → 2.1GB → 3.4GB over last 10 minutes",
        "ERROR config reload introduced unreferenced object cache (size: 847MB)",
        "WARN  GC overhead limit exceeded — 98% of time spent in garbage collection",
        "ERROR heap space: java.lang.OutOfMemoryError after config hot-reload",
    ],
    "rate_limit": [
        "WARN  rate limiter threshold breached: 847 req/s (limit: 500 req/s)",
        "ERROR 429 Too Many Requests — dropping 38% of incoming traffic",
        "WARN  error rate spike triggered auto rate-limit (current error rate: 42%)",
        "INFO  rate limiter engaged at 14:23:07 UTC",
    ],
    "replica_lag": [
        "WARN  replication lag: 47 seconds behind primary",
        "ERROR stale read detected: query returned data 52s old",
        "WARN  replica falling behind due to write load on primary",
        "ERROR read-your-writes consistency violated for user_id=48291",
    ],
    "memory_warning": [
        "WARN  memory usage at 78% — approaching threshold",
        "INFO  minor GC pauses increasing: avg 120ms over last 5 minutes",
        "WARN  RSS memory: 3.1GB (limit: 4GB) — monitor closely",
    ],
    "healthy": [
        "INFO  health check passed",
        "INFO  request processed in 45ms",
        "INFO  all systems nominal",
    ],
}


# ---------------------------------------------------------------------------
# System Simulator
# ---------------------------------------------------------------------------

class SystemSimulator:
    """
    Simulates a distributed system with 6 services.

    Usage:
        sim = SystemSimulator(seed=42)
        sim.apply_scenario("task1")
        print(sim.get_metrics("database"))
        sim.restart_service("database")
        print(sim.get_system_health())
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self._services: Dict[str, dict] = {}
        self._step: int = 0
        self._action_history: List[dict] = []
        self._topology_requested: bool = False
        self._active_scenario: Optional[str] = None
        self._scenario_config: dict = {}
        self._initialize_services()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize_services(self) -> None:
        """Set all services to healthy baseline."""
        for name in ALL_SERVICES:
            self._services[name] = {
                "name": name,
                "health": 1.0,
                "status": "healthy",
                "error_rate": 0.0,
                "latency_ms": self.rng.randint(20, 60),
                "logs": list(LOG_TEMPLATES["healthy"]),
                "failure_type": None,
                "is_root_cause": False,
                "secondary_failure": None,
            }

    def _set_service_failure(
        self,
        service: str,
        failure_type: str,
        health: float,
        error_rate: float,
        latency_ms: int,
        is_root_cause: bool = False,
    ) -> None:
        """Apply a specific failure state to a service."""
        self._services[service].update({
            "health": round(health, 3),
            "status": "down" if health < 0.2 else "degraded",
            "error_rate": round(error_rate, 3),
            "latency_ms": latency_ms,
            "logs": list(LOG_TEMPLATES.get(failure_type, LOG_TEMPLATES["healthy"])),
            "failure_type": failure_type if failure_type != "healthy" else None,
            "is_root_cause": is_root_cause,
        })

    def _diagnose_from_logs(self, svc: dict) -> str:
        """
        Generate a plain-English diagnosis summary from service state.
        Appended to get_logs output so agents can connect evidence to action.
        This mirrors what real monitoring tools like Datadog or PagerDuty do —
        they show raw logs AND a summary diagnosis.
        """
        failure = svc.get("failure_type")
        secondary = svc.get("secondary_failure")
        name = svc.get("name", "unknown")

        parts = []

        if failure == "oom":
            parts.append(
                f"DIAGNOSIS: {name} has OUT-OF-MEMORY error. "
                f"Recommended fix: restart_service({name})"
            )
        elif failure == "connection_pool":
            parts.append(
                f"DIAGNOSIS: {name} has CONNECTION POOL EXHAUSTION. "
                f"Recommended fix: clear_connections({name})"
            )
        elif failure == "config_leak":
            parts.append(
                f"DIAGNOSIS: {name} has CONFIG-INDUCED MEMORY LEAK. "
                f"Recommended fix: rollback_config({name})"
            )
        elif failure == "rate_limit":
            parts.append(
                f"DIAGNOSIS: {name} has RATE LIMITER ACTIVE. "
                f"Recommended fix: reset_ratelimit({name})"
            )
        elif failure == "replica_lag":
            parts.append(
                f"DIAGNOSIS: {name} has REPLICA LAG. "
                f"Recommended fix: sync_replica({name})"
            )
        elif failure == "memory_warning":
            parts.append(
                f"DIAGNOSIS: {name} has memory pressure (non-critical). "
                f"Monitor closely — no immediate fix required."
            )

        if secondary == "rate_limit":
            parts.append(
                f"SECONDARY ISSUE: rate limiting also active on {name}. "
                f"Apply reset_ratelimit({name}) AFTER fixing primary issue."
            )
        elif secondary == "replica_lag":
            parts.append(
                f"SECONDARY ISSUE: replica lag also present on {name}. "
                f"Apply sync_replica({name}) after primary fix."
            )

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------

    def apply_scenario(self, scenario_id: str, task_config: Optional[dict] = None) -> None:
        """Load a failure scenario for the given task."""
        self._initialize_services()
        self._step = 0
        self._action_history = []
        self._topology_requested = False
        self._active_scenario = scenario_id
        self._scenario_config = task_config or {}

        loaders = {
            "task1": self._load_task1,
            "task2": self._load_task2,
            "task3": self._load_task3,
            "task4": self._load_task4,
        }

        loader = loaders.get(scenario_id)
        if loader is None:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        loader()

    def _load_task1(self) -> None:
        """Task 1: Database OOM. Payment and api-gateway show symptoms."""
        self._set_service_failure(
            "database", "oom",
            health=0.05, error_rate=1.0, latency_ms=0,
            is_root_cause=True,
        )
        self._set_service_failure(
            "payment-service", "healthy",
            health=0.55, error_rate=0.45, latency_ms=8500,
        )
        self._set_service_failure(
            "api-gateway", "healthy",
            health=0.65, error_rate=0.30, latency_ms=6200,
        )

    def _load_task2(self) -> None:
        """Task 2: Database connection pool + auth-service red herring."""
        self._set_service_failure(
            "database", "connection_pool",
            health=0.15, error_rate=0.90, latency_ms=30000,
            is_root_cause=True,
        )
        self._set_service_failure(
            "payment-service", "healthy",
            health=0.40, error_rate=0.60, latency_ms=31000,
        )
        self._set_service_failure(
            "api-gateway", "healthy",
            health=0.45, error_rate=0.50, latency_ms=28000,
        )
        self._set_service_failure(
            "auth-service", "memory_warning",
            health=0.75, error_rate=0.05, latency_ms=180,
        )

    def _load_task3(self) -> None:
        """Task 3: Multi-vector with order dependency."""
        self._set_service_failure(
            "api-gateway", "config_leak",
            health=0.35, error_rate=0.42, latency_ms=4500,
            is_root_cause=True,
        )
        self._services["api-gateway"]["secondary_failure"] = "rate_limit"
        self._services["api-gateway"]["logs"] = (
            LOG_TEMPLATES["config_leak"] + LOG_TEMPLATES["rate_limit"]
        )
        self._set_service_failure(
            "database", "replica_lag",
            health=0.60, error_rate=0.20, latency_ms=2100,
            is_root_cause=True,
        )
        self._set_service_failure(
            "payment-service", "healthy",
            health=0.50, error_rate=0.35, latency_ms=5800,
        )
        self._scenario_config["fix_order"] = []
        self._scenario_config["required_order"] = [
            "rollback_config:api-gateway",
            "reset_ratelimit:api-gateway",
            "sync_replica:database",
        ]

    def _load_task4(self) -> None:
        """Task 4: Non-stationary with events at steps 5 and 8."""
        self._set_service_failure(
            "database", "connection_pool",
            health=0.20, error_rate=0.85, latency_ms=28000,
            is_root_cause=True,
        )
        self._set_service_failure(
            "payment-service", "healthy",
            health=0.45, error_rate=0.55, latency_ms=29000,
        )
        self._set_service_failure(
            "api-gateway", "healthy",
            health=0.50, error_rate=0.45, latency_ms=25000,
        )
        self._services["database"]["secondary_failure"] = "replica_lag"
        self._scenario_config["nonstationary_events"] = {
            5: {
                "service": "cache",
                "failure": "oom",
                "health": 0.10,
                "error_rate": 0.95,
                "latency_ms": 0,
            },
            8: {
                "service": "notification-service",
                "failure": "healthy",
                "health": 0.55,
                "error_rate": 0.30,
                "latency_ms": 3200,
            },
        }

    # ------------------------------------------------------------------
    # Agent-callable tools
    # ------------------------------------------------------------------

    def get_metrics(self, service: str) -> dict:
        """Returns current health metrics for a service."""
        if service not in self._services:
            return {"error": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("get_metrics", service)
        return {
            "service": service,
            "health": svc["health"],
            "status": svc["status"],
            "error_rate": svc["error_rate"],
            "latency_ms": svc["latency_ms"],
            "message": (
                f"Metrics for {service}: health={svc['health']:.2f}, "
                f"status={svc['status']}, error_rate={svc['error_rate']:.2f}, "
                f"latency={svc['latency_ms']}ms"
            ),
        }

    def get_logs(self, service: str, lines: int = 10) -> dict:
        """
        Returns recent log entries for a service.
        Appends a diagnosis summary to help agents connect
        log evidence to the correct remediation action.
        This mirrors real monitoring tools (Datadog, PagerDuty)
        that show both raw logs and a summary diagnosis.
        """
        if service not in self._services:
            return {"error": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("get_logs", service)
        log_lines = svc["logs"][:lines]

        message = f"Logs for {service}:\n" + "\n".join(log_lines)

        # Append diagnosis if there is an active failure
        diagnosis = self._diagnose_from_logs(svc)
        if diagnosis:
            message += f"\n\n{diagnosis}"

        return {
            "service": service,
            "logs": log_lines,
            "count": len(log_lines),
            "message": message,
        }

    def get_topology(self) -> dict:
        """Returns the dependency graph. FREE action — no step cost."""
        self._topology_requested = True
        self._log_action("get_topology", "system")
        return {
            "services": ALL_SERVICES,
            "dependencies": SERVICE_DEPENDENCIES,
            "message": "Topology retrieved. Use this to trace failure paths.",
        }

    def restart_service(self, service: str) -> dict:
        """Restarts a service. Correct fix for OOM failures."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("restart_service", service)
        failure = svc.get("failure_type")

        if failure == "oom":
            svc.update({
                "health": 0.95,
                "status": "healthy",
                "error_rate": 0.02,
                "latency_ms": self.rng.randint(30, 70),
                "logs": [
                    "INFO  service restarted successfully",
                    "INFO  health check passed",
                ],
                "failure_type": None,
            })
            return {
                "success": True,
                "message": f"{service} restarted — OOM resolved",
                "correct_fix": True,
            }
        elif failure is None or failure == "healthy":
            original_health = svc["health"]
            svc["health"] = max(0.0, original_health - 0.05)
            svc["latency_ms"] = self.rng.randint(100, 300)
            return {
                "success": True,
                "message": f"{service} restarted — was healthy, brief disruption",
                "correct_fix": False,
                "wasted_action": True,
            }
        else:
            svc["health"] = min(1.0, svc["health"] + 0.10)
            svc["error_rate"] = max(0.0, svc["error_rate"] - 0.05)
            return {
                "success": True,
                "message": f"{service} restarted — wrong fix for {failure}, partial improvement",
                "correct_fix": False,
            }

    def rollback_config(self, service: str) -> dict:
        """Rolls back config. Correct fix for config_leak failures."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("rollback_config", service)
        failure = svc.get("failure_type")

        if failure == "config_leak":
            svc.update({
                "health": 0.80,
                "status": "degraded",
                "error_rate": 0.15,
                "latency_ms": 800,
                "logs": [
                    "INFO  config rolled back to v2.1.4",
                    "INFO  memory usage stabilizing: 2.1GB → 1.3GB",
                    "INFO  GC overhead normalized",
                    "WARN  rate limiter still active — reset_ratelimit required",
                ],
                "failure_type": None,
            })
            if "fix_order" in self._scenario_config:
                self._scenario_config["fix_order"].append(f"rollback_config:{service}")
            return {
                "success": True,
                "message": (
                    f"{service} config rolled back — memory leak stopped. "
                    f"WARNING: service still degraded (health=0.80). "
                    f"Rate limiter still active — apply reset_ratelimit({service}) next."
                ),
                "correct_fix": True,
            }
        else:
            return {
                "success": True,
                "message": f"{service} config rollback had no effect — no config issue detected",
                "correct_fix": False,
            }

    def reset_ratelimit(self, service: str) -> dict:
        """Resets rate limiter. Correct fix for rate_limit failures."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("reset_ratelimit", service)

        has_rate_limit = (
            svc.get("failure_type") == "rate_limit"
            or svc.get("secondary_failure") == "rate_limit"
        )

        if self._active_scenario == "task3" and has_rate_limit:
            fix_order = self._scenario_config.get("fix_order", [])
            config_fixed = any("rollback_config" in f for f in fix_order)

            if config_fixed:
                svc.update({
                    "health": 0.95,
                    "status": "healthy",
                    "error_rate": 0.02,
                    "latency_ms": self.rng.randint(30, 80),
                    "logs": [
                        "INFO  rate limiter thresholds reset",
                        "INFO  traffic flowing normally: 312 req/s",
                        "INFO  error rate normalized to 1.8%",
                    ],
                    "secondary_failure": None,
                })
                fix_order.append(f"reset_ratelimit:{service}")
                return {
                    "success": True,
                    "message": "Rate limiter reset — traffic restored",
                    "correct_fix": True,
                }
            else:
                svc["health"] = min(1.0, svc["health"] + 0.10)
                svc["error_rate"] = max(0.0, svc["error_rate"] - 0.10)
                return {
                    "success": True,
                    "message": "Rate limiter reset — partial improvement. Root config issue still active.",
                    "correct_fix": False,
                    "wrong_order": True,
                }
        else:
            return {
                "success": True,
                "message": f"No active rate limit issue on {service}",
                "correct_fix": False,
            }

    def sync_replica(self, service: str) -> dict:
        """Forces replica sync. Correct fix for replica_lag failures."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("sync_replica", service)
        failure = svc.get("failure_type")

        if failure == "replica_lag":
            svc.update({
                "health": 0.95,
                "status": "healthy",
                "error_rate": 0.01,
                "latency_ms": self.rng.randint(40, 90),
                "logs": [
                    "INFO  replica sync initiated",
                    "INFO  replication lag: 0 seconds",
                    "INFO  read consistency restored",
                ],
                "failure_type": None,
            })
            if "fix_order" in self._scenario_config:
                self._scenario_config["fix_order"].append(f"sync_replica:{service}")
            return {
                "success": True,
                "message": f"{service} replica synced — stale reads resolved",
                "correct_fix": True,
            }
        else:
            return {
                "success": True,
                "message": f"No replica lag detected on {service}",
                "correct_fix": False,
            }

    def clear_connections(self, service: str) -> dict:
        """Clears connection pool. Correct fix for connection_pool failures."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        svc = self._services[service]
        self._log_action("clear_connections", service)
        failure = svc.get("failure_type")

        if failure == "connection_pool":
            svc.update({
                "health": 0.92,
                "status": "healthy",
                "error_rate": 0.03,
                "latency_ms": self.rng.randint(40, 100),
                "logs": [
                    "INFO  connection pool cleared and reinitialized",
                    "INFO  pool size: 20 connections available",
                    "INFO  query throughput restored",
                ],
                "failure_type": None,
            })
            return {
                "success": True,
                "message": f"{service} connection pool cleared — queries flowing",
                "correct_fix": True,
            }
        else:
            return {
                "success": True,
                "message": f"No connection pool issue on {service}",
                "correct_fix": False,
            }

    def scale_service(self, service: str) -> dict:
        """Adds replicas to a service."""
        if service not in self._services:
            return {"success": False, "message": f"Unknown service: {service}"}
        self._log_action("scale_service", service)
        svc = self._services[service]
        svc["health"] = min(1.0, svc["health"] + 0.12)
        svc["error_rate"] = max(0.0, svc["error_rate"] - 0.08)
        svc["latency_ms"] = max(50, int(svc["latency_ms"] * 0.80))
        return {
            "success": True,
            "message": f"{service} scaled up — additional replicas added",
        }

    def mark_resolved(self) -> dict:
        """Agent declares incident resolved. Ends episode."""
        self._log_action("mark_resolved", "system")
        return {
            "success": True,
            "message": "Incident marked as resolved",
            "system_health": self.get_system_health(),
        }

    # ------------------------------------------------------------------
    # World tick (autonomous degradation)
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        Autonomous world update. Called after every agent action
        (except get_topology which is free).
        """
        self._step += 1

        for name, svc in self._services.items():
            if svc["failure_type"] is not None:
                rate = DEGRADATION_RATES[name]
                svc["health"] = max(0.0, round(svc["health"] - rate, 3))
                svc["error_rate"] = min(1.0, round(svc["error_rate"] + rate * 0.5, 3))
                if svc["health"] < 0.2:
                    svc["status"] = "down"
                elif svc["health"] < 0.7:
                    svc["status"] = "degraded"

        if self._active_scenario == "task4":
            events = self._scenario_config.get("nonstationary_events", {})
            if self._step in events:
                event = events[self._step]
                self._set_service_failure(
                    event["service"],
                    event["failure"],
                    health=event["health"],
                    error_rate=event["error_rate"],
                    latency_ms=event["latency_ms"],
                )

    # ------------------------------------------------------------------
    # Read-only helpers
    # ------------------------------------------------------------------

    def get_system_health(self) -> float:
        """Weighted aggregate health across all services."""
        weights = {
            "api-gateway": 0.25,
            "database": 0.30,
            "payment-service": 0.20,
            "auth-service": 0.10,
            "cache": 0.10,
            "notification-service": 0.05,
        }
        total = sum(
            self._services[svc]["health"] * w
            for svc, w in weights.items()
        )
        return round(total, 4)

    def get_active_alerts(self) -> List[dict]:
        """Returns alerts for services with health < 0.85."""
        alerts = []
        for name, svc in self._services.items():
            if svc["health"] < 0.85:
                severity = (
                    "critical" if svc["health"] < 0.25
                    else "high" if svc["health"] < 0.50
                    else "medium"
                )
                alerts.append({
                    "service": name,
                    "metric": "health",
                    "value": svc["health"],
                    "severity": severity,
                    "status": svc["status"],
                })
        order = {"critical": 0, "high": 1, "medium": 2}
        return sorted(alerts, key=lambda a: order.get(a["severity"], 3))

    def get_all_service_statuses(self) -> Dict[str, dict]:
        """Returns status snapshot of all services."""
        return {
            name: {
                "name": svc["name"],
                "health": svc["health"],
                "status": svc["status"],
                "error_rate": svc["error_rate"],
                "latency_ms": svc["latency_ms"],
            }
            for name, svc in self._services.items()
        }

    def get_step(self) -> int:
        return self._step

    def get_action_history(self) -> List[dict]:
        return list(self._action_history)

    def get_scenario_config(self) -> dict:
        return copy.deepcopy(self._scenario_config)

    def is_root_cause(self, service: str) -> bool:
        return self._services.get(service, {}).get("is_root_cause", False)

    def _log_action(self, tool: str, target: str) -> None:
        self._action_history.append({
            "step": self._step,
            "tool": tool,
            "target": target,
        })