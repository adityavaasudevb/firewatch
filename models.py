"""
Data models for the FireWatch SRE Incident Response Environment.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


# =============================================================================
# Action Model
# =============================================================================

class FireWatchAction(Action):
    """
    Action for the FireWatch environment.

    The agent selects a diagnostic or remediation tool and a target service.
    """

    tool: Literal[
        "get_metrics",
        "get_logs",
        "get_topology",
        "restart_service",
        "rollback_config",
        "scale_service",
        "reset_ratelimit",
        "sync_replica",
        "clear_connections",
        "mark_resolved",
    ] = Field(..., description="The diagnostic or remediation tool to use")

    target: Literal[
        "api-gateway",
        "auth-service",
        "payment-service",
        "database",
        "cache",
        "notification-service",
        "system",
    ] = Field(..., description="The target service or 'system' for global actions")

    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters for the action",
    )


# =============================================================================
# Sub-models (plain BaseModel, NOT Action)
# =============================================================================

class AlertModel(BaseModel):
    """Represents a firing alert for a degraded or down service."""

    service: str
    metric: str
    value: float
    severity: str
    status: str


class ServiceStatusModel(BaseModel):
    """Current status snapshot of a single service."""

    name: str
    health: float
    status: str
    error_rate: float
    latency_ms: int


class TopologyModel(BaseModel):
    """Service dependency graph returned by get_topology."""

    services: List[str]
    dependencies: Dict[str, List[str]]


# =============================================================================
# Observation Model
# =============================================================================

class FireWatchObservation(Observation):
    """
    Observation from the FireWatch environment.

    Extends the OpenEnv base Observation which provides:
      - done: bool
      - reward: Optional[float]
      - metadata: Dict[str, Any]
    """

    step: int = Field(default=0, description="Current step number")
    system_health: float = Field(default=1.0, description="Aggregate health 0.0-1.0")
    active_alerts: List[AlertModel] = Field(
        default_factory=list,
        description="Currently firing alerts sorted by severity",
    )
    services: Dict[str, ServiceStatusModel] = Field(
        default_factory=dict,
        description="Status of all 6 services",
    )
    last_action_result: str = Field(
        default="",
        description="Human-readable result of the last action",
    )
    incident_summary: str = Field(
        default="",
        description="Natural language description of the incident",
    )
    topology: Optional[TopologyModel] = Field(
        default=None,
        description="Dependency graph (None until get_topology is called)",
    )
    step_budget: Optional[int] = Field(
        default=None,
        description="Steps remaining (hidden in some tasks)",
    )