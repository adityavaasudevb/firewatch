"""
server/firewatch_environment.py
===============================
FireWatch SRE Incident Response Environment.
"""

from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FireWatchAction, FireWatchObservation, AlertModel, ServiceStatusModel, TopologyModel
except ImportError:
    from models import FireWatchAction, FireWatchObservation, AlertModel, ServiceStatusModel, TopologyModel

try:
    from ..firewatch.simulation import SystemSimulator
    from ..firewatch.reward import compute_reward, reset_episode_tracking
    from ..firewatch.tasks import TASK_CONFIGS
    from ..firewatch.graders import GRADERS
except ImportError:
    from firewatch.simulation import SystemSimulator
    from firewatch.reward import compute_reward, reset_episode_tracking
    from firewatch.tasks import TASK_CONFIGS
    from firewatch.graders import GRADERS


def clamp_score(raw: float) -> float:
    """Clamp ONLY final grader scores to strictly within (0, 1)."""
    return round(max(0.01, min(0.99, raw)), 4)


class FireWatchEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.sim = SystemSimulator(seed=42)
        self._task_id: str = "task1"
        self._task_config: dict = TASK_CONFIGS["task1"]
        self._episode_id: str = str(uuid4())[:8]
        self._step: int = 0
        self._step_budget: int = 10
        self._done: bool = False
        self._final_score: float = 0.05
        self._last_action_result: str = "Episode not started. Call reset() first."
        self._last_action_error: Optional[str] = None
        self._topology_snapshot: dict = {}
        self._state = State(episode_id=self._episode_id, step_count=0)

    def reset(self, **kwargs) -> FireWatchObservation:
        task_id = kwargs.get("task_id", "task1")
        if task_id not in TASK_CONFIGS:
            task_id = "task1"

        self._task_id = task_id
        self._task_config = TASK_CONFIGS[task_id]
        self._episode_id = str(uuid4())[:8]
        self._step = 0
        self._step_budget = self._task_config["step_budget"]
        self._done = False
        self._final_score = 0.05
        self._last_action_result = "Incident detected. Begin investigation."
        self._last_action_error = None
        self._topology_snapshot = {}
        self._state = State(episode_id=self._episode_id, step_count=0)

        self.sim.apply_scenario(self._task_config["scenario_id"])
        reset_episode_tracking(self._episode_id)

        # reward=0.0 on reset is fine — not checked by validator
        return self._build_observation(reward=0.0, done=False)

    def step(self, action: FireWatchAction) -> FireWatchObservation:
        if self._done:
            return self._build_observation(
                reward=0.0,
                done=True,
                extra_metadata={
                    "message": "Episode already ended.",
                    "final_score": self._final_score,
                },
            )

        prev_health = self.sim.get_system_health()
        action_result = self._execute_action(action)
        self._last_action_result = action_result.get("message", "Action completed.")

        if action_result.get("success") is False or action_result.get("error"):
            self._last_action_error = (
                action_result.get("error") or action_result.get("message")
            )
        else:
            self._last_action_error = None

        if action.tool == "get_topology":
            self._topology_snapshot = action_result

        curr_health = self.sim.get_system_health()

        # Raw reward — negatives are intentional and meaningful
        # r(t) = health_delta*2.0 + correct_fix_bonus + diagnosis_bonus
        #       - wrong_fix_penalty - step_cost
        reward = compute_reward(
            episode_id=self._episode_id,
            prev_health=prev_health,
            curr_health=curr_health,
            action_tool=action.tool,
            action_target=action.target,
            action_result=action_result,
            root_cause_services=self._task_config.get("root_causes", []),
        )

        if action.tool != "get_topology":
            self.sim.tick()
            self._step += 1
            self._state.step_count = self._step

        done = False
        extra_metadata = {}

        if action.tool == "mark_resolved":
            done = True
            extra_metadata["termination"] = "agent_resolved"
        elif self._step >= self._step_budget:
            done = True
            extra_metadata["termination"] = "budget_exhausted"

        if done:
            self._done = True
            grader_fn = GRADERS.get(self._task_id)
            if grader_fn:
                grader_output = grader_fn(
                    action_history=self.sim.get_action_history(),
                    final_system_health=self.sim.get_system_health(),
                    scenario_config=self.sim.get_scenario_config(),
                    steps_taken=self._step,
                    max_steps=self._step_budget,
                )
                # Only grader scores are clamped to (0.01, 0.99)
                self._final_score = clamp_score(grader_output["score"])
                grader_output["score"] = self._final_score
                extra_metadata["final_score"] = self._final_score
                extra_metadata["grader_details"] = grader_output
            else:
                extra_metadata["final_score"] = self._final_score

        return self._build_observation(
            reward=reward,
            done=done,
            extra_metadata=extra_metadata,
        )

    @property
    def state(self) -> State:
        return self._state

    def close(self):
        self._done = True

    def _execute_action(self, action: FireWatchAction) -> dict:
        t = action.tool
        tgt = action.target
        p = action.parameters or {}

        dispatch = {
            "get_metrics": lambda: self.sim.get_metrics(tgt),
            "get_logs": lambda: self.sim.get_logs(tgt, lines=int(p.get("lines", 10))),
            "get_topology": lambda: self.sim.get_topology(),
            "restart_service": lambda: self.sim.restart_service(tgt),
            "rollback_config": lambda: self.sim.rollback_config(tgt),
            "scale_service": lambda: self.sim.scale_service(tgt),
            "reset_ratelimit": lambda: self.sim.reset_ratelimit(tgt),
            "sync_replica": lambda: self.sim.sync_replica(tgt),
            "clear_connections": lambda: self.sim.clear_connections(tgt),
            "mark_resolved": lambda: self.sim.mark_resolved(),
        }

        handler = dispatch.get(t)
        if handler:
            return handler()
        return {"success": False, "message": f"Unknown tool: {t}"}

    def _build_observation(self, reward, done, extra_metadata=None):
        services = {
            name: ServiceStatusModel(**s)
            for name, s in self.sim.get_all_service_statuses().items()
        }
        alerts = [AlertModel(**a) for a in self.sim.get_active_alerts()]

        hide_budget = self._task_config.get("hide_step_budget", False)
        budget = None if hide_budget else max(0, self._step_budget - self._step)

        topology = None
        if self._topology_snapshot:
            topology = TopologyModel(
                services=self._topology_snapshot.get("services", []),
                dependencies=self._topology_snapshot.get("dependencies", {}),
            )

        metadata = {
            "episode_id": self._episode_id,
            "task_id": self._task_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return FireWatchObservation(
            step=self._step,
            system_health=self.sim.get_system_health(),
            active_alerts=alerts,
            services=services,
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            incident_summary=self._task_config.get("description", ""),
            topology=topology,
            step_budget=budget,
            done=done,
            reward=reward,  # raw, never clamped — negatives are valid
            metadata=metadata,
        )