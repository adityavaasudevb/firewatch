"""FireWatch Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FireWatchAction, FireWatchObservation


class FireWatchClient(
    EnvClient[FireWatchAction, FireWatchObservation, State]
):
    """
    Client for the FireWatch Environment.

    Example:
        >>> with FireWatchClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.system_health)
        ...
        ...     action = FireWatchAction(tool="get_logs", target="database")
        ...     result = client.step(action)
        ...     print(result.observation.last_action_result)
    """

    def _step_payload(self, action: FireWatchAction) -> Dict:
        """Convert FireWatchAction to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[FireWatchObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", payload)
        observation = FireWatchObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
    