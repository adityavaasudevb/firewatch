"""FireWatch Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import FireWatchAction, FireWatchObservation


class FireWatchClient(EnvClient[FireWatchAction, FireWatchObservation, State]):
    """Client for the FireWatch Environment."""

    def _step_payload(self, action: FireWatchAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[FireWatchObservation]:
        obs_data = payload.get("observation", payload)
        observation = FireWatchObservation(**obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )