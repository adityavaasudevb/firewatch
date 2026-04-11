---
title: FireWatch Environment Server
emoji: 🔥
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FireWatch - SRE Incident Response RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-orange)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A real-world reinforcement learning environment where AI agents act as on-call Site Reliability Engineers (SREs), diagnosing and remediating live distributed system failures under time pressure.

FireWatch simulates a production distributed system of 6 interconnected services. Incidents occur, services degrade, and the agent must diagnose the root cause and apply the correct remediation - all while the system continues to deteriorate autonomously every step.

**The world ticks without waiting for the agent.** Unlike static environments, FireWatch implements autonomous degradation via a `tick()` mechanism: unhealthy services lose health every step whether or not the agent acts. This creates a genuine non-stationary MDP and forces the agent to balance investigation speed against action quality.

---

## Quick Start

The simplest way to use the FireWatch environment from this repository is through the `FireWatchClient` class:

```python
from client import FireWatchClient
from models import FireWatchAction

try:
    # Create environment from Docker image
    env = FireWatchClient.from_docker_image("firewatch-env:latest")

    # Reset to task1
    result = env.reset()
    print(f"System Health: {result.observation.system_health}")
    print(f"Active Alerts: {len(result.observation.active_alerts)}")

    # Take diagnostic actions
    actions = [
        FireWatchAction(tool="get_topology", target="system"),
        FireWatchAction(tool="get_logs", target="database"),
        FireWatchAction(tool="restart_service", target="database"),
        FireWatchAction(tool="mark_resolved", target="system"),
    ]

    for action in actions:
        result = env.step(action)
        print(f"Action: {action.tool}({action.target})")
        print(f"  -> Reward: {result.reward:.2f}")
        print(f"  -> Health: {result.observation.system_health:.2f}")
        print(f"  -> Done: {result.done}")
        if result.done:
            break

finally:
    # Always clean up
    env.close()
```

That's it! The `FireWatchClient.from_docker_image()` method handles:

- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

The examples in this README assume you are working from the repository root, where [client.py](/c:/Users/adiva/Desktop/firewatch/client.py) and [models.py](/c:/Users/adiva/Desktop/firewatch/models.py) are directly importable.

---

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t firewatch-env:latest .
```

---

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --repo-id your-username/firewatch --private
```

The `openenv push` command will:

1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face before pushing
- Verify that `openenv validate` passes locally
- Verify that the Docker image builds and runs locally

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment
- `--repo-id`, `-r`: Repository ID in format `username/repo-name`
- `--base-image`, `-b`: Override the Docker base image
- `--private`: Deploy the Space as private

### Examples

```bash
# Push to your default personal namespace
openenv push

# Push to a specific repository
openenv push --repo-id pepparrr/firewatch

# Push as a private Space
openenv push --private
```

After deployment, your space will be available at:

`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:

- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

---

## System Architecture

FireWatch simulates 6 services with the following dependency graph:

```text
api-gateway --> auth-service
api-gateway --> payment-service --> database
                                \--> cache
api-gateway --> notification-service
```

Each service has: `health (0.0-1.0)`, `status (healthy/degraded/down)`, `error_rate`, `latency_ms`, and `log_entries`.

---

## Environment Details

### Action

**FireWatchAction**: Contains tool selection and target service

| Tool | Target | Description |
|------|--------|-------------|
| `get_metrics` | any service | Observe health, error rate, latency |
| `get_logs` | any service | Read recent log entries (reveals failure type) |
| `get_topology` | `system` | View full dependency graph - **FREE, costs no step** |
| `restart_service` | any service | Restart (correct fix for OOM failures) |
| `rollback_config` | any service | Rollback deployment config (fixes memory leaks) |
| `reset_ratelimit` | any service | Reset rate limiter thresholds |
| `sync_replica` | any service | Force database replica sync (fixes replica lag) |
| `clear_connections` | any service | Reset connection pool (fixes connection exhaustion) |
| `scale_service` | any service | Add replicas to help with load |
| `mark_resolved` | `system` | End the episode (agent declares incident resolved) |

**Valid targets:** `api-gateway`, `auth-service`, `payment-service`, `database`, `cache`, `notification-service`, `system`

### Observation

**FireWatchObservation**: Contains full system state and metadata

| Field | Type | Description |
|-------|------|-------------|
| `step` | `int` | Current step number |
| `system_health` | `float` | Aggregate health 0.0-1.0 (weighted across all services) |
| `active_alerts` | `List[AlertModel]` | Firing alerts sorted by severity |
| `services` | `Dict[str, ServiceStatusModel]` | Full status of all 6 services |
| `last_action_result` | `str` | Plain English result of the last action |
| `last_action_error` | `Optional[str]` | Raw error string from the last action, or `None` |
| `incident_summary` | `str` | Natural language description of the incident |
| `topology` | `Optional[TopologyModel]` | Dependency graph (None until `get_topology` is called) |
| `step_budget` | `Optional[int]` | Steps remaining (hidden in Task 4) |
| `done` | `bool` | True when episode has ended |
| `reward` | `float` | Reward for the current step |
| `metadata` | `dict` | Additional info like episode_id, task_id, final_score |

### Reward

The reward function provides signal on **every single step** - never sparse, never binary.

```text
r(t) = health_delta * 2.0          # Health improvement/degradation
     + correct_fix_bonus           # +1.0 one-time for applying the correct fix
     + diagnosis_bonus             # +0.3 one-time for investigating the root cause service
     - wrong_fix_penalty           # -0.3 for restarting a healthy service
     - step_cost                   # -0.02 per step (prevents stalling)
```

`get_topology` has zero step cost - agents that plan before acting are rewarded by not wasting their budget.

Step rewards shape the trajectory, but the final benchmark score comes from a deterministic task grader. The final score is **not** the sum of the per-step rewards printed during inference.

---

## Tasks

FireWatch provides four tasks spanning increasing operational difficulty:

| Task | Difficulty | Primary Challenge |
|------|------------|-------------------|
| `task1` | Easy | Single root-cause service failure |
| `task2` | Medium | Cascading failure with a red herring |
| `task3` | Hard | Ordered multi-fix remediation |
| `task4` | Expert | Non-stationary incident with hidden step budget |

### Task 1 - Single Service Failure (Easy)

The database is down due to an OOM error. Payment-service and api-gateway show elevated latency as downstream symptoms. The agent must investigate logs, identify the root cause, and apply the correct fix within 10 steps.

- Root cause: `database` (OOM)
- Correct fix: `restart_service("database")`
- Step budget: 10

### Task 2 - Cascading Failure with Red Herring (Medium)

The database has connection pool exhaustion, causing payment-service and api-gateway to time out. Auth-service has an unrelated memory warning - it is **not** causing anything. The agent must trace the dependency graph, identify the true root cause, apply the correct fix, and avoid wasting actions on the red herring.

- Root cause: `database` (connection pool)
- Correct fix: `clear_connections("database")`
- Red herring: `auth-service` (memory warning, not causal)
- Step budget: 15

### Task 3 - Multi-vector Incident with Order Dependency (Hard)

Three simultaneous failures that **must be fixed in a specific order**:

1. `rollback_config("api-gateway")` - stops the memory leak first
2. `reset_ratelimit("api-gateway")` - only works after config is rolled back
3. `sync_replica("database")` - fixes stale reads

Applying fixes out of order reduces their effectiveness. At step 12, `notification-service` begins degrading as a distraction. The agent must prioritize correctly.

- Step budget: 20

### Task 4 - Non-stationary Adaptive Incident (Expert)

The world changes during the episode on a fixed schedule:

- **Step 0:** Database has two simultaneous failures (connection pool + replica lag)
- **Step 5:** Cache fails independently (new problem appears)
- **Step 8:** Notification-service begins degrading (downstream cascade)

The step budget is **hidden** from the agent (partial observability). The agent must handle the initial root cause while adapting to a changing system.

Scoring is split to ensure fairness:

- 60% - Did the agent fix the initial database failure?
- 25% - Did the agent respond to the new failures at steps 5 and 8?
- 15% - Final system health

- Step budget: 25 (hidden)

---

## Baseline Scores

Scores produced by a hybrid LLM + heuristic baseline:

| Task | Name | Difficulty | Baseline Score |
|------|------|------------|----------------|
| task1 | Single Service Failure | Easy | 0.79 |
| task2 | Cascading Failure with Red Herring | Medium | 0.76 |
| task3 | Multi-vector Ordered Incident | Hard | 0.84 |
| task4 | Non-stationary Adaptive Incident | Expert | 0.95 |
| **Average** | | | **0.83** |

These scores were produced with the verified submission-time setup using the Hugging Face router and `Qwen/Qwen2.5-7B-Instruct-1M`.

Measured full-run inference time with this verified setup is approximately **28 seconds** on the tested machine.

To reproduce baseline scores, run:

```bash
export HF_TOKEN="your_token"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

---

## Advanced Usage

### Connecting to an Existing Server

If you already have a FireWatch environment server running, you can connect directly:

```python
from client import FireWatchClient
from models import FireWatchAction

# Connect to existing server
env = FireWatchClient(base_url="http://localhost:8000")

# Use as normal
result = env.reset()
result = env.step(FireWatchAction(tool="get_logs", target="database"))
```

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from client import FireWatchClient
from models import FireWatchAction

with FireWatchClient(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Health: {result.observation.system_health}")

    # Multiple steps with low latency
    for tool, target in [("get_topology", "system"), ("get_logs", "database")]:
        result = env.step(FireWatchAction(tool=tool, target=target))
        print(f"Reward: {result.reward:.2f}")
```

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```python
from server.firewatch_environment import FireWatchEnvironment
from models import FireWatchAction

env = FireWatchEnvironment()

# Test all 4 tasks
for task_id in ["task1", "task2", "task3", "task4"]:
    obs = env.reset(task_id=task_id)
    print(f"{task_id}: health={obs.system_health:.2f}, alerts={len(obs.active_alerts)}")
```

---

## Development & Testing

### Running Locally

Run the server locally for development:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Test health endpoint
curl http://localhost:8000/health
```

### Running via Docker

```bash
# Build the image
docker build -t firewatch-env:latest .

# Run the container (web UI enabled by default)
docker run -p 8000:8000 firewatch-env:latest

# Verify
curl http://localhost:8000/health
```

### Run Baseline Inference

```bash
# Set environment variables
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="Qwen/Qwen2.5-7B-Instruct-1M"
export API_BASE_URL="https://router.huggingface.co/v1"

# Run inference
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns `{"status":"healthy"}` |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute action, returns observation + reward |
| `/state` | GET | Internal state snapshot |
| `/docs` | GET | Interactive API documentation |
| `/web` | GET | OpenEnv web interface |
| `/ws` | WS | WebSocket endpoint for persistent sessions |

---

## Project Structure

```text
firewatch/
|-- .dockerignore          # Docker build exclusions
|-- .gitignore             # Git ignore rules
|-- Dockerfile             # Container image definition (root)
|-- LICENSE                # MIT License
|-- README.md              # This file
|-- openenv.yaml           # OpenEnv manifest
|-- pyproject.toml         # Project metadata and dependencies
|-- uv.lock                # Locked dependencies (generated)
|-- requirements.txt       # Python dependencies
|-- client.py              # FireWatchClient
|-- models.py              # Action and Observation models
|-- inference.py           # Baseline inference script
|-- firewatch/             # Core simulation package
|   |-- __init__.py
|   |-- simulation.py      # The distributed system simulator
|   |-- reward.py          # Per-step reward function
|   |-- tasks.py           # Task scenario configurations
|   `-- graders.py         # Deterministic grading functions
`-- server/
    |-- __init__.py
    |-- firewatch_environment.py  # OpenEnv environment class
    |-- app.py                    # FastAPI application
    `-- Dockerfile                # Container image (backup copy)
```

---

## What Makes FireWatch Unique

**Non-stationary world:** Every existing OpenEnv environment waits patiently for the agent. FireWatch degrades autonomously every step via `tick()`. Passivity is penalized by the environment itself, not by a sparse end-reward. This makes FireWatch a genuine non-stationary MDP - the property the RL research community has identified as most missing from current benchmarks.

Unlike static or single-step benchmarks, delayed action in FireWatch changes the environment itself, so the agent is evaluated on triage under evolving system state rather than on isolated decisions.

**Structured traps for LLMs:** Task 2's red herring exploits the LLM tendency to fix the most visible alert rather than trace root causation. Task 3's order dependency breaks models that jump to action without planning. Task 4's partial observability (hidden step budget + changing topology) collapses static reasoners. These are documented, research-validated failure modes of frontier models - not arbitrary difficulty.

**Honest partial credit:** Every grader provides partial credit at multiple granularities. An agent that investigates correctly but applies the wrong fix still scores 0.20-0.35. No grader produces binary 0/1 outcomes.

FireWatch evaluates not just correctness, but sequencing, causality reasoning, and adaptation under changing system dynamics.

---

## Verified Deployment

FireWatch has been verified in the following ways:

- `openenv validate` passes
- `python inference.py` produces compliant structured logs
- `docker build` succeeds
- `docker run` succeeds
- `/health`, `/docs`, and `/web` respond successfully
- the Hugging Face Space deploys and runs successfully at `pepparrr/firewatch`

---

## OpenEnv Compliance

- Implements typed Action, Observation models (Pydantic)
- Supports `step()`, `reset()`, `state()`
- Deterministic graders with 0.0-1.0 scoring
- `openenv.yaml` provided
- Fully compatible with OpenEnv validation and deployment
- Passes `openenv validate`

---

## License

MIT
