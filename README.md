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

# FireWatch — SRE Incident Response RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-orange)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A real-world reinforcement learning environment where AI agents act as on-call Site Reliability Engineers (SREs), diagnosing and remediating live distributed system failures under time pressure.

FireWatch simulates a production distributed system of 6 interconnected services. Incidents occur, services degrade, and the agent must diagnose the root cause and apply the correct remediation — all while the system continues to deteriorate autonomously every step.

**The world ticks without waiting for the agent.** Unlike static environments, FireWatch implements autonomous degradation via a `tick()` mechanism: unhealthy services lose health every step whether or not the agent acts. This creates a genuine non-stationary MDP and forces the agent to balance investigation speed against action quality.

---

## Why FireWatch Matters

### The Problem
Every production team runs incident response. When systems break at 3am, an on-call engineer must diagnose root causes across interconnected services, under time pressure, with incomplete information. This is one of the highest-stakes human tasks in software engineering — and one of the least benchmarked for AI agents.

### What's Missing Today
Existing agent benchmarks evaluate static reasoning: answer a question, write some code, fill a form. None evaluate **triage under non-stationary conditions** — where the system degrades while you think, where symptoms mislead you toward the wrong service, and where the order of your actions changes their effectiveness.

### What FireWatch Provides
- **For RL researchers:** A non-stationary MDP with dense rewards, partial observability, and structured traps — properties identified as critical gaps in current benchmarks.
- **For agent developers:** A realistic evaluation of causal reasoning, prioritization, and adaptation — the skills that separate useful AI assistants from toy demos.
- **For SRE teams:** A training ground for evaluating whether AI agents can handle real incident patterns (cascading failures, red herrings, ordered remediation).

---

## Quick Start

```python
from client import FireWatchClient
from models import FireWatchAction

try:
    env = FireWatchClient.from_docker_image("firewatch-env:latest")
    result = env.reset()
    print(f"System Health: {result.observation.system_health}")
    print(f"Active Alerts: {len(result.observation.active_alerts)}")

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
    env.close()
```

---

## Building the Docker Image

```bash
docker build -t firewatch-env:latest .
```

---

## Deploying to Hugging Face Spaces

```bash
openenv push
# or
openenv push --repo-id your-username/firewatch --private
```

The deployed space includes:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`

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

### Action Space

**FireWatchAction**: Contains tool selection and target service

| Tool | Target | Description |
|------|--------|-------------|
| `get_metrics` | any service | Observe health, error rate, latency |
| `get_logs` | any service | Read recent log entries (reveals failure type) |
| `get_topology` | `system` | View full dependency graph — **FREE, costs no step** |
| `restart_service` | any service | Correct fix for OOM failures |
| `rollback_config` | any service | Correct fix for config/memory leak failures |
| `reset_ratelimit` | any service | Correct fix for rate limiting (apply after rollback_config) |
| `sync_replica` | any service | Correct fix for replica lag |
| `clear_connections` | any service | Correct fix for connection pool exhaustion |
| `scale_service` | any service | Add replicas to help with load |
| `mark_resolved` | `system` | End the episode (agent declares incident resolved) |

**Valid targets:** `api-gateway`, `auth-service`, `payment-service`, `database`, `cache`, `notification-service`, `system`

### Observation Space

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

### Reward Function

The reward function provides signal on **every single step** — never sparse, never binary.

```text
r(t) = health_delta * 2.0       # Health improvement/degradation
     + correct_fix_bonus        # +1.0 one-time for applying the correct fix
     + diagnosis_bonus          # +0.3 one-time for investigating the root cause
     - wrong_fix_penalty        # -0.3 for restarting a healthy service
     - step_cost                # -0.02 per step (prevents stalling)
```

`get_topology` has zero step cost — agents that plan before acting are rewarded by not wasting their budget.

**Reward Examples:**

| Action | Scenario | Reward |
|--------|----------|--------|
| `get_topology(system)` | Any | `0.00` — free action |
| `get_logs(database)` | DB is root cause | `+0.28` — diagnosis bonus |
| `get_logs(cache)` | Cache is healthy | `-0.02` — step cost only |
| `restart_service(database)` | DB has OOM | `+1.55` — health delta + fix bonus |
| `restart_service(cache)` | Cache is healthy | `-0.37` — wrong fix penalty |
| `mark_resolved(system)` | End of episode | `-0.02` — step cost only |

---

## Tasks

FireWatch provides four tasks spanning increasing operational difficulty:

| Task | Difficulty | Primary Challenge | Step Budget |
|------|------------|-------------------|-------------|
| `task1` | Easy | Single root-cause service failure | 10 |
| `task2` | Medium | Cascading failure with a red herring | 15 |
| `task3` | Hard | Ordered multi-fix remediation | 20 |
| `task4` | Expert | Non-stationary incident with hidden step budget | 25 (hidden) |

### Task 1 — Single Service Failure (Easy)

The database is down due to an OOM error. Payment-service and api-gateway show elevated latency as downstream symptoms.

- **Root cause:** `database` (OOM)
- **Correct fix:** `restart_service("database")`

### Task 2 — Cascading Failure with Red Herring (Medium)

The database has connection pool exhaustion, causing payment-service and api-gateway to time out. Auth-service has an unrelated memory warning — it is **not** causing anything. The agent must trace the dependency graph, identify the true root cause, and avoid wasting actions on the red herring.

- **Root cause:** `database` (connection pool)
- **Correct fix:** `clear_connections("database")`
- **Red herring:** `auth-service` (memory warning, not causal)

### Task 3 — Multi-vector Incident with Order Dependency (Hard)

Three simultaneous failures that **must be fixed in a specific order**:

1. `rollback_config("api-gateway")` — stops the memory leak first
2. `reset_ratelimit("api-gateway")` — only works after config is rolled back
3. `sync_replica("database")` — fixes stale reads

Applying fixes out of order reduces their effectiveness.

### Task 4 — Non-stationary Adaptive Incident (Expert)

The world changes during the episode on a fixed schedule:

- **Step 0:** Database has connection pool exhaustion
- **Step 5:** Cache fails independently (new problem appears)
- **Step 8:** Notification-service begins degrading (downstream cascade)

The step budget is **hidden** from the agent (partial observability). The agent must handle the initial root cause while adapting to a changing system.

---

## Baseline Scores

Scores produced using `meta-llama/Llama-3.3-70B-Instruct` via the Hugging Face router with a multi-turn LLM baseline and investigation fallback:

| Task | Name | Difficulty | Score |
|------|------|------------|-------|
| task1 | Single Service Failure | Easy | 0.91 |
| task2 | Cascading Failure with Red Herring | Medium | 0.45 |
| task3 | Multi-vector Ordered Incident | Hard | 0.69 |
| task4 | Non-stationary Adaptive Incident | Expert | 0.60 |

**Score interpretation:**

- **Task 1 (0.91):** Single failure, read logs, apply fix, resolve — optimal behavior demonstrated.
- **Task 2 (0.45):** Model correctly fixes root cause but consistently falls for the red herring (auth-service), applying unnecessary fixes. This is a documented LLM failure mode — models trained on helpful-assistant data compulsively address every visible alert. The grader correctly penalizes this.
- **Task 3 (0.69):** Model applies all three fixes in correct order when it reads the right logs. Partial scores reflect cases where logs are read but fixes miss.
- **Task 4 (0.60):** Model handles primary database fix and cache failure. Non-stationary nature and hidden budget create genuine expert-level difficulty.

**Note on Task 2 vs Task 3/4:** Task 2 scores lower not because it is harder conceptually, but because the red herring specifically exploits a well-documented LLM bias. A model that avoids the red herring trap scores ~0.70+. Task 3 and 4 require multi-step planning which is a different kind of difficulty.

To reproduce baseline scores:

```bash
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

---

## What Makes FireWatch Unique

**Non-stationary world:** Every existing OpenEnv environment waits patiently for the agent. FireWatch degrades autonomously every step via `tick()`. Passivity is penalized by the environment itself, not by a sparse end-reward.

**Structured traps for LLMs:** Task 2's red herring exploits the LLM tendency to fix the most visible alert. Task 3's order dependency breaks models that jump to action without planning. Task 4's partial observability collapses static reasoners. These are documented, research-validated failure modes of frontier models — not arbitrary difficulty.

**Diagnosis-action gap:** Our baseline reveals that even 70B-parameter models correctly diagnose failures but fail to bridge from evidence to remediation. This gap is exactly what RL training is designed to close.

**Honest partial credit:** Every grader provides partial credit at multiple granularities. No grader produces binary 0/1 outcomes. An agent that investigates correctly but applies the wrong fix still scores 0.20-0.40.

---

## Difficulty Analysis

### Why Task 2 Breaks Naive Agents
Auth-service shows `WARN memory usage at 78%` — a salient but irrelevant alert. LLMs trained on helpful-assistant data compulsively address every visible problem. Applying any fix to auth-service wastes steps and gets penalized while the real root cause continues degrading.

### Why Task 3 Requires Planning
The three required fixes have a strict dependency chain. `reset_ratelimit` only works fully after `rollback_config`. Applying fixes in wrong order reduces effectiveness and the grader penalizes wrong-order attempts.

### Why Task 4 Defeats Static Reasoners
New failures appear at steps 5 and 8. The step budget is hidden. An agent that solves the initial database failure but ignores the cache crash at step 5 scores only ~0.45. Continuous re-assessment is required.

---

## Development & Testing

### Running Locally

```bash
pip install -r requirements.txt
pip install -e .
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

### Running via Docker

```bash
docker build -t firewatch-env:latest .
docker run -p 8000:8000 firewatch-env:latest
curl http://localhost:8000/health
```

### Run Baseline Inference

```bash
export HF_TOKEN="your_huggingface_token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status":"healthy"}` |
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
|-- Dockerfile
|-- LICENSE
|-- README.md
|-- openenv.yaml
|-- pyproject.toml
|-- requirements.txt
|-- client.py
|-- models.py
|-- inference.py
|-- firewatch/
|   |-- __init__.py
|   |-- simulation.py
|   |-- reward.py
|   |-- tasks.py
|   `-- graders.py
`-- server/
    |-- __init__.py
    |-- firewatch_environment.py
    |-- app.py
    `-- Dockerfile
```

---

## OpenEnv Compliance

- Implements typed `Action`, `Observation` models (Pydantic)
- Supports `step()`, `reset()`, `state()`
- Deterministic graders with scores strictly in (0, 1)
- `openenv.yaml` provided
- Passes `openenv validate`
- Deploys successfully at `pepparrr/firewatch`

---

## License

MIT