"""
inference.py — FireWatch Baseline Inference Script

MANDATORY COMPLIANCE:
  - Uses OpenAI API client for all LLM calls
  - Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars
  - Default values for API_BASE_URL and MODEL_NAME
  - Emits [START] [STEP] [END] structured logs (lowercase booleans)
  - Runtime under 20 minutes on 2 vCPU / 8GB RAM
"""

import json
import os
import re
import sys
import textwrap
import time
from typing import Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# MANDATORY env vars (with defaults where required)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")

if HF_TOKEN is None:
    print("[WARN] HF_TOKEN not set — LLM calls will fail, using heuristic fallback.", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_STEPS_PER_TASK = 30
TEMPERATURE = 0.0
MAX_TOKENS = 350
SUCCESS_SCORE_THRESHOLD = 0.5

VALID_TOOLS = [
    "get_metrics", "get_logs", "get_topology",
    "restart_service", "rollback_config", "scale_service",
    "reset_ratelimit", "sync_replica", "clear_connections",
    "mark_resolved",
]
VALID_TARGETS = [
    "api-gateway", "auth-service", "payment-service",
    "database", "cache", "notification-service", "system",
]

# ---------------------------------------------------------------------------
# Structured logging — [START] [STEP] [END]
# Uses lowercase true/false as required by spec
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a live
production incident. Diagnose the root cause and restore system health.

AVAILABLE TOOLS:
  get_metrics(service)       check health, error_rate, latency
  get_logs(service)          read log entries — REVEALS the failure type
  get_topology()             view service dependency graph (FREE, costs no step)
  restart_service(service)   fixes OOM errors (look for 'OOMKiller' in logs)
  rollback_config(service)   fixes config/memory-leak failures
  reset_ratelimit(service)   fixes rate limiter issues (apply AFTER rollback_config)
  sync_replica(service)      fixes database replica lag
  clear_connections(service) fixes connection pool exhaustion
  scale_service(service)     adds replicas to help with load
  mark_resolved()            CALL THIS to end the episode when health > 0.80

VALID SERVICES:
  api-gateway, auth-service, payment-service, database, cache,
  notification-service, system

DECISION RULES:
  1. ALWAYS call get_topology() first (it is FREE)
  2. Read logs on every alerting service before taking any fix action
  3. Match the fix to the log evidence:
       - "OOMKiller" in logs       -> restart_service
       - "connection pool"         -> clear_connections
       - "config reload" / memory  -> rollback_config, THEN reset_ratelimit
       - "replication lag"         -> sync_replica
  4. Fix multiple issues: apply fixes in order, one per step
  5. When ALL critical services are healthy (health > 0.80), call mark_resolved
  6. NEVER repeat the same action on the same service twice

RESPONSE FORMAT — respond with ONLY valid JSON, no other text:
{
  "reasoning": "one sentence explaining your decision",
  "tool": "tool_name",
  "target": "service_name"
}
""").strip()


# ---------------------------------------------------------------------------
# Observation -> text prompt
# ---------------------------------------------------------------------------

def observation_to_prompt(obs_dict: dict, step: int, action_history: list) -> str:
    """Convert observation dict to a text prompt for the LLM."""
    alerts = obs_dict.get("active_alerts", [])
    services = obs_dict.get("services", {})
    last = obs_dict.get("last_action_result", "")
    budget = obs_dict.get("step_budget")
    health = obs_dict.get("system_health", 0.0)
    summary = obs_dict.get("incident_summary", "")
    topology = obs_dict.get("topology")

    alert_lines = "\n".join(
        f"  [{a['severity'].upper()}] {a['service']}: "
        f"health={a['value']:.2f}, status={a['status']}"
        for a in alerts
    ) or "  None — system is healthy"

    svc_lines = "\n".join(
        f"  {name}: health={s['health']:.2f}, status={s['status']}, "
        f"error_rate={s['error_rate']:.2f}, latency={s['latency_ms']}ms"
        for name, s in services.items()
    )

    topo_section = ""
    if topology:
        deps = (
            topology.get("dependencies", {})
            if isinstance(topology, dict)
            else getattr(topology, "dependencies", {})
        )
        topo_lines = "\n".join(
            f"  {svc} -> {d or '(no deps)'}" for svc, d in deps.items()
        )
        topo_section = f"\nDEPENDENCY GRAPH:\n{topo_lines}"

    history_lines = ""
    if action_history:
        recent = action_history[-6:]
        history_lines = "\nRECENT ACTIONS (do not repeat):\n" + "\n".join(
            f"  step {a['step']}: {a['tool']}({a['target']})"
            for a in recent
        )

    budget_str = str(budget) if budget is not None else "HIDDEN"

    urgency = ""
    if budget is not None and budget <= 5:
        urgency = f"\n⚠ URGENT: Only {budget} steps left. Apply fixes immediately or call mark_resolved."
    elif health >= 0.82:
        urgency = "\n✓ System health is good. Consider calling mark_resolved."

    return textwrap.dedent(f"""
STEP {step} | System Health: {health:.3f} | Steps Remaining: {budget_str}
{urgency}

INCIDENT DESCRIPTION:
{summary.strip()[:300]}

ACTIVE ALERTS:
{alert_lines}

ALL SERVICES:
{svc_lines}
{topo_section}
{history_lines}

LAST ACTION RESULT:
{last}

What is your next action? Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# LLM response parser
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[dict]:
    """Parse LLM JSON response into tool/target dict."""
    if not response_text:
        return None

    try:
        match = re.search(r"\{.*?\}", response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            tool = str(parsed.get("tool", "")).strip()
            target = str(parsed.get("target", "")).strip()
            if tool in VALID_TOOLS and target in VALID_TARGETS:
                return {"tool": tool, "target": target}
    except Exception:
        pass

    tm = re.search(r'"tool"\s*:\s*"([^"]+)"', response_text)
    xm = re.search(r'"target"\s*:\s*"([^"]+)"', response_text)
    if tm and xm:
        t, x = tm.group(1).strip(), xm.group(1).strip()
        if t in VALID_TOOLS and x in VALID_TARGETS:
            return {"tool": t, "target": x}

    return None


# ---------------------------------------------------------------------------
# Heuristic fallback agent
# ---------------------------------------------------------------------------

def heuristic_fallback(obs: dict, history: list) -> dict:
    """
    Deterministic fallback when LLM is unavailable or gives bad output.
    """
    done_actions = {(a["tool"], a["target"]) for a in history}
    services = obs.get("services", {})
    health = obs.get("system_health", 0.0)

    if ("get_topology", "system") not in done_actions:
        return {"tool": "get_topology", "target": "system"}

    critical_order = [
        "database", "api-gateway", "payment-service",
        "cache", "auth-service", "notification-service",
    ]
    for svc in critical_order:
        svc_data = services.get(svc, {})
        status = svc_data.get("status", "healthy") if isinstance(svc_data, dict) else "healthy"
        if status != "healthy" and ("get_logs", svc) not in done_actions:
            return {"tool": "get_logs", "target": svc}

    if services.get("api-gateway", {}).get("status") != "healthy":
        if ("rollback_config", "api-gateway") not in done_actions:
            return {"tool": "rollback_config", "target": "api-gateway"}
        if ("reset_ratelimit", "api-gateway") not in done_actions:
            return {"tool": "reset_ratelimit", "target": "api-gateway"}

    db_status = services.get("database", {}).get("status", "healthy")
    if db_status != "healthy":
        if ("clear_connections", "database") not in done_actions:
            return {"tool": "clear_connections", "target": "database"}
        if ("restart_service", "database") not in done_actions:
            return {"tool": "restart_service", "target": "database"}
        if ("sync_replica", "database") not in done_actions:
            return {"tool": "sync_replica", "target": "database"}

    if services.get("cache", {}).get("status") != "healthy":
        if ("restart_service", "cache") not in done_actions:
            return {"tool": "restart_service", "target": "cache"}

    if services.get("notification-service", {}).get("status") != "healthy":
        if ("scale_service", "notification-service") not in done_actions:
            return {"tool": "scale_service", "target": "notification-service"}

    degraded = [
        s for s, v in services.items()
        if isinstance(v, dict) and v.get("status") != "healthy"
    ]
    if health > 0.82 or not degraded:
        return {"tool": "mark_resolved", "target": "system"}

    for svc in degraded:
        if ("get_metrics", svc) not in done_actions:
            return {"tool": "get_metrics", "target": svc}

    return {"tool": "mark_resolved", "target": "system"}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(env, llm_client: OpenAI, task_id: str, task_name: str) -> dict:
    """Run a full episode and emit structured logs."""
    from models import FireWatchAction

    BENCHMARK = "firewatch"
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()

    rewards: List[float] = []
    action_history: List[dict] = []
    steps_taken: int = 0
    final_score: float = 0.0
    success: bool = False

    for step_num in range(1, MAX_STEPS_PER_TASK + 1):
        if obs_dict.get("done", False):
            break

        error_msg: Optional[str] = None
        user_prompt = observation_to_prompt(obs_dict, step_num, action_history)

        # Call LLM
        action = None
        try:
            if HF_TOKEN:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_action(response_text)
            else:
                response_text = ""
        except Exception as exc:
            response_text = ""
            error_msg = str(exc)

        # Prevent repetition
        if action and len(action_history) >= 3:
            last_3 = [(a["tool"], a["target"]) for a in action_history[-3:]]
            if last_3.count((action["tool"], action["target"])) >= 2:
                action = heuristic_fallback(obs_dict, action_history)

        # Fallback to heuristic if LLM failed
        if action is None:
            action = heuristic_fallback(obs_dict, action_history)

        action_str = f"{action['tool']}({action['target']})"

        step_reward = 0.0
        done = False

        try:
            fw_action = FireWatchAction(
                tool=action["tool"],
                target=action["target"],
                parameters={},
            )
            result = env.step(fw_action)

            # result is a FireWatchObservation (OpenEnv compliant)
            obs_dict = result.model_dump()
            step_reward = float(result.reward or 0.0)
            done = result.done
            steps_taken = step_num

            if done and result.metadata:
                final_score = float(result.metadata.get("final_score", 0.0))

        except Exception as exc:
            error_msg = str(exc)
            done = True

        rewards.append(step_reward)
        action_history.append({
            "step": step_num,
            "tool": action["tool"],
            "target": action["target"],
        })

        log_step(
            step=step_num,
            action=action_str,
            reward=step_reward,
            done=done,
            error=error_msg,
        )

        if done:
            break

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "task_name": task_name,
        "score": final_score,
        "steps_taken": steps_taken,
        "rewards": rewards,
        "success": success,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> float:
    print("=" * 60, flush=True)
    print("  FireWatch — Baseline Inference Script", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model:    {MODEL_NAME}", flush=True)
    print(f"  Base URL: {API_BASE_URL}", flush=True)
    print(f"  Max steps per task: {MAX_STEPS_PER_TASK}", flush=True)
    print("=" * 60, flush=True)

    from server.firewatch_environment import FireWatchEnvironment

    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "placeholder",
    )
    env = FireWatchEnvironment()

    tasks = [
        ("task1", "Single Service Failure"),
        ("task2", "Cascading Failure with Red Herring"),
        ("task3", "Multi-vector Ordered Incident"),
        ("task4", "Non-stationary Adaptive Incident"),
    ]

    start_time = time.time()
    results = []

    for task_id, task_name in tasks:
        print(f"\n  ── {task_name} ──", flush=True)
        result = run_task(env, llm_client, task_id, task_name)
        results.append(result)

    elapsed = time.time() - start_time
    avg_score = sum(r["score"] for r in results) / len(results)

    print("\n" + "=" * 60, flush=True)
    print("  SCORE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        bar = "█" * int(r["score"] * 20) + "░" * (20 - int(r["score"] * 20))
        print(f"  {r['task_id']:<10} {r['score']:.4f}  [{bar}]", flush=True)
    print(f"\n  Average: {avg_score:.4f} | Time: {elapsed:.1f}s", flush=True)
    print("=" * 60, flush=True)

    output = {
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "average_score": avg_score,
        "results": results,
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n  Saved -> baseline_scores.json", flush=True)
    return avg_score


if __name__ == "__main__":
    main()