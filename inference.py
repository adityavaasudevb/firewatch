"""
inference.py — FireWatch Baseline Inference Script
"""

import json
import os
import re
import textwrap

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

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


def clamp_strict(raw: float) -> float:
    """Clamp to strictly within (0, 1) — never exactly 0.0 or 1.0."""
    return round(max(0.01, min(0.99, raw)), 2)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    safe_reward = clamp_strict(reward)
    print(
        f"[STEP] step={step} action={action} reward={safe_reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    success_str = "true" if success else "false"
    safe_score = clamp_strict(score)
    safe_rewards = [clamp_strict(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(
        f"[END] success={success_str} steps={steps} score={safe_score:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a live
production incident. Diagnose the root cause and restore system health.

AVAILABLE TOOLS:
  get_metrics(service)       check health, error_rate, latency
  get_logs(service)          read log entries - REVEALS the failure type
  get_topology()             view service dependency graph (FREE, costs no step)
  restart_service(service)   fixes OOM errors (look for OOMKiller in logs)
  rollback_config(service)   fixes config/memory-leak failures
  reset_ratelimit(service)   fixes rate limiter issues (apply AFTER rollback_config)
  sync_replica(service)      fixes database replica lag
  clear_connections(service) fixes connection pool exhaustion
  scale_service(service)     adds replicas to help with load
  mark_resolved()            CALL THIS to end the episode when health > 0.80

VALID SERVICES:
  api-gateway, auth-service, payment-service, database, cache,
  notification-service, system

RULES:
  1. ALWAYS call get_topology() first (FREE)
  2. Read logs on alerting services before fixing
  3. Match fix to log evidence
  4. When health > 0.80, call mark_resolved
  5. NEVER repeat same action twice

Respond with ONLY valid JSON:
{"reasoning": "...", "tool": "tool_name", "target": "service_name"}
""").strip()


def observation_to_prompt(obs_dict, step, action_history):
    alerts = obs_dict.get("active_alerts", [])
    services = obs_dict.get("services", {})
    last = obs_dict.get("last_action_result", "")
    budget = obs_dict.get("step_budget")
    health = obs_dict.get("system_health", 0.0)
    summary = obs_dict.get("incident_summary", "")
    topology = obs_dict.get("topology")

    alert_lines = "\n".join(
        f"  [{a['severity'].upper()}] {a['service']}: health={a['value']:.2f}, status={a['status']}"
        for a in alerts
    ) or "  None"

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
        history_lines = "\nRECENT ACTIONS:\n" + "\n".join(
            f"  step {a['step']}: {a['tool']}({a['target']})" for a in recent
        )

    budget_str = str(budget) if budget is not None else "HIDDEN"

    return f"""STEP {step} | Health: {health:.3f} | Budget: {budget_str}

INCIDENT: {summary.strip()[:200]}

ALERTS:
{alert_lines}

SERVICES:
{svc_lines}
{topo_section}
{history_lines}

LAST RESULT: {last}

Respond with JSON only."""


def parse_action(response_text):
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


def heuristic_fallback(obs, history):
    done_actions = {(a["tool"], a["target"]) for a in history}
    services = obs.get("services", {})
    health = obs.get("system_health", 0.0)

    if ("get_topology", "system") not in done_actions:
        return {"tool": "get_topology", "target": "system"}

    for svc in [
        "database", "api-gateway", "payment-service",
        "cache", "auth-service", "notification-service",
    ]:
        svc_data = services.get(svc, {})
        status = (
            svc_data.get("status", "healthy")
            if isinstance(svc_data, dict)
            else "healthy"
        )
        if status != "healthy" and ("get_logs", svc) not in done_actions:
            return {"tool": "get_logs", "target": svc}

    if services.get("api-gateway", {}).get("status") != "healthy":
        if ("rollback_config", "api-gateway") not in done_actions:
            return {"tool": "rollback_config", "target": "api-gateway"}
        if ("reset_ratelimit", "api-gateway") not in done_actions:
            return {"tool": "reset_ratelimit", "target": "api-gateway"}

    if services.get("database", {}).get("status", "healthy") != "healthy":
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


TASK_NAMES = {
    "task1": "Single Service Failure",
    "task2": "Cascading Failure with Red Herring",
    "task3": "Multi-vector Ordered Incident",
    "task4": "Non-stationary Adaptive Incident",
}


def run_task(env, llm_client, task_id):
    """Run one task episode. Returns dict with results. Does NOT emit [END]."""
    from models import FireWatchAction

    task_name = TASK_NAMES.get(task_id, task_id)
    log_start(task=task_name, env="firewatch", model=MODEL_NAME)

    rewards = []
    action_history = []
    steps_taken = 0
    final_score = 0.05

    try:
        obs = env.reset(task_id=task_id)
        obs_dict = obs.model_dump()

        for step_num in range(1, MAX_STEPS_PER_TASK + 1):
            if obs_dict.get("done", False):
                break

            user_prompt = observation_to_prompt(obs_dict, step_num, action_history)

            action = None
            try:
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
            except Exception:
                response_text = ""

            if action and len(action_history) >= 3:
                last_3 = [(a["tool"], a["target"]) for a in action_history[-3:]]
                if last_3.count((action["tool"], action["target"])) >= 2:
                    action = heuristic_fallback(obs_dict, action_history)

            if action is None:
                action = heuristic_fallback(obs_dict, action_history)

            action_str = f"{action['tool']}({action['target']})"
            step_reward = 0.01
            done = False
            last_action_error = None

            try:
                fw_action = FireWatchAction(
                    tool=action["tool"],
                    target=action["target"],
                    parameters={},
                )
                result = env.step(fw_action)
                obs_dict = result.model_dump()
                step_reward = clamp_strict(float(result.reward or 0.01))
                done = result.done
                steps_taken = step_num
                last_action_error = obs_dict.get("last_action_error", None)

                if done and result.metadata:
                    raw_score = float(result.metadata.get("final_score", 0.05))
                    final_score = clamp_strict(raw_score)
            except Exception:
                step_reward = 0.01
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
                error=last_action_error,
            )

            if done:
                break

    except Exception:
        final_score = 0.05

    final_score = clamp_strict(final_score)
    success = final_score >= SUCCESS_SCORE_THRESHOLD

    return {
        "task_id": task_id,
        "score": final_score,
        "success": success,
        "steps_taken": steps_taken,
        "rewards": rewards,
    }


def main():
    from server.firewatch_environment import FireWatchEnvironment

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    for task_id in ["task1", "task2", "task3", "task4"]:
        env = FireWatchEnvironment()
        result = None
        try:
            result = run_task(env, llm_client, task_id)
        finally:
            if hasattr(env, "close"):
                env.close()
            if result is None:
                result = {
                    "score": 0.05,
                    "success": False,
                    "steps_taken": 0,
                    "rewards": [0.01],
                }
            log_end(
                success=result["success"],
                steps=result["steps_taken"],
                score=result["score"],        # ← THIS WAS MISSING
                rewards=result["rewards"],
            )


if __name__ == "__main__":
    main()