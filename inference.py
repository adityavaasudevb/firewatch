"""
inference.py — FireWatch Baseline Inference Script
"""

import json
import os
import re
import textwrap

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

MAX_STEPS_PER_TASK = 25
TEMPERATURE = 0.0
MAX_TOKENS = 512
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

FIX_TOOLS = {
    "restart_service", "rollback_config", "reset_ratelimit",
    "sync_replica", "clear_connections", "scale_service",
}


def clamp_score(raw: float) -> float:
    return round(max(0.01, min(0.99, raw)), 2)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    done_str = "true" if done else "false"
    error_str = error if error is not None else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    success_str = "true" if success else "false"
    safe_score = clamp_score(score)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={safe_score:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Site Reliability Engineer (SRE) responding to a live
production incident. The system is actively degrading — every step costs time.

CRITICAL WORKFLOW — FOLLOW THIS EXACTLY:
  Step 1: get_topology(system)     — FREE, always do this first
  Step 2: get_logs(<alerting svc>) — MANDATORY before any fix
  Step 3: apply the correct fix based on what logs reveal
  Step 4: mark_resolved(system)    — ONLY when health > 0.80

TOOL → FIX MAPPING (from log evidence):
  Log contains "OOMKiller" or "out-of-memory"             → restart_service(service)
  Log contains "connection pool" or "HikariPool"          → clear_connections(service)
  Log contains "config" or "memory leak" or "GC overhead" → rollback_config(service)
  Log contains "rate limit" or "429" or "req/s"           → reset_ratelimit(service)
  Log contains "replica lag" or "stale read"              → sync_replica(service)
  Service is overwhelmed with traffic                     → scale_service(service)

AVAILABLE TOOLS:
  get_metrics(service)       — check health numbers (NOT enough to diagnose)
  get_logs(service)          — READ THIS to know the correct fix
  get_topology(system)       — FREE dependency graph, no step cost
  restart_service(service)   — fixes OOM
  rollback_config(service)   — fixes memory leaks, config issues
  reset_ratelimit(service)   — fixes rate limiting (only AFTER rollback_config)
  sync_replica(service)      — fixes replica lag
  clear_connections(service) — fixes connection pool exhaustion
  scale_service(service)     — adds capacity
  mark_resolved(system)      — ends episode, ONLY when health > 0.80

VALID SERVICES:
  api-gateway, auth-service, payment-service, database, cache,
  notification-service, system

STRICT RULES:
  1. get_topology(system) MUST be your first action — it is FREE
  2. get_logs MUST be called on a degraded service before applying any fix
  3. NEVER guess a fix — only act on log evidence
  4. NEVER call mark_resolved unless health > 0.80 AND you applied at least one fix
  5. NEVER repeat the same action twice
  6. get_metrics alone is NOT sufficient — logs reveal root cause
  7. If FINDINGS show a problem with a fix mapping — APPLY THAT FIX NOW
  8. Multiple services can fail simultaneously — fix each one based on its logs
  9. After fixing one service, check if others still need fixing

Respond with ONLY valid JSON, no other text:
{"reasoning": "<cite specific log evidence and explain your fix choice>", "tool": "tool_name", "target": "service_name"}
""").strip()


def extract_findings(last_action_result: str):
    """Parse log output and return a short actionable finding."""
    if not last_action_result:
        return None

    text = last_action_result.lower()

    if "oomkiller" in text or "out-of-memory" in text or "out of memory" in text:
        return "OOM error → APPLY restart_service"
    if "connection pool" in text or "hikaripool" in text or "connection is not available" in text:
        return "connection pool exhausted → APPLY clear_connections"
    if ("config" in text or "reload" in text) and ("leak" in text or "gc overhead" in text or "unreferenced" in text):
        return "config/memory leak → APPLY rollback_config"
    if "rate limit" in text or "429" in text or ("req/s" in text and "limit" in text):
        return "rate limiting breached → APPLY reset_ratelimit (after rollback_config)"
    if "replication lag" in text or "replica lag" in text or "stale read" in text:
        return "replica lag → APPLY sync_replica"
    if "memory usage" in text and ("78%" in text or "threshold" in text or "monitor" in text):
        return "memory warning only — low severity, may not need fix"
    if "health check passed" in text or "all systems nominal" in text or "restarted successfully" in text:
        return "service is healthy — no action needed"

    return None


def build_observation_message(
    obs_dict,
    step,
    findings,
    fixes_applied,
    last_action_was_log,
    last_log_finding,
    last_log_target,
    is_first_step=False,
):
    """Build the user message for this conversation turn."""
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
        topo_lines = "\n".join(f"  {svc} -> {deps[svc]}" for svc in deps)
        topo_section = f"\nDEPENDENCY GRAPH:\n{topo_lines}"

    summary_section = ""
    if is_first_step and summary:
        summary_section = f"\nINCIDENT: {summary.strip()[:300]}\n"

    findings_section = ""
    if findings:
        lines = [f"  {svc}: {finding}" for svc, finding in findings.items()]
        findings_section = "\nFINDINGS FROM LOGS:\n" + "\n".join(lines)

    fixes_section = ""
    if fixes_applied:
        fixes_section = "\nFIXES APPLIED:\n" + "\n".join(f"  {f}" for f in fixes_applied)

    action_directive = ""
    if last_action_was_log and last_log_finding and "APPLY" in (last_log_finding or ""):
        action_directive = (
            f"\n🚨 IMMEDIATE ACTION REQUIRED 🚨\n"
            f"You just read logs for {last_log_target}.\n"
            f"Finding: {last_log_finding}\n"
            f"DO NOT investigate more services. APPLY THE FIX NOW.\n"
            f"Respond with the fix action for {last_log_target}."
        )
    elif findings and not fixes_applied:
        actionable = {
            svc: f for svc, f in findings.items()
            if "APPLY" in f
            and services.get(svc, {}).get("status", "healthy") != "healthy"
        }
        if actionable:
            svc, finding = next(iter(actionable.items()))
            action_directive = (
                f"\n You have evidence for {svc}: {finding}\n"
                f"Apply the fix before investigating more."
            )
    elif fixes_applied and health > 0.80:
        action_directive = "\n Health > 0.80 and fixes applied. Call mark_resolved(system)."
    elif fixes_applied:
        still_broken = [
            svc for svc, f in findings.items()
            if "APPLY" in f
            and services.get(svc, {}).get("status", "healthy") != "healthy"
            and not any(svc in fix for fix in fixes_applied)
        ]
        if still_broken:
            action_directive = (
                f"\n Still degraded: {', '.join(still_broken)}. "
                f"Get logs and fix them."
            )

    budget_str = str(budget) if budget is not None else "HIDDEN"

    return f"""STEP {step} | Health: {health:.3f} | Budget: {budget_str}
{summary_section}
ALERTS:
{alert_lines}

SERVICES:
{svc_lines}
{topo_section}
{findings_section}
{fixes_section}

LAST RESULT: {last}
{action_directive}

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


def investigation_fallback(obs_dict, action_history, findings):
    """Investigation-only fallback — ZERO fix logic."""
    done_actions = {(a["tool"], a["target"]) for a in action_history}
    services = obs_dict.get("services", {})

    if ("get_topology", "system") not in done_actions:
        return {"tool": "get_topology", "target": "system"}

    degraded_without_logs = sorted(
        [
            (name, data)
            for name, data in services.items()
            if isinstance(data, dict)
            and data.get("status") != "healthy"
            and ("get_logs", name) not in done_actions
        ],
        key=lambda x: x[1].get("health", 1.0),
    )
    if degraded_without_logs:
        return {"tool": "get_logs", "target": degraded_without_logs[0][0]}

    degraded_without_metrics = [
        (name, data)
        for name, data in services.items()
        if isinstance(data, dict)
        and data.get("status") != "healthy"
        and ("get_metrics", name) not in done_actions
    ]
    if degraded_without_metrics:
        worst = min(degraded_without_metrics, key=lambda x: x[1].get("health", 1.0))
        return {"tool": "get_metrics", "target": worst[0]}

    return {"tool": "mark_resolved", "target": "system"}


TASK_NAMES = {
    "task1": "Single Service Failure",
    "task2": "Cascading Failure with Red Herring",
    "task3": "Multi-vector Ordered Incident",
    "task4": "Non-stationary Adaptive Incident",
}


def run_task(env, llm_client, task_id):
    """Run one task episode using multi-turn conversation with persistent findings."""
    from models import FireWatchAction

    task_name = TASK_NAMES.get(task_id, task_id)
    log_start(task=task_name, env="firewatch", model=MODEL_NAME)

    rewards = []
    action_history = []
    steps_taken = 0
    final_score = 0.05

    findings = {}
    fixes_applied = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    last_action_was_log = False
    last_log_finding = None
    last_log_target = None

    try:
        obs = env.reset(task_id=task_id)
        obs_dict = obs.model_dump()

        for step_num in range(1, MAX_STEPS_PER_TASK + 1):
            if obs_dict.get("done", False):
                break

            user_msg = build_observation_message(
                obs_dict=obs_dict,
                step=step_num,
                findings=findings,
                fixes_applied=fixes_applied,
                last_action_was_log=last_action_was_log,
                last_log_finding=last_log_finding,
                last_log_target=last_log_target,
                is_first_step=(step_num == 1),
            )
            messages.append({"role": "user", "content": user_msg})

            action = None
            response_text = ""
            try:
                completion = llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
                action = parse_action(response_text)
            except Exception:
                response_text = ""
                action = None

            if action is None:
                action = investigation_fallback(obs_dict, action_history, findings)
                response_text = json.dumps({
                    "reasoning": "Fallback: LLM did not produce valid JSON.",
                    "tool": action["tool"],
                    "target": action["target"],
                })

            messages.append({"role": "assistant", "content": response_text})

            action_str = f"{action['tool']}({action['target']})"
            step_reward = -0.02
            done = False
            last_action_error = None

            last_action_was_log = False
            last_log_finding = None
            last_log_target = None

            try:
                fw_action = FireWatchAction(
                    tool=action["tool"],
                    target=action["target"],
                    parameters={},
                )
                result = env.step(fw_action)
                obs_dict = result.model_dump()
                step_reward = float(result.reward or -0.02)
                done = result.done
                steps_taken = step_num
                last_action_error = obs_dict.get("last_action_error", None)

                if action["tool"] == "get_logs":
                    last_result = obs_dict.get("last_action_result", "")
                    finding = extract_findings(last_result)
                    if finding:
                        findings[action["target"]] = finding
                        last_action_was_log = True
                        last_log_finding = finding
                        last_log_target = action["target"]

                if action["tool"] in FIX_TOOLS:
                    fixes_applied.append(action_str)

                if done and result.metadata:
                    raw_score = float(result.metadata.get("final_score", 0.05))
                    final_score = clamp_score(raw_score)

            except Exception:
                step_reward = -0.02
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

    final_score = clamp_score(final_score)
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
                    "rewards": [-0.02],
                }
            log_end(
                success=result["success"],
                steps=result["steps_taken"],
                score=result["score"],
                rewards=result["rewards"],
            )


if __name__ == "__main__":
    main()