#!/usr/bin/env python3
"""Performative smoke test — invoke every CloudWatch MCP tool one by one (direct + chat).

Direct mode calls boto3/MCP handlers in-process (no LLM).
Chat mode sends one forced prompt per tool through /chat and verifies the tool was called.

Usage:
  python scripts/e2e_all_tools_performative.py direct
  python scripts/e2e_all_tools_performative.py chat [base_url]
  python scripts/e2e_all_tools_performative.py all [base_url]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import (
    ALLOWED_TOOL_NAMES,
    analyze_log_group,
    analyze_metric,
    cancel_logs_insight_query,
    describe_log_groups,
    execute_log_insights_query,
    get_active_alarms,
    get_alarm_history,
    get_logs_insight_query_results,
    get_metric_data,
    get_metric_metadata,
    get_recommended_metric_alarms,
    iso_utc,
    query_log_group,
    search_logs_across_groups,
    utc_now,
)

DEFAULT_LOG_GROUP = "/aws/apigateway/era-api-gateway-dev/access-logs"
DEFAULT_CHAT = "http://localhost:8080/chat"
TIMEOUT = 180


def _ok(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    return True


def _preview(result: dict[str, Any], limit: int = 120) -> str:
    try:
        text = json.dumps(result, default=str)
    except TypeError:
        text = str(result)
    return text[:limit].replace("\n", " ")


def run_direct_suite() -> list[dict[str, Any]]:
    """Force-invoke each allowed tool sequentially with minimal valid arguments."""
    now = utc_now()
    start = iso_utc(now - timedelta(hours=1))
    end = iso_utc(now)
    ctx: dict[str, Any] = {"log_group": DEFAULT_LOG_GROUP}
    results: list[dict[str, Any]] = []

    steps: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        (
            "describe_log_groups",
            lambda: describe_log_groups(log_group_name_prefix="apigateway", max_items=10),
        ),
        (
            "query_log_group",
            lambda: query_log_group(
                log_group_name=ctx["log_group"],
                hours=1,
                search_filter="errors",
                response_mode="summary",
                max_result_lines=5,
            ),
        ),
        (
            "analyze_log_group",
            lambda: analyze_log_group(
                log_group_name=ctx["log_group"],
                start_time=start,
                end_time=end,
            ),
        ),
        (
            "search_logs_across_groups",
            lambda: search_logs_across_groups(
                search_filter="http_500",
                hours=1,
                response_mode="summary",
                max_result_lines=10,
            ),
        ),
        (
            "execute_log_insights_query",
            lambda: execute_log_insights_query(
                log_group_names=[ctx["log_group"]],
                query_string="fields @timestamp, @message | limit 3",
                start_time=start,
                end_time=end,
                wait_for_completion=True,
            ),
        ),
    ]

    for name, runner in steps:
        started = time.time()
        row: dict[str, Any] = {"tool": name, "mode": "direct"}
        try:
            data = runner()
            if name == "describe_log_groups":
                groups = data.get("log_groups") or []
                if groups:
                    ctx["log_group"] = groups[0].get("logGroupName") or ctx["log_group"]
            if name == "execute_log_insights_query" and data.get("query_id"):
                ctx["query_id"] = data["query_id"]
            row["pass"] = _ok(data)
            row["elapsed"] = round(time.time() - started, 2)
            row["preview"] = _preview(data)
            if not row["pass"]:
                row["error"] = data.get("error") or "unexpected response"
        except Exception as exc:
            row["pass"] = False
            row["elapsed"] = round(time.time() - started, 2)
            row["error"] = str(exc)
        results.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  [{len(results)}/{len(ALLOWED_TOOL_NAMES)}] {status} {name} ({row.get('elapsed', '?')}s)")

    query_id = ctx.get("query_id")
    insight_tail: list[tuple[str, Callable[[], dict[str, Any]]]] = []
    if query_id:
        insight_tail.append(
            (
                "get_logs_insight_query_results",
                lambda: get_logs_insight_query_results(query_id=query_id),
            )
        )
    else:
        results.append(
            {
                "tool": "get_logs_insight_query_results",
                "mode": "direct",
                "pass": False,
                "skipped": True,
                "error": "no query_id from execute_log_insights_query",
            }
        )
        print(f"  [{len(results)}/{len(ALLOWED_TOOL_NAMES)}] SKIP get_logs_insight_query_results (no query_id)")

    for name, runner in insight_tail:
        started = time.time()
        row = {"tool": name, "mode": "direct"}
        try:
            data = runner()
            row["pass"] = _ok(data)
            row["elapsed"] = round(time.time() - started, 2)
            row["preview"] = _preview(data)
        except Exception as exc:
            row["pass"] = False
            row["elapsed"] = round(time.time() - started, 2)
            row["error"] = str(exc)
        results.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  [{len(results)}/{len(ALLOWED_TOOL_NAMES)}] {status} {name} ({row.get('elapsed', '?')}s)")

    cancel_id: str | None = None
    try:
        pending = execute_log_insights_query(
            log_group_names=[ctx["log_group"]],
            query_string="fields @timestamp | limit 1",
            start_time=start,
            end_time=end,
            wait_for_completion=False,
        )
        cancel_id = pending.get("query_id")
    except Exception:
        cancel_id = query_id

    started = time.time()
    row = {"tool": "cancel_logs_insight_query", "mode": "direct"}
    if cancel_id:
        try:
            data = cancel_logs_insight_query(query_id=cancel_id)
            row["pass"] = _ok(data)
            row["preview"] = _preview(data)
        except Exception as exc:
            row["pass"] = False
            row["error"] = str(exc)
    else:
        row["pass"] = False
        row["skipped"] = True
        row["error"] = "no query_id to cancel"
    row["elapsed"] = round(time.time() - started, 2)
    results.append(row)
    status = "PASS" if row["pass"] else ("SKIP" if row.get("skipped") else "FAIL")
    print(f"  [{len(results)}/{len(ALLOWED_TOOL_NAMES)}] {status} cancel_logs_insight_query ({row.get('elapsed', '?')}s)")

    metric_steps: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("get_active_alarms", lambda: get_active_alarms(max_items=10)),
        (
            "get_alarm_history",
            lambda: get_alarm_history(
                start_time=iso_utc(now - timedelta(hours=24)),
                end_time=end,
                max_records=20,
            ),
        ),
        (
            "get_metric_metadata",
            lambda: get_metric_metadata(namespace="AWS/ApiGateway", max_items=10),
        ),
        (
            "get_metric_data",
            lambda: get_metric_data(
                namespace="AWS/ApiGateway",
                metric_name="Count",
                start_time=start,
                end_time=end,
                statistic="Sum",
                period=300,
            ),
        ),
        (
            "analyze_metric",
            lambda: analyze_metric(
                namespace="AWS/ApiGateway",
                metric_name="Count",
                start_time=start,
                end_time=end,
                statistic="Sum",
                period=300,
            ),
        ),
        (
            "get_recommended_metric_alarms",
            lambda: get_recommended_metric_alarms(
                namespace="AWS/ApiGateway",
                metric_name="Count",
                lookback_hours=24,
            ),
        ),
    ]

    for name, runner in metric_steps:
        started = time.time()
        row = {"tool": name, "mode": "direct"}
        try:
            data = runner()
            row["pass"] = _ok(data) or bool(data.get("message"))
            row["elapsed"] = round(time.time() - started, 2)
            row["preview"] = _preview(data)
        except Exception as exc:
            row["pass"] = False
            row["elapsed"] = round(time.time() - started, 2)
            row["error"] = str(exc)
        results.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  [{len(results)}/{len(ALLOWED_TOOL_NAMES)}] {status} {name} ({row.get('elapsed', '?')}s)")

    covered = {row["tool"] for row in results}
    missing = sorted(ALLOWED_TOOL_NAMES - covered)
    if missing:
        print(f"\n  WARNING: not covered in direct suite: {missing}")

    return results


CHAT_TOOL_PROMPTS: list[tuple[str, str]] = [
    ("describe_log_groups", "Lambda ile ilgili log gruplarını listele"),
    (
        "query_log_group",
        "/aws/apigateway/era-api-gateway-dev/access-logs son 1 saatte error var mı",
    ),
    (
        "search_logs_across_groups",
        "Sistemde son 1 saatte 500 dönenleri getir — tüm log gruplarında sayıları göster",
    ),
    ("get_active_alarms", "Şu an aktif CloudWatch alarmları neler"),
    ("get_alarm_history", "Son 24 saatte hangi alarmlar tetiklendi"),
    (
        "get_metric_metadata",
        "AWS/ApiGateway namespace'inde hangi metrikler var",
    ),
    (
        "analyze_metric",
        "AWS/ApiGateway Count metriğini son 1 saatte analiz et",
    ),
]


def post_chat(base: str, message: str, history: list | None = None) -> dict[str, Any]:
    body = json.dumps({"message": message, "history": history or []}).encode()
    req = urllib.request.Request(
        base,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def run_chat_suite(base: str = DEFAULT_CHAT) -> list[dict[str, Any]]:
    """Force each tool through the chat agent with an explicit operational prompt."""
    results: list[dict[str, Any]] = []
    print(f"\n=== Chat performative ({base}) ===")
    for index, (expected_tool, message) in enumerate(CHAT_TOOL_PROMPTS, 1):
        started = time.time()
        row: dict[str, Any] = {
            "tool": expected_tool,
            "mode": "chat",
            "message": message,
        }
        try:
            data = post_chat(base, message)
            tool_names = [t.get("name") for t in (data.get("tool_calls") or [])]
            row["pass"] = expected_tool in tool_names
            row["elapsed"] = round(time.time() - started, 2)
            row["tools_called"] = tool_names
            row["preview"] = (data.get("response") or "")[:160].replace("\n", " ")
            if not row["pass"]:
                row["error"] = f"expected {expected_tool}, got {tool_names}"
        except urllib.error.URLError as exc:
            row["pass"] = False
            row["elapsed"] = round(time.time() - started, 2)
            row["error"] = str(exc.reason or exc)
        except Exception as exc:
            row["pass"] = False
            row["elapsed"] = round(time.time() - started, 2)
            row["error"] = str(exc)
        results.append(row)
        status = "PASS" if row["pass"] else "FAIL"
        print(f"  [{index}/{len(CHAT_TOOL_PROMPTS)}] {status} {expected_tool} ({row.get('elapsed', '?')}s)")
        if row.get("error"):
            print(f"       {row['error']}")
    return results


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "direct").lower()
    base = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CHAT
    all_results: list[dict[str, Any]] = []

    if mode in {"direct", "all"}:
        print(f"=== Direct performative ({len(ALLOWED_TOOL_NAMES)} tools) ===")
        all_results.extend(run_direct_suite())

    if mode in {"chat", "all"}:
        all_results.extend(run_chat_suite(base))

    passed = sum(1 for row in all_results if row.get("pass"))
    total = len(all_results)
    print(f"\n{'=' * 50}")
    print(f"SONUÇ: {passed}/{total} geçti")
    for row in all_results:
        if not row.get("pass"):
            print(f"  FAIL {row.get('mode')}::{row.get('tool')} — {row.get('error', row.get('preview', ''))[:100]}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
