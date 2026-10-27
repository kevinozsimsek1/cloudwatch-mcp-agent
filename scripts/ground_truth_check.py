#!/usr/bin/env python3
"""Ground-truth check: compare /chat answers with direct CloudWatch queries.

Each question is asked N times (consistency) and the numbers in the answer are compared with an
independent boto3 query (correctness). Needs CloudWatch read access (IRSA in-cluster, or AWS_PROFILE).

Usage:
  python scripts/ground_truth_check.py [--chat-url http://localhost:8080/chat] [--runs 3]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import boto3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AWS_REGION, SYSTEM_SCAN_EXCLUDE_PREFIXES  # noqa: E402
from app.queries import _FILTER_CLAUSES  # noqa: E402
from app.sre_context import calendar_window  # noqa: E402

ACCESS_LOG = "/aws/apigateway/era-api-gateway-dev/access-logs"

logs = boto3.client("logs", region_name=AWS_REGION)
cloudwatch = boto3.client("cloudwatch", region_name=AWS_REGION)


# ---------------------------------------------------------------------------
# Independent truth queries
# ---------------------------------------------------------------------------


def _insights(groups: list[str], query: str, start: float, end: float) -> list[dict[str, str]]:
    query_id = logs.start_query(
        logGroupNames=groups, startTime=int(start), endTime=int(end), queryString=query
    )["queryId"]
    while True:
        response = logs.get_query_results(queryId=query_id)
        if response["status"] in {"Complete", "Failed", "Cancelled", "Timeout"}:
            return [{f["field"]: f["value"] for f in row} for row in response["results"]]
        time.sleep(1)


def _all_runtime_groups() -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {}
    for page in logs.get_paginator("describe_log_groups").paginate():
        for group in page["logGroups"]:
            name = group["logGroupName"]
            if any(name.startswith(p) for p in SYSTEM_SCAN_EXCLUDE_PREFIXES):
                continue
            by_class.setdefault(group.get("logGroupClass", "STANDARD"), []).append(name)
    return by_class


def truth_ranking(search_filter: str, hours: int) -> dict[str, Any]:
    now = time.time()
    query = f"fields @log\n{_FILTER_CLAUSES[search_filter]}\n| stats count() as n by @log"
    counts: dict[str, int] = {}
    for names in _all_runtime_groups().values():
        for i in range(0, len(names), 50):
            for row in _insights(names[i : i + 50], query, now - hours * 3600, now):
                counts[row["@log"].split(":", 1)[-1]] = int(row["n"])
    ranked = sorted(counts.items(), key=lambda item: -item[1])
    return {"total": sum(counts.values()), "top": ranked[0][0] if ranked else None}


def truth_group_count(group: str, search_filter: str, start: float, end: float) -> int:
    rows = _insights([group], f"fields @message\n{_FILTER_CLAUSES[search_filter]}\n| stats count() as n", start, end)
    return int(rows[0]["n"]) if rows else 0


def truth_metric_sum(namespace: str, schema: str, metric: str, hours: int) -> float:
    end = datetime.now(timezone.utc)
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {"Id": "m", "Expression": f'SELECT SUM("{metric}") FROM SCHEMA({schema})', "Period": 3600}
        ],
        StartTime=end - timedelta(hours=hours),
        EndTime=end,
    )
    return sum(response["MetricDataResults"][0]["Values"])


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------


def _int(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1).replace(".", "")) if match else None


def answer_total(text: str) -> int | None:
    for pattern in (r"Toplam: (\d+)\*\*", r"\*\*(\d+)\*\* adet", r"\*\*(\d+)\*\* kayıt"):
        value = _int(pattern, text)
        if value is not None:
            return value
    if "eşleşme bulunamadı" in text or "eşleşme bulamadım" in text:
        return 0
    return None


def answer_top_group(text: str) -> str | None:
    match = re.search(r"\|\s*1\s*\|\s*`([^`]+)`", text)
    return match.group(1).split(":", 1)[-1] if match else None


def close(actual: int | float | None, expected: int | float, tolerance: float = 0.03) -> bool:
    """Truth and agent run seconds apart on live logs; allow a small drift."""
    if actual is None:
        return False
    return abs(actual - expected) <= max(3, expected * tolerance)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

Check = Callable[[str, list[dict[str, Any]]], tuple[bool, str]]


def case_top_errors() -> Check:
    truth = truth_ranking("errors", 6)

    def check(text: str, _tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total, top = answer_total(text), answer_top_group(text)
        ok = top == truth["top"] and close(total, truth["total"], 0.05)
        return ok, f"agent top={top} total={total} | truth top={truth['top']} total={truth['total']}"

    return check


def case_group_count(search_filter: str, hours: int) -> Check:
    now = time.time()
    expected = truth_group_count(ACCESS_LOG, search_filter, now - hours * 3600, now)

    def check(text: str, _tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total = answer_total(text)
        return close(total, expected), f"agent={total} truth={expected}"

    return check


def case_yesterday_500() -> Check:
    window = calendar_window("yesterday")
    expected = truth_group_count(ACCESS_LOG, "http_500", window.start.timestamp(), window.end.timestamp())

    def check(text: str, _tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total = answer_total(text)
        return total == expected, f"agent={total} truth(TR calendar day)={expected}"

    return check


def case_system_total(search_filter: str, hours: int) -> Check:
    truth = truth_ranking(search_filter, hours)

    def check(text: str, _tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total = answer_total(text)
        return close(total, truth["total"]), f"agent={total} truth={truth['total']}"

    return check


def case_named_service(group: str, hours: int) -> Check:
    now = time.time()
    expected = truth_group_count(group, "errors", now - hours * 3600, now)

    def check(text: str, tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total = answer_total(text)
        top = answer_top_group(text)
        scoped = top in (None, group)
        return close(total, expected) and scoped, f"agent={total} top={top} truth={expected}"

    return check


def case_minutes(minutes: int) -> Check:
    def check(text: str, tools: list[dict[str, Any]]) -> tuple[bool, str]:
        used = [t.get("arguments", {}).get("minutes") for t in tools]
        return minutes in used and f"son {minutes} dakika" in text, f"tool minutes={used}"

    return check


def case_metric_sum(namespace: str, schema: str, metric: str, hours: int) -> Check:
    expected = truth_metric_sum(namespace, schema, metric, hours)

    def check(text: str, tools: list[dict[str, Any]]) -> tuple[bool, str]:
        total = _int(r"Toplam: ([\d.]+)\*\*", text)
        return close(total, expected, 0.05), f"agent={total} truth={expected:.0f} tools={[t['name'] for t in tools]}"

    return check


CASES: list[tuple[str, str, Callable[[], Check]]] = [
    ("top_errors_6h", "Son 6 saatte hangi log grubunda en çok error var? Top 5 göster", case_top_errors),
    ("single_500_24h", f"{ACCESS_LOG} son 24 saatte kaç tane 500 var", lambda: case_group_count("http_500", 24)),
    ("single_2xx_detail_24h", f"{ACCESS_LOG} son 24 saatte 2xx istekleri göster", lambda: case_group_count("http_2xx", 24)),
    ("single_errors_12h", f"{ACCESS_LOG} son 12 saatte kaç hata almış", lambda: case_group_count("errors", 12)),
    ("system_5xx_24h", "Son 24 saatte sistemde 5xx hataları hangi log gruplarında? sayılarıyla", lambda: case_system_total("http_5xx", 24)),
    ("yesterday_500", "Dün api gateway access loglarında kaç tane 500 hatası oldu?", case_yesterday_500),
    ("named_service", "plt-comm-catalog servisinde son 6 saatte error var mı?",
     lambda: case_named_service("/aws/containerinsights/era-eks-dev/plt-comm-catalog", 6)),
    ("minutes_window", "Son 30 dakikada tüm sistemde 500 hatası var mı?", lambda: case_minutes(30)),
    ("english_500", f"How many 500 errors did {ACCESS_LOG} have in the last 24 hours?", lambda: case_group_count("http_500", 24)),
    ("metric_apigw_5xx", "API Gateway 5xx metriği son 24 saatte nasıl?",
     lambda: case_metric_sum("AWS/ApiGateway", '"AWS/ApiGateway", ApiName', "5XXError", 24)),
]


def post_chat(url: str, message: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"message": message, "history": []}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-url", default="http://localhost:8080/chat")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--only", nargs="*", help="Case ids to run")
    args = parser.parse_args()

    summary: list[tuple[str, int, int, bool]] = []
    for case_id, question, make_check in CASES:
        if args.only and case_id not in args.only:
            continue
        print(f"\n=== {case_id}: {question}")
        check = make_check()
        passed, answers = 0, set()
        for run in range(args.runs):
            started = time.time()
            data = post_chat(args.chat_url, question)
            text = data.get("response") or ""
            ok, detail = check(text, data.get("tool_calls") or [])
            passed += ok
            answers.add(answer_total(text))
            print(f"  run {run + 1}: {'PASS' if ok else 'FAIL'} ({time.time() - started:.1f}s) {detail}")
        # Rolling windows slide between runs; treat small live-data drift as consistent.
        numeric = [a for a in answers if isinstance(a, int)]
        consistent = len({a is None for a in answers}) == 1 and (
            not numeric or close(min(numeric), max(numeric))
        )
        print(f"  consistency: {'OK' if consistent else 'DIFF'} totals={sorted(answers, key=str)}")
        summary.append((case_id, passed, args.runs, consistent))

    print("\n=== SUMMARY")
    for case_id, passed, runs, consistent in summary:
        print(f"  {case_id:24s} correct {passed}/{runs}  consistent={'yes' if consistent else 'no'}")
    all_ok = all(p == r and c for _, p, r, c in summary)
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
