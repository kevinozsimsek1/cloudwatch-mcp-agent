#!/usr/bin/env python3
"""Regression chat tests for CloudWatch SRE agent. Usage: python scripts/sre_chat_test.py [base_url]"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/chat"

TESTS = [
    {
        "id": "T1_system_500_summary",
        "message": "Sistemde son 1 saatte 500 dönenleri getir — tüm log gruplarında sayıları göster",
        "expect_tools": ["search_logs_across_groups"],
        "expect_args": {"search_filter": "http_500", "response_mode": "summary"},
    },
    {
        "id": "T2_rank_errors",
        "message": "Son 6 saatte hangi log grubunda en çok error var? Top 5 göster",
        "expect_tools": ["search_logs_across_groups"],
        "expect_args": {"response_mode": "summary"},
    },
    {
        "id": "T3_ambiguous_500",
        "message": "500 var mı?",
        "expect_clarification": True,
    },
    {
        "id": "T4_active_alarms",
        "message": "Şu an aktif CloudWatch alarmları neler?",
        "expect_tools": ["get_active_alarms"],
    },
    {
        "id": "T5_alarm_history_24h",
        "message": "Son 24 saatte hangi alarmlar tetiklendi?",
        "expect_tools": ["get_alarm_history"],
    },
    {
        "id": "T6_comparison_today_yesterday",
        "message": "Bugün vs dün API Gateway 500 hatalarını karşılaştır",
        "expect_tools": ["search_logs_across_groups"],
        "expect_min_tool_calls": 2,
    },
    {
        "id": "T7_success_2xx",
        "message": "Son 1 saatte başarılı 2xx istekleri göster",
        "expect_tools": ["search_logs_across_groups"],
        "expect_args": {"search_filter": "http_2xx"},
    },
    {
        "id": "T8_specific_groups_count",
        "message": "Sadece api-gateway ve lambda log gruplarında son 1 saatte 500 sayısını getir, detay istemiyorum",
        "expect_tools": ["search_logs_across_groups"],
        "expect_args": {"response_mode": "count_only"},
    },
    {
        "id": "T9_specific_group_detail",
        "message": "/aws/apigateway/era-api-gateway-dev/access-logs grubunda son 1 saatte 500 olanları detaylı getir",
        "expect_tools": ["query_log_group"],
        "expect_args_any": {"search_filter": "http_500"},
    },
    {
        "id": "T10_log_groups_list",
        "message": "Lambda ile ilgili log gruplarını listele",
        "expect_tools": ["describe_log_groups"],
        "expect_args_keywords": ["lambda"],
    },
    {
        "id": "T11_cross_group_500_count",
        "message": "status kodu 500 olan kaç tane log grubum var",
        "expect_tools": ["search_logs_across_groups"],
        "expect_args": {"search_filter": "http_500", "response_mode": "summary"},
    },
    {
        "id": "T12_single_group_12h_errors",
        "message": "aws/apigateway/era-api-gateway-dev/access-logs son 12 saatde kaç hata almış",
        "expect_tools": ["query_log_group"],
        "expect_args": {
            "search_filter": "errors",
            "response_mode": "summary",
            "hours": 12,
            "log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs",
        },
    },
]


def post_chat(message: str) -> dict:
    body = json.dumps({"message": message}).encode()
    req = urllib.request.Request(
        BASE, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    results = []
    for t in TESTS:
        print(f"\n=== {t['id']} ===")
        started = time.time()
        try:
            data = post_chat(t["message"])
            elapsed = time.time() - started
            tool_calls = data.get("tool_calls") or []
            tool_names = [tc.get("name") for tc in tool_calls]
            response = (data.get("response") or "").lower()
            status = "PASS"
            notes: list[str] = []

            if t.get("expect_clarification"):
                markers = ["?", "tüm sistem", "belirli", "sayı", "detay", "hangi"]
                if tool_calls and not any(m in response for m in markers):
                    status = "FAIL"
                    notes.append("called tools without clarifying vague question")

            if t.get("expect_tools"):
                if not any(n in tool_names for n in t["expect_tools"]):
                    status = "FAIL"
                    notes.append(f"expected {t['expect_tools']}, got {tool_names}")

            if t.get("expect_min_tool_calls") and len(tool_calls) < t["expect_min_tool_calls"]:
                status = "FAIL"
                notes.append(f"need >={t['expect_min_tool_calls']} calls, got {len(tool_calls)}")

            for key, val in t.get("expect_args", {}).items():
                if not any(tc.get("arguments", {}).get(key) == val for tc in tool_calls):
                    status = "FAIL"
                    notes.append(f"missing {key}={val}")

            if t.get("expect_args_keywords"):
                blob = json.dumps(tool_calls).lower()
                if not all(kw in blob for kw in t["expect_args_keywords"]):
                    status = "FAIL"
                    notes.append(f"expected keywords {t['expect_args_keywords']} in args")

            print(f"{status} ({elapsed:.1f}s) tools={tool_names}")
            if notes:
                print(" ", "; ".join(notes))
            results.append(status)
        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append("ERROR")

    passed = sum(1 for s in results if s == "PASS")
    print(f"\nSUMMARY: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
