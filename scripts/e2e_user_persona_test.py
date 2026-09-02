#!/usr/bin/env python3
"""E2E persona tests — technical vs non-technical user flows."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/chat"
TIMEOUT = 120


def post_chat(message: str, history: list | None = None) -> dict:
    body = json.dumps({"message": message, "history": history or []}).encode()
    req = urllib.request.Request(
        BASE,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def check(name: str, message: str, response: str, *, must: list[str] | None = None,
          must_not: list[str] | None = None, tool: str | None = None,
          tools: list | None = None, max_sec: float = 60) -> dict:
    text = response.lower()
    issues: list[str] = []
    for pat in must or []:
        if not re.search(pat, text, re.I):
            issues.append(f"missing: /{pat}/")
    for pat in must_not or []:
        if re.search(pat, text, re.I):
            issues.append(f"forbidden: /{pat}/")
    if tool:
        names = [t.get("name") for t in (tools or [])]
        if tool not in names:
            issues.append(f"expected tool {tool}, got {names}")
    return {
        "name": name,
        "message": message,
        "pass": not issues,
        "issues": issues,
        "preview": response[:220].replace("\n", " "),
    }


NON_TECH = [
    {
        "name": "NT-01 selam",
        "message": "selam",
        "must": [r"merhaba|cloudwatch|log"],
        "must_not": [r"i can't help", r"omlet"],
    },
    {
        "name": "NT-02 naber",
        "message": "naber",
        "must": [r"merhaba|cloudwatch"],
        "must_not": [r"i can't help"],
    },
    {
        "name": "NT-03 ne iş yapıyosun",
        "message": "sen ne iş yapıyosun",
        "must": [r"cloudwatch|log|sre|asistan"],
        "must_not": [r"i can't help"],
    },
    {
        "name": "NT-04 omlet reddi",
        "message": "omlet nasıl yapılır",
        "must": [r"yardımcı olam|cloudwatch|log"],
        "must_not": [r"yumurta|tava|pişir"],
    },
    {
        "name": "NT-05 belirsiz soru",
        "message": "500 var mı",
        "must": [r"netleştir|belirli|tüm sistem|hangi|kaç saat|\?"],
        "must_not": [r"yumurta"],
    },
    {
        "name": "NT-06 basit log listesi",
        "message": "apigateway log gruplarımı listele",
        "must": [r"log grub"],
        "tool": "describe_log_groups",
    },
    {
        "name": "NT-07 hata var mı (doğal dil)",
        "message": "api gateway loglarında son 6 saatte hata var mı",
        "must": [r"saat|hata|error|500|eşleş|bulunamad"],
        "tool": "search_logs_across_groups",
    },
]

TECH = [
    {
        "name": "T-01 tam log group path + 500",
        "message": "/aws/apigateway/era-api-gateway-dev/access-logs son 1 saatte 500 var mı",
        "must": [r"500|http_500|kayıt|eşleş|bulunamad"],
        "tool": "query_log_group",
    },
    {
        "name": "T-02 cross-group 500 özet",
        "message": "Son 6 saatte hangi log grubunda en çok 500 var",
        "must": [r"log group|grup|500|saat"],
        "tool": "search_logs_across_groups",
    },
    {
        "name": "T-03 aktif alarmlar",
        "message": "Şu an aktif CloudWatch alarmları neler",
        "must": [r"alarm"],
        "tool": "get_active_alarms",
    },
    {
        "name": "T-04 alarm geçmişi",
        "message": "Son 24 saatte hangi alarmlar tetiklendi",
        "must": [r"alarm"],
        "tool": "get_alarm_history",
    },
    {
        "name": "T-05 follow-up detay (context)",
        "message": "detaylı göster",
        "history_setup": [
            {"role": "user", "content": "/aws/apigateway/era-api-gateway-dev/access-logs son 6 saatte 500 var mı"},
            {"role": "assistant", "content": "Evet — son 6 saat içinde eşleşme var.", "log_context": {
                "log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs",
                "hours": 6,
                "search_filter": "http_500",
                "match_count": 5,
            }},
        ],
        "must": [r"@timestamp|kayıt|500|detay|cloudwatch"],
        "tool": "query_log_group",
    },
    {
        "name": "T-06 off-topic reddi EN",
        "message": "how to bake a cake",
        "must": [r"can't help|cloudwatch|cannot"],
        "must_not": [r"flour|oven|bake for"],
    },
    {
        "name": "T-08 ranked group selection (1 numarayı)",
        "message": "1 numarayı",
        "history_setup": [
            {"role": "user", "content": "400 lü hataları getir son 4 saatde"},
            {
                "role": "assistant",
                "content": (
                    "Sistem özeti — http_400 (son 4 saat, durum: Complete)\n"
                    "2 log grubunda eşleşme var · Toplam: **739** kayıt\n\n"
                    "| # | Log group | Adet |\n"
                    "|---|---|---|\n"
                    "| 1 | `/aws/apigateway/era-api-gateway-dev/access-logs` | **655** |\n"
                    "| 2 | `/212845026981:API-Gateway-Execution-Logs_rmfixdooo6/dev` | **84** |"
                ),
                "log_context": {
                    "hours": 4,
                    "search_filter": "http_400",
                    "match_count": 739,
                    "ranked_groups": [
                        ["/aws/apigateway/era-api-gateway-dev/access-logs", 655],
                        ["/212845026981:API-Gateway-Execution-Logs_rmfixdooo6/dev", 84],
                    ],
                },
            },
            {"role": "user", "content": "detaylı getir"},
            {
                "role": "assistant",
                "content": (
                    "Önceki aramada **birden fazla** log grubu bulunmuştu. Hangisinin detayını göstereyim?\n\n"
                    "1. `/aws/apigateway/era-api-gateway-dev/access-logs` — **655** eşleşme\n"
                    "2. `/212845026981:API-Gateway-Execution-Logs_rmfixdooo6/dev` — **84** eşleşme"
                ),
                "log_context": {
                    "hours": 4,
                    "search_filter": "http_400",
                    "match_count": 739,
                    "ranked_groups": [
                        ["/aws/apigateway/era-api-gateway-dev/access-logs", 655],
                        ["/212845026981:API-Gateway-Execution-Logs_rmfixdooo6/dev", 84],
                    ],
                },
            },
        ],
        "must": [r"access-logs|655|400|kayıt|http"],
        "must_not": [r"yardımcı olam"],
        "tool": "query_log_group",
    },
    {
        "name": "T-07 lambda filtre listesi",
        "message": "Lambda ile ilgili log gruplarını listele",
        "must": [r"lambda|log grub"],
        "tool": "describe_log_groups",
    },
]

CONVERSATION = [
    {
        "name": "C-01 multi-turn drilldown",
        "turns": [
            ("apigateway log gruplarımı listele", {"must": [r"log grub"], "tool": "describe_log_groups"}),
            ("bu log grupları içerisinde son 6 saatte 500 hata dönen var mı", {"must": [r"500|saat"], "tool": "search_logs_across_groups"}),
        ],
    },
    {
        "name": "C-02 system-wide 500 + delegated follow-up",
        "turns": [
            ("500 hatalarını göster son 1 saatde", {"must": [r"500|saat|log group|grup|eşleş|bulunamad"], "tool": "search_logs_across_groups"}),
            ("sen varmı yokmu göster", {"must": [r"500|saat|log group|grup|eşleş|bulunamad|kayıt"], "tool": "search_logs_across_groups"}),
        ],
    },
    {
        "name": "C-03 time window adjustment",
        "turns": [
            ("500 lü errorleri göster son 1 saatleri göster", {"must": [r"saat|eşleş|bulunamad|log group|grup"], "tool": "search_logs_across_groups"}),
            ("3 saat önceye bak", {"must": [r"3\s*saat|saat|eşleş|bulunamad|log group|grup"], "tool": "search_logs_across_groups"}),
        ],
    },
    {
        "name": "C-05 count challenge + show all + fresh 4xx",
        "turns": [
            ("500 lü hataları göstersene son 1 saatde", {"must": [r"http_500|500|saat"], "tool": "search_logs_across_groups"}),
            ("son 9 saat", {"must": [r"9\s*saat|143|141|142|kayıt"], "tool": "search_logs_across_groups"}),
            ("tamamını göstersene", {"must": [r"kayıt|500|http_500"]}),
            ("143 kayıt dedin", {"must": [r"143|kayıt|15|önizleme|tamam"], "must_not": [r"yardımcı olam"]}),
            ("tamam peki 4xx lü. hataları göster son 2 saatde", {"must": [r"2\s*saat|4xx|400|http_4"], "tool": "search_logs_across_groups"}),
        ],
    },
    {
        "name": "C-06 multi-group detail + number pick",
        "turns": [
            ("400 lü hataları getir son 4 saatde", {"must": [r"400|http_400|739|655"], "tool": "search_logs_across_groups"}),
            ("detaylı getir", {"must": [r"birden fazla|hangisinin|655"], "must_not": [r"yardımcı olam"]}),
            ("1 numarayı", {"must": [r"access-logs|655|400|kayıt"], "must_not": [r"yardımcı olam"], "tool": "query_log_group"}),
        ],
    },
]


def run_case(case: dict) -> dict:
    started = time.time()
    try:
        history = case.get("history_setup")
        data = post_chat(case["message"], history)
        elapsed = time.time() - started
        result = check(
            case["name"],
            case["message"],
            data.get("response") or "",
            must=case.get("must"),
            must_not=case.get("must_not"),
            tool=case.get("tool"),
            tools=data.get("tool_calls"),
        )
        result["elapsed"] = round(elapsed, 1)
        result["tools"] = [t.get("name") for t in (data.get("tool_calls") or [])]
        if elapsed > case.get("max_sec", 60):
            result["pass"] = False
            result["issues"].append(f"slow: {elapsed:.1f}s")
        return result
    except Exception as exc:
        return {
            "name": case["name"],
            "message": case["message"],
            "pass": False,
            "issues": [str(exc)],
            "preview": "",
            "elapsed": round(time.time() - started, 1),
        }


def run_conversation(conv: dict) -> list[dict]:
    results = []
    history: list[dict] = []
    for i, (msg, expect) in enumerate(conv["turns"], 1):
        started = time.time()
        try:
            data = post_chat(msg, history)
            elapsed = time.time() - started
            result = check(
                f"{conv['name']} turn {i}",
                msg,
                data.get("response") or "",
                must=expect.get("must"),
                must_not=expect.get("must_not"),
                tool=expect.get("tool"),
                tools=data.get("tool_calls"),
            )
            result["elapsed"] = elapsed
            result["tools"] = [t.get("name") for t in (data.get("tool_calls") or [])]
            results.append(result)
            history.append({"role": "user", "content": msg})
            history.append({
                "role": "assistant",
                "content": data.get("response") or "",
                "log_context": data.get("log_context"),
            })
        except Exception as exc:
            results.append({
                "name": f"{conv['name']} turn {i}",
                "message": msg,
                "pass": False,
                "issues": [str(exc)],
                "preview": "",
            })
            break
    return results


def main() -> int:
    print(f"E2E persona tests → {BASE}\n")
    all_results: list[dict] = []

    print("=== Teknik olmayan kullanıcı ===")
    for case in NON_TECH:
        r = run_case(case)
        all_results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {status} {r['name']} ({r.get('elapsed', '?')}s) tools={r.get('tools', [])}")
        if r["issues"]:
            print(f"       {'; '.join(r['issues'])}")
        if not r["pass"]:
            print(f"       preview: {r.get('preview', '')[:120]}")

    print("\n=== Teknik kullanıcı ===")
    for case in TECH:
        r = run_case(case)
        all_results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {status} {r['name']} ({r.get('elapsed', '?')}s) tools={r.get('tools', [])}")
        if r["issues"]:
            print(f"       {'; '.join(r['issues'])}")
        if not r["pass"]:
            print(f"       preview: {r.get('preview', '')[:120]}")

    print("\n=== Çok turlu konuşma ===")
    for conv in CONVERSATION:
        for r in run_conversation(conv):
            all_results.append(r)
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  {status} {r['name']} ({r.get('elapsed', '?')}s) tools={r.get('tools', [])}")
            if r["issues"]:
                print(f"       {'; '.join(r['issues'])}")

    passed = sum(1 for r in all_results if r["pass"])
    total = len(all_results)
    print(f"\n{'='*50}")
    print(f"SONUÇ: {passed}/{total} geçti")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
