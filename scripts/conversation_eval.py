#!/usr/bin/env python3
"""Multi-turn conversation eval: does the agent resolve references, follow-ups and "why" questions?

Each scenario is a short conversation; the follow-up turn is graded on the tool call it produced
(right group / filter / window), so the score does not depend on wording.

Usage:
  python scripts/conversation_eval.py [--chat-url http://localhost:8080/chat] [--runs 1]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from typing import Any, Callable

ACCESS_LOG = "/aws/apigateway/era-api-gateway-dev/access-logs"

Check = Callable[[dict[str, Any], dict[str, Any]], tuple[bool, str]]


def _calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    return data.get("tool_calls") or []


def _ranked(prev: dict[str, Any]) -> list[str]:
    return [g for g, _ in (prev.get("log_context") or {}).get("ranked_groups") or []]


def _find(data: dict[str, Any], name: str | tuple[str, ...], **want: Callable[[Any], bool]) -> dict[str, Any] | None:
    names = (name,) if isinstance(name, str) else name
    for call in _calls(data):
        if call.get("name") not in names:
            continue
        args = call.get("arguments") or {}
        if all(test(args.get(key)) for key, test in want.items()):
            return call
    return None


def _summary(data: dict[str, Any]) -> str:
    return json.dumps(
        [(c.get("name"), {k: v for k, v in (c.get("arguments") or {}).items() if k != "query_string"}) for c in _calls(data)],
        ensure_ascii=False,
    )[:300]


def expect_call(name, **want) -> Check:
    def check(data, prev):
        resolved = {k: (v(prev) if getattr(v, "needs_prev", False) else v) for k, v in want.items()}
        return _find(data, name, **resolved) is not None, _summary(data)
    return check


def expect_no_tools() -> Check:
    return lambda data, prev: (not _calls(data), _summary(data) or "no tools")


def expect_text(pattern: str) -> Check:
    def check(data, prev):
        text = f"{data.get('insight') or ''}\n{data.get('response') or ''}"
        return bool(re.search(pattern, text, re.I)), (data.get("insight") or data.get("response") or "")[:160].replace("\n", " ")
    return check


def ranked_group(index: int):
    """Arg predicate built from the previous turn's ranked groups."""
    def build(prev):
        groups = _ranked(prev)
        target = groups[index] if len(groups) > index else "<missing>"
        return lambda value: isinstance(value, str) and value.endswith(target.split(":", 1)[-1])
    build.needs_prev = True
    return build


eq = lambda expected: (lambda value: value == expected)  # noqa: E731
one_of = lambda *expected: (lambda value: value in expected)  # noqa: E731
contains = lambda part: (lambda value: isinstance(value, str) and part.lower() in value.lower())  # noqa: E731
yesterday_start = lambda value: isinstance(value, str) and value.endswith("T21:00:00Z")  # noqa: E731
deep = one_of("analysis", "detail")

SCENARIOS: list[tuple[str, list[tuple[str, Check | None]]]] = [
    ("2. grup referansı", [
        ("Son 6 saatte hangi log grubunda en çok error var? Top 5 göster", None),
        ("2. grubun detayını göster", expect_call("query_log_group", log_group_name=ranked_group(1))),
    ]),
    ("neden sorusu", [
        ("Son 6 saatte hangi log grubunda en çok error var?", None),
        ("neden bu kadar çok hata var orada?", expect_call(
            ("query_log_group", "search_logs_across_groups"), response_mode=deep)),
    ]),
    ("aynısı dün için", [
        ("Son 6 saatte hangi log grubunda en çok error var?", None),
        ("peki dün ne durumdaydı?", expect_call(
            "search_logs_across_groups", search_filter=eq("errors"), start_time=yesterday_start)),
    ]),
    ("metrikten endpoint'e", [
        ("API Gateway 5xx metriği son 24 saatte nasıl?", None),
        ("bunlar hangi endpoint'lerden geliyor?", expect_call(
            ("query_log_group", "search_logs_across_groups"), search_filter=one_of("http_5xx", "http_500", "http_502", "http_504"))),
    ]),
    ("bunların kaçı 403", [
        (f"{ACCESS_LOG} son 12 saatte kaç hata almış", None),
        ("bunların kaçı 403?", expect_call("query_log_group", search_filter=eq("http_403"), hours=eq(12))),
    ]),
    ("peki 4xx", [
        ("son 1 saatte sistemde 500 var mı", None),
        ("peki 4xx?", expect_call("search_logs_across_groups", search_filter=one_of("http_4xx", "http_400"))),
    ]),
    ("alarm takibi", [
        ("aktif alarm var mı", None),
        ("son 24 saatte tetiklenenler hangileri?", expect_call("get_alarm_history")),
    ]),
    ("lambda → EraAuthorizer", [
        ("Lambda fonksiyonlarında son 24 saatte hata var mı?", None),
        ("EraAuthorizer'daki hatalar ne diyor?", expect_call("query_log_group", log_group_name=contains("EraAuthorizer"))),
    ]),
    ("servis → hata türü", [
        ("plt-comm-catalog loglarında son 6 saatte hata var mı", None),
        ("ne tür hatalar bunlar?", expect_call("query_log_group", log_group_name=contains("plt-comm-catalog"), response_mode=deep)),
    ]),
    ("english follow-up", [
        ("Which log group had the most errors in the last 6 hours?", None),
        ("show me the details of the first one", expect_call("query_log_group", log_group_name=ranked_group(0))),
    ]),
    ("504 path yoğunluğu", [
        (f"{ACCESS_LOG} son 24 saatte 504 var mı", None),
        ("hangi path'lerde yoğunlaşıyor?", expect_call(
            "query_log_group", search_filter=eq("http_504"), hours=eq(24), response_mode=deep)),
    ]),
    ("ikinci servisin son 1 saati", [
        ("Son 6 saatte en çok hata veren servis hangisi?", None),
        ("ikinci servisin son 1 saatine bak", expect_call("query_log_group", log_group_name=ranked_group(1), hours=eq(1))),
    ]),
    ("debezium hatası ne demek", [
        ("debezium loglarında son 6 saatte hata var mı", None),
        ("bu hata ne anlama geliyor, nasıl çözülür?", expect_text(r"binlog|kafka|connector|mysql|timeout|offset")),
    ]),
    ("kapsam dışı", [("bana pilav tarifi ver", expect_no_tools())]),
    ("belirsiz soru", [("500 var mı?", expect_no_tools())]),
]


def post_chat(url: str, message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps({"message": message, "history": history}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode())


def history_entry(data: dict[str, Any]) -> dict[str, Any]:
    content = (f"{data['insight']}\n\n" if data.get("insight") else "") + (data.get("response") or "")
    entry: dict[str, Any] = {"role": "assistant", "content": content[:2500]}
    if data.get("log_context"):
        entry["log_context"] = data["log_context"]
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-url", default="http://localhost:8080/chat")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    passed = total = insights = 0
    for name, turns in SCENARIOS:
        for run in range(args.runs):
            history: list[dict[str, Any]] = []
            prev: dict[str, Any] = {}
            print(f"\n=== {name} (run {run + 1})")
            for message, check in turns:
                started = time.time()
                try:
                    data = post_chat(args.chat_url, message, history)
                except Exception as exc:  # noqa: BLE001
                    data = {"response": f"ERROR {exc}", "tool_calls": []}
                elapsed = time.time() - started
                if check:
                    ok, detail = check(data, prev)
                    total += 1
                    passed += ok
                    insights += bool(data.get("insight"))
                    print(f"  {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s) {message!r}\n       {detail}")
                    if data.get("insight"):
                        print(f"       yorum: {data['insight'][:200]!r}")
                else:
                    print(f"  ..   ({elapsed:.1f}s) {message!r} → {_summary(data)}")
                history += [{"role": "user", "content": message}, history_entry(data)]
                prev = data
    print(f"\nRESULT: {passed}/{total} follow-up turns correct · {insights} with model interpretation")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
