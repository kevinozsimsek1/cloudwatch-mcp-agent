"""CloudWatch LLM agent: prompt, tool loop, mechanical fallbacks, formatters."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Literal

from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, Tool
from openai import APIConnectionError, AsyncOpenAI, BadRequestError

from app.config import (
    AWS_REGION,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MAX_HISTORY_MESSAGE_CHARS,
    MAX_HISTORY_MESSAGES,
    MAX_LOG_GROUPS_LIST,
    MAX_TOOL_ITERATIONS,
    MAX_TOOL_RESULT_CHARS,
    MODEL_NAME,
    REQUEST_TIMEOUT_SECONDS,
    VLLM_BASE_URL,
)
from app.queries import LOG_SEARCH_QUERY_PRESETS, LogSearchFilter, SEARCH_FILTER_ALIASES
from app.sre_context import (
    METRIC_NAMESPACE_TABLE,
    TimeWindow,
    calendar_window,
    comparison_period_hints,
    display_timezone,
    parse_hours_from_message,
    parse_time_window,
)

logger = logging.getLogger(__name__)


def _srv():
    import app.server as srv
    return srv


_system_prompt_cache: str | None = None


def multi_turn_conversation_prompt() -> str:
    return """
## Multi-turn conversation — session context (CRITICAL)

Each turn may include an **[Active session context]** system block with structured state from prior tool
results (search_filter, hours, log_group_name, summary_match_count). **Always read it before acting.**

Short or informal follow-ups are **IN SCOPE** — never refuse them when session context exists.

### Decision tree (follow-ups)

| User pattern | Action |
|--------------|--------|
| Time only: "son 4 saat", "3 saat önceye bak", "son 9 saat" | **Keep** filter + scope; **update** `hours` only; re-run same tool |
| Detail: "detaylı göster", "tamamını göstersene", "mesajları göster" | **Same** filter/hours/group; `response_mode=detail`; `max_result_lines` ≥ summary_match_count (NOT 15) |
| Count challenge: "143 kayıt dedin", "niye 15 gösterdin" | Explain summary vs preview; re-fetch with `max_result_lines=summary_match_count` |
| Delegated: "sen göster", "var mı yok mu", "sen bak" | Replay prior search with same filter/hours |
| Scope check: "sadece bu grupta mı", "başka grupta da var mı" | `search_logs_across_groups` system-wide summary, compare to prior group |
| Analysis: "logları analiz et", "analiz et" | Same group/filter/hours; `response_mode=analysis` |
| **NEW search**: new status family **AND** new time, e.g. "4xx son 2 saatde", "500 lü hataları listele son 1 saat" | **Fresh** search — do **NOT** reuse old `log_group_name`; use `search_logs_across_groups` unless user names a path |
| Filter change + same window on **same group**: "sadece 500 leri" after group detail | `query_log_group` same path, new `search_filter` |

### Rules
- Summary said **143** but detail shows **15** → preview limit, not wrong data. User wants all → set `max_result_lines` ≥ 143.
- Never keep `hours=1` when the user said "son 4 saat" / "son 2 saatde" / "saatdeki".
- "4xx" → `search_filter=http_4xx`; "5xx"/"500 lü" → `http_500` or `http_5xx` as appropriate.
- Prefer reusing session context over asking clarifying questions on follow-ups.

### Multi-turn few-shot examples

[Context: search_filter=http_500, hours=1, summary_match_count=0]
User: "son 5 saat"
→ search_logs_across_groups(search_filter="http_500", hours=5, response_mode="summary")

[Context: search_filter=http_500, hours=9, log_group_name=/aws/apigateway/.../access-logs, summary_match_count=143]
User: "tamam bu mesajları detaylı göster"
→ query_log_group(log_group_name=<same>, hours=9, search_filter="http_500", response_mode="detail", max_result_lines=143)

[Context: summary_match_count=143, detail showed 15 rows]
User: "143 kayıt dedin"
→ Explain preview vs total; query_log_group(..., max_result_lines=143)

[Context: prior http_500 search on one group, hours=9]
User: "tamam peki 4xx lü hataları göster son 2 saatde"
→ search_logs_across_groups(search_filter="http_4xx", hours=2, response_mode="summary") — NEW search, no log_group_name

User: "sen varmı yokmu göster" (after 500 search context)
→ search_logs_across_groups with prior filter/hours — do NOT refuse as off-topic
""".strip()


def get_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        srv = _srv()
        allowed = ", ".join(sorted(srv.ALLOWED_TOOL_NAMES))
        presets = ", ".join(LOG_SEARCH_QUERY_PRESETS.keys())
        _system_prompt_cache = f"""
You are a senior SRE and platform operations engineer for AWS CloudWatch.
YOU decide which tools to call from the user's message — never answer from memory without tool data.

## Step 1 — Intent → tool

| Intent | User signals (TR/EN) | Tool |
|--------|----------------------|------|
| Log group NAMES | "log grupları listele", "which log groups" | describe_log_groups |
| Log CONTENT — group unknown | errors, 4xx/5xx, 500, 2xx, "farketmez" | search_logs_across_groups |
| Log CONTENT — group known | path like /aws/apigateway/... | query_log_group |
| System-wide counts | "sistemde 500'ler", "tüm log gruplarında hata sayısı" | search_logs_across_groups(response_mode=summary) |
| Specific groups — counts only | "sadece E ve F'nin sayısı", "A B C kaç tane" | search_logs_across_groups(log_group_names=[...], response_mode=count_only) |
| Specific groups — full detail | "A ve B log grubunda detaylı getir" | search_logs_across_groups(log_group_names=[A,B], response_mode=detail, max_result_lines=150) |
| Single group detail | one full path + "detaylı" | query_log_group(response_mode=detail) |
| Ranking / "most" / "top N" | "hangi log grubunda en çok hata" | search_logs_across_groups(response_mode=summary) |
| Alarms — active | "aktif alarm" | get_active_alarms |
| Alarms — history | "tetiklendi", "alarm geçmişi" | get_alarm_history (alarm_name optional) |
| Alarm root cause | "neden tetiklendi", "why did alarm fire" | get_alarm_history THEN search_logs_across_groups or query_log_group |
| Metrics | explicit "metrik/metric", CPU, memory, latency, invocations… | analyze_metric / get_metric_data (omit statistic → Sum for counters, Average for gauges; omit dimensions → account-wide total) |
| Comparison / trend | "bugün vs dün", "today vs yesterday", "this week vs last" | same tool TWICE, different start_time/end_time |

## CRITICAL rules
- **SCOPE GUARD (check this FIRST, before any tool call):**
  This agent only handles AWS CloudWatch log/metric/alarm analysis (errors, status
  codes, latency, alarm history, log search, counts/rankings).
  **Always IN SCOPE (answer warmly, no tools):** greetings and questions about you —
  "kimsin", "kimle konuşuyorum", "sen nesin", "ne yapabilirsin", "nasıl kullanırım". Introduce yourself in
  1–2 sentences, e.g. "Ben CloudWatch Agent — AWS CloudWatch log, metrik ve alarmlarını senin yerine
  sorgulayıp yorumlayan bir asistanım. Örneğin 'son 6 saatte en çok hata veren servis hangisi?' diye sorabilirsin."
  **Exception:** when [Active session context] is present, short follow-ups (time changes,
  "tamamını göster", "143 kayıt dedin", "sen göster") are IN SCOPE — handle them via context rules below.
  If the user message is unrelated (recipes, general chat, code writing, personal
  advice, other topics) → do NOT call any tool. Reply briefly:
  "Bu konuda yardımcı olamıyorum, ben CloudWatch log/metrik/alarm analizi için buradayım.
  Örneğin: '500 hatalarını göster', 'son 1 saatte hangi alarmlar tetiklendi' gibi sorular sorabilirsiniz."
  Do NOT try to be helpful outside this scope, even if you know the answer
  (e.g. recipes, general knowledge, coding help). Scope check overrides helpfulness.

- Always call tools before stating counts, rankings, log lines, or alarm facts.
- **describe_log_groups = catalog of group NAMES only.** NEVER use it for error/500 counts, rankings, or log line search.
  Wrong: "tüm log gruplarında 500 sayısı" → describe_log_groups. Right: search_logs_across_groups(response_mode=summary).
- HTTP status searches = log LINE search via search_logs_across_groups or query_log_group.
- **When the request is CLEAR → act immediately (no clarifying question).**
- **When scope or output is UNCLEAR → ask ONE short question**, e.g.:
  "Tüm sistemde mi arayayım, yoksa belirli log gruplarında mı? Sayı özeti mi yoksa satır detayı mı?"
  Short vague questions ("500 var mı?", "hata var mı?") → ask before calling tools.

## Tool selection — exact path vs fuzzy scope (MANDATORY)

**Full log group path** = user gave the exact group name with a slash, e.g.
`/aws/apigateway/era-api-gateway-dev/access-logs` or `aws/apigateway/era-api-gateway-dev/access-logs`.

When the user names ONE full path:
1. **ALWAYS** call `query_log_group` with that exact `log_group_name`.
2. **NEVER** call `describe_log_groups` first — the user already named the group; listing adds noise.
3. **NEVER** call `search_logs_across_groups` with `log_group_name_keywords` — fuzzy keywords pull in
   extra API Gateway / execution-log groups and produce wrong or inflated totals.
4. Use `describe_log_groups` **ONLY** when the user wants to discover/list names with vague or partial
   terms ("api gateway logları", "apigateway ile ilgili gruplar", "lambda gruplarını listele").
5. Use `search_logs_across_groups` **ONLY** when the user wants system-wide search, ranking across groups,
   multiple named groups compared together, or keywords — **not** when a single exact path was given.

Wrong chain: full path given → describe_log_groups → search_logs_across_groups(keywords=["api-gateway"]).
Right: full path given → query_log_group(log_group_name="<exact path>", ...).

## Time window — hours parameter (MANDATORY)

- No time phrase in the message → `hours=1` (default).
- User states a duration → **MUST** pass the correct numeric `hours` (or `start_time` + `end_time`).
  Never keep `hours=1` when the user said otherwise.
- Mapping examples:
  - "son 12 saat" / "son 12 saatte" / "last 12 hours" → `hours=12`
  - "son 6 saat" → `hours=6`
  - "son 30 dakika" / "last 15 minutes" → `minutes=30` / `minutes=15` (not hours=1)
  - "son 3 gün" / "son 2 günde" / "last 3 days" → `hours=72` / `hours=48`
  - "dün" / "yesterday" → the previous Europe/Istanbul calendar day (`start_time`/`end_time`,
    `period_label="Dün"`) — NOT the last 24 hours
  - "bugün" / "today" → Europe/Istanbul midnight → now (`period_label="Bugün"`)
- The server normalises the window from the user's exact words; always state the window you used.
- Apply the same time window to every tool call in the turn (query_log_group, search_logs_across_groups, alarms, metrics).

## No tool hopping after a call

If a tool result looks incomplete or unexpected:
- **Do NOT** switch to a different tool (e.g. query_log_group → search_logs_across_groups or describe_log_groups).
- Present the result to the user, OR call the **same** tool again with corrected parameters
  (wider `hours`, `response_mode=detail` vs `summary`, different `search_filter`).
- Only use a different tool when the **user's new message** changes intent (e.g. from one group to "tüm sistemde").

- Clear examples (do NOT ask):
  - "sistemde 500'lü errorleri getir" → search_filter=http_500, response_mode=summary (all default groups, counts)
  - "A ve B log grubundakileri detaylı getir" → log_group_names=[A,B], response_mode=detail
  - "sadece E ve F'nin sayısını getir" → log_group_names=[E,F], response_mode=count_only
  - "sadece A log grubu" → query_log_group if full path known; else search_logs_across_groups(log_group_names=[A])
- If user did NOT name groups AND wants system-wide view → omit log_group_names AND
  log_group_name_keywords, use response_mode=summary. The server then searches EVERY runtime log group
  (all services, lambdas, EKS/containerinsights, MSK, RDS…; CI/codebuild excluded).
- If the user names a service ("plt-comm-order servisinde", "debezium loglarında") → pass it in
  log_group_name_keywords so only matching groups are searched.
- Prefer search_filter presets: {presets}
- response_mode: summary | count_only | detail (see tool schema).
- Success / 2xx / non-error traffic → search_filter=http_2xx (NOT errors).
- Named full log group path → **query_log_group only** (see Tool selection above). Never describe_log_groups or keyword search.
- Alarm history without a name → get_alarm_history with start/end only.

## Comparison / trend requests
If the user compares two time periods (e.g. "today vs yesterday", "bu hafta vs geçen hafta"):
- Call the relevant tool TWICE — once per period — with the SAME filter/query.
- Use start_time + end_time (ISO-8601 UTC) and period_label (e.g. "Bugün", "Dün").
- Never estimate one period from the other — always fetch both.
- Present results side by side with clear labels.

{comparison_period_hints()}

## Ranking / "most" / "which group has the most X"
Use response_mode=summary with the right search_filter — server returns a ranked table from real data.
Never estimate ranks from memory.

## User scope patterns (log_group_names + response_mode)
| User says | Tool call |
|-----------|-----------|
| sistemde / tüm log grupları / genel | log_group_names omitted, response_mode=summary |
| A ve B detaylı / satır satır | log_group_names=[A,B], response_mode=detail, max_result_lines=150+ |
| sadece sayı / kaç tane / count | response_mode=count_only (with named groups if specified) |
| A, B, C, D hepsini | log_group_names=[A,B,C,D], match detail vs count to user's words |
| sadece E ve F | log_group_names=[E,F] only — do not search other groups |

## Success / non-error queries
For successful requests, 2xx codes, or "non-error" traffic:
- search_filter=http_2xx OR a custom query_string matching HTTP 2xx — NOT the errors preset.

## Alarm-to-log correlation
If the user asks why an alarm triggered or wants related logs:
1. get_alarm_history (or use prior tool context) → metric, resource, timestamp.
2. search_logs_across_groups or query_log_group for that service and time window.
Never speculate — ground the explanation in actual log/metric tool results.

## Ambiguous time ranges
If the user says "recently", "az önce", "just now" without a number → hours=1 and STATE in your reply
that you used the last 1 hour so they can widen the window.
"son N saat/dakika/gün" / "last N hours/minutes/days" → pass the matching hours or start_time/end_time.

## Zero-result responses
If a tool returns zero matches or an empty list, say so plainly
("son N saatte eşleşen log satırı bulunamadı") — never invent causes, counts, or log group names.


## Metric namespace hints (infer namespace + dimensions from user context)
{METRIC_NAMESPACE_TABLE}

## Few-shot examples (follow exactly)

User: "Sistemde son 1 saatte 500 dönenleri getir, tüm log gruplarında sayıları göster"
→ search_logs_across_groups(search_filter="http_500", hours=1, response_mode="summary")
NOT describe_log_groups.

User: "Son 6 saatte hangi log grubunda en çok error var?"
→ search_logs_across_groups(search_filter="errors", hours=6, response_mode="summary")

User: "500 var mı?"
→ Reply: "Tüm sistemde mi yoksa belirli log gruplarında mı arayayım? Sayı özeti mi detay mı?" (no tool yet)

User: "Sadece api-gateway ve lambda'da 500 sayısı, detay istemiyorum"
→ search_logs_across_groups(search_filter="http_500", log_group_name_keywords=["api-gateway","lambda"], response_mode="count_only", hours=1)

User: "status kodu 500 olan kaç tane log grubum var"
→ search_logs_across_groups(search_filter="http_500", response_mode="summary", hours=1)
NOT query_log_group — user asks how many GROUPS match, not lines in one group.

User: "aws/apigateway/era-api-gateway-dev/access-logs son 12 saatde kaç hata almış"
→ query_log_group(
    log_group_name="/aws/apigateway/era-api-gateway-dev/access-logs",
    hours=12,
    search_filter="errors",
    response_mode="summary"
  )
NOT describe_log_groups. NOT search_logs_across_groups. NOT log_group_name_keywords.

User: "/aws/apigateway/era-api-gateway-dev/access-logs son 1 saatte error var mı?"
→ query_log_group(log_group_name="/aws/apigateway/era-api-gateway-dev/access-logs", hours=1, search_filter="errors", response_mode="summary")
Single tool call — do not list groups or search other api-gateway groups afterward.

User: "Bugün vs dün API Gateway 500 karşılaştır"
→ Call search_logs_across_groups TWICE with search_filter="http_500", log_group_name_keywords=["api-gateway"], different start_time/end_time, period_label each time.

User: "Lambda ile ilgili log gruplarını listele"
→ describe_log_groups(log_group_name_keywords=["lambda"], max_items=200)

User: "sadece bozkurteradev status 500 olanları göster"
→ query_log_group(
    log_group_name="<apigateway access-logs path>",
    tenant_filter="bozkurteradev",
    search_filter="http_500",
    response_mode="detail",
    hours=<from user>
  )

User: "logları analiz et" (after a prior log query)
→ query_log_group again with same log_group_name, hours, search_filter, tenant_filter; response_mode="analysis"

## Other examples
- "tüm log gruplarımı listele" → describe_log_groups(max_items=1000)
- "codebuild ve lambda log grupları" → describe_log_groups(log_group_name_keywords=[...])
- Allowed tools: {allowed}
- Default region: {srv.AWS_REGION}

## Language
Reply in the same language as the user's latest message (Turkish in → Turkish out; English in → English out).
Never switch to English when the user wrote in Turkish.

## Off-topic & casual messages
You are primarily a CloudWatch SRE assistant, but you MUST answer brief harmless casual questions
(greetings, identity, "hangi takımlısın", light small talk) in a friendly, natural tone.
- NEVER refuse harmless personal or opinion questions with "I'm sorry, I can't share that" or similar.
- If the topic is unrelated to AWS/observability, answer in 1–3 sentences, then offer help with logs/metrics/alarms.
- Refuse ONLY for: credentials/secrets, bypassing security, illegal or harmful actions.

## Response format
Search/log-line and ranking answers are formatted server-side from real AWS data.
NEVER invent log group names, counts, rankings, or placeholder rows.
NEVER paste raw CloudWatch Insights query strings in the user-facing reply — users see results only.
Summary (count) replies are conversational: confirm the finding, then offer drill-down — do not dump log lines until the user asks.

{multi_turn_conversation_prompt()}

Use response_mode=analysis when user asks to analyze / explain root cause — server aggregates real log fields.
""".strip()
    return _system_prompt_cache


# Alias for imports


# Tools whose output is formatted on the server from real AWS data (no LLM synthesis).
# Search tools: return formatted answer immediately after the tool runs (prevents hallucination).
IMMEDIATE_FORMAT_TOOLS = frozenset({
    "search_logs_across_groups",
    "execute_log_insights_query",
    "query_log_group",
})

LOG_GROUP_PATH_PATTERN = re.compile(
    r"(?P<path>/aws/[\w\-\./]+|aws/[\w\-\./]+)",
    re.IGNORECASE,
)


def normalize_log_group_path(raw: str) -> str:
    path = (raw or "").strip().strip("`'\"").rstrip("`")
    path = _srv().normalize_insights_log_group(path)
    if path.startswith("aws/"):
        return "/" + path
    if path and not path.startswith("/"):
        return "/" + path.lstrip("/")
    return path


_SINGLE_GROUP_SUMMARY_RE = re.compile(
    r"\*\*`(?P<group>[^`]+)`\*\*\s*—\s*son\s*\*\*(?P<hours>\d+)\s*saat\*\*\s*·\s*"
    r"\*\*(?P<filter>[^*]+)\*\*:\s*\*\*(?P<count>\d+)\*\*",
    re.I,
)

_CONVERSATIONAL_SUMMARY_RE = re.compile(
    r"Evet — son \*\*(?P<hours>\d+) saat\*\* içinde `(?P<group>[^`]+)` log grubunda "
    r"\*\*(?P<count>\d+)\*\* adet \*\*(?P<filter>[^*]+)\*\*",
    re.I,
)

_DETAIL_PREVIEW_LINES = 15
_MAX_RESULT_LINES = 500

_SHOW_ALL_RESULTS_RE = re.compile(
    r"tamamını|tamamini|tümünü|tumunu|hepsini|tamamı|tamami|"
    r"tüm\s+kayıt|tum\s+kayit|all\s+(?:of\s+them|records|lines)|hepsini\s+göster",
    re.I,
)

_DETAIL_HEADER_RE = re.compile(
    r"\*\*`(?P<group>[^`]+)`\*\*\s*—\s*son\s*\*\*(?P<hours>\d+)\s*saat\*\*",
    re.I,
)

_PRIOR_GROUP_LINE_RE = re.compile(
    r"\*\*Önceki grup\*\*\s*`(?P<group>[^`]+)`",
    re.I,
)


def _merge_log_ctx(
    recent: dict[str, Any],
    older: dict[str, Any],
) -> dict[str, Any]:
    """Merge an older assistant turn into a more recent one (newest-first walk)."""
    out = {**recent}
    for key in (
        "hours",
        "search_filter",
        "tenant_filter",
        "log_group_name",
        "log_group_names",
        "scope",
        "log_group_name_keywords",
    ):
        if older.get(key) and (
            key not in out
            or (
                key == "search_filter"
                and out.get("search_filter") == "errors"
                and older["search_filter"] != "errors"
            )
            or (key == "hours" and int(out.get("hours") or 1) == 1 and int(older.get("hours") or 1) != 1)
        ):
            out[key] = older[key]
    recent_ranked = list(out.get("ranked_groups") or [])
    older_ranked = list(older.get("ranked_groups") or [])
    if len(older_ranked) > len(recent_ranked):
        out["ranked_groups"] = older_ranked
    elif not recent_ranked and older_ranked:
        out["ranked_groups"] = older_ranked
    out["match_count"] = max(int(out.get("match_count") or 0), int(older.get("match_count") or 0))
    return out


def _extract_ranked_groups_from_markdown(content: str) -> list[tuple[str, int]]:
    groups: list[tuple[str, int]] = []
    for match in re.finditer(
        r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*\*\*(\d+)\*\*\s*\|",
        content,
    ):
        groups.append((normalize_log_group_path(match.group(1)), int(match.group(2))))
    if groups:
        return groups
    for match in re.finditer(r"^\d+\.\s+`([^`]+)`:\s+\*\*(\d+)\*\*", content, re.M):
        groups.append((normalize_log_group_path(match.group(1)), int(match.group(2))))
    if groups:
        return groups
    for match in re.finditer(
        r"^\d+\.\s+`([^`]+)`\s*[—–-]\s*\*\*(\d+)\*\*",
        content,
        re.M,
    ):
        groups.append((normalize_log_group_path(match.group(1)), int(match.group(2))))
    if groups:
        return groups
    for match in re.finditer(
        r"^\d+\.\s+(/[^\s—–-]+)\s*[—–-]\s*(\d+)\s+eşleşme",
        content,
        re.M | re.I,
    ):
        groups.append((normalize_log_group_path(match.group(1)), int(match.group(2))))
    return groups


def _extract_hours_and_filter_from_assistant(content: str) -> tuple[int, LogSearchFilter]:
    hours = 1
    hours_match = re.search(r"son\s*\*\*(\d+)\s*saat\*\*|\(son\s*(\d+)\s*saat", content, re.I)
    if hours_match:
        hours = int(next(g for g in hours_match.groups() if g))

    filter_name: LogSearchFilter = "errors"
    for preset in LOG_SEARCH_QUERY_PRESETS:
        if re.search(rf"\b{re.escape(preset)}\b", content, re.I):
            filter_name = preset
            break
    else:
        lowered = content.lower()
        if "500" in lowered or "http_500" in lowered:
            filter_name = "http_500"
        elif "error" in lowered or "hata" in lowered:
            filter_name = "errors"
    return hours, filter_name


def _parse_assistant_log_context(content: str) -> dict[str, Any] | None:
    """Extract log query context from various server-formatted assistant replies."""
    for pattern in (_CONVERSATIONAL_SUMMARY_RE, _SINGLE_GROUP_SUMMARY_RE):
        match = pattern.search(content)
        if match:
            return {
                "log_group_name": normalize_log_group_path(match.group("group")),
                "hours": int(match.group("hours")),
                "search_filter": parse_filter_from_display_label(match.group("filter")),
                "match_count": int(match.group("count")),
            }

    compact = re.search(
        r"\[Önceki:\s*`([^`]+)`\s+son\s+(\d+)\s+saat\s+([^:]+):\s+(\d+)\s+eşleşme",
        content,
    )
    if compact:
        return {
            "log_group_name": normalize_log_group_path(compact.group(1)),
            "hours": int(compact.group(2)),
            "search_filter": parse_filter_from_display_label(compact.group(3)),
            "match_count": int(compact.group(4)),
        }

    multi_compact = re.search(
        r"\[Önceki:.*?(\d+)\s+log grubu:\s*(.+?)\.\s*Detay için grup adını belirt\.\]",
        content,
        re.I | re.S,
    )
    if multi_compact:
        names = [
            normalize_log_group_path(name)
            for name in re.findall(r"`([^`]+)`", multi_compact.group(2))
        ]
        if names:
            hours, search_filter = _extract_hours_and_filter_from_assistant(content)
            ctx = {
                "hours": hours,
                "search_filter": search_filter,
                "log_group_names": names,
                "ranked_groups": [(name, 0) for name in names],
            }
            if len(names) == 1:
                ctx["log_group_name"] = names[0]
            return ctx

    detail = _DETAIL_HEADER_RE.search(content)
    if detail:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        return {
            "log_group_name": normalize_log_group_path(detail.group("group")),
            "hours": int(detail.group("hours")) if detail.group("hours") else hours,
            "search_filter": search_filter,
        }

    prior = _PRIOR_GROUP_LINE_RE.search(content)
    if prior:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        return {
            "log_group_name": normalize_log_group_path(prior.group("group")),
            "hours": hours,
            "search_filter": search_filter,
        }

    only_group = re.search(r"yalnızca\*?\*?\s*`([^`]+)`", content, re.I)
    if only_group:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        return {
            "log_group_name": normalize_log_group_path(only_group.group(1)),
            "hours": hours,
            "search_filter": search_filter,
        }

    ranked = _extract_ranked_groups_from_markdown(content)
    if ranked:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        ctx: dict[str, Any] = {
            "hours": hours,
            "search_filter": search_filter,
            "ranked_groups": ranked,
            "log_group_names": [g for g, _ in ranked],
        }
        if len(ranked) == 1:
            ctx["log_group_name"] = ranked[0][0]
            ctx["match_count"] = ranked[0][1]
        return ctx

    inline_group = re.search(
        r"(?:İşte|Detay için)\s+`(?P<group>/aws/[^`]+)`",
        content,
        re.I,
    )
    if inline_group:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        return {
            "log_group_name": normalize_log_group_path(inline_group.group("group")),
            "hours": hours,
            "search_filter": search_filter,
        }

    empty_search = re.search(
        r"Son\s+(\d+)\s+saat\s+içinde\s+\d+\s+log\s+grubunda\s+arama\s+yapıldı",
        content,
        re.I,
    )
    if empty_search:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        return {
            "hours": int(empty_search.group(1)) if empty_search.group(1) else hours,
            "search_filter": search_filter,
            "match_count": 0,
        }

    ranked = _extract_ranked_groups_from_markdown(content)
    total_match = re.search(r"Toplam:\s*\*\*(\d+)\*\*", content, re.I)
    if ranked and total_match:
        hours, search_filter = _extract_hours_and_filter_from_assistant(content)
        ctx: dict[str, Any] = {
            "hours": hours,
            "search_filter": search_filter,
            "ranked_groups": ranked,
            "log_group_names": [group for group, _ in ranked],
            "match_count": int(total_match.group(1)),
        }
        if len(ranked) == 1:
            ctx["log_group_name"] = ranked[0][0]
        return ctx

    return None


def parse_filter_from_display_label(label: str) -> LogSearchFilter:
    lowered = (label or "").strip().lower()
    if "backend" in lowered or "auth hariç" in lowered:
        return "http_500_backend"
    for preset in LOG_SEARCH_QUERY_PRESETS:
        if preset in lowered or lowered == preset:
            return preset  # type: ignore[return-value]
    alias = SEARCH_FILTER_ALIASES.get(lowered)
    if alias:
        return alias
    if "error" in lowered:
        return "errors"
    return "errors"


def is_system_wide_log_show_request(message: str) -> bool:
    """Show/count HTTP errors across groups without naming a log group path."""
    if extract_log_group_paths(message):
        return False
    if is_vague_log_question(message):
        return False
    if re.search(
        r"detaylı|satır\s*satır|tamamını|hepsini|json\s*export|daha\s+detay|"
        r"show\s+detail|more\s+detail",
        message,
        re.I,
    ):
        return False
    has_status = is_http_status_search(message) or bool(
        re.search(r"\berror|\bhata|\bexception", message, re.I)
    )
    if not has_status:
        return False
    return bool(
        re.search(
            r"göster|getir|say|count|kaç|var\s*m|listele|show|check|bul",
            message,
            re.I,
        )
    )


def is_log_drilldown_followup(message: str) -> bool:
    if is_vague_log_question(message):
        return False
    if is_metric_analysis_request(message):
        return False
    if is_scope_uniqueness_followup(message):
        return False
    if is_system_wide_log_show_request(message):
        return False
    if re.search(r"var\s*m[ıi]|kaç\s*tane|listele", message, re.I):
        return False
    if re.search(r"hangi\s+log\s+gru", message, re.I):
        return False
    stripped = message.strip()
    if re.fullmatch(
        r"(?i)(detaylı\s*)?göster(ir\s*misin)?\.?|show(\s+details?)?\.?|details?\.?",
        stripped,
    ):
        return True
    if is_log_analysis_request(message):
        return True
    if parse_explicit_line_limit(message) and re.search(
        r"göster|getir|ver|listele", message, re.I
    ):
        return True
    if wants_show_all_results(message) and re.search(r"göster|getir|ver", message, re.I):
        return True
    return bool(
        re.search(
            r"detaylı|detay\s*li|detaylı\s+yaz|satır\s*satır|göstersene|gösterir\s*misin|göstermeni\s+istiyorum|"
            r"daha\s+detay|show\s+detail|more\s+detail|tümünü\s+göster|hepsini\s+göster|"
            r"tamamını\s+göster|tamamını\s+getir|detaylı\s+getir",
            message,
            re.I,
        )
    )


_ORDINAL_GROUP_WORDS = {
    "birinci": 1,
    "ikinci": 2,
    "üçüncü": 3,
    "dördüncü": 4,
    "first": 1,
    "second": 2,
    "third": 3,
}


def parse_ranked_group_selection(
    message: str,
    ranked_groups: list[tuple[str, int]],
) -> str | None:
    if not ranked_groups:
        return None
    stripped = message.strip()
    if not stripped or len(stripped) > 40:
        return None

    ordinal = _ORDINAL_GROUP_WORDS.get(stripped.lower())
    if ordinal and 1 <= ordinal <= len(ranked_groups):
        return ranked_groups[ordinal - 1][0]

    for pattern in (
        r"(?i)^(?:numara\s+)?(\d{1,2})(?:\s*numara(?:yı|yi|yu|yü)?)?\.?$",
        r"(?i)^(\d{1,2})\s*(?:numara|nolu|no|\.?\s*gru[pb])",
        r"(?i)^(\d{1,2})\.?\s*$",
    ):
        match = re.match(pattern, stripped)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(ranked_groups):
                return ranked_groups[index][0]
    return None


def is_scope_uniqueness_followup(
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> bool:
    """User asks whether hits are ONLY in the prior log group or also elsewhere."""
    if is_metric_analysis_request(message):
        return False
    if not re.search(
        r"(?i)("
        r"(bir\s+tek|yalnızca|sadece|only).{0,40}(bu|this|şu).{0,20}(log\s*)?gru|"
        r"(bu|this|şu).{0,30}(log\s*)?gru.{0,30}(tek|yalnız|only|alone)|"
        r"başka.{0,30}(log\s*)?gru.{0,20}(da|de)?\s*(var|bulun)|"
        r"başka\s+yer(de|de)?\s+(var|bulun)|"
        r"(sonuç|result).{0,20}(sınırlı|limited|only\s+this)|"
        r"sadece\s+burada\s+mı|only\s+here"
        r")",
        message,
    ):
        return False
    ctx = merge_log_query_context(history)
    return bool(ctx.get("log_group_name") or ctx.get("search_filter"))


def build_log_context_from_tool_data(data: dict[str, Any]) -> dict[str, Any] | None:
    """Structured carry-over context from a log tool JSON payload."""
    if not isinstance(data, dict) or data.get("error"):
        return None

    hours = int(data.get("hours") or 1)
    search_filter = str(data.get("search_filter") or "errors")
    ctx: dict[str, Any] = {"hours": hours, "search_filter": search_filter}
    if data.get("tenant_filter"):
        ctx["tenant_filter"] = str(data["tenant_filter"])
    if data.get("scope") in {"system", "keywords"}:
        # The question was about a scope, not about whichever single group happened to match.
        ctx["scope"] = data["scope"]
        if data["scope"] == "keywords" and data.get("keywords_tried"):
            ctx["log_group_name_keywords"] = list(data["keywords_tried"])
    if data.get("window_label"):
        ctx["window_label"] = str(data["window_label"])
        if data.get("window_minutes"):
            ctx["window_minutes"] = int(data["window_minutes"])
        elif data.get("start_time") and data.get("end_time"):
            ctx["start_time"] = str(data["start_time"])
            ctx["end_time"] = str(data["end_time"])

    if data.get("log_group_name"):
        group = normalize_log_group_path(
            _srv().normalize_insights_log_group(str(data["log_group_name"]))
        )
        ctx["log_group_name"] = group
        ctx["match_count"] = int(data.get("match_count") or 0)
        return ctx

    ranked: list[tuple[str, int]] = []
    for row in _collect_log_rows_from_data(data):
        group = normalize_log_group_path(
            _srv().normalize_insights_log_group(str(row.get("log_group") or "").strip())
        )
        if not group:
            continue
        count = int(float(row.get("matches") or 0))
        ranked.append((group, count))
    ranked.sort(key=lambda item: (-item[1], item[0]))

    if ranked:
        ctx["ranked_groups"] = ranked
        ctx["log_group_names"] = [group for group, _ in ranked]
        if len(ranked) == 1:
            ctx["log_group_name"] = ranked[0][0]
            ctx["match_count"] = ranked[0][1]
        return ctx

    ctx["match_count"] = int(data.get("match_count") or 0)
    return ctx


def build_log_export_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Structured JSON export from log tool results."""
    rows = _collect_log_rows_from_data(data)
    events: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("message", "")).strip()
        if not raw:
            continue
        entry: dict[str, Any] = {
            "timestamp": row.get("timestamp", ""),
            "log_group": _srv().normalize_insights_log_group(
                str(row.get("log_group") or data.get("log_group_name") or "")
            ),
            "log_stream": row.get("log_stream", ""),
        }
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                entry.update(parsed)
            else:
                entry["raw_message"] = raw
        except json.JSONDecodeError:
            entry["raw_message"] = raw
        events.append(entry)

    if not events:
        return None

    return {
        "log_group_name": data.get("log_group_name"),
        "hours": data.get("hours"),
        "search_filter": data.get("search_filter"),
        "tenant_filter": data.get("tenant_filter"),
        "match_count": int(data.get("match_count") or len(events)),
        "exported_count": len(events),
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "region": data.get("region"),
        "events": events,
    }


_FILTER_LABELS = {
    "errors": "Hatalar",
    "http_2xx": "2xx başarılı istekler",
    "http_4xx": "4xx yanıtlar",
    "http_5xx": "5xx yanıtlar",
    "http_500_backend": "HTTP 500 (backend)",
}
_FILTER_PHRASES = {
    "errors": "hataları",
    "http_2xx": "2xx isteklerini",
    "http_4xx": "4xx hatalarını",
    "http_5xx": "5xx hatalarını",
    "http_500_backend": "backend 500 hatalarını",
}


def _filter_label(search_filter: str) -> str:
    if search_filter in _FILTER_LABELS:
        return _FILTER_LABELS[search_filter]
    if search_filter.startswith("http_"):
        return f"HTTP {search_filter[5:]}"
    return search_filter or "Arama"


def _filter_phrase(search_filter: str) -> str:
    if search_filter in _FILTER_PHRASES:
        return _FILTER_PHRASES[search_filter]
    if search_filter.startswith("http_"):
        return f"{search_filter[5:]} hatalarını"
    return "hataları"


def build_response_card(tool_name: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Structured summary of a tool result for the UI answer card (real tool data only)."""
    if not isinstance(data, dict) or data.get("error"):
        return None
    if tool_name == "analyze_metric":
        card: dict[str, Any] = {
            "kind": "metric",
            "title": f"{data.get('namespace', '')} / {data.get('metric_name', '')}",
            "namespace": data.get("namespace"),
            "metric_name": data.get("metric_name"),
            "statistic": data.get("statistic"),
            "window": (
                f"{data['start_time']} → {data['end_time']}"
                if data.get("start_time") and data.get("end_time")
                else None
            ),
            "aggregation": data.get("aggregation"),
            "tool": tool_name,
        }
        if not data.get("datapoint_count"):
            return {**card, "empty": True}
        first, second = data.get("first_half_avg"), data.get("second_half_avg")
        change = (second - first) / first * 100 if first and second is not None else None
        return {
            **card,
            "sum": data.get("sum"),
            "average": data.get("average"),
            "min": data.get("min"),
            "max": data.get("max"),
            "trend": data.get("trend"),
            "change_pct": round(change, 1) if change is not None else None,
            "datapoints": data.get("datapoint_count"),
            "series": data.get("series") or [],
        }

    if tool_name not in {"query_log_group", "search_logs_across_groups"}:
        return None
    search_filter = str(data.get("search_filter") or "")
    hours = int(data.get("hours") or 1)
    card = {
        "tool": tool_name,
        "filter": search_filter,
        "filter_label": _filter_label(search_filter),
        "window": data.get("window_label") or f"son {hours} saat",
        "hours": hours,
        "start_time": data.get("start_time"),
        "end_time": data.get("end_time"),
        "scope": data.get("scope") or ("group" if data.get("log_group_name") else None),
        "log_group": data.get("log_group_name"),
        "groups_searched": data.get("log_groups_searched"),
        "keywords": data.get("keywords_tried") or [],
        "excluded_prefixes": data.get("excluded_prefixes") or [],
        "tenant": data.get("tenant_filter"),
        "total": int(data.get("total_matches") or data.get("match_count") or 0),
        "bytes_scanned": data.get("bytes_scanned"),
        "batches": data.get("insights_batches"),
        "status": data.get("status"),
    }
    mode = data.get("response_mode")
    if mode == "analysis":
        return {**card, "kind": "analysis"}
    if mode in {"summary", "count_only"}:
        ranked = sorted(
            (
                (
                    _srv().normalize_insights_log_group(str(row.get("log_group") or "").strip()),
                    int(float(row.get("matches") or 0)),
                )
                for row in _collect_log_rows_from_data(data)
                if str(row.get("log_group") or "").strip()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if data.get("log_group_name") and len(ranked) <= 1:
            return {
                **card,
                "kind": "count",
                "status_breakdown": data.get("status_breakdown") or [],
            }
        if not ranked:
            return {**card, "kind": "empty"}
        return {
            **card,
            "kind": "ranking",
            "groups_with_matches": max(len(ranked), int(data.get("groups_with_matches") or 0)),
            "top": [{"group": g, "count": c} for g, c in ranked[:10]],
        }
    rows_returned = len(_collect_log_rows_from_data(data))
    return {**card, "kind": "detail" if rows_returned else "empty", "shown": rows_returned}


_METRIC_LOG_FILTERS = (
    (re.compile(r"5xx|5\d\d", re.I), "http_5xx"),
    (re.compile(r"4xx|4\d\d", re.I), "http_4xx"),
)
_METRIC_NAMESPACE_KEYWORDS = {
    "AWS/ApiGateway": ["api-gateway"],
    "AWS/Lambda": ["lambda"],
    "AWS/ApplicationELB": ["alb"],
    "AWS/ECS": ["containerinsights"],
    "ContainerInsights": ["containerinsights"],
    "AWS/CloudFront": ["cloudfront"],
    "AWS/RDS": ["rds"],
}


def metric_log_context(data: dict[str, Any]) -> dict[str, Any] | None:
    """Map a metric answer to the logs behind it, so "detaylı göster" has somewhere to go."""
    namespace = str(data.get("namespace") or "")
    metric = str(data.get("metric_name") or "")
    if not namespace:
        return None
    search_filter = next(
        (flt for pattern, flt in _METRIC_LOG_FILTERS if pattern.search(metric)), "errors"
    )
    hours = 1
    try:
        start = _srv().parse_time(str(data["start_time"]))
        end = _srv().parse_time(str(data["end_time"]))
        hours = max(1, -(-int((end - start).total_seconds()) // 3600))
    except (KeyError, ValueError):
        pass
    return {
        "source": "metric",
        "metric": f"{namespace}/{metric}",
        "search_filter": search_filter,
        "hours": min(168, hours),
        "log_group_name_keywords": _METRIC_NAMESPACE_KEYWORDS.get(namespace, []),
        # Explicit None stops older turns' groups from leaking into this context.
        "log_group_name": None,
        "log_group_names": None,
    }


def build_suggestions(card: dict[str, Any] | None) -> list[str]:
    """Clickable follow-ups; each phrase is handled by an existing follow-up path."""
    if not card:
        return []
    kind = card.get("kind")
    hours = int(card.get("hours") or 1)
    wider = "son 24 saate bak" if hours < 24 else "son 7 güne bak"
    phrase = _filter_phrase(str(card.get("filter") or ""))
    if kind == "metric":
        return [
            "İlgili logları detaylı göster",
            f"{card.get('namespace')} {card.get('metric_name')} metriği son 7 günde nasıl?",
        ]
    if kind == "ranking":
        return ["1. grubun detayını göster", "logları analiz et", f"{phrase[0].upper()}{phrase[1:]} dünle karşılaştır"]
    if kind == "count":
        suggestions = ["detaylı göster", "logları analiz et"] if card.get("total") else []
        return suggestions + [wider]
    if kind == "detail":
        total, shown = int(card.get("total") or 0), int(card.get("shown") or 0)
        extra = ["tamamını göster"] if shown < total <= _MAX_RESULT_LINES else []
        return ["logları analiz et", *extra]
    if kind == "empty":
        return [wider]
    return []


def attach_log_context(
    result: dict[str, Any],
    tool_name: str,
    tool_result: str,
) -> dict[str, Any]:
    if tool_name not in {"query_log_group", "search_logs_across_groups", "analyze_metric"}:
        return result
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return result
    enriched = dict(result)
    card = build_response_card(tool_name, data)
    if card:
        enriched["card"] = card
        enriched["suggestions"] = build_suggestions(card)
    if tool_name == "analyze_metric":
        metric_ctx = metric_log_context(data) if isinstance(data, dict) else None
        if metric_ctx:
            enriched["log_context"] = metric_ctx
        return enriched
    log_context = build_log_context_from_tool_data(data)
    if not log_context:
        return enriched
    enriched["log_context"] = log_context
    export_payload = build_log_export_payload(data)
    if export_payload:
        enriched["log_export"] = export_payload
    return enriched


def extract_last_log_query_context(
    history: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Recover log group, hours, and filter from recent chat (for drill-down follow-ups)."""
    user_ctx: dict[str, Any] | None = None
    merged: dict[str, Any] | None = None
    for item in reversed(history or []):
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role == "assistant":
            chunk: dict[str, Any] | None = None
            structured = item.get("log_context")
            if isinstance(structured, dict) and structured:
                chunk = dict(structured)
            else:
                parsed = _parse_assistant_log_context(content)
                if parsed:
                    chunk = parsed
            if chunk:
                merged = _merge_log_ctx(merged, chunk) if merged else chunk
        elif role == "user" and user_ctx is None:
            paths = extract_log_group_paths(content)
            if paths:
                user_ctx = {
                    "log_group_name": paths[0],
                    "hours": parse_hours_from_message(content),
                    "search_filter": infer_log_search_filter(content),
                }
            elif re.search(r"500|error|hata|status", content, re.I):
                user_ctx = {
                    "hours": parse_hours_from_message(content),
                    "search_filter": infer_log_search_filter(content),
                }
    return merged or user_ctx


def extract_user_log_intent_from_history(
    history: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Recover tenant/filter/time from earlier user turns when assistant had no tool result."""
    for item in reversed(history or []):
        if str(item.get("role") or "") != "user":
            continue
        content = str(item.get("content") or "")
        tenant = extract_tenant_filter(content)
        paths = extract_log_group_paths(content)
        if not tenant and not paths and not re.search(
            r"500|5\s*xx|error|hata|analiz|status", content, re.I
        ):
            continue
        intent: dict[str, Any] = {
            "hours": parse_hours_from_message(content),
            "search_filter": infer_log_search_filter(content),
        }
        if tenant:
            intent["tenant_filter"] = tenant
        if paths:
            intent["log_group_name"] = paths[0]
        return intent
    return None


def _extract_peak_match_count_from_history(
    history: list[dict[str, Any]] | None,
) -> int:
    """Highest match total seen in the thread (summary counts beat detail previews)."""
    peak = 0
    for item in history or []:
        structured = item.get("log_context")
        if isinstance(structured, dict):
            peak = max(peak, int(structured.get("match_count") or 0))
            for row in structured.get("ranked_groups") or []:
                if isinstance(row, (list, tuple)) and len(row) >= 2:
                    peak = max(peak, int(row[1]))
        content = str(item.get("content") or "")
        total = re.search(r"Toplam:\s*\*\*(\d+)\*\*", content, re.I)
        if total:
            peak = max(peak, int(total.group(1)))
        for match in re.finditer(
            r"\|\s*\d+\s*\|\s*`[^`]+`\s*\|\s*\*\*(\d+)\*\*\s*\|",
            content,
        ):
            peak = max(peak, int(match.group(1)))
    return peak


def merge_log_query_context(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    assistant_ctx = extract_last_log_query_context(history) or {}
    user_ctx = extract_user_log_intent_from_history(history) or {}
    merged = {**user_ctx, **assistant_ctx}
    peak = _extract_peak_match_count_from_history(history)
    if peak > int(merged.get("match_count") or 0):
        merged["match_count"] = peak
    return merged


def build_session_context_message(
    history: list[dict[str, Any]] | None,
) -> dict[str, str] | None:
    """Structured session state injected into the LLM thread (prompt + context, not regex-only)."""
    ctx = merge_log_query_context(history)
    if not ctx:
        return None

    lines = [
        "[Active session context — apply to follow-ups unless the user clearly starts a NEW search]",
    ]
    if ctx.get("search_filter"):
        lines.append(f"- search_filter: {ctx['search_filter']}")
    if ctx.get("hours"):
        lines.append(f"- hours: {ctx['hours']}")
    if ctx.get("log_group_name"):
        lines.append(f"- log_group_name: {ctx['log_group_name']}")
    if ctx.get("tenant_filter"):
        lines.append(f"- tenant_filter: {ctx['tenant_filter']}")
    peak = int(ctx.get("match_count") or 0)
    if peak:
        lines.append(
            f"- summary_match_count: {peak} "
            f"(detail preview may show only {_DETAIL_PREVIEW_LINES}; use max_result_lines≥{peak} when user wants all)"
        )
    if ctx.get("window_label"):
        window = f"- window: {ctx['window_label']}"
        if ctx.get("start_time") and ctx.get("end_time"):
            window += f" (start_time={ctx['start_time']}, end_time={ctx['end_time']})"
        lines.append(window)
    if ctx.get("log_group_name_keywords"):
        lines.append(f"- log_group_name_keywords: {', '.join(ctx['log_group_name_keywords'])}")
    if ctx.get("source") == "metric":
        lines.append(
            f"- previous answer was the metric {ctx.get('metric')}; its related logs = "
            f"search_filter {ctx.get('search_filter')} in groups matching the keywords above"
        )
    ranked = ctx.get("ranked_groups") or []
    if len(ranked) > 1:
        lines.append('- ranked_groups (numbered — "2. grup" / "ikinci servis" means entry 2):')
        for index, (group, count) in enumerate(ranked[:10], 1):
            lines.append(f"  {index}. {group} ({count})")

    lines.extend(
        [
            "",
            "Follow-up hints for this turn:",
            "- Time-only → update hours, keep filter/scope.",
            "- detaylı/tamamını → detail + max_result_lines ≥ summary_match_count.",
            "- X kayıt dedin → explain preview vs total; fetch all rows.",
            "- New status + new time → fresh search_logs_across_groups (drop log_group_name).",
        ]
    )
    return {"role": "system", "content": "\n".join(lines)}


def summarize_access_log_error_types(rows: list[dict[str, str]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = str(row.get("message", "")).strip()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise json.JSONDecodeError("not object", raw, 0)
            err_type = str(data.get("errorResponseType") or "").strip()
            status = str(data.get("status") or "").strip()
            key = err_type if err_type and err_type not in ("-", "") else f"HTTP {status or '?'}"
        except json.JSONDecodeError:
            key = "diğer"
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def format_filter_display_name(filter_name: str) -> str:
    if filter_name == "http_500_backend":
        return "http_500 (backend — auth hariç)"
    return filter_name


def format_status_breakdown(breakdown: list[dict[str, Any]] | None) -> str:
    if not breakdown:
        return ""
    parts = " · ".join(f"**{item['status']}**: {item['count']}" for item in breakdown[:8])
    return (
        f"\n\n**HTTP 4xx/5xx yanıt dağılımı:** {parts}\n"
        "_Hata sayısına uygulama hataları ve tüm 5xx yanıtları dahil; "
        "4xx yanıtlar yukarıda ayrıca gösteriliyor._"
    )


def format_single_group_summary_reply(
    *,
    group_name: str,
    hours: int,
    filter_name: str,
    count: int,
    status: str,
    window_note: str,
    prefix: str,
    tenant_filter: str | None = None,
    window_label: str | None = None,
    status_breakdown: list[dict[str, Any]] | None = None,
) -> str:
    display_filter = format_filter_display_name(filter_name)
    tenant_note = f" (tenant: **{tenant_filter}**)" if tenant_filter else ""
    window = f"**{window_label}**" if window_label else f"son **{hours} saat**"
    breakdown = format_status_breakdown(status_breakdown)
    if count == 0:
        return (
            f"{prefix}{window[0].upper() + window[1:]} içinde `{group_name}` log grubunda{tenant_note} "
            f"**{display_filter}** için eşleşme bulamadım (durum: {status}){window_note}.{breakdown}"
        )
    return (
        f"{prefix}Evet — {window} içinde `{group_name}` log grubunda{tenant_note} "
        f"**{count}** adet **{display_filter}** kaydı bulundu (durum: {status}){window_note}.{breakdown}\n\n"
        "İstersen satır satır detayını gösterebilirim; **\"detaylı göster\"** yazman yeterli.\n"
        "Kök neden özeti için **\"logları analiz et\"** de."
    )


def format_search_scope(data: dict[str, Any]) -> str:
    """One line telling the user exactly which log groups were searched."""
    searched = int(data.get("log_groups_searched") or 0)
    scope = data.get("scope")
    excluded = ", ".join(f"`{p}`" for p in data.get("excluded_prefixes") or [])
    excluded_note = f" ({excluded} hariç)" if excluded else ""
    if scope == "system":
        return f"Kapsam: **tüm sistem** — {searched} log grubu tarandı{excluded_note}"
    if scope == "keywords":
        keywords = ", ".join(f"`{k}`" for k in data.get("keywords_tried") or [])
        return f"Kapsam: {keywords} ile eşleşen **{searched}** log grubu{excluded_note}"
    if data.get("requested_log_group_names"):
        return f"Kapsam: istenen **{searched}** log grubu"
    return ""


def extract_log_group_paths(message: str) -> list[str]:
    paths: list[str] = []
    for match in LOG_GROUP_PATH_PATTERN.finditer(message):
        paths.append(normalize_log_group_path(match.group("path")))
    return list(dict.fromkeys(paths))


def infer_log_search_filter(message: str) -> LogSearchFilter:
    lowered = message.lower()
    if re.search(r"\b2\s*xx\b|başarılı|successful|non[- ]?error|2xx", lowered):
        return "http_2xx"
    if re.search(r"\b500\b|statusu\s*500|status\s*500", lowered) and re.search(
        r"integration|backend|entegrasyon|gerçek|integrationstatus",
        lowered,
    ):
        return "http_500_backend"
    for code in ("504", "503", "502", "500", "404", "403", "401", "400"):
        if re.search(
            rf"\b{code}\b|{code}\s*l|status\s*kodu.*{code}|statusu\s*{code}|status\s*{code}",
            lowered,
        ):
            return f"http_{code}"  # type: ignore[return-value]
    if re.search(r"5\s*xx|status\s*kodu.*5", lowered):
        return "http_5xx"
    if re.search(r"4\s*xx|status\s*kodu.*4", lowered):
        return "http_4xx"
    for alias, preset in SEARCH_FILTER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return preset
    return "errors"


def is_http_status_search(message: str) -> bool:
    return bool(
        re.search(
            r"2\s*xx|4\s*xx|5\s*xx|\b[245]\d{2}\b|başarılı|successful|status\s*kodu",
            message,
            re.I,
        )
    )


def is_named_log_content_query(message: str) -> bool:
    if not extract_log_group_paths(message):
        return False
    if re.search(r"\blog\s*gr", message, re.I) and re.search(
        r"\b(listele|liste|hangi|tüm)\b", message, re.I
    ):
        if not re.search(r"error|hata|5\d{2}|400|exception", message, re.I):
            return False
    return bool(
        re.search(
            r"error|hata|hatalar|exception|5\s*xx|5\d{2}|400|4\s*xx|fail|fatal|"
            r"status\s*kodu|getir|var\s*mı|içerik|satır|timeout",
            message,
            re.I,
        )
    )


def format_active_alarms_list(data: dict[str, Any]) -> str:
    metric_alarms = data.get("metric_alarms") or []
    composite_alarms = data.get("composite_alarms") or []
    region = data.get("region", AWS_REGION)
    total = len(metric_alarms) + len(composite_alarms)
    lines = [f"**{total} aktif alarm** ({region})", ""]
    index = 1
    for alarm in metric_alarms:
        name = alarm.get("AlarmName", "")
        state = alarm.get("StateValue", "ALARM")
        metric = alarm.get("MetricName", "")
        suffix = f" — metric: `{metric}`" if metric else ""
        lines.append(f"{index}. `{name}` ({state}){suffix}")
        index += 1
    for alarm in composite_alarms:
        name = alarm.get("AlarmName", "")
        lines.append(f"{index}. `{name}` (composite)")
        index += 1
    if not total:
        lines.append("Aktif alarm bulunamadı.")
    return "\n".join(lines)


def format_alarm_history(data: dict[str, Any]) -> str:
    if data.get("error"):
        return f"Alarm geçmişi alınamadı: {data['error']}"

    items = data.get("history_items") or []
    hours = data.get("hours")
    region = data.get("region", AWS_REGION)
    alarm_name = data.get("alarm_name")

    if not items:
        scope = f"`{alarm_name}`" if alarm_name else "tüm alarmlar"
        window = f"son {hours} saat" if hours else "seçili aralık"
        return f"{scope} için {window} içinde alarm geçişi bulunamadı ({region})."

    lines = []
    if alarm_name:
        lines.append(f"**`{alarm_name}` alarm geçmişi** ({region}) — {len(items)} kayıt")
    else:
        window = f"son {hours} saat" if hours else "seçili aralık"
        lines.append(f"**Alarm geçişleri** ({region}) — {window} — {len(items)} kayıt")
    lines.append("")

    for index, item in enumerate(items[:80], start=1):
        name = item.get("AlarmName", "")
        ts = item.get("Timestamp", "")
        summary = item.get("HistorySummary", "")
        if hasattr(ts, "isoformat"):
            ts = _srv().iso_utc(ts.astimezone(timezone.utc))
        elif isinstance(ts, str) and "T" not in ts:
            ts = str(ts)
        detail = summary or item.get("HistoryItemType", "")
        lines.append(f"{index}. `{name}` — {detail}")
        if ts:
            lines[-1] += f" ({ts})"

    if len(items) > 80:
        lines.append(f"\n... ve {len(items) - 80} kayıt daha")
    return "\n".join(lines)


def format_analyze_metric(data: dict[str, Any]) -> str:
    if data.get("error"):
        return f"Metrik analizi başarısız: {data['error']}"
    ns = data.get("namespace", "")
    metric = data.get("metric_name", "")
    stat = data.get("statistic", "Average")
    window = ""
    if data.get("start_time") and data.get("end_time"):
        window = f" ({data['start_time']} → {data['end_time']})"
    if data.get("message") and not data.get("datapoint_count"):
        prefix = f"**{ns} / {metric}** ({stat}){window}\n" if ns and metric else ""
        return f"{prefix}Bu aralıkta veri noktası bulunamadı."

    trend_map = {
        "increasing": "artıyor ↑",
        "decreasing": "azalıyor ↓",
        "stable": "stabil",
    }
    trend = trend_map.get(str(data.get("trend", "")), data.get("trend", ""))
    first_avg = data.get("first_half_avg")
    second_avg = data.get("second_half_avg")
    if isinstance(first_avg, (int, float)) and isinstance(second_avg, (int, float)) and first_avg:
        change = (second_avg - first_avg) / first_avg * 100
        trend = f"{trend} (son yarı, ilk yarıya göre {change:+.0f}%)"
    scope = {
        "account-wide": "tüm kaynaklar (hesap geneli toplam)",
        "dimensions": "seçili boyutlar",
    }.get(str(data.get("aggregation") or ""), "")
    lines = [f"**{ns} / {metric}** ({stat}){window}"]
    if scope:
        lines.append(f"- Kapsam: {scope}")
    if stat == "Sum":
        lines.append(f"- **Toplam: {data.get('sum', 0):,.0f}**".replace(",", "."))
    lines.extend(
        [
            f"- Veri noktası: {data.get('datapoint_count', 0)}",
            f"- Min: {data.get('min', 0):.2f} · Max: {data.get('max', 0):.2f} · "
            f"Ortalama: {data.get('average', 0):.2f}",
            f"- Trend: {trend}",
        ]
    )
    if data.get("first_timestamp") and data.get("last_timestamp"):
        lines.append(f"- İlk / son veri noktası: {data['first_timestamp']} → {data['last_timestamp']}")
    return "\n".join(lines)


def is_alarm_history_request(message: str) -> bool:
    lowered = message.lower()
    if re.search(r"\baktif\s+alarm", lowered) and not re.search(
        r"tetik|geçmiş|history|son\s+\d+\s+saat", lowered
    ):
        return False
    return bool(
        re.search(
            r"alarm.*tetik|tetiklenen\s+alarm|alarm\s+geçmiş|alarm\s+history|"
            r"hangi\s+alarm|alarmların\s+tamam|tamamını\s+göster.*alarm|göster.*alarmların|"
            r"son\s+\d+\s+saat.*alarm|alarm.*son\s+\d+\s+saat",
            lowered,
        )
    )


def is_active_alarms_request(message: str) -> bool:
    lowered = message.lower()
    return bool(
        re.search(r"\baktif\s+(?:cloudwatch\s+)?alarml", lowered)
        or re.search(r"\bactive\s+(?:cloudwatch\s+)?alarm", lowered)
    ) and not is_alarm_history_request(message)


_METRIC_WORD_RE = re.compile(r"metri[kğ]|metric", re.I)
_METRIC_SIGNAL_RE = re.compile(
    r"\bcpu\b|memory|bellek|\bmem\b|latency|gecikme|throttl|invocation|çağrı\s+sayı|"
    r"duration|utilization|kullanım\s+oran|bağlantı\s+sayı|connection\s+count|"
    r"concurrent|queue\s+depth|kuyruk",
    re.I,
)
_LOG_CONTENT_RE = re.compile(
    r"\blog|access|satır|kayıt|record|line|(?<!\d)[1-5]\d{2}(?!\d)|[1-5]\s*xx\b",
    re.I,
)


def is_metric_analysis_request(message: str) -> bool:
    """Explicit metric questions only — "error" alone is a log question, not a metric one."""
    lowered = message.lower()
    if _METRIC_WORD_RE.search(lowered):
        return True
    if _LOG_CONTENT_RE.search(lowered):
        return False
    return bool(
        re.search(
            r"\b(ecs|lambda|rds|alb|api\s*gateway|ec2|sqs|cloudfront|dynamo|eks|kubernetes|elasticache)\b",
            lowered,
        )
        and _METRIC_SIGNAL_RE.search(lowered)
    )


def wants_show_all_results(message: str) -> bool:
    return bool(_SHOW_ALL_RESULTS_RE.search(message))


def parse_explicit_line_limit(message: str) -> int | None:
    """User-requested row count, e.g. '10 tane', '21 ini' — not time windows like 'son 1 saat'."""
    patterns = (
        r"(\d{1,4})\s*(?:'|')?(?:ini|ını|unu|ünü|inde|ında)\b",
        r"\b(\d{1,4})\s*(?:tane|kayıt|kayit|adet|satır|satir)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.I)
        if match:
            return min(_MAX_RESULT_LINES, max(1, int(match.group(1))))

    order_match = re.search(r"\b(?:ilk|son)\s+(\d{1,4})\b", message, re.I)
    if order_match:
        after = message[order_match.end() :].lstrip()
        if not re.match(
            r"(?:saat|hr|dakika|dk|minute|gün|day)",
            after,
            re.I,
        ):
            return min(_MAX_RESULT_LINES, max(1, int(order_match.group(1))))
    return None


def wants_log_detail(message: str) -> bool:
    return bool(
        re.search(
            r"detaylı|detay\s*li|satır\s*satır|gösterir\s*misin|göstermeni\s+istiyorum|"
            r"göstersene|göster\s*bana|show\s+detail|satırları\s+göster|birde\s+detay|"
            r"bir\s+de\s+detay|tamamını\s+göster|hepsini\s+göster",
            message,
            re.I,
        )
    )


APIGW_ACCESS_LOG_FIELD_ORDER = [
    "status",
    "httpMethod",
    "path",
    "resourcePath",
    "stage",
    "errorResponseType",
    "errorMessage",
    "integrationStatus",
    "integrationLatency",
    "integrationErrorMessage",
    "responseLatency",
    "responseLength",
    "ip",
    "requestId",
    "requestTime",
    "domainName",
    "protocol",
    "userAgent",
    "tenantId",
    "userId",
    "tenantDomain",
]


def _meaningful_log_field_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return text not in ("", "-", "null")


def format_access_log_cloudwatch_block(
    raw: str,
    *,
    timestamp: str = "",
    log_group: str = "",
    log_stream: str = "",
) -> str:
    """CloudWatch console-style field list for API Gateway JSON access logs."""
    lines: list[str] = []
    if timestamp:
        lines.append(f"@timestamp\t{timestamp}")
    if log_group:
        lines.append(f"@log\t{log_group}")
    if log_stream:
        lines.append(f"@logStream\t{log_stream}")

    stripped = raw.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        if stripped:
            lines.append(f"@message\t{stripped[:2000]}")
        return "\n".join(lines)
    if not isinstance(data, dict):
        if stripped:
            lines.append(f"@message\t{stripped[:2000]}")
        return "\n".join(lines)

    lines.append(f"@message\t{json.dumps(data, ensure_ascii=False)}")

    shown: set[str] = set()
    for key in APIGW_ACCESS_LOG_FIELD_ORDER:
        value = data.get(key)
        if _meaningful_log_field_value(value):
            lines.append(f"{key}\t{value}")
            shown.add(key)
    for key in sorted(data):
        if key in shown:
            continue
        value = data.get(key)
        if _meaningful_log_field_value(value):
            lines.append(f"{key}\t{value}")
    return "\n".join(lines)


def normalize_tenant_slug(raw: str) -> str:
    slug = (raw or "").strip().strip("'\"").lower()
    slug = re.sub(
        r"(?:'?(?:nin|nın|nun|nün|in|ın|un|ün|i|ı|u|ü))$",
        "",
        slug,
        flags=re.I,
    )
    return slug


_NON_TENANT_TOKENS = {
    "api-gateway", "apigateway", "api", "lambda", "containerinsights", "codebuild", "alb", "eks",
    "ecs", "rds", "cloudfront", "log", "loglar", "logları", "error", "errors", "hata", "hataları",
    "son", "bugün", "dün", "sistem", "sistemde", "tüm", "sayı", "sayısını", "status", "http",
}
# "sadece api-gateway ve lambda log gruplarında" names log groups, not a tenant.
_SCOPE_AFTER_SLUG_RE = re.compile(
    r"^['\"]?\s+(?:ve\s+[\w\-\.]+\s+|and\s+[\w\-\.]+\s+)?(?:log|grub|grup|servis|service|pod)",
    re.I,
)


def _is_valid_tenant_slug(slug: str) -> bool:
    """Reject HTTP status codes and other non-tenant tokens."""
    if len(slug) < 3:
        return False
    if slug.isdigit():
        return False
    if slug in _NON_TENANT_TOKENS or re.fullmatch(r"[1-5](?:\d\d|xx)", slug):
        return False
    return True


def extract_tenant_filter(message: str) -> str | None:
    only = re.search(r"(?:sadece|only)\s+['\"]?([\w\-\.]+)", message, re.I)
    if only and _SCOPE_AFTER_SLUG_RE.match(message[only.end():]):
        message = message[: only.start()] + message[only.end():]
    patterns = (
        r"(?:sadece|only)\s+['\"]?([\w\-\.]+)['\"]?",
        r"['\"]?([\w\-\.]+)['\"]?\s+tenn?ant[ıi]na\s+ait",
        r"['\"]?([\w\-\.]+)['\"]?\s+(?:tenant|tennant|kirac[ıi]|müşteri)",
        r"(?:tenant|tennant|kirac[ıi])\s+['\"]?([\w\-\.]+)['\"]?",
        r"([\w\-\.]+)\s+tenn?ant[ıi]",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.I)
        if match:
            slug = normalize_tenant_slug(match.group(1))
            if _is_valid_tenant_slug(slug):
                return slug
    return None


def is_scoped_log_search_without_group(message: str) -> bool:
    """Tenant/status/analysis query without a full log group path."""
    if is_metric_analysis_request(message):
        return False
    if extract_log_group_paths(message):
        return False
    tenant = extract_tenant_filter(message)
    has_log_signal = bool(
        re.search(
            r"500|5\s*xx|4\s*xx|error|hata|status|log|detay|tenant|kirac",
            message,
            re.I,
        )
    )
    if tenant and has_log_signal:
        return True
    if is_log_analysis_request(message) and has_log_signal:
        return True
    return False


def is_log_analysis_request(message: str) -> bool:
    if is_metric_analysis_request(message):
        return False
    return bool(
        re.search(
            r"analiz\s*et|analyze|yorumla|sebeb|neden|kök\s*neden|root\s*cause|"
            r"değerlendir|incele|özetle\s+ve\s+analiz",
            message,
            re.I,
        )
    )


def _collect_log_rows_from_data(data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    default_group = str(data.get("log_group_name") or "")
    for item in data.get("results") or []:
        if isinstance(item, dict) and not any(
            key in item for key in ("log_groups", "metric_alarms", "error_patterns")
        ):
            rows.append(
                {
                    "timestamp": str(item.get("timestamp", "")),
                    "log_group": str(item.get("log_group") or default_group),
                    "log_stream": str(item.get("log_stream", "")),
                    "message": str(item.get("message", "")),
                    "matches": str(item.get("matches", "")),
                }
            )
        elif isinstance(item, list):
            fields = {
                str(field.get("field", "")): str(field.get("value", ""))
                for field in item
                if isinstance(field, dict)
            }
            rows.append(
                {
                    "timestamp": fields.get("@timestamp", ""),
                    "log_group": fields.get("@log", fields.get("@logGroup", default_group)),
                    "log_stream": fields.get("@logStream", ""),
                    "message": fields.get("@message", ""),
                    "matches": fields.get("matches", ""),
                }
            )
    return rows


def _parse_access_log_events(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("message", "")).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append(data)
    return events


_APP_MESSAGE_KEYS = ("body", "m", "message", "msg", "log", "errorMessage", "mt", "error")
_APP_LEVEL_KEYS = ("severity_text", "l", "level", "Level", "severity")
_PATTERN_SUBS = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<id>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?\b"), "<time>"),
    (re.compile(r"(https?://[^\s?\"']+)\?[^\s\"']*"), r"\1?…"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.I), "<hex>"),
    (re.compile(r"^\[[^\]]{3,60}\]\s*"), ""),
    (re.compile(r"\b\d+\b"), "<n>"),
)


def _app_log_fields(raw: str) -> tuple[str, str, dict[str, str]]:
    """(message, level, context) for JSON app logs (Serilog, OTel, Python) or plain text."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return raw.strip(), "", {}
    message = next((str(data[k]) for k in _APP_MESSAGE_KEYS if isinstance(data.get(k), str) and data[k].strip()), raw)
    level = next((str(data[k]) for k in _APP_LEVEL_KEYS if data.get(k)), "")
    resource = data.get("resource") if isinstance(data.get("resource"), dict) else {}
    context = {
        "service": str(resource.get("service.name") or data.get("service") or data.get("logger") or ""),
        "environment": str(resource.get("deployment.environment.name") or data.get("environment") or ""),
    }
    return message.strip(), level, {k: v for k, v in context.items() if v}


def message_pattern(message: str) -> str:
    """Collapse ids, numbers, timestamps and query strings so repeats of one error group together."""
    pattern = message.splitlines()[0] if message else ""
    for regex, replacement in _PATTERN_SUBS:
        pattern = regex.sub(replacement, pattern)
    return re.sub(r"\s+", " ", pattern).strip()[:160]


def format_app_log_analysis(rows: list[dict[str, str]]) -> list[str]:
    """Analysis lines for non-access logs: top message patterns, levels, services, environments."""
    patterns: dict[str, list[str]] = {}
    levels: dict[str, int] = {}
    services: dict[str, int] = {}
    environments: dict[str, int] = {}
    for row in rows:
        raw = str(row.get("message", "")).strip()
        if not raw:
            continue
        message, level, context = _app_log_fields(raw)
        patterns.setdefault(message_pattern(message), []).append(message)
        if level:
            levels[level.upper()] = levels.get(level.upper(), 0) + 1
        if context.get("service"):
            services[context["service"]] = services.get(context["service"], 0) + 1
        if context.get("environment"):
            environments[context["environment"]] = environments.get(context["environment"], 0) + 1
    total = sum(len(v) for v in patterns.values())
    if not total:
        return []

    def dist(counter: dict[str, int]) -> str:
        return ", ".join(f"**{k}**: {v}" for k, v in sorted(counter.items(), key=lambda x: -x[1])[:5])

    lines = ["### Bulgular (gerçek log verisi)", "", "**En sık hata kalıpları:**"]
    ranked = sorted(patterns.items(), key=lambda item: -len(item[1]))
    for pattern, examples in ranked[:6]:
        share = round(100 * len(examples) / total)
        lines.append(f"- **{len(examples)}×** (%{share}) `{pattern}`")
    if len(ranked) > 6:
        lines.append(f"- … ve {len(ranked) - 6} farklı kalıp daha")
    lines.append("")
    if levels:
        lines.append(f"**Seviye:** {dist(levels)}")
    if services:
        lines.append(f"**Servis:** {dist(services)}")
    if environments:
        lines.append(f"**Ortam:** {dist(environments)}")
        if any(env.lower() in {"local", "localhost", "dev-local"} for env in environments):
            lines.append(
                "_Kayıtların bir kısmı **local** ortamdan geliyor — bunlar geliştirici makinelerinden "
                "gönderilen istemci logları olabilir, sunucu tarafı bir sorun olmayabilir._"
            )
    lines.append("")
    top_pattern, top_examples = ranked[0]
    lines.extend(["### Öne çıkan hata", "", f"`{top_examples[0][:400]}`", ""])
    return lines


def _analysis_insight_for_error_type(err_type: str, count: int, total: int) -> str:
    share = int(round(100 * count / total)) if total else 0
    hints = {
        "AUTHORIZER_FAILURE": (
            "Authorizer/Lambda authorizer hatası veya geçersiz/eksik token olabilir."
        ),
        "ACCESS_DENIED": (
            "IAM veya resource policy tarafında explicit deny (yetki/policy kontrolü)."
        ),
        "INTEGRATION_TIMEOUT": "Backend/integration zaman aşımı — downstream servis yavaş veya erişilemiyor.",
        "INTEGRATION_FAILURE": "Backend/integration çağrısı başarısız — downstream hata dönüyor olabilir.",
    }
    hint = hints.get(err_type, "")
    base = f"**{err_type}**: {count} kayıt (%{share})"
    return f"{base} — {hint}" if hint else base


def _build_root_cause_summary(
    events: list[dict[str, Any]],
    *,
    by_error_type: dict[str, int],
    by_path: dict[str, int],
    by_integration: dict[str, int],
) -> list[str]:
    """Heuristic root-cause hints from aggregated access-log fields."""
    if not events:
        return []

    total = len(events)
    hints: list[str] = []

    slow = [
        event
        for event in events
        if int(str(event.get("responseLatency") or "0").split(".")[0] or 0) >= 3000
    ]
    if slow:
        hints.append(
            f"- **Yüksek gecikme:** {len(slow)}/{total} istek ≥3s — integration/backend timeout "
            "veya yavaş downstream servis olası."
        )

    if by_path:
        top_path, top_count = max(by_path.items(), key=lambda item: (item[1], item[0]))
        if top_count >= 2 and top_count / total >= 0.25:
            hints.append(
                f"- **Tekrarlayan endpoint:** `{top_path}` → **{top_count}** istek "
                f"(toplamın %{int(round(100 * top_count / total))}) — bu servis loglarına odaklan."
            )

    if by_integration:
        bad = {
            code: count
            for code, count in by_integration.items()
            if str(code).isdigit() and int(code) >= 500
        }
        if bad:
            codes = ", ".join(f"**{code}** ({count})" for code, count in sorted(bad.items()))
            hints.append(f"- **integrationStatus:** {codes} — backend doğrudan hata dönüyor.")

    for err_type, count in sorted(by_error_type.items(), key=lambda item: (-item[1], item[0])):
        if err_type == "INTEGRATION_FAILURE":
            hints.append(
                f"- **INTEGRATION_FAILURE** ({count}): Backend/integration çağrısı başarısız — "
                "EKS/container veya downstream API loglarını kontrol et."
            )
        elif err_type == "INTEGRATION_TIMEOUT":
            hints.append(
                f"- **INTEGRATION_TIMEOUT** ({count}): Backend zaman aşımı — timeout/latency ve "
                "connection pool ayarlarını incele."
            )
        elif err_type in {"AUTHORIZER_FAILURE", "ACCESS_DENIED"}:
            hints.append(
                f"- **{err_type}** ({count}): Yetkilendirme/policy — token, IAM veya authorizer "
                "kaynaklı; HTTP 500 görünse de backend hatası olmayabilir."
            )

    tenants: dict[str, int] = {}
    for event in events:
        tenant = str(event.get("tenantDomain") or "").strip()
        if tenant and tenant != "-":
            tenants[tenant] = tenants.get(tenant, 0) + 1
    if len(tenants) == 1:
        only_tenant, tenant_count = next(iter(tenants.items()))
        hints.append(
            f"- **Tenant:** tüm örnekler `{only_tenant}` üzerinde ({tenant_count} kayıt) — "
            "tenant-spesifik veri veya konfigürasyon sorunu olabilir."
        )

    return hints


def format_log_analysis_report(data: dict[str, Any]) -> str:
    """Data-backed log analysis — aggregates only from real matched lines."""
    if data.get("error"):
        return f"Log analizi başarısız: {data['error']}"

    rows = _collect_log_rows_from_data(data)
    events = _parse_access_log_events(rows)
    hours = int(data.get("hours") or 1)
    groups = data.get("all_log_group_names") or data.get("log_group_names") or []
    if data.get("log_group_name"):
        group = str(data["log_group_name"])
    elif len(groups) == 1:
        group = str(groups[0])
    elif groups:
        sample = ", ".join(f"`{g}`" for g in groups[:3])
        extra = f" +{len(groups) - 3}" if len(groups) > 3 else ""
        group = f"{len(groups)} log grubu ({sample}{extra})"
    else:
        group = "seçili log grupları"
    tenant = str(data.get("tenant_filter") or "").strip()
    filter_name = format_filter_display_name(str(data.get("search_filter") or "errors"))
    match_count = int(data.get("match_count") or len(rows))
    window_label = str(data.get("window_label") or "").strip()
    window_note = ""
    if data.get("start_time") and data.get("end_time"):
        window_note = f" ({data['start_time']} → {data['end_time']})"

    tenant_note = f" · tenant: **{tenant}**" if tenant else ""
    lines = [
        f"## Log analizi — `{group}`",
        f"**Dönem:** {f'**{window_label}**' if window_label else f'son **{hours} saat**'} · "
        f"**filtre:** {filter_name}{tenant_note}{window_note}",
        f"**Toplam eşleşme:** **{match_count}** kayıt (analiz örneği: **{len(events)}** satır)",
        "",
    ]

    is_access_log = any(
        isinstance(event, dict) and (event.get("httpMethod") or event.get("status") or event.get("path"))
        for event in events
    )
    if not is_access_log:
        app_lines = format_app_log_analysis(rows)
        if not app_lines:
            lines.append("Bu filtreyle eşleşen kayıt bulunamadı.")
            return "\n".join(lines)
        lines[2] = f"**Toplam eşleşme:** **{match_count}** kayıt (analiz örneği: **{len(rows)}** satır)"
        lines.extend(app_lines)
        if match_count > len(rows):
            lines.append(f"_Analiz en yeni {len(rows)} kayıt üzerinden yapıldı._")
        return "\n".join(lines)

    by_error_type: dict[str, int] = {}
    by_path: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_method: dict[str, int] = {}
    by_integration: dict[str, int] = {}
    error_messages: dict[str, int] = {}

    for event in events:
        err_type = str(event.get("errorResponseType") or "").strip()
        if _meaningful_log_field_value(err_type):
            by_error_type[err_type] = by_error_type.get(err_type, 0) + 1
        path = str(event.get("path") or event.get("resourcePath") or "").strip()
        if path:
            by_path[path] = by_path.get(path, 0) + 1
        status = str(event.get("status") or "").strip()
        if status:
            by_status[status] = by_status.get(status, 0) + 1
        method = str(event.get("httpMethod") or "").strip()
        if method:
            by_method[method] = by_method.get(method, 0) + 1
        integration = str(event.get("integrationStatus") or "").strip()
        if _meaningful_log_field_value(integration):
            by_integration[integration] = by_integration.get(integration, 0) + 1
        err_msg = str(event.get("errorMessage") or "").strip()
        if _meaningful_log_field_value(err_msg):
            error_messages[err_msg] = error_messages.get(err_msg, 0) + 1

    sample_n = len(events)
    lines.extend(["### Bulgular (gerçek log verisi)", ""])

    if by_error_type:
        lines.append("**errorResponseType dağılımı:**")
        for err_type, count in sorted(by_error_type.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {_analysis_insight_for_error_type(err_type, count, sample_n)}")
        lines.append("")

    if by_path:
        lines.append("**En çok etkilenen path'ler:**")
        for path, count in sorted(by_path.items(), key=lambda x: (-x[1], x[0]))[:8]:
            lines.append(f"- `{path}`: **{count}** istek")
        lines.append("")

    if by_status:
        status_line = ", ".join(f"**{code}**: {count}" for code, count in sorted(by_status.items()))
        lines.append(f"**HTTP status:** {status_line}")
        lines.append("")

    if by_integration:
        int_line = ", ".join(
            f"**{code}**: {count}" for code, count in sorted(by_integration.items(), key=lambda x: -x[1])
        )
        lines.append(f"**integrationStatus:** {int_line}")
        lines.append("")

    if error_messages:
        lines.append("**Öne çıkan errorMessage:**")
        for msg, count in sorted(error_messages.items(), key=lambda x: (-x[1], x[0]))[:5]:
            lines.append(f"- ({count}×) {msg[:160]}")
        lines.append("")

    root_causes = _build_root_cause_summary(
        events,
        by_error_type=by_error_type,
        by_path=by_path,
        by_integration=by_integration,
    )
    if root_causes:
        lines.extend(["### Muhtemel kök nedenler", ""])
        lines.extend(root_causes)
        lines.append("")

    lines.extend(["### Örnek kayıtlar (CloudWatch alan görünümü)", ""])
    for index, row in enumerate(rows[:5], 1):
        block = format_access_log_cloudwatch_block(
            str(row.get("message", "")),
            timestamp=str(row.get("timestamp", "")),
            log_group=_srv().normalize_insights_log_group(str(row.get("log_group") or group)),
            log_stream=str(row.get("log_stream", "")),
        )
        lines.extend([f"**{index}.**", "```", block, "```", ""])

    if match_count > 5:
        lines.append(
            f"Detaylı satır listesi için **\"detaylı göster\"** yazabilirsin "
            f"({match_count - 5} kayıt daha)."
        )

    if str(data.get("search_filter") or "") == "http_500":
        lines.append(
            "\n_Not: `AUTHORIZER_FAILURE` / `ACCESS_DENIED` kayıtları HTTP 500 olarak görünebilir; "
            "backend 500 için \"integration 500\" veya `http_500_backend` filtresi kullan._"
        )

    return "\n".join(line for line in lines if line is not None)


def format_log_search_results(data: dict[str, Any]) -> str:
    """Format Insights search hits with real log group names only."""
    if data.get("error"):
        return f"Log araması başarısız: {data['error']}"

    hours = data.get("hours", 1)
    match_count = int(data.get("match_count") or 0)
    groups_searched = int(data.get("log_groups_searched") or 0)
    status = data.get("status", "Unknown")
    period_label = (data.get("period_label") or "").strip()
    window_label = (data.get("window_label") or "").strip()
    window = window_label or f"son {hours} saat"
    scope_line = format_search_scope(data)
    window_note = ""
    if data.get("start_time") and data.get("end_time"):
        window_note = f" ({data['start_time']} → {data['end_time']})"

    rows = _collect_log_rows_from_data(data)

    all_groups = data.get("all_log_group_names") or data.get("log_group_names") or []
    if data.get("log_group_name"):
        all_groups = [str(data["log_group_name"])] + [g for g in all_groups if g != data["log_group_name"]]

    prefix = f"**{period_label}** — " if period_label and period_label != window_label else ""
    response_mode = data.get("response_mode") or (
        "summary" if data.get("rank_by_log_group") else "detail"
    )
    if response_mode == "analysis":
        return format_log_analysis_report(data)

    filter_name = data.get("search_filter") or "arama"
    if filter_name == "http_500_backend":
        filter_name = "http_500 (backend — auth hariç)"
    total_matches = data.get("total_matches", match_count)

    def _ranked_rows() -> list[tuple[str, int]]:
        ranked = [
            (
                _srv().normalize_insights_log_group((row.get("log_group") or "").strip()),
                int(float(row.get("matches") or 0)),
            )
            for row in rows
            if (row.get("log_group") or "").strip()
        ]
        ranked = [(g, c) for g, c in ranked if g]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    if response_mode in {"summary", "count_only"}:
        ranked = _ranked_rows()
        if not ranked:
            scope = (
                f"istenen gruplar: {', '.join(f'`{n}`' for n in data.get('requested_log_group_names') or [])}"
                if data.get("requested_log_group_names")
                else f"{groups_searched} log grubu"
            )
            scope_suffix = f"\n{scope_line}" if scope_line else ""
            return (
                f"{prefix}**{filter_name} — sayım** ({window}, {scope}) — "
                f"eşleşme bulunamadı (durum: {status}).{window_note}{scope_suffix}"
            )
        if data.get("log_group_name") and len(ranked) <= 1:
            group_name = str(data["log_group_name"])
            count = ranked[0][1] if ranked else 0
            return format_single_group_summary_reply(
                group_name=group_name,
                hours=hours,
                filter_name=filter_name,
                count=count,
                status=status,
                window_note=window_note,
                prefix=prefix,
                tenant_filter=str(data.get("tenant_filter") or "") or None,
                window_label=window_label or None,
                status_breakdown=data.get("status_breakdown"),
            )

        title = "Sistem özeti" if response_mode == "summary" else "Seçili gruplar — sayım"
        groups_with_matches = max(len(ranked), int(data.get("groups_with_matches") or 0))
        lines = [
            f"{prefix}**{title} — {filter_name}** ({window}, durum: {status}){window_note}",
            f"**{groups_with_matches} log grubunda** eşleşme var · **Toplam: {total_matches}** kayıt",
            scope_line,
            "",
            "| # | Log group | Adet |",
            "|---|-----------|------|",
        ]
        for index, (log_group, count) in enumerate(ranked, 1):
            lines.append(f"| {index} | `{log_group}` | **{count}** |")
        result = "\n".join(line for line in lines if line)
        if groups_with_matches > len(ranked):
            result += (
                f"\n\n_İlk {len(ranked)} grup gösteriliyor; {groups_with_matches - len(ranked)} grupta "
                "daha az sayıda eşleşme var (toplama dahil)._"
            )
        return result

    if data.get("rank_by_log_group"):
        ranked = _ranked_rows()
        if not ranked:
            return (
                f"{prefix}Son {hours} saat içinde {groups_searched} log grubunda sıralama yapıldı — "
                f"eşleşme bulunamadı (durum: {status}).{window_note}"
            )
        lines = [
            f"{prefix}**Sıralama — {filter_name}** (son {hours} saat, {groups_searched} log grubu, "
            f"durum: {status}){window_note}",
            f"**Toplam: {total_matches}** eşleşme",
            "",
            "**Log gruplarına göre (gerçek tool verisi):**",
        ]
        for index, (log_group, count) in enumerate(ranked, 1):
            lines.append(f"{index}. `{log_group}`: **{count}** eşleşme")
        return "\n".join(lines)

    if not rows:
        sample = ", ".join(f"`{name}`" for name in all_groups[:12])
        suffix = f"\n\nLog group: {sample}" if sample else ""
        if data.get("log_group_name"):
            return (
                f"{prefix}`{data['log_group_name']}` — {window} içinde eşleşen satır bulunamadı "
                f"(durum: {status}).{window_note}{suffix}"
            )
        if scope_line:
            suffix = f"\n\n{scope_line}"
        return (
            f"{prefix}{window[0].upper() + window[1:]} içinde {groups_searched} log grubunda arama yapıldı — "
            f"eşleşen satır bulunamadı (durum: {status}).{window_note}{suffix}"
        )

    by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        log_group = _srv().normalize_insights_log_group(
            (row.get("log_group") or "").strip() or "log-group-bilinmiyor"
        )
        by_group.setdefault(log_group, []).append(row)

    lines = []
    display_filter = format_filter_display_name(filter_name)
    default_group = str(data.get("log_group_name") or "")
    window_md = f"**{window_label}**" if window_label else f"son **{hours} saat**"
    rows_returned = len(rows)
    preview_note = (
        f" · en yeni **{min(rows_returned, _MAX_RESULT_LINES)}** satır getirildi"
        if match_count > rows_returned
        else ""
    )
    if data.get("log_group_name"):
        lines.append(
            f"{prefix}**`{data['log_group_name']}`** — {window_md} · "
            f"**{display_filter}** · **{match_count}** kayıt{preview_note}{window_note}"
        )
        lines.append("")
        lines.append("**Kayıtlar (CloudWatch alan görünümü):**")
    else:
        lines.append(
            f"{prefix}**Detay — {display_filter}** · {window} · **{match_count}** kayıt{preview_note} "
            f"({groups_searched} log grubu, durum: {status}){window_note}"
        )

    error_breakdown = summarize_access_log_error_types(rows)
    if error_breakdown and len(error_breakdown) > 1:
        breakdown = ", ".join(f"**{name}**: {count}" for name, count in error_breakdown[:6])
        lines.extend(["", f"**Tür dağılımı:** {breakdown}"])

    if len(by_group) > 1:
        lines.extend(["", "**Özet:**"])
        for log_group, group_rows in sorted(
            by_group.items(), key=lambda item: (-len(item[1]), item[0])
        ):
            lines.append(f"- `{log_group}`: {len(group_rows)} satır")

    display_limit = min(
        _MAX_RESULT_LINES,
        max(1, int(data.get("max_result_lines") or _DETAIL_PREVIEW_LINES)),
    )
    if (
        match_count > _DETAIL_PREVIEW_LINES
        and display_limit <= _DETAIL_PREVIEW_LINES
        and not data.get("show_all")
    ):
        display_limit = _DETAIL_PREVIEW_LINES
    shown = 0
    for log_group, group_rows in sorted(
        by_group.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if len(by_group) > 1:
            lines.extend(["", f"### `{log_group}` ({len(group_rows)} satır)", ""])
        for row in group_rows:
            if shown >= display_limit:
                break
            message = str(row.get("message", "")).strip()
            if not message:
                continue
            shown += 1
            timestamp = row.get("timestamp", "")
            log_group = _srv().normalize_insights_log_group(
                str(row.get("log_group") or default_group)
            )
            block = format_access_log_cloudwatch_block(
                message,
                timestamp=timestamp,
                log_group=log_group,
                log_stream=str(row.get("log_stream", "")),
            )
            if len(by_group) == 1:
                lines.extend([f"", f"**{shown}.**", "```", block, "```"])
            else:
                lines.extend([f"", f"**{timestamp}** — `{log_group}`", "```", block, "```"])
        if shown >= display_limit:
            break

    if data.get("search_filter") == "http_500" and rows:
        lines.append(
            "\n_Not: API Gateway access log'larında `AUTHORIZER_FAILURE` veya `ACCESS_DENIED` "
            "kayıtları da HTTP **status 500** dönebilir (authorizer hatası, backend 500 değil). "
            "Sadece backend/integration 500 istiyorsan \"integration 500\" veya "
            "\"backend 500\" de._"
        )

    if data.get("truncated") or match_count > shown:
        remaining = max(0, match_count - shown)
        if remaining and match_count <= _MAX_RESULT_LINES:
            lines.append(
                f"\n... ve **{remaining}** kayıt daha. "
                f"**\"tamamını göster\"** veya **\"{remaining} ini getir\"** yazabilirsin."
            )
        elif remaining:
            lines.append(
                f"\n... ve **{remaining}** kayıt daha. Tek seferde en fazla {_MAX_RESULT_LINES} satır "
                "gösterebiliyorum — zaman aralığını daraltabilir, tenant filtresi verebilir veya "
                "**\"logları analiz et\"** diyerek kök neden özetini alabilirsin."
            )

    if scope_line and data.get("scope") == "system":
        lines.append(f"\n{scope_line}")
    elif all_groups:
        lines.append(f"\n**Aranan log grupları ({len(all_groups)}):**")
        for name in all_groups[:40]:
            lines.append(f"- `{_srv().normalize_insights_log_group(name)}`")
        if len(all_groups) > 40:
            lines.append(f"- ... ve {len(all_groups) - 40} grup daha")

    return "\n".join(lines)


def try_direct_tool_response(tool_name: str, tool_result: str) -> str | None:
    """Format tool output on the server — real AWS data only, no LLM synthesis."""
    try:
        parsed = json.loads(tool_result)
    except json.JSONDecodeError:
        parsed = None

    if tool_name in IMMEDIATE_FORMAT_TOOLS and isinstance(parsed, dict):
        return format_log_search_results(parsed)

    data = _srv().parse_log_groups_tool_result(tool_result)
    if data is None:
        data = parsed if isinstance(parsed, dict) else None
    if not data:
        return None
    if tool_name == "describe_log_groups" and "log_groups" in data:
        return _srv().format_log_groups_list(data)
    if tool_name == "get_active_alarms":
        return format_active_alarms_list(data)
    if tool_name == "get_alarm_history":
        return format_alarm_history(data)
    if tool_name == "analyze_metric":
        return format_analyze_metric(data)
    return None


def is_comparison_request(message: str) -> bool:
    return bool(
        re.search(
            r"\bvs\.?\b|karşılaştır|compare|versus|"
            r"bugün.*dün|dün.*bugün|this\s+week.*last\s+week|geçen\s+hafta.*bu\s+hafta",
            message,
            re.I,
        )
    )


def is_log_search_intent(message: str) -> bool:
    return bool(
        re.search(
            r"500|5\s*xx|4\s*xx|2\s*xx|error|hata|sayı|count|kaç\s+tane|"
            r"en\s+çok|sıralama|detaylı\s+getir|satır\s+satır|http",
            message,
            re.I,
        )
    )


def is_cross_group_log_search(message: str) -> bool:
    """User wants counts/ranking across log groups — not single-group drill-down."""
    # gru[bp] covers Turkish inflections: grubu, grupları, gruplarımda...
    if re.search(
        r"kaç\s+tane\s+log\s+gru[bp]|kaç\s+log\s+gru[bp]|"
        r"hangi\s+log\s+gru[bp]|tüm\s+log\s+gru[bp]|sistemde|"
        r"how\s+many\s+log\s+groups|which\s+log\s+group",
        message,
        re.I,
    ):
        return True
    if not extract_log_group_paths(message) and is_http_status_search(message):
        return bool(re.search(r"kaç|sayı|count|hangi|tüm|sistemde|en\s+çok", message, re.I))
    return False


def is_single_named_group_log_query(message: str) -> bool:
    """User named one log group path and wants content/counts in that group only."""
    paths = extract_log_group_paths(message)
    if len(paths) != 1 or is_cross_group_log_search(message):
        return False
    return is_named_log_content_query(message) or bool(
        re.search(r"kaç\s+hata|kaç\s+error|hata\s+alm|error\s+count", message, re.I)
    )


def is_log_catalog_intent(message: str) -> bool:
    if is_log_search_intent(message) and re.search(
        r"sayı|count|500|5\s*xx|error|hata|detay", message, re.I
    ):
        return False
    # gru[bp]\w* covers Turkish possessives: grubu, grupları, gruplarımı, gruplarımızı...
    return bool(
        re.search(
            r"log\s*gru[bp]\w*\s*(listele|göster)|"
            r"listele.*log\s*gr|hangi\s+log\s*gruplar|which\s+log\s+groups",
            message,
            re.I,
        )
    )


def is_vague_log_question(message: str) -> bool:
    stripped = message.strip()
    if len(stripped) > 40:
        return False
    return bool(re.search(r"^(500|5\s*xx|4\s*xx|hata|error)\s+var\s+m[ıi]\??$", stripped, re.I))


_SCOPE_STOPWORDS = {
    "son", "tüm", "tum", "bütün", "butun", "sistem", "sistemdeki", "sistemin", "access", "api",
    "bu", "şu", "hangi", "error", "errors", "hata", "hataları", "http", "the", "all", "uygulama",
    "uygulamanın", "app", "application", "gateway", "aws", "cloudwatch", "insights", "detaylı",
    "en", "çok", "her", "kaç", "o", "bir", "dev", "prod", "lambda", "ecs", "eks", "log", "logs",
    "veren", "eden", "olan", "alan", "gelen", "çalışan", "hangisi", "ilgili", "diğer", "bazı",
    "which", "what", "my", "any", "these", "those", "this", "that", "show", "give", "last", "recent",
    "error", "failed", "most",
}
_HYPHENATED_NAME_RE = re.compile(r"(?<![/\w-])([a-z0-9]+(?:-[a-z0-9]+)+)(?![/\w-])")
_NAMED_SCOPE_RE = re.compile(
    r"(?<![/\w-])([a-z0-9][a-z0-9_-]{2,})\s+(?:servis\w*|service\w*|log\w*|grub\w*|grup\w*|pod\w*)",
    re.I,
)


def infer_keywords_from_message(message: str) -> list[str]:
    lowered = message.lower()
    mapping = [
        (r"api[- ]?gateway", "api-gateway"),
        (r"\blambda\b", "lambda"),
        (r"\becs\b|container\s*insights", "containerinsights"),
        (r"\balb\b|load\s*balancer", "alb"),
        (r"cloudfront", "cloudfront"),
        (r"codebuild", "codebuild"),
        (r"\beks\b|kubernetes|\bk8s\b", "containerinsights"),
    ]
    found: list[str] = []
    for pattern, keyword in mapping:
        if re.search(pattern, lowered):
            found.append(keyword)

    # Service names the user typed ("plt-comm-order servisinde", "debezium loglarında") must
    # scope the search; otherwise an LLM that drops them returns an unrelated system-wide table.
    path_text = " ".join(extract_log_group_paths(message)).lower()
    candidates = _HYPHENATED_NAME_RE.findall(lowered) + _NAMED_SCOPE_RE.findall(lowered)
    for token in candidates:
        token = token.strip("-_")
        if (
            len(token) < 3
            or token in _SCOPE_STOPWORDS
            or token.isdigit()
            or re.fullmatch(r"\d+[a-z]{0,3}", token)
            or token in path_text
            or token in {"api-gateway", "api-gw"}
        ):
            continue
        if token not in found:
            found.append(token)
    return found


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def time_window_args(window: TimeWindow | None) -> dict[str, Any]:
    """Tool time arguments for a parsed window (rolling hours, minutes, or fixed calendar range)."""
    if window is None:
        return {"hours": 1}
    if window.is_fixed:
        return {
            "hours": window.hours,
            "start_time": _iso(window.start),
            "end_time": _iso(window.end),
            "period_label": window.label,
        }
    if window.is_sub_hour_precision:
        return {"hours": window.hours, "minutes": window.minutes}
    return {"hours": window.hours}


_PERIOD_LABEL_CALENDAR = (
    (re.compile(r"geçen\s+hafta|last\s+week", re.I), "last_week"),
    (re.compile(r"bu\s+hafta|this\s+week", re.I), "this_week"),
    (re.compile(r"dün|dun\b|yesterday", re.I), "yesterday"),
    (re.compile(r"bugün|bugun|today", re.I), "today"),
)


def apply_time_window_args(message: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Make the tool window match the user's words; the LLM's own guess only fills gaps."""
    args = dict(arguments)
    label = str(args.get("period_label") or "")
    for pattern, name in _PERIOD_LABEL_CALENDAR:
        if label and pattern.search(label):
            # Comparison legs ("Bugün" / "Dün"): exact TR-calendar boundaries, not LLM-guessed UTC.
            window = calendar_window(name)
            args.update(time_window_args(window))
            args["period_label"] = label
            args.pop("minutes", None)
            return args
    if is_comparison_request(message):
        return args

    window = parse_time_window(message)
    if window is None:
        if args.get("start_time") and args.get("end_time"):
            return args
        if not args.get("hours"):
            args["hours"] = 1
        return args
    for key in ("start_time", "end_time", "minutes"):
        args.pop(key, None)
    if not window.is_fixed:
        args.pop("period_label", None)
    args.update(time_window_args(window))
    return args


def adjust_search_logs_args(message: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Enrich LLM-chosen search args from clear user phrasing (not tool routing)."""
    args = dict(arguments)
    if is_cross_group_log_search(message) and not extract_log_group_paths(message):
        args.pop("log_group_name", None)
        args.pop("log_group_names", None)
        if not infer_keywords_from_message(message):
            args.pop("log_group_name_keywords", None)

    if is_log_analysis_request(message):
        args["response_mode"] = "analysis"
    elif re.search(r"detay\s*istemiyorum|sadece\s+say[ıi]|count\s+only", message, re.I):
        # Explicit user wording wins over the model's choice.
        args["response_mode"] = "count_only"
    elif not args.get("response_mode"):
        if re.search(r"kaç\s+tane\s+log\s+grub|kaç\s+log\s+grub", message, re.I):
            args["response_mode"] = "summary"
        elif re.search(
            r"sistemde|tüm\s+log|sayıları\s+göster|en\s+çok|hangi\s+log\s+gru|top\s+\d+",
            message,
            re.I,
        ):
            args["response_mode"] = "summary"
        elif re.search(r"detaylı|satır\s+satır|full\s+detail", message, re.I):
            args["response_mode"] = "detail"
        elif is_cross_group_log_search(message):
            args["response_mode"] = "summary"

    tenant = extract_tenant_filter(message)
    if tenant:
        args["tenant_filter"] = tenant

    if not args.get("search_filter"):
        if re.search(r"\b500\b|500\s*l", message, re.I):
            args["search_filter"] = "http_500"
        elif re.search(r"2\s*xx|başarılı|successful", message, re.I):
            args["search_filter"] = "http_2xx"
        elif re.search(r"5\s*xx", message, re.I):
            args["search_filter"] = "http_5xx"
        elif re.search(r"\b400\b|4\s*xx", message, re.I):
            args["search_filter"] = "http_400"
        elif re.search(r"error|hata", message, re.I):
            args["search_filter"] = "errors"

    named_paths = extract_log_group_paths(message)
    if named_paths and not is_cross_group_log_search(message):
        args["log_group_names"] = named_paths
        args.pop("log_group_name_keywords", None)
        args.pop("log_group_name", None)
    elif not args.get("log_group_names"):
        keywords = infer_keywords_from_message(message)
        if keywords and not args.get("log_group_name_keywords"):
            args["log_group_name_keywords"] = keywords

    args = apply_time_window_args(message, args)

    max_lines = int(args.get("max_result_lines") or 50)
    args["max_result_lines"] = min(200, max(50, max_lines))
    if args.get("response_mode") == "detail":
        args["max_result_lines"] = max(args["max_result_lines"], 150)
    return args


def adjust_query_log_group_args(message: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep single-group queries scoped to the current message, not chat history."""
    args = dict(arguments)
    paths = extract_log_group_paths(message)
    if paths:
        args["log_group_name"] = paths[0]

    inferred = infer_log_search_filter(message)
    current = args.get("search_filter")
    if not current or (current == "errors" and inferred != "errors"):
        args["search_filter"] = inferred

    if is_log_analysis_request(message):
        args["response_mode"] = "analysis"
    elif wants_log_detail(message):
        args["response_mode"] = "detail"
    elif not args.get("response_mode"):
        if re.search(r"var\s*m[ıi]|sayı|kaç|count|özet|summary", message, re.I):
            args["response_mode"] = "summary"
        else:
            args["response_mode"] = "detail"

    tenant = extract_tenant_filter(message)
    if tenant:
        args["tenant_filter"] = tenant

    args = apply_time_window_args(message, args)

    max_lines = int(args.get("max_result_lines") or 50)
    if args.get("response_mode") == "analysis":
        args["max_result_lines"] = min(200, max(100, max_lines))
    elif args.get("response_mode") == "detail":
        if re.search(r"tümünü|hepsini|all\s+lines", message, re.I):
            args["max_result_lines"] = 200
        else:
            args["max_result_lines"] = min(200, max(_DETAIL_PREVIEW_LINES, max_lines))
    else:
        args["max_result_lines"] = min(200, max(50, max_lines))
    if args.get("response_mode") == "detail" and not re.search(
        r"tümünü|hepsini|all\s+lines", message, re.I
    ):
        args["max_result_lines"] = _DETAIL_PREVIEW_LINES
    return args


def adjust_describe_log_groups_args(
    user_message: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Clamp limits and infer keywords when user named a service type."""
    args = dict(arguments)
    requested = int(args.get("max_items") or 50)
    args["max_items"] = min(MAX_LOG_GROUPS_LIST, max(1, requested))

    raw_keywords = args.get("log_group_name_keywords")
    if isinstance(raw_keywords, list):
        args["log_group_name_keywords"] = _srv().dedupe_search_keywords(raw_keywords)
    elif not args.get("log_group_name_prefix"):
        inferred = infer_keywords_from_message(user_message)
        if inferred:
            args["log_group_name_keywords"] = _srv().dedupe_search_keywords(inferred)
            args["max_items"] = min(MAX_LOG_GROUPS_LIST, max(args["max_items"], 200))

    return args


def normalize_tool_call(
    message: str,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    keep_llm_args: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Fix the model's tool choice/arguments against the user's exact words.

    keep_llm_args=True (llm_first) carries the model's arguments through redirects, so values it
    resolved from conversation context (group from "2. grup", hours from the previous turn) survive.
    """
    carried = dict(arguments) if keep_llm_args else {}
    if tool_name == "describe_log_groups":
        if is_log_search_intent(message) and not is_log_catalog_intent(message):
            logger.warning(
                "Redirecting describe_log_groups -> search_logs_across_groups (log search intent): %s",
                message[:120],
            )
            tool_name = "search_logs_across_groups"
            keywords = arguments.get("log_group_name_keywords") or (
                [arguments["log_group_name_prefix"]] if arguments.get("log_group_name_prefix") else None
            )
            arguments = adjust_search_logs_args(
                message, {"log_group_name_keywords": keywords} if keep_llm_args and keywords else {}
            )
        else:
            arguments = adjust_describe_log_groups_args(message, arguments)
            if arguments.get("log_group_name_prefix"):
                arguments["log_group_name_prefix"] = _srv().normalize_log_group_prefix_arg(
                    arguments["log_group_name_prefix"]
                )

    if tool_name == "query_log_group":
        if is_cross_group_log_search(message):
            logger.warning(
                "Redirecting query_log_group -> search_logs_across_groups (cross-group intent): %s",
                message[:120],
            )
            tool_name = "search_logs_across_groups"
            carried.pop("log_group_name", None)
            arguments = adjust_search_logs_args(message, carried)
        else:
            arguments = adjust_query_log_group_args(message, arguments)

    if tool_name == "search_logs_across_groups":
        if is_single_named_group_log_query(message):
            logger.warning(
                "Redirecting search_logs_across_groups -> query_log_group (single named group): %s",
                message[:120],
            )
            tool_name = "query_log_group"
            for key in ("log_group_names", "log_group_name_keywords"):
                carried.pop(key, None)
            arguments = adjust_query_log_group_args(message, carried)
        else:
            arguments = adjust_search_logs_args(message, arguments)
    return tool_name, arguments


def summarize_message_for_history(content: str) -> str:
    """Shrink huge list replies so the next turn is not polluted."""
    if re.search(r"^\*\*\d+\s+log grubu\*\*", content, re.M):
        count = re.search(r"\*\*(\d+)\s+log grubu\*\*", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} log grubu listelendi. Sonraki soru bağımsız değerlendir.]"
    summary_match = _CONVERSATIONAL_SUMMARY_RE.search(content)
    if summary_match:
        return (
            f"[Önceki: `{summary_match.group('group')}` son {summary_match.group('hours')} saat "
            f"{summary_match.group('filter')}: {summary_match.group('count')} eşleşme. "
            "Detay için aynı grubu kullan.]"
        )
    legacy_summary = _SINGLE_GROUP_SUMMARY_RE.search(content)
    if legacy_summary:
        return (
            f"[Önceki: `{legacy_summary.group('group')}` son {legacy_summary.group('hours')} saat "
            f"{legacy_summary.group('filter')}: {legacy_summary.group('count')} eşleşme. "
            "Detay için aynı grubu kullan.]"
        )
    if re.search(r"^\*\*\d+\s+aktif alarm\*\*", content, re.M):
        count = re.search(r"\*\*(\d+)\s+aktif alarm\*\*", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} aktif alarm listelendi.]"
    if re.search(r"^\*\*Son \d+ saat — \d+ eşleşen satır\*\*", content, re.M):
        count = re.search(r"— (\d+) eşleşen satır", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} log satırı arandı ve gerçek gruplarla listelendi.]"
    if re.search(r"\*\*Sistem özeti", content, re.M):
        ranked = _extract_ranked_groups_from_markdown(content)
        if len(ranked) == 1:
            group, count = ranked[0]
            hours, filter_name = _extract_hours_and_filter_from_assistant(content)
            return (
                f"[Önceki: `{group}` son {hours} saat {filter_name}: {count} eşleşme. "
                "Detay için aynı grubu kullan.]"
            )
        groups = re.search(r"·\s*(\d+)\s+log grubunda", content)
        matches = re.search(r"\*\*Toplam:\s*(\d+)\*\*", content)
        g = groups.group(1) if groups else "?"
        m = matches.group(1) if matches else "?"
        if ranked:
            sample = ", ".join(f"`{g}`" for g, _ in ranked[:5])
            return (
                f"[Önceki: {m} eşleşme, {len(ranked)} log grubu: {sample}. "
                "Detay için grup adını belirt.]"
            )
        return (
            f"[Önceki: {m} eşleşme, {g} log grubunda özetlendi. "
            "Sonraki soru bağımsız değerlendir.]"
        )
    detail_header = _DETAIL_HEADER_RE.search(content)
    if detail_header:
        hours, filter_name = _extract_hours_and_filter_from_assistant(content)
        return (
            f"[Önceki: `{detail_header.group('group')}` son {detail_header.group('hours') or hours} saat "
            f"{filter_name} detay gösterildi. Tekrar detay için aynı grubu kullan.]"
        )
    prior_group = _PRIOR_GROUP_LINE_RE.search(content)
    if prior_group:
        hours, filter_name = _extract_hours_and_filter_from_assistant(content)
        return (
            f"[Önceki: `{prior_group.group('group')}` son {hours} saat {filter_name} kapsam sorgusu. "
            "Detay için aynı grubu kullan.]"
        )
    only_group = re.search(r"yalnızca\*?\*?\s*`([^`]+)`", content, re.I)
    if only_group:
        hours, filter_name = _extract_hours_and_filter_from_assistant(content)
        return (
            f"[Önceki: yalnızca `{only_group.group(1)}` son {hours} saat {filter_name}. "
            "Detay için aynı grubu kullan.]"
        )
    return trim_message_content(content)


def compact_tool_result_for_context(tool_name: str, tool_result: str) -> str:
    """Keep LLM context small after tool calls (mechanical, not intent-based)."""
    srv = _srv()
    if tool_name == "describe_log_groups":
        data = srv.parse_log_groups_tool_result(tool_result)
        if data:
            names = [
                group.get("logGroupName", "")
                for group in data.get("log_groups", [])
                if group.get("logGroupName")
            ]
            payload = {
                "count": len(names),
                "log_group_names": names[: srv.INSIGHTS_MAX_LOG_GROUPS],
                "region": data.get("region", AWS_REGION),
                "note": "Names for follow-up Insights query. Do not dump full catalog to user unless they asked.",
            }
            return json.dumps(payload, ensure_ascii=False)

    if tool_name in {"execute_log_insights_query", "search_logs_across_groups", "query_log_group"}:
        try:
            data = json.loads(tool_result)
        except json.JSONDecodeError:
            return trim_message_content(tool_result, max_chars=MAX_TOOL_RESULT_CHARS)
        if isinstance(data, dict):
            results = data.get("results") or []
            # Tools return already-formatted dict rows; raw Insights rows are lists.
            rows = [
                {k: (v[:300] if k == "message" else v) for k, v in row.items() if v}
                if isinstance(row, dict)
                else row
                for row in results[:20]
            ]
            if any(isinstance(row, list) for row in rows):
                rows = srv.format_insights_rows(results, limit=20)
            payload = {
                key: data.get(key)
                for key in (
                    "status",
                    "log_group_name",
                    "search_filter",
                    "hours",
                    "window_label",
                    "start_time",
                    "end_time",
                    "period_label",
                    "scope",
                    "log_groups_searched",
                    "groups_with_matches",
                    "status_breakdown",
                )
                if data.get(key) not in (None, "", [])
            }
            payload["match_count"] = data.get("match_count", len(results))
            payload["rows_returned"] = len(results)
            payload["results"] = rows
            payload["truncated"] = bool(data.get("truncated")) or len(results) > 20
            if data.get("error"):
                payload["error"] = data["error"]
            return json.dumps(payload, ensure_ascii=False)

    return trim_message_content(tool_result, max_chars=MAX_TOOL_RESULT_CHARS)


def trim_message_content(content: str, max_chars: int = MAX_HISTORY_MESSAGE_CHARS) -> str:
    if len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + f"\n...[mesaj {len(content) - max_chars} karakter kısaltıldı]"
    )


def trim_conversation_history(
    history: list[dict[str, Any]] | None,
    max_messages: int = MAX_HISTORY_MESSAGES,
) -> list[dict[str, Any]]:
    if not history:
        return []
    trimmed = history[-max_messages:]
    return [
        {
            "role": message["role"],
            "content": summarize_message_for_history(message.get("content", "")),
        }
        for message in trimmed
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]


_openai_tools_cache: list[dict[str, Any]] | None = None
_llm_client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    timeout=REQUEST_TIMEOUT_SECONDS,
)


# --- Scope classifier: isolated LLM call to reject fully off-topic requests
# (recipes, general chat, coding help, politics, etc.) BEFORE they reach the
# main tool-loop LLM, which tends to "helpfully" answer them anyway. ---


def _mcp_tool_to_openai(tool: Tool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": _srv().CLOUDWATCH_TOOLS.get(tool.name, tool.description or ""),
            "parameters": tool.inputSchema,
        },
    }


async def get_openai_tools() -> list[dict[str, Any]]:
    global _openai_tools_cache
    if _openai_tools_cache is None:
        tools = await _srv().mcp.list_tools()
        _openai_tools_cache = [
            _mcp_tool_to_openai(tool)
            for tool in tools
            if tool.name in _srv().ALLOWED_TOOL_NAMES
        ]
        logger.info("Loaded %d CloudWatch tools for agent", len(_openai_tools_cache))
    return _openai_tools_cache


def serialize_tool_result(result: CallToolResult) -> str:
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, default=str)

    if not result.content:
        return "Tool returned no content."

    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))

    content = "\n".join(parts)
    if len(content) > MAX_TOOL_RESULT_CHARS:
        content = (
            content[:MAX_TOOL_RESULT_CHARS]
            + f"\n...[truncated {len(content) - MAX_TOOL_RESULT_CHARS} chars]"
        )
    if result.isError:
        return f"Tool error: {content}"
    return content


LOG_TOOL_RESULT_PREVIEW_CHARS = 4000


def find_api_connection_error(exc: BaseException) -> APIConnectionError | None:
    """Unwrap ExceptionGroup chains from MCP session teardown."""
    if isinstance(exc, APIConnectionError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            found = find_api_connection_error(sub)
            if found is not None:
                return found
    cause = exc.__cause__
    if isinstance(cause, BaseException):
        return find_api_connection_error(cause)
    return None


def summarize_tool_result_for_log(tool_name: str, result_str: str) -> str:
    """Compact one-line summary for pod logs (args are logged separately)."""
    if result_str.startswith("Tool error:"):
        return result_str[:500]

    try:
        data = json.loads(result_str)
    except json.JSONDecodeError:
        return result_str[:LOG_TOOL_RESULT_PREVIEW_CHARS]

    if not isinstance(data, dict):
        return result_str[:LOG_TOOL_RESULT_PREVIEW_CHARS]

    if tool_name == "describe_log_groups":
        groups = data.get("log_groups") or []
        names = [
            g.get("logGroupName", g) if isinstance(g, dict) else str(g)
            for g in groups[:10]
        ]
        parts = [f"count={data.get('count', len(groups))}"]
        if names:
            parts.append(f"sample={names}")
        if data.get("message"):
            parts.append(f"message={data['message']!r}")
        return " | ".join(parts)

    if tool_name in {"query_log_group", "search_logs_across_groups", "analyze_log_group"}:
        parts: list[str] = []
        for key in (
            "status",
            "match_count",
            "record_count",
            "count",
            "response_mode",
            "period_label",
            "search_filter",
        ):
            if key in data and data[key] is not None:
                parts.append(f"{key}={data[key]}")
        if data.get("query_string"):
            parts.append(f"query_string={data['query_string']!r}")
        if data.get("log_group_name"):
            parts.append(f"log_group_name={data['log_group_name']!r}")
        if data.get("log_group_names"):
            parts.append(f"log_group_names={data['log_group_names']!r}")
        results = data.get("results") or data.get("records") or []
        if results:
            parts.append(f"results_rows={len(results)}")
            parts.append(
                "first_row="
                + json.dumps(results[0], default=str, ensure_ascii=False)[:400]
            )
        return " | ".join(parts) if parts else result_str[:LOG_TOOL_RESULT_PREVIEW_CHARS]

    if tool_name in {"get_active_alarms", "describe_alarms", "get_alarm_history"}:
        alarms = data.get("alarms") or data.get("alarm_history") or []
        return f"items={len(alarms)} keys={list(data.keys())[:6]}"

    if tool_name in {"get_metric_data", "get_metric_statistics"}:
        series = data.get("metric_data_results") or data.get("datapoints") or []
        return f"series={len(series)} keys={list(data.keys())[:6]}"

    preview = json.dumps(
        {k: data[k] for k in list(data.keys())[:8] if k not in ("results", "log_groups")},
        default=str,
        ensure_ascii=False,
    )
    return preview[:LOG_TOOL_RESULT_PREVIEW_CHARS]


class CloudWatchMcpSession:
    def __init__(self) -> None:
        self._session_cm = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "CloudWatchMcpSession":
        self._session_cm = create_connected_server_and_client_session(_srv().mcp._mcp_server)
        self.session = await self._session_cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        self.session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError("MCP session is not initialized")
        if name not in _srv().ALLOWED_TOOL_NAMES:
            raise ValueError(f"Tool '{name}' is not allowed. Use one of: {sorted(_srv().ALLOWED_TOOL_NAMES)}")

        args_json = json.dumps(arguments, ensure_ascii=False)
        logger.info("Tool CALL [%s] args=%s", name, args_json)
        started = time.time()
        result = await self.session.call_tool(name, arguments)
        serialized = serialize_tool_result(result)
        elapsed = time.time() - started
        summary = summarize_tool_result_for_log(name, serialized)
        if result.isError:
            logger.error(
                "Tool RESULT [%s] elapsed=%.2fs is_error=true summary=%s",
                name,
                elapsed,
                summary,
            )
        else:
            logger.info(
                "Tool RESULT [%s] elapsed=%.2fs is_error=false summary=%s",
                name,
                elapsed,
                summary,
            )
        if len(serialized) > LOG_TOOL_RESULT_PREVIEW_CHARS:
            logger.debug(
                "Tool RESULT [%s] full_payload=%s",
                name,
                serialized[: LOG_TOOL_RESULT_PREVIEW_CHARS * 2],
            )
        return serialized


def humanize_response(text: str) -> str:
    """Convert accidental JSON-only LLM replies into readable text for the UI."""
    if not text:
        return ""
    stripped = text.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return text

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return text

    if isinstance(data, dict):
        if data.get("error"):
            return str(data["error"])

        log_groups = data.get("log_groups") or data.get("logGroups")
        if isinstance(log_groups, list):
            if not log_groups:
                return str(data.get("message") or "Eşleşen log group bulunamadı.")
            lines = ["Bulunan log grupları:"]
            for item in log_groups:
                if isinstance(item, dict):
                    name = item.get("logGroupName", "")
                    retention = item.get("retentionInDays")
                    suffix = f" (retention: {retention} gün)" if retention else ""
                    lines.append(f"- {name}{suffix}")
                else:
                    lines.append(f"- {item}")
            if data.get("message"):
                lines.insert(1, str(data["message"]))
            return "\n".join(lines)

        metric_alarms = data.get("metric_alarms") or data.get("metricAlarms")
        if isinstance(metric_alarms, list):
            if not metric_alarms:
                return "Aktif alarm bulunamadı."
            lines = ["Aktif alarmlar:"]
            for alarm in metric_alarms[:20]:
                if isinstance(alarm, dict):
                    lines.append(f"- {alarm.get('AlarmName', alarm)}")
            return "\n".join(lines)

        if data.get("message") and len(data) <= 4:
            return str(data["message"])

    return text


_MODEL_REFUSAL_RE = re.compile(
    r"(?i)("
    r"i'?m sorry.*(can'?t|cannot|unable)|"
    r"i can'?t share|i cannot share|"
    r"i'?m unable to|cannot assist|can'?t assist|"
    r"as an ai( language model)?, i (can'?t|cannot)|"
    r"i'?m not able to (share|disclose|provide)|"
    r"that'?s (not something|something) i (can'?t|cannot)|"
    r"üzgünüm.*(paylaşam|söyleyem|yardımcı olam)|"
    r"maalesef.*(paylaşam|söyleyem|yardımcı olam)"
    r")",
)

_ENGLISH_DOMINANT_RE = re.compile(
    r"(?i)\b(i'?m sorry|i cannot|i can't|however|please note|as an ai|let me know)\b",
)


_CASUAL_GREETING_RE = re.compile(
    r"(?i)\b("
    r"merhaba|selam|slm|mrb|sa|as|"
    r"naber|n'aber|ne haber|nabıyorsun|naber\s+nasıl|napiyorsun|"
    r"nasılsın|nasılsınız|iyi misin|iyi misiniz|"
    r"günaydın|tünaydın|iyi akşamlar|iyi günler|"
    r"hey|hi|hello|how are you|hey there|what'?s up|sup"
    r")\b",
)


_EN_WORDS_RE = re.compile(
    r"(?i)\b(the|is|are|was|what|which|who|how|why|show|me|my|please|can|could|you|give|last|hours?|"
    r"minutes?|days?|errors?|any|there|in|of|for|and|with|did|does|have|list|logs?|groups?|i|it|this)\b"
)
_TR_WORDS_RE = re.compile(
    r"(?i)\b(bana|ben|sen|bir|ve|ile|mi|mı|mu|mü|ne|neden|nasıl|kaç|son|var|yok|ver|getir|göster|"
    r"listele|merhaba|selam|naber|hangi|grup\w*|log\w*|hata\w*|alarm\w*|metri\w+|saat\w*|gün\w*|"
    r"detay\w*|analiz\w*|tarif\w*|lütfen|bak|peki|tamam|şu|bu|o)\b"
)


def detect_message_language(message: str) -> Literal["tr", "en"]:
    """Turkish unless the message is clearly English (users of this tool write Turkish by default)."""
    if re.search(r"[çğıöşüÇĞİÖŞÜ]", message):
        return "tr"
    english = len(_EN_WORDS_RE.findall(message))
    turkish = len(_TR_WORDS_RE.findall(message))
    return "en" if english > turkish else "tr"


def is_operational_intent(message: str) -> bool:
    return bool(
        is_log_search_intent(message)
        or is_metric_analysis_request(message)
        or is_log_catalog_intent(message)
        or is_alarm_history_request(message)
        or is_active_alarms_request(message)
        or is_named_log_content_query(message)
        or is_cross_group_log_search(message)
        or is_scoped_log_search_without_group(message)
        or is_system_wide_log_show_request(message)
        or is_log_drilldown_followup(message)
        or extract_log_group_paths(message)
    )


_OUT_OF_SCOPE_REPLY = (
    "Bu konuda yardımcı olamıyorum — ben CloudWatch log, metrik ve alarm analizi için "
    "buradayım. Örneğin: **500 hatalarını göster**, **log gruplarımı listele**, "
    "**son 1 saatte hangi alarmlar tetiklendi**."
)


def _has_cloudwatch_signal(message: str) -> bool:
    return bool(
        re.search(
            r"(?i)cloudwatch|aws|log\s*gr|alarm|metric|metrik|500|error|hata|eks|lambda|"
            r"apigateway|container|ecs|rds|status\s*kod",
            message,
        )
    )


def is_clearly_off_topic(message: str) -> bool:
    """Fast mechanical reject for obvious non-CloudWatch requests (no LLM)."""
    if is_operational_intent(message) or is_harmless_off_topic(message):
        return False
    stripped = message.strip()
    if not stripped or len(stripped) > 220:
        return False
    if _has_cloudwatch_signal(stripped):
        return False
    return bool(
        re.search(
            r"(?i)(tarif|yemek|omlet|recipe|cook|mutfak|"
            r"nasıl\s+yapıl|nasil\s+yapil|how\s+to\s+make|"
            r"hangi\s+film|dizi\s+öner|futbol|maç\s+skor|"
            r"python\s+öğren|code\s+review|şiir\s+yaz|poem|"
            r"flört|ilişki|siyaset|politik|borsa|kripto)",
            stripped,
        )
    )


def build_out_of_scope_reply(message: str) -> str:
    if detect_message_language(message) == "en":
        return (
            "I can't help with that — I'm here for CloudWatch log, metric, and alarm analysis. "
            "Try: 'show 500 errors', 'list log groups', or 'alarms in the last hour'."
        )
    return _OUT_OF_SCOPE_REPLY


def is_harmless_off_topic(message: str) -> bool:
    """Casual / identity questions unrelated to CloudWatch operations."""
    if is_operational_intent(message):
        return False
    stripped = message.strip()
    if not stripped or len(stripped) > 160:
        return False
    if _CASUAL_GREETING_RE.search(stripped):
        return True
    return bool(
        re.search(
            r"(?i)(kimsin|sen kimsin|who are you|what are you|"
            r"hangi takım|which team|"
            r"ne iş yap|what do you do|"
            r"adın ne|what'?s your name|"
            r"sen nesin|what kind of assistant)",
            stripped,
        )
    )


def build_off_topic_reply(message: str) -> str:
    lang = detect_message_language(message)
    sports = bool(re.search(r"(?i)takım|team", message))

    if lang == "tr":
        if sports:
            return (
                "Ben bir futbol takımı tutmuyorum — AWS CloudWatch odaklı bir SRE asistanıyım. "
                "İstersen log, metrik veya alarm konularında yardımcı olabilirim."
            )
        if re.search(r"(?i)kimsin|nesin|ne iş", message):
            return (
                "Ben CloudWatch SRE asistanıyım: log grupları, hata/500 aramaları, metrikler ve "
                "alarmlar konusunda yardımcı olurum. Ne bakmamı istersin?"
            )
        if _CASUAL_GREETING_RE.search(message):
            return (
                "Merhaba! CloudWatch log, metrik ve alarm konularında yardımcı olabilirim. "
                "Ne inceleyelim?"
            )
        return (
            "CloudWatch SRE asistanıyım — log, metrik ve alarm sorularında yardımcı olurum. "
            "Ne bakmamı istersin?"
        )

    if sports:
        return (
            "I don't follow a sports team — I'm a CloudWatch-focused SRE assistant. "
            "Happy to help with logs, metrics, or alarms."
        )
    if re.search(r"(?i)who are you|what are you|what do you do", message):
        return (
            "I'm a CloudWatch SRE assistant for logs, error/500 searches, metrics, and alarms. "
            "What would you like to check?"
        )
    if re.search(r"(?i)^hi\b|^hello\b|how are you", message):
        return "Hello! I can help with CloudWatch logs, metrics, and alarms. What should we look at?"
    return "I'm a CloudWatch SRE assistant. Ask me about logs, metrics, or alarms."


def is_model_refusal(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    return bool(_MODEL_REFUSAL_RE.search(stripped))


def sanitize_user_facing_response(message: str, response: str) -> str:
    """Replace unnecessary model refusals and fix obvious language mismatches."""
    if not response:
        return response

    if is_model_refusal(response):
        if is_harmless_off_topic(message):
            logger.warning(
                "Model refused harmless off-topic message; using canned reply. user=%r refusal=%r",
                message[:120],
                response[:200],
            )
            return build_off_topic_reply(message)
        logger.warning(
            "Model refusal on operational message (keeping response). user=%r refusal=%r",
            message[:120],
            response[:200],
        )

    user_lang = detect_message_language(message)
    if user_lang == "tr" and _ENGLISH_DOMINANT_RE.search(response):
        if is_harmless_off_topic(message):
            logger.warning(
                "English reply to Turkish off-topic message; using Turkish canned reply. user=%r",
                message[:120],
            )
            return build_off_topic_reply(message)
        logger.info(
            "English-leaning reply to Turkish operational message (not auto-replaced). user=%r",
            message[:80],
        )

    return response


LLM_FIRST_PROMPT = """
## Answer mode: interpreting SRE

You handle EVERY message yourself — follow-ups, references and "why" questions included.

### Resolving what the user refers to
- Read [Active session context] and earlier tool results before acting. Resolve "bu", "bunlar", "o servis",
  "aynısını dün için", "peki 4xx?", "neden?" from them.
- "2. grubun detayı", "ikinci servis", "ilkine bak" → the entry with that number in ranked_groups →
  query_log_group(log_group_name=<that group>, same search_filter and hours, response_mode="detail").
- Keep the previous search_filter / hours / scope unless the user changes them. If the context has
  start_time/end_time for the window, pass them again.
- "bunlar", "bunların kaçı …", "orada" after a single-group answer → query_log_group on that same group.
- Only use log group names that appear in the user's message, the context or earlier tool results —
  never take one from the examples in these instructions.
- Metrics: call analyze_metric (not get_metric_data) unless the user asks for raw datapoints.
- "neden", "ne anlama geliyor", "hangi endpoint/path", "ne tür hatalar" → fetch real lines
  (response_mode="analysis" or "detail") in the relevant group, then explain from those lines.
- After a metric answer, "detay" / "ilgili loglar" → search_logs_across_groups with the context's
  search_filter and log_group_name_keywords.
- Ask ONE short question only when you genuinely cannot tell scope or time (e.g. a bare "500 var mı?").

### Writing the reply after tool calls
- The server already renders the tool data (numbers, tables, log lines) as a card under your text.
  Do NOT repeat tables, lists of groups or log lines.
- Write 2–4 short sentences in natural Turkish (English if the user wrote English): what was found,
  what it most likely means, the probable cause if the log lines show one, and one useful next step.
- Use only numbers that appear in tool results; never estimate or compute new figures (no percentages
  or sums you calculated). Sentences containing numbers that are not in the tool results are deleted.
- No headings, no bullet lists unless comparing two periods or listing fix steps.
- Never mention tool names, parameters or modes (response_mode, detail, summary…); speak like an SRE.
- Do not claim something "increased" or "decreased" unless you fetched two periods to compare.
- Your reply goes straight to the user: no notes to yourself, no "The tool returned…" or "We need to…".
""".strip()

_CARD_TOOLS = (
    "search_logs_across_groups",
    "query_log_group",
    "analyze_metric",
    "get_active_alarms",
    "get_alarm_history",
    "describe_log_groups",
)
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)*")


def _number_variants(token: str) -> set[str]:
    """'1.209' / '1,209' / '1209' all mean the same count; '3,5' may be a decimal."""
    plain = re.sub(r"[.,]", "", token)
    variants = {token, plain}
    if re.fullmatch(r"\d+[.,]\d+", token):
        variants.add(token.replace(",", "."))
    return variants


_DEMONSTRATIVE_START_RE = re.compile(r"^(bu|bunlar|bunun|bu da|this|that|these)\b", re.I)
_PLACEHOLDER_RE = re.compile(r"^[\s|…·.\-–—*]*$")
# "3. **/aws/...plt-comm-catalog** – 688 hata": the card already lists groups with counts.
_CARD_ROW_RE = re.compile(r"^\s*(\d+[.)]|[-*•])\s+.*(/aws/|log[- ]grou?p|_rmfix|`[^`]*/[^`]*`).*\d", re.I)


_REASONING_LEAK_RE = re.compile(
    r"^\s*(the tool|tool returned|we need to|we should|we have to|the user (asked|wants|is asking)|"
    r"let'?s |i need to|i should|analysis:|okay,? so)",
    re.I,
)
_TECH_JARGON_RE = re.compile(
    r"response_mode|search_filter|max_result_lines|log_group_name|query_log_group|"
    r"search_logs_across_groups|analyze_metric|get_alarm_history|\btool\b|`detail`|`summary`|`analysis`",
    re.I,
)
_EMPTY_LIST_ITEM_RE = re.compile(r"^\s*(\d+[.)]|[-*•])\s*$")


def finalize_insight(message: str, text: str) -> str | None:
    """Last pass: no leaked chain-of-thought, no tool jargon, no hollow list skeletons."""
    if not text or _REASONING_LEAK_RE.match(text):
        return None
    if detect_message_language(message) == "tr" and detect_message_language(text) == "en":
        return None
    lines: list[str] = []
    for line in text.splitlines():
        sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", line)
            if not _TECH_JARGON_RE.search(sentence)
        ]
        line = " ".join(sentences)
        if _EMPTY_LIST_ITEM_RE.match(line):
            continue
        lines.append(line)
    # Drop lead-ins ("şunlardı:") whose list vanished.
    kept: list[str] = []
    for index, line in enumerate(lines):
        if line.rstrip().endswith(":"):
            following = next((l for l in lines[index + 1 :] if l.strip()), "")
            if not re.match(r"^\s*(\d+[.)]|[-*•])\s+\S", following):
                continue
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned or None


def clean_insight(text: str) -> str:
    """The card already shows the data: drop tables, headings and placeholder rows the model adds."""
    text = re.sub(r"[\u00a0\u202f\u2009\u2011]", lambda m: "-" if m.group(0) == "\u2011" else " ", text or "")
    lines = [
        line
        for line in text.splitlines()
        if not _CARD_ROW_RE.match(line)
        and not line.lstrip().startswith("|")
        and not line.lstrip().startswith("#")
        and not (line.strip() and _PLACEHOLDER_RE.match(line))
    ]
    # A lead-in like "En sık hatalar şunlar:" is meaningless once the table under it is gone.
    while lines and lines[-1].rstrip().endswith(":"):
        lines.pop()
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return cleaned[:1500]


def verify_insight_numbers(text: str, sources: list[str]) -> str:
    """Drop sentences whose numbers do not occur in the tool results (no invented or computed figures)."""
    known: set[str] = set()
    for source in sources:
        for token in _NUMBER_TOKEN_RE.findall(source or ""):
            known |= _number_variants(token)
    kept_lines: list[str] = []
    for line in (text or "").splitlines():
        kept: list[str] = []
        dropped_previous = False
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            bad = [
                match.group(0)
                for match in _NUMBER_TOKEN_RE.finditer(sentence)
                if not (match.group(0).isdigit() and int(match.group(0)) < 10)
                and not (_number_variants(match.group(0)) & known)
            ]
            if bad or (dropped_previous and _DEMONSTRATIVE_START_RE.match(sentence.strip())):
                logger.warning("Dropping insight sentence (unverified numbers %s): %r", bad, sentence[:160])
                dropped_previous = True
                continue
            dropped_previous = False
            kept.append(sentence)
        if kept or not line.strip():
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip()


def compose_llm_first_answer(
    message: str,
    text: str,
    tool_results: list[tuple[str, dict[str, Any], str]],
    tool_calls_made: list[dict[str, Any]],
    iterations: int,
) -> dict[str, Any]:
    """Data from tools (card/markdown) + the model's interpretation (verified insight)."""
    if not tool_results:
        return {"response": text or build_out_of_scope_reply(message), "tool_calls": tool_calls_made, "iterations": iterations}

    insight = finalize_insight(
        message,
        verify_insight_numbers(clean_insight(text), [message] + [result for _, _, result in tool_results]),
    )
    data_results = [(name, result) for name, _, result in tool_results if name in _CARD_TOOLS]
    if not data_results:
        return {"response": insight or text, "tool_calls": tool_calls_made, "iterations": iterations}

    log_searches = [(n, r) for n, r in data_results if n in {"search_logs_across_groups", "query_log_group"}]
    if len(log_searches) >= 2:
        # Comparison / multi-step: show every result; the model's text ties them together.
        sections = [try_direct_tool_response(n, r) or r for n, r in log_searches]
        result = attach_log_context(
            {"response": "\n\n---\n\n".join(sections), "insight": insight, "tool_calls": tool_calls_made, "iterations": iterations},
            *log_searches[-1],
        )
        result.pop("card", None)
        result.pop("suggestions", None)
        return result

    name, raw = data_results[-1]
    direct = try_direct_tool_response(name, raw) or raw
    return attach_log_context(
        {"response": direct, "insight": insight, "tool_calls": tool_calls_made, "iterations": iterations},
        name,
        raw,
    )


def _known_log_groups(ctx: dict[str, Any], history: list[dict[str, Any]] | None) -> set[str]:
    groups = {str(ctx.get("log_group_name") or "")}
    groups |= {str(g) for g in ctx.get("log_group_names") or []}
    groups |= {str(g) for g, _ in ctx.get("ranked_groups") or []}
    for item in history or []:
        groups |= set(extract_log_group_paths(str(item.get("content") or "")))
        structured = item.get("log_context") or {}
        groups |= {str(g) for g, _ in structured.get("ranked_groups") or []}
    return {normalize_log_group_path(g) for g in groups if g}


def align_scope_with_context(
    message: str,
    tool_name: str,
    arguments: dict[str, Any],
    ctx: dict[str, Any],
    history: list[dict[str, Any]] | None,
) -> tuple[str, dict[str, Any]]:
    """Keep follow-ups on the scope the conversation is about (llm_first guardrail).

    - A log group the user never mentioned and no previous answer contained is invented → search
      the system (or the services named in the message) instead.
    - "bunların kaçı 403?" right after a single-group answer → stay on that group.
    """
    if tool_name == "get_metric_data":
        # The summarising tool returns the same series plus sum/trend and renders a card.
        allowed = {"namespace", "metric_name", "start_time", "end_time", "statistic", "period", "dimensions", "region"}
        return "analyze_metric", {k: v for k, v in arguments.items() if k in allowed}

    if tool_name == "query_log_group":
        group = normalize_log_group_path(str(arguments.get("log_group_name") or ""))
        named = set(extract_log_group_paths(message))
        known = _known_log_groups(ctx, history) | named
        mentioned = any(part and part.lower() in message.lower() for part in group.split("/")[-1:])
        if group and group not in known and not mentioned:
            logger.warning("llm_first: model picked unseen log group %s; searching instead", group)
            args = {k: v for k, v in arguments.items() if k != "log_group_name"}
            return "search_logs_across_groups", adjust_search_logs_args(message, args)

    if tool_name == "query_log_group" and not extract_log_group_paths(message):
        group = normalize_log_group_path(str(arguments.get("log_group_name") or ""))
        scope = ctx.get("scope")
        same_group = group == normalize_log_group_path(str(ctx.get("log_group_name") or ""))
        new_filter = arguments.get("search_filter") and arguments.get("search_filter") != ctx.get("search_filter")
        if scope in {"system", "keywords"} and same_group and new_filter:
            # "peki 4xx?" after a system-wide 500 search keeps the system scope.
            logger.info("llm_first: keeping previous %s scope for new filter %s", scope, arguments.get("search_filter"))
            args = {k: v for k, v in arguments.items() if k != "log_group_name"}
            if scope == "keywords" and ctx.get("log_group_name_keywords"):
                args["log_group_name_keywords"] = list(ctx["log_group_name_keywords"])
            args["response_mode"] = "summary"
            return "search_logs_across_groups", adjust_search_logs_args(message, args)

    if tool_name == "search_logs_across_groups":
        names = arguments.get("log_group_names") or []
        if len(names) == 1 and not arguments.get("log_group_name_keywords"):
            args = {k: v for k, v in arguments.items() if k != "log_group_names"}
            args["log_group_name"] = names[0]
            return "query_log_group", adjust_query_log_group_args(message, args)
        single = ctx.get("log_group_name") if ctx.get("scope") not in {"system", "keywords"} else None
        # Only explicit widening words count here; "kaç" alone ("bunların kaçı 403?") is a follow-up.
        new_scope = (
            extract_log_group_paths(message)
            or infer_keywords_from_message(message)
            or re.search(
                r"sistem|tüm|bütün|hepsi|genel|diğer|hangi\s+log\s+gru|log\s+grupları|all\b|every|across",
                message,
                re.I,
            )
        )
        if single and not new_scope and not arguments.get("log_group_names") and not arguments.get("log_group_name_keywords"):
            logger.info("llm_first: keeping follow-up on previous group %s", single)
            args = {k: v for k, v in arguments.items() if k not in {"log_group_names", "log_group_name_keywords"}}
            args["log_group_name"] = single
            return "query_log_group", adjust_query_log_group_args(message, args)
    return tool_name, arguments


_DATA_QUESTION_RE = re.compile(
    r"hangi|nereden|nerede|neden|niye|kaç|göster|detay|listele|endpoint|path|tür|hata|"
    r"which|where|why|how many|show|detail",
    re.I,
)
_FETCH_NUDGE = (
    "[Reminder] The user is asking about data from the current conversation. Answer it by calling the "
    "appropriate tool with the context's filter, window and scope — do not answer from memory."
)


async def _run_agent_core(
    message: str,
    conversation_history: list[dict[str, Any]] | None = None,
    *,
    _retry_without_history: bool = False,
) -> dict[str, Any]:
    history = None if _retry_without_history else conversation_history
    if not history and is_harmless_off_topic(message):
        return {"response": build_off_topic_reply(message), "tool_calls": [], "iterations": 0}
    if is_clearly_off_topic(message):
        return {"response": build_out_of_scope_reply(message), "tool_calls": [], "iterations": 0}

    tools = await get_openai_tools()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{get_system_prompt()}\n\n{LLM_FIRST_PROMPT}"},
        {"role": "system", "content": current_time_note()},
        *trim_conversation_history(history, max_messages=MAX_HISTORY_MESSAGES),
    ]
    session_ctx = build_session_context_message(history)
    if session_ctx:
        messages.append(session_ctx)
    messages.append({"role": "user", "content": message})
    ctx = merge_log_query_context(history) if history else {}

    tool_calls_made: list[dict[str, Any]] = []
    tool_results: list[tuple[str, dict[str, Any], str]] = []
    nudged = False
    async with CloudWatchMcpSession() as mcp_session:
        for iteration in range(MAX_TOOL_ITERATIONS):
            try:
                response = await _llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=LLM_MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                )
            except BadRequestError as exc:
                err_text = str(exc).lower()
                if not _retry_without_history and ("context length" in err_text or "maximum context" in err_text):
                    logger.warning("Context length exceeded; retrying without history")
                    return await _run_agent_core(message, conversation_history, _retry_without_history=True)
                raise
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                if (
                    iteration == 0
                    and not tool_results
                    and ctx.get("search_filter")
                    and _DATA_QUESTION_RE.search(message)
                    and not nudged
                ):
                    # Follow-up about conversation data answered without fetching: ask the model once more.
                    nudged = True
                    messages.append({"role": "system", "content": _FETCH_NUDGE})
                    continue
                text = sanitize_user_facing_response(message, humanize_response(assistant_message.content or ""))
                return compose_llm_first_answer(message, text, tool_results, tool_calls_made, iteration + 1)

            messages.append(assistant_message.model_dump(exclude_none=True))
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                if "<|" in tool_name or tool_name not in _srv().ALLOWED_TOOL_NAMES:
                    logger.warning("Skipping invalid tool call name: %s", tool_name)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {"error": f"Unknown tool: {tool_name}", "allowed_tools": sorted(_srv().ALLOWED_TOOL_NAMES)},
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tool_name, arguments = normalize_tool_call(message, tool_name, arguments, keep_llm_args=True)
                tool_name, arguments = align_scope_with_context(message, tool_name, arguments, ctx, history)
                try:
                    tool_result = await mcp_session.call_tool(tool_name, arguments)
                except Exception as exc:
                    logger.exception("Tool %s failed", tool_name)
                    tool_result = json.dumps({"error": str(exc), "tool": tool_name}, ensure_ascii=False)
                tool_results.append((tool_name, arguments, tool_result))
                tool_calls_made.append(
                    {"name": tool_name, "arguments": arguments, "result_preview": tool_result[:500]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": compact_tool_result_for_context(tool_name, tool_result),
                    }
                )

    return compose_llm_first_answer(message, "", tool_results, tool_calls_made, MAX_TOOL_ITERATIONS)


def current_time_note() -> str:
    """The model has no clock; give it "now" so comparison windows use real dates."""
    now_utc = datetime.now(timezone.utc)
    local = now_utc.astimezone(display_timezone())
    return (
        f"[Current time] {local:%Y-%m-%d %H:%M} Europe/Istanbul (UTC{local:%z}) = "
        f"{now_utc:%Y-%m-%dT%H:%M:%SZ}. Calendar words (bugün, dün, bu hafta) mean Europe/Istanbul days."
    )


async def run_agent(
    message: str,
    conversation_history: list[dict[str, Any]] | None = None,
    *,
    _retry_without_history: bool = False,
) -> dict[str, Any]:
    try:
        return await _run_agent_core(
            message,
            conversation_history,
            _retry_without_history=_retry_without_history,
        )
    except BaseException as exc:
        if find_api_connection_error(exc) is not None:
            logger.exception("vLLM connection failed during agent session")
            return {
                "response": (
                    "Model servisine (vLLM) bağlanılamadı. "
                    "vllm-gptoss pod'unun ayakta ve hazır olduğunu kontrol edip tekrar dene."
                ),
                "tool_calls": [],
                "iterations": 0,
            }
        raise


