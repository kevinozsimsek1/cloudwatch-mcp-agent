"""CloudWatch LLM agent: prompt, tool loop, mechanical fallbacks, formatters."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, Tool
from openai import APIConnectionError, AsyncOpenAI, BadRequestError

from app.config import (
    AWS_REGION,
    ENABLE_MECHANICAL_FALLBACKS,
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
    comparison_period_hints,
    infer_metric_targets,
    is_ambiguous_time_reference,
    has_explicit_time_window,
    parse_hours_from_message,
)

logger = logging.getLogger(__name__)


def _srv():
    import app.server as srv
    return srv


_system_prompt_cache: str | None = None


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
| Metrics | CPU, latency, errors, ECS/Lambda/RDS… | analyze_metric / get_metric_data |
| Comparison / trend | "bugün vs dün", "today vs yesterday", "this week vs last" | same tool TWICE, different start_time/end_time |

## CRITICAL rules
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
  - "son 12 saat" / "son 12 saatde" / "last 12 hours" → `hours=12`
  - "son 6 saat" → `hours=6`
  - "son 3 gün" / "last 3 days" → `hours=72`
  - "dün" / "yesterday" → `hours=24`
  - "bugün" / "today" → hours from midnight UTC to now (use parsed value)
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
- If user did NOT name groups AND wants system-wide view → omit log_group_names, use response_mode=summary.
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
Follow-ups like "detaylı göster" or "logları analiz et" reuse the same log group, hours, filter, and tenant from the previous turn.
If the user asks whether results are ONLY in one group ("sadece bu grupta mı", "başka gruplarda da var mı"),
call search_logs_across_groups WITHOUT log_group_names (system-wide summary) and compare to the prior group.
Use response_mode=analysis when user asks to analyze / explain root cause — server aggregates real log fields.
""".strip()
    return _system_prompt_cache


# Alias for imports
SYSTEM_PROMPT = property(lambda self: get_system_prompt())  # noqa - not used


# Tools whose output is formatted on the server from real AWS data (no LLM synthesis).
DIRECT_FORMAT_TOOLS = frozenset({
    "describe_log_groups",
    "get_active_alarms",
    "get_alarm_history",
    "analyze_metric",
    "search_logs_across_groups",
    "execute_log_insights_query",
    "query_log_group",
})
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

SHOW_LOG_GROUPS_AGAIN = re.compile(
    r"göstersene|göster\s*abi|listeyi\s*göster|tekrar\s*listele|hepsini\s*göster|göster\s*bana",
    re.I,
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

_DETAIL_HEADER_RE = re.compile(
    r"\*\*`(?P<group>[^`]+)`\*\*\s*—\s*son\s*\*\*(?P<hours>\d+)\s*saat\*\*",
    re.I,
)

_PRIOR_GROUP_LINE_RE = re.compile(
    r"\*\*Önceki grup\*\*\s*`(?P<group>[^`]+)`",
    re.I,
)


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


def is_log_drilldown_followup(message: str) -> bool:
    if is_vague_log_question(message):
        return False
    if is_metric_analysis_request(message):
        return False
    if is_scope_uniqueness_followup(message):
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
    return bool(
        re.search(
            r"detaylı|detay\s*li|satır\s*satır|göstersene|gösterir\s*misin|daha\s+detay|"
            r"show\s+detail|more\s+detail|tümünü\s+göster|hepsini\s+göster",
            message,
            re.I,
        )
    )


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


def _log_groups_equivalent(left: str, right: str) -> bool:
    a = _srv().normalize_insights_log_group((left or "").strip())
    b = _srv().normalize_insights_log_group((right or "").strip())
    if not a or not b:
        return False
    return a == b or a.endswith(b) or b.endswith(a)


def format_scope_uniqueness_reply(
    *,
    prior_group: str,
    hours: int,
    filter_name: str,
    ranked: list[tuple[str, int]],
    window_note: str,
    status: str,
) -> str:
    display_filter = format_filter_display_name(filter_name)
    norm_prior = _srv().normalize_insights_log_group(prior_group)
    groups_with_hits = [(g, c) for g, c in ranked if c > 0]
    prior_count = next((c for g, c in groups_with_hits if _log_groups_equivalent(g, norm_prior)), 0)
    other_groups = [
        (g, c) for g, c in groups_with_hits if not _log_groups_equivalent(g, norm_prior)
    ]

    if not groups_with_hits:
        return (
            f"Son **{hours} saat** içinde sistem genelinde **{display_filter}** eşleşmesi "
            f"bulamadım (durum: {status}){window_note}."
        )

    if not other_groups:
        count_note = f" (**{prior_count}** eşleşme)" if prior_count else ""
        return (
            f"Evet — son **{hours} saat** içinde **{display_filter}** kayıtları "
            f"**yalnızca** `{norm_prior}` log grubunda görünüyor{count_note}. "
            f"Taradığım diğer log gruplarında eşleşme yok (durum: {status}){window_note}."
        )

    lines = [
        f"Hayır — son **{hours} saat** içinde **{display_filter}** yalnızca "
        f"`{norm_prior}` ile sınırlı değil (durum: {status}){window_note}.",
        "",
        f"**Önceki grup** `{norm_prior}`: **{prior_count}** eşleşme",
        "",
        f"**Başka {len(other_groups)} log grubunda da var:**",
    ]
    for group, count in other_groups[:15]:
        lines.append(f"- `{group}`: **{count}** eşleşme")
    if len(other_groups) > 15:
        lines.append(f"- ... ve **{len(other_groups) - 15}** grup daha")
    return "\n".join(lines)


def build_log_context_from_tool_data(data: dict[str, Any]) -> dict[str, Any] | None:
    """Structured carry-over context from a log tool JSON payload."""
    if not isinstance(data, dict) or data.get("error"):
        return None

    hours = int(data.get("hours") or 1)
    search_filter = str(data.get("search_filter") or "errors")
    ctx: dict[str, Any] = {"hours": hours, "search_filter": search_filter}
    if data.get("tenant_filter"):
        ctx["tenant_filter"] = str(data["tenant_filter"])

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

    return None


def attach_log_context(
    result: dict[str, Any],
    tool_name: str,
    tool_result: str,
) -> dict[str, Any]:
    if tool_name not in {"query_log_group", "search_logs_across_groups"}:
        return result
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return result
    log_context = build_log_context_from_tool_data(data)
    if not log_context:
        return result
    enriched = dict(result)
    enriched["log_context"] = log_context
    return enriched


def make_log_tool_response(
    *,
    response: str,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_result: str,
    iterations: int = 1,
) -> dict[str, Any]:
    return attach_log_context(
        {
            "response": response,
            "tool_calls": [{"name": tool_name, "arguments": tool_args}],
            "iterations": iterations,
        },
        tool_name,
        tool_result,
    )


def extract_last_log_query_context(
    history: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Recover log group, hours, and filter from recent chat (for drill-down follow-ups)."""
    user_ctx: dict[str, Any] | None = None
    for item in reversed(history or []):
        role = str(item.get("role") or "")
        content = str(item.get("content") or "")
        if role == "assistant":
            structured = item.get("log_context")
            if isinstance(structured, dict) and structured:
                return dict(structured)
            parsed = _parse_assistant_log_context(content)
            if parsed:
                return parsed
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
    return user_ctx


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


def merge_log_query_context(history: list[dict[str, Any]] | None) -> dict[str, Any]:
    assistant_ctx = extract_last_log_query_context(history) or {}
    user_ctx = extract_user_log_intent_from_history(history) or {}
    return {**user_ctx, **assistant_ctx}


def synthesize_scoped_search_message(message: str, ctx: dict[str, Any]) -> str:
    """Rebuild a scoped query when follow-up omits tenant/filter."""
    tenant = extract_tenant_filter(message) or str(ctx.get("tenant_filter") or "")
    parts: list[str] = []
    if tenant and not extract_tenant_filter(message):
        parts.append(f"sadece {tenant} tenant")
    if not re.search(r"500|5\s*xx|4\s*xx|error|hata|status", message, re.I):
        sf = str(ctx.get("search_filter") or "")
        if sf and sf.startswith("http_"):
            parts.append(sf.replace("http_", "") + " hataları")
    parts.append(message)
    return " ".join(parts)


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
) -> str:
    display_filter = format_filter_display_name(filter_name)
    tenant_note = f" (tenant: **{tenant_filter}**)" if tenant_filter else ""
    if count == 0:
        return (
            f"{prefix}Son **{hours} saat** içinde `{group_name}` log grubunda{tenant_note} "
            f"**{display_filter}** için eşleşme bulamadım (durum: {status}){window_note}."
        )
    return (
        f"{prefix}Evet — son **{hours} saat** içinde `{group_name}` log grubunda{tenant_note} "
        f"**{count}** adet **{display_filter}** kaydı bulundu (durum: {status}){window_note}.\n\n"
        "İstersen satır satır detayını gösterebilirim; **\"detaylı göster\"** yazman yeterli.\n"
        "Kök neden özeti için **\"logları analiz et\"** de."
    )


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
        return "http_400"
    for alias, preset in SEARCH_FILTER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return preset
    return "errors"


def should_list_log_groups(message: str, history: list[dict[str, Any]] | None) -> bool:
    if re.search(r"\blog\s*gr", message, re.I) and re.search(
        r"listele|liste|tüm|tamam|hepsi|göster", message, re.I
    ):
        return True
    if not SHOW_LOG_GROUPS_AGAIN.search(message):
        return False
    for item in (history or [])[-8:]:
        content = item.get("content", "")
        if item.get("role") == "user" and re.search(r"\blog\s*gr", content, re.I):
            return True
        if "log grubu" in content or "kayıt listelendi" in content:
            return True
    return False


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
    if data.get("message") and not data.get("datapoint_count"):
        ns = data.get("namespace", "")
        metric = data.get("metric_name", "")
        prefix = f"**{ns} / {metric}**\n" if ns and metric else ""
        return f"{prefix}{data['message']}"

    ns = data.get("namespace", "")
    metric = data.get("metric_name", "")
    stat = data.get("statistic", "Average")
    trend_map = {
        "increasing": "artıyor",
        "decreasing": "azalıyor",
        "stable": "stabil",
    }
    trend = trend_map.get(str(data.get("trend", "")), data.get("trend", ""))
    lines = [
        f"**{ns} / {metric}** ({stat})",
        f"- Veri noktası: {data.get('datapoint_count', 0)}",
        f"- Min: {data.get('min', 0):.2f}",
        f"- Max: {data.get('max', 0):.2f}",
        f"- Ortalama: {data.get('average', 0):.2f}",
        f"- Trend: {trend}",
    ]
    if data.get("first_timestamp"):
        lines.append(f"- Başlangıç: {data['first_timestamp']}")
    if data.get("last_timestamp"):
        lines.append(f"- Bitiş: {data['last_timestamp']}")
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
    return bool(re.search(r"\baktif\s+alarm", lowered)) and not is_alarm_history_request(message)


def is_metric_analysis_request(message: str) -> bool:
    lowered = message.lower()
    if re.search(r"metrik|metric", lowered) and re.search(
        r"cpu|memory|mem|latency|gecikme|error|invoc|throttl|duration|süre|bağlant",
        lowered,
    ):
        return True
    return bool(
        re.search(
            r"\b(ecs|lambda|rds|alb|api\s*gateway|ec2|sqs|cloudfront|dynamo|eks|kubernetes)\b",
            lowered,
        )
        and re.search(r"cpu|memory|mem|metrik|metric|latency|gecikme|error|invoc", lowered)
    )


def ecs_metrics_from_message(message: str) -> list[tuple[str, str]]:
    return infer_metric_targets(message)


def is_ecs_metric_request(message: str) -> bool:
    return is_metric_analysis_request(message)


def wants_log_detail(message: str) -> bool:
    return bool(
        re.search(
            r"detaylı|detay\s*li|satır\s*satır|gösterir\s*misin|göstersene|"
            r"göster\s*bana|show\s+detail|satırları\s+göster|birde\s+detay|bir\s+de\s+detay",
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
        lines.append(f"@message\t{stripped[:500]}")
        return "\n".join(lines)
    if not isinstance(data, dict):
        lines.append(f"@message\t{stripped[:500]}")
        return "\n".join(lines)

    lines.append("@message")
    lines.append(json.dumps(data, ensure_ascii=False, indent=2))

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


def format_access_log_message(raw: str) -> str:
    """Readable one-liner for API Gateway / JSON access logs."""
    stripped = raw.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped[:300]
    if not isinstance(data, dict):
        return stripped[:300]

    status = data.get("status", "?")
    err_type = data.get("errorResponseType", "-")
    method = data.get("httpMethod", "?")
    path = data.get("path") or data.get("resourcePath") or "?"
    integration_status = data.get("integrationStatus", "-")
    err_msg = str(data.get("errorMessage", "")).strip()
    parts = [f"**HTTP {status}**", f"{method} `{path}`"]
    if err_type and err_type not in ("-", ""):
        parts.append(f"`{err_type}`")
    if integration_status and integration_status not in ("-", ""):
        parts.append(f"integration={integration_status}")
    if err_msg and err_msg not in ("-", "null", ""):
        parts.append(err_msg[:100])
    return " · ".join(parts)


def normalize_tenant_slug(raw: str) -> str:
    slug = (raw or "").strip().strip("'\"").lower()
    slug = re.sub(
        r"(?:'?(?:nin|nın|nun|nün|in|ın|un|ün|i|ı|u|ü))$",
        "",
        slug,
        flags=re.I,
    )
    return slug


def extract_tenant_filter(message: str) -> str | None:
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
            if len(slug) >= 3:
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
    window_note = ""
    if data.get("start_time") and data.get("end_time"):
        window_note = f" ({data['start_time']} → {data['end_time']})"

    tenant_note = f" · tenant: **{tenant}**" if tenant else ""
    lines = [
        f"## Log analizi — `{group}`",
        f"**Dönem:** son **{hours} saat** · **filtre:** {filter_name}{tenant_note}{window_note}",
        f"**Toplam eşleşme:** **{match_count}** kayıt (analiz örneği: **{len(events)}** satır)",
        "",
    ]

    if not events:
        lines.append("Bu filtreyle eşleşen kayıt bulunamadı veya mesajlar parse edilemedi.")
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
    window_note = ""
    if data.get("start_time") and data.get("end_time"):
        window_note = f" ({data['start_time']} → {data['end_time']})"

    rows = _collect_log_rows_from_data(data)

    all_groups = data.get("all_log_group_names") or data.get("log_group_names") or []
    if data.get("log_group_name"):
        all_groups = [str(data["log_group_name"])] + [g for g in all_groups if g != data["log_group_name"]]

    prefix = f"**{period_label}** — " if period_label else ""
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
            return (
                f"{prefix}**{filter_name} — sayım** (son {hours} saat, {scope}) — "
                f"eşleşme bulunamadı (durum: {status}).{window_note}"
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
            )

        title = "Sistem özeti" if response_mode == "summary" else "Seçili gruplar — sayım"
        lines = [
            f"{prefix}**{title} — {filter_name}** (son {hours} saat, durum: {status}){window_note}",
            f"**{len(ranked)} log grubunda** eşleşme var · **Toplam: {total_matches}** kayıt",
            "",
            "| # | Log group | Adet |",
            "|---|-----------|------|",
        ]
        for index, (log_group, count) in enumerate(ranked, 1):
            lines.append(f"| {index} | `{log_group}` | **{count}** |")
        if response_mode == "count_only":
            pass
        return "\n".join(line for line in lines if line)

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
                f"{prefix}`{data['log_group_name']}` — son {hours} saat içinde eşleşen satır bulunamadı "
                f"(durum: {status}).{window_note}{suffix}"
            )
        return (
            f"{prefix}Son {hours} saat içinde {groups_searched} log grubunda arama yapıldı — "
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
    if data.get("log_group_name"):
        lines.append(
            f"{prefix}**`{data['log_group_name']}`** — son **{hours} saat** · "
            f"**{display_filter}** · **{match_count}** kayıt{window_note}"
        )
        lines.append("")
        lines.append("**Kayıtlar (CloudWatch alan görünümü):**")
    else:
        lines.append(
            f"{prefix}**Detay — {display_filter}** · son {hours} saat · **{match_count}** satır "
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

    display_limit = min(200, max(1, int(data.get("max_result_lines") or _DETAIL_PREVIEW_LINES)))
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
            shown += 1
            timestamp = row.get("timestamp", "")
            log_group = _srv().normalize_insights_log_group(
                str(row.get("log_group") or default_group)
            )
            block = format_access_log_cloudwatch_block(
                str(row.get("message", "")),
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
        if remaining:
            lines.append(
                f"\n... ve **{remaining}** kayıt daha. Hepsini görmek için **\"tümünü göster\"** yaz."
            )

    if all_groups:
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


def is_alarm_correlation_request(message: str) -> bool:
    return bool(
        re.search(
            r"neden\s+tetik|why\s+did.*alarm|alarm.*neden|root\s+cause|kök\s+neden",
            message,
            re.I,
        )
    )


def should_defer_immediate_format(message: str, tool_calls_made: list[dict[str, Any]]) -> bool:
    """Allow multi-step tool loops before server-formatted final reply."""
    if is_comparison_request(message):
        return True
    if is_alarm_correlation_request(message):
        names = {tc.get("name") for tc in tool_calls_made}
        if "get_alarm_history" in names and "search_logs_across_groups" not in names and "query_log_group" not in names:
            return True
        if not names & {"get_alarm_history", "get_active_alarms"}:
            return True
    return False


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


def infer_keywords_from_message(message: str) -> list[str]:
    lowered = message.lower()
    mapping = [
        (r"api[- ]?gateway", "api-gateway"),
        (r"\blambda\b", "lambda"),
        (r"\becs\b|container\s*insights", "containerinsights"),
        (r"\balb\b|load\s*balancer", "alb"),
        (r"cloudfront", "cloudfront"),
        (r"codebuild", "codebuild"),
        (r"\beks\b", "containerinsights"),
    ]
    found: list[str] = []
    for pattern, keyword in mapping:
        if re.search(pattern, lowered):
            found.append(keyword)
    return found


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
    elif not args.get("response_mode"):
        if re.search(r"kaç\s+tane\s+log\s+grub|kaç\s+log\s+grub", message, re.I):
            args["response_mode"] = "summary"
        elif re.search(r"detay\s*istemiyorum|sadece\s+say[ıi]|count\s+only", message, re.I):
            args["response_mode"] = "count_only"
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

    if has_explicit_time_window(message):
        args["hours"] = parse_hours_from_message(message)
    elif "hours" not in args or args.get("hours") in (None, 1):
        args["hours"] = parse_hours_from_message(message)

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

    if has_explicit_time_window(message):
        args["hours"] = parse_hours_from_message(message)
    elif "hours" not in args or args.get("hours") in (None, 1):
        args["hours"] = parse_hours_from_message(message)

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
            payload = {
                "status": data.get("status"),
                "log_groups_searched": data.get("log_groups_searched"),
                "match_count": data.get("match_count", len(results)),
                "results": srv.format_insights_rows(results, limit=20),
                "truncated": len(results) > 20,
            }
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


def trim_conversation_history(history: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not history:
        return []
    trimmed = history[-MAX_HISTORY_MESSAGES:]
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


def detect_message_language(message: str) -> Literal["tr", "en"]:
    if re.search(r"[çğıöşüÇĞİÖŞÜ]", message):
        return "tr"
    if re.search(
        r"(?i)\b(merhaba|selam|nasılsın|hangi|grup|log|hata|alarm|metrik|saat|var mı|"
        r"detaylı|analiz|listele|göster|söyler)\b",
        message,
    ):
        return "tr"
    return "en"


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
        or is_log_drilldown_followup(message)
        or extract_log_group_paths(message)
    )


def is_harmless_off_topic(message: str) -> bool:
    """Casual / identity questions unrelated to CloudWatch operations."""
    if is_operational_intent(message):
        return False
    stripped = message.strip()
    if not stripped or len(stripped) > 160:
        return False
    return bool(
        re.search(
            r"(?i)(\bmerhaba\b|\bselam\b|\bnasılsın\b|how are you|"
            r"^hi\b|^hello\b|hey there|"
            r"kimsin|sen kimsin|who are you|what are you|"
            r"hangi takım|which team|"
            r"ne iş yaparsın|what do you do|"
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
        if re.search(r"(?i)merhaba|selam|nasılsın", message):
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


async def run_log_group_list_request(
    message: str,
    history: list[dict[str, Any]] | None,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not should_list_log_groups(message, history):
        return None

    tool_args = adjust_describe_log_groups_args(message, {"max_items": MAX_LOG_GROUPS_LIST})
    try:
        tool_result = await mcp_session.call_tool("describe_log_groups", tool_args)
    except Exception as exc:
        logger.exception("describe_log_groups list request failed")
        return {
            "response": f"Log group listesi alınamadı: {exc}",
            "tool_calls": [{"name": "describe_log_groups", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("describe_log_groups", tool_result)
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "describe_log_groups", "arguments": tool_args}],
        "iterations": 1,
    }


async def run_http_status_search(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not is_http_status_search(message):
        return None

    search_filter = infer_log_search_filter(message)
    tool_args = {
        "query_string": LOG_SEARCH_QUERY_PRESETS[search_filter],
        "hours": parse_hours_from_message(message),
        "max_result_lines": 50,
    }
    try:
        tool_result = await mcp_session.call_tool("search_logs_across_groups", tool_args)
    except Exception as exc:
        logger.exception("search_logs_across_groups status search failed")
        return {
            "response": f"HTTP durum araması başarısız: {exc}",
            "tool_calls": [{"name": "search_logs_across_groups", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("search_logs_across_groups", tool_result)
    return make_log_tool_response(
        response=direct or tool_result,
        tool_name="search_logs_across_groups",
        tool_args=tool_args,
        tool_result=tool_result,
    )


async def run_named_log_group_query(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not is_named_log_content_query(message):
        return None

    paths = extract_log_group_paths(message)
    if not paths:
        return None

    tool_args = {
        "log_group_name": paths[0],
        "hours": parse_hours_from_message(message),
        "search_filter": infer_log_search_filter(message),
    }
    tenant = extract_tenant_filter(message)
    if tenant:
        tool_args["tenant_filter"] = tenant
    if is_log_analysis_request(message):
        tool_args["response_mode"] = "analysis"
        tool_args["max_result_lines"] = 100
    elif wants_log_detail(message):
        tool_args["response_mode"] = "detail"
        tool_args["max_result_lines"] = (
            200 if re.search(r"tümünü|hepsini", message, re.I) else 50
        )
    else:
        tool_args["response_mode"] = "summary"
    try:
        tool_result = await mcp_session.call_tool("query_log_group", tool_args)
    except Exception as exc:
        logger.exception("query_log_group fallback failed")
        return {
            "response": f"Log sorgusu başarısız: {exc}",
            "tool_calls": [{"name": "query_log_group", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("query_log_group", tool_result)
    return make_log_tool_response(
        response=direct or tool_result,
        tool_name="query_log_group",
        tool_args=tool_args,
        tool_result=tool_result,
    )


async def run_scoped_log_search_without_group(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    """Tenant or analysis query without a named log group — search API Gateway access logs."""
    if not is_scoped_log_search_without_group(message):
        return None

    tenant = extract_tenant_filter(message)
    keywords = infer_keywords_from_message(message) or ["apigateway"]
    tool_args: dict[str, Any] = {
        "search_filter": infer_log_search_filter(message),
        "hours": parse_hours_from_message(message),
        "log_group_name_keywords": keywords,
    }
    if tenant:
        tool_args["tenant_filter"] = tenant
    if is_log_analysis_request(message):
        tool_args["response_mode"] = "analysis"
        tool_args["max_result_lines"] = 100
    elif wants_log_detail(message):
        tool_args["response_mode"] = "detail"
        tool_args["max_result_lines"] = 50
    else:
        tool_args["response_mode"] = "summary"

    try:
        tool_result = await mcp_session.call_tool("search_logs_across_groups", tool_args)
    except Exception as exc:
        logger.exception("scoped log search without group failed")
        return {
            "response": f"Log araması başarısız: {exc}",
            "tool_calls": [{"name": "search_logs_across_groups", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("search_logs_across_groups", tool_result)
    return make_log_tool_response(
        response=direct or tool_result,
        tool_name="search_logs_across_groups",
        tool_args=tool_args,
        tool_result=tool_result,
    )


async def run_scope_uniqueness_followup(
    message: str,
    history: list[dict[str, Any]] | None,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    """Check if prior single-group hits also appear in other log groups."""
    if not is_scope_uniqueness_followup(message, history):
        return None

    ctx = merge_log_query_context(history) or {}
    prior_group = str(ctx.get("log_group_name") or "").strip()
    if not prior_group:
        return {
            "response": (
                "Hangi log grubunu kastediyorsun? Önce bir log grubunda arama yap, "
                "sonra \"sadece bu grupta mı\" diye sor."
            ),
            "tool_calls": [],
            "iterations": 0,
        }

    hours = int(ctx.get("hours") or parse_hours_from_message(message))
    search_filter = str(
        ctx.get("search_filter")
        or (infer_log_search_filter(message) if re.search(r"500|error|hata|status", message, re.I) else "errors")
    )
    tool_args: dict[str, Any] = {
        "search_filter": search_filter,
        "hours": hours,
        "response_mode": "summary",
        "max_result_lines": 50,
    }
    tenant = extract_tenant_filter(message) or str(ctx.get("tenant_filter") or "") or None
    if tenant:
        tool_args["tenant_filter"] = tenant

    logger.info(
        "Scope-uniqueness follow-up: expanding search beyond prior group=%s filter=%s hours=%s",
        prior_group,
        search_filter,
        hours,
    )

    try:
        tool_result = await mcp_session.call_tool("search_logs_across_groups", tool_args)
    except Exception as exc:
        logger.exception("scope uniqueness follow-up failed")
        return {
            "response": f"Sistem geneli karşılaştırma başarısız: {exc}",
            "tool_calls": [{"name": "search_logs_across_groups", "arguments": tool_args}],
            "iterations": 1,
        }

    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, dict) and not data.get("error"):
        ranked: list[tuple[str, int]] = []
        for row in data.get("results") or []:
            if isinstance(row, dict):
                group = _srv().normalize_insights_log_group(str(row.get("log_group") or ""))
                try:
                    count = int(float(row.get("matches") or 0))
                except ValueError:
                    count = 0
                if group and count > 0:
                    ranked.append((group, count))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        window_note = ""
        if data.get("start_time") and data.get("end_time"):
            window_note = f" ({data['start_time']} → {data['end_time']})"
        reply = format_scope_uniqueness_reply(
            prior_group=prior_group,
            hours=hours,
            filter_name=search_filter,
            ranked=ranked,
            window_note=window_note,
            status=str(data.get("status") or "Unknown"),
        )
        return make_log_tool_response(
            response=reply,
            tool_name="search_logs_across_groups",
            tool_args=tool_args,
            tool_result=tool_result,
        )

    direct = try_direct_tool_response("search_logs_across_groups", tool_result)
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "search_logs_across_groups", "arguments": tool_args}],
        "iterations": 1,
    }


async def run_log_drilldown_followup(
    message: str,
    history: list[dict[str, Any]] | None,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    """Reuse prior log query context when user asks for detail without repeating the path."""
    if not is_log_drilldown_followup(message):
        return None

    paths = extract_log_group_paths(message)
    ctx = merge_log_query_context(history)
    ranked_groups: list[tuple[str, int]] = list(ctx.get("ranked_groups") or [])
    if not ranked_groups and len(ctx.get("log_group_names") or []) > 1:
        ranked_groups = [(name, 0) for name in ctx["log_group_names"]]

    if paths:
        group = paths[0]
        hours = (
            parse_hours_from_message(message)
            if has_explicit_time_window(message)
            else int(ctx.get("hours") or parse_hours_from_message(message))
        )
        search_filter = (
            infer_log_search_filter(message)
            if re.search(r"error|500|4\d{2}|hata|status", message, re.I)
            else str(ctx.get("search_filter") or infer_log_search_filter(message))
        )
    elif ctx.get("log_group_name"):
        group = normalize_log_group_path(str(ctx["log_group_name"]))
        hours = int(ctx.get("hours") or parse_hours_from_message(message))
        search_filter = str(ctx.get("search_filter") or "errors")
    elif len(ranked_groups) > 1:
        lines = [
            "Önceki aramada **birden fazla** log grubu bulunmuştu. Hangisinin detayını göstereyim?",
            "",
        ]
        for index, (log_group, count) in enumerate(ranked_groups[:12], 1):
            lines.append(f"{index}. `{log_group}` — **{count}** eşleşme")
        lines.append("")
        lines.append("Numara veya tam log group path yazman yeterli.")
        return {
            "response": "\n".join(lines),
            "tool_calls": [],
            "iterations": 0,
        }
    elif extract_tenant_filter(message) or ctx.get("tenant_filter"):
        return await run_scoped_log_search_without_group(
            synthesize_scoped_search_message(message, ctx),
            mcp_session,
        )
    elif is_scoped_log_search_without_group(message):
        return None
    else:
        return {
            "response": (
                "Hangi log grubunun detayını göstereyim? Tam log group path'ini yazabilir "
                "veya önce \"X log grubunda error var mı\" diye sorup ardından \"detaylı göster\" de.\n\n"
                "Tenant + status için path gerekmez — örn: "
                "**\"bozkurt tenant 500 hatalarını analiz et\"**"
            ),
            "tool_calls": [],
            "iterations": 0,
        }

    show_all = bool(re.search(r"tümünü|hepsini|all\s+lines", message, re.I))
    if is_log_analysis_request(message):
        response_mode = "analysis"
        max_lines = 100
    elif show_all:
        response_mode = "detail"
        max_lines = 200
    else:
        response_mode = "detail"
        max_lines = _DETAIL_PREVIEW_LINES

    tool_args = {
        "log_group_name": group,
        "hours": hours,
        "search_filter": search_filter,
        "response_mode": response_mode,
        "max_result_lines": max_lines,
    }
    tenant = extract_tenant_filter(message) or str(ctx.get("tenant_filter") or "") or None
    if tenant:
        tool_args["tenant_filter"] = tenant
    try:
        tool_result = await mcp_session.call_tool("query_log_group", tool_args)
    except Exception as exc:
        logger.exception("log drilldown follow-up failed")
        return {
            "response": f"Log detayı alınamadı: {exc}",
            "tool_calls": [{"name": "query_log_group", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("query_log_group", tool_result)
    return make_log_tool_response(
        response=direct or tool_result,
        tool_name="query_log_group",
        tool_args=tool_args,
        tool_result=tool_result,
    )


async def run_active_alarms_request(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not is_active_alarms_request(message):
        return None

    tool_args = {"max_items": 100}
    try:
        tool_result = await mcp_session.call_tool("get_active_alarms", tool_args)
    except Exception as exc:
        logger.exception("get_active_alarms fallback failed")
        return {
            "response": f"Aktif alarmlar alınamadı: {exc}",
            "tool_calls": [{"name": "get_active_alarms", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("get_active_alarms", tool_result)
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "get_active_alarms", "arguments": tool_args}],
        "iterations": 1,
    }


async def run_alarm_history_request(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not is_alarm_history_request(message):
        return None

    hours = parse_hours_from_message(message)
    end_dt = _srv().utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    tool_args = {
        "start_time": _srv().iso_utc(start_dt),
        "end_time": _srv().iso_utc(end_dt),
        "max_records": 100,
    }
    try:
        tool_result = await mcp_session.call_tool("get_alarm_history", tool_args)
    except Exception as exc:
        logger.exception("get_alarm_history fallback failed")
        return {
            "response": f"Alarm geçmişi alınamadı: {exc}",
            "tool_calls": [{"name": "get_alarm_history", "arguments": tool_args}],
            "iterations": 1,
        }

    direct = try_direct_tool_response("get_alarm_history", tool_result)
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "get_alarm_history", "arguments": tool_args}],
        "iterations": 1,
    }


async def run_ecs_metric_analysis(
    message: str,
    mcp_session: "CloudWatchMcpSession",
) -> dict[str, Any] | None:
    if not is_ecs_metric_request(message):
        return None

    hours = parse_hours_from_message(message)
    end_dt = _srv().utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    metrics = ecs_metrics_from_message(message)
    tool_calls: list[dict[str, Any]] = []
    sections: list[str] = []

    for namespace, metric_name in metrics:
        tool_args = {
            "namespace": namespace,
            "metric_name": metric_name,
            "start_time": _srv().iso_utc(start_dt),
            "end_time": _srv().iso_utc(end_dt),
            "statistic": "Average",
            "period": 300,
        }
        try:
            tool_result = await mcp_session.call_tool("analyze_metric", tool_args)
        except Exception as exc:
            logger.exception("analyze_metric fallback failed for %s/%s", namespace, metric_name)
            sections.append(f"**{namespace} / {metric_name}** — analiz başarısız: {exc}")
            tool_calls.append({"name": "analyze_metric", "arguments": tool_args})
            continue

        tool_calls.append({"name": "analyze_metric", "arguments": tool_args})
        direct = try_direct_tool_response("analyze_metric", tool_result)
        sections.append(direct or tool_result)

    return {
        "response": "\n\n".join(sections),
        "tool_calls": tool_calls,
        "iterations": 1,
    }


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


async def _run_agent_core(
    message: str,
    conversation_history: list[dict[str, Any]] | None = None,
    *,
    _retry_without_history: bool = False,
) -> dict[str, Any]:
    tools = await get_openai_tools()
    messages: list[dict[str, Any]] = [{"role": "system", "content": get_system_prompt()}]

    trimmed_history = [] if _retry_without_history else trim_conversation_history(conversation_history)
    if trimmed_history:
        messages.extend(trimmed_history)

    messages.append({"role": "user", "content": message})
    tool_calls_made: list[dict[str, Any]] = []
    last_direct_format: tuple[str, str] | None = None

    async with CloudWatchMcpSession() as mcp_session:
        ecs_metrics = await run_ecs_metric_analysis(message, mcp_session)
        if ecs_metrics is not None:
            return ecs_metrics

        scoped = await run_scoped_log_search_without_group(message, mcp_session)
        if scoped is not None:
            return scoped

        scope_check = await run_scope_uniqueness_followup(
            message,
            conversation_history if not _retry_without_history else None,
            mcp_session,
        )
        if scope_check is not None:
            return scope_check

        drilldown = await run_log_drilldown_followup(
            message,
            conversation_history if not _retry_without_history else None,
            mcp_session,
        )
        if drilldown is not None:
            return drilldown

        if is_harmless_off_topic(message):
            reply = build_off_topic_reply(message)
            logger.info("Off-topic reply without LLM: %r", message[:100])
            return {
                "response": reply,
                "tool_calls": [],
                "iterations": 0,
            }

        if is_vague_log_question(message):
            return {
                "response": (
                    "Netleştirebilir miyim?\n"
                    "- **Tüm sistemde** mi arayayım, yoksa **belirli log gruplarında** mı?\n"
                    "- **Sayı özeti** (grup başına adet) mi, yoksa **satır detayı** mı?\n"
                    "- Zaman aralığı: son kaç saat?"
                ),
                "tool_calls": [],
                "iterations": 0,
            }

        if ENABLE_MECHANICAL_FALLBACKS:
            named_query = await run_named_log_group_query(message, mcp_session)
            if named_query is not None:
                return named_query

            list_query = await run_log_group_list_request(message, trimmed_history, mcp_session)
            if list_query is not None:
                return list_query

            status_query = await run_http_status_search(message, mcp_session)
            if status_query is not None:
                return status_query

            alarm_history = await run_alarm_history_request(message, mcp_session)
            if alarm_history is not None:
                return alarm_history

            active_alarms = await run_active_alarms_request(message, mcp_session)
            if active_alarms is not None:
                return active_alarms

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
            except APIConnectionError as exc:
                logger.exception("vLLM connection failed")
                return {
                    "response": (
                        "Model servisine (vLLM) bağlanılamadı. "
                        "vllm-gptoss pod'unun ayakta ve hazır olduğunu kontrol edip tekrar dene."
                    ),
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }
            except BadRequestError as exc:
                err_text = str(exc).lower()
                if (
                    not _retry_without_history
                    and ("context length" in err_text or "maximum context" in err_text)
                ):
                    logger.warning("Context length exceeded; retrying without history")
                    return await _run_agent_core(
                        message,
                        conversation_history,
                        _retry_without_history=True,
                    )
                if "context length" in err_text or "maximum context" in err_text:
                    return {
                        "response": (
                            "Sohbet geçmişi çok uzun olduğu için model yanıt veremedi. "
                            "Sol panelden **Yeni** ile yeni sohbet başlatıp tekrar dene."
                        ),
                        "tool_calls": tool_calls_made,
                        "iterations": iteration + 1,
                    }
                raise
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                content = (assistant_message.content or "").strip()
                if (
                    iteration == 0
                    and not tool_calls_made
                    and is_log_search_intent(message)
                    and not is_log_catalog_intent(message)
                    and not is_comparison_request(message)
                ):
                    rescue_args = adjust_search_logs_args(message, {})
                    try:
                        tool_result = await mcp_session.call_tool(
                            "search_logs_across_groups", rescue_args
                        )
                        tool_calls_made.append(
                            {
                                "name": "search_logs_across_groups",
                                "arguments": rescue_args,
                                "result_preview": tool_result[:500],
                            }
                        )
                        direct = try_direct_tool_response(
                            "search_logs_across_groups", tool_result
                        )
                        if direct is not None:
                            return attach_log_context(
                                {
                                    "response": direct,
                                    "tool_calls": tool_calls_made,
                                    "iterations": iteration + 1,
                                },
                                "search_logs_across_groups",
                                tool_result,
                            )
                    except Exception as exc:
                        logger.exception("search_logs rescue failed")
                        return {
                            "response": f"Log araması başarısız: {exc}",
                            "tool_calls": tool_calls_made,
                            "iterations": iteration + 1,
                        }

                if not content and ENABLE_MECHANICAL_FALLBACKS:
                    named_query = await run_named_log_group_query(message, mcp_session)
                    if named_query is not None:
                        return named_query

                last_tool_name = tool_calls_made[-1]["name"] if tool_calls_made else None
                if (
                    last_direct_format is not None
                    and last_tool_name in DIRECT_FORMAT_TOOLS
                ):
                    direct_name, direct_result = last_direct_format
                    if direct_name == last_tool_name:
                        direct = try_direct_tool_response(direct_name, direct_result)
                        if direct is not None:
                            return attach_log_context(
                                {
                                    "response": direct,
                                    "tool_calls": tool_calls_made,
                                    "iterations": iteration + 1,
                                },
                                direct_name,
                                direct_result,
                            )
                return {
                    "response": sanitize_user_facing_response(
                        message,
                        humanize_response(assistant_message.content or ""),
                    ),
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

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
                                {
                                    "error": f"Unknown tool: {tool_name}",
                                    "allowed_tools": sorted(_srv().ALLOWED_TOOL_NAMES),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if tool_name == "describe_log_groups":
                    if is_log_search_intent(message) and not is_log_catalog_intent(message):
                        logger.warning(
                            "Redirecting describe_log_groups -> search_logs_across_groups "
                            "(log search intent): %s",
                            message[:120],
                        )
                        tool_name = "search_logs_across_groups"
                        arguments = adjust_search_logs_args(message, {})
                    else:
                        arguments = adjust_describe_log_groups_args(message, arguments)
                        if arguments.get("log_group_name_prefix"):
                            arguments["log_group_name_prefix"] = _srv().normalize_log_group_prefix_arg(
                                arguments["log_group_name_prefix"]
                            )

                if tool_name == "query_log_group":
                    if is_cross_group_log_search(message):
                        logger.warning(
                            "Redirecting query_log_group -> search_logs_across_groups "
                            "(cross-group intent): %s",
                            message[:120],
                        )
                        tool_name = "search_logs_across_groups"
                        arguments = adjust_search_logs_args(message, {})
                    else:
                        arguments = adjust_query_log_group_args(message, arguments)

                if tool_name == "search_logs_across_groups":
                    if is_single_named_group_log_query(message):
                        logger.warning(
                            "Redirecting search_logs_across_groups -> query_log_group "
                            "(single named group): %s",
                            message[:120],
                        )
                        tool_name = "query_log_group"
                        arguments = adjust_query_log_group_args(message, {})
                    else:
                        arguments = adjust_search_logs_args(message, arguments)

                try:
                    tool_result = await mcp_session.call_tool(tool_name, arguments)
                except Exception as exc:
                    logger.exception("Tool %s failed", tool_name)
                    tool_result = json.dumps(
                        {"error": str(exc), "tool": tool_name},
                        ensure_ascii=False,
                    )

                if tool_name in DIRECT_FORMAT_TOOLS:
                    last_direct_format = (tool_name, tool_result)
                else:
                    last_direct_format = None
                tool_calls_made.append(
                    {
                        "name": tool_name,
                        "arguments": arguments,
                        "result_preview": tool_result[:500],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": compact_tool_result_for_context(tool_name, tool_result),
                    }
                )

            # Use the FINAL tool name (post-redirect), not the LLM's original choice,
            # so redirected calls (e.g. describe_log_groups -> search) also return immediately.
            if (
                len(assistant_message.tool_calls) == 1
                and last_direct_format is not None
                and last_direct_format[0] in IMMEDIATE_FORMAT_TOOLS
                and not should_defer_immediate_format(message, tool_calls_made)
            ):
                direct_name, direct_result = last_direct_format
                direct = try_direct_tool_response(direct_name, direct_result)
                if direct is not None:
                    return attach_log_context(
                        {
                            "response": direct,
                            "tool_calls": tool_calls_made,
                            "iterations": iteration + 1,
                        },
                        direct_name,
                        direct_result,
                    )

    return {
        "response": "Maximum tool iterations reached. Partial investigation completed.",
        "tool_calls": tool_calls_made,
        "iterations": MAX_TOOL_ITERATIONS,
    }


