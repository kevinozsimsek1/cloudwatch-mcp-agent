import json
import logging
import os
import re
import time
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, Tool
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL", "http://vllm-gptoss.llm-model.svc.cluster.local:8080/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "10"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOGS_QUERY_POLL_INTERVAL = float(os.getenv("LOGS_QUERY_POLL_INTERVAL", "1.0"))
LOGS_QUERY_MAX_WAIT = int(os.getenv("LOGS_QUERY_MAX_WAIT", "60"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_HISTORY_MESSAGE_CHARS = int(os.getenv("MAX_HISTORY_MESSAGE_CHARS", "2500"))
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "8000"))
MAX_LOG_GROUPS_LIST = int(os.getenv("MAX_LOG_GROUPS_LIST", "1000"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
DASHBOARD_CACHE_TTL = int(os.getenv("DASHBOARD_CACHE_TTL", "45"))
DASHBOARD_ERROR_DETAILS_LIMIT = int(os.getenv("DASHBOARD_ERROR_DETAILS_LIMIT", "150"))

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools registry
# ---------------------------------------------------------------------------

CLOUDWATCH_TOOLS = {
    "describe_log_groups": (
        "List CloudWatch log groups. Single filter: log_group_name_prefix (user's term as-is). "
        "Multiple types in one question: log_group_name_keywords (OR). "
        "All results: max_items=1000, prefix null. Fuzzy matching is automatic."
    ),
    "analyze_log_group": (
        "Legacy error analysis for one log group. Prefer query_log_group when user gives a log group path."
    ),
    "query_log_group": (
        "Query log LINES in ONE named log group. Use when user gives a path like "
        "/aws/apigateway/... or aws/lambda/.... Params: log_group_name, hours (default 1), "
        "search_filter: errors | http_400 | http_5xx. Do NOT ask clarifying questions."
    ),
    "execute_log_insights_query": (
        "Search log LINES in SPECIFIC named log groups (you must already have names). "
        "For HTTP codes (400, 5XX) or errors when user did NOT name a group, prefer "
        "search_logs_across_groups instead — do NOT ask which log group."
    ),
    "search_logs_across_groups": (
        "Search log LINES across many log groups when user wants errors, HTTP status codes "
        "(400, 4xx, 5XX), or log patterns but did NOT specify a log group. "
        "Call this immediately — do NOT ask 'hangi log grubu'. "
        "Provide query_string (Insights syntax). Use max_result_lines=150–200 when user wants detail. "
        "Optional log_group_name_keywords; if omitted, searches containerinsights, api-gateway, "
        "alb, lambda, cloudfront."
    ),
    "get_logs_insight_query_results": (
        "Fetch results for a running/completed Insights query_id."
    ),
    "cancel_logs_insight_query": (
        "Cancel a running Insights query by query_id."
    ),
    "get_active_alarms": (
        "List CloudWatch alarms currently in ALARM state. "
        "Use max_items=100 when user wants all; default 50 for a sample."
    ),
    "get_alarm_history": (
        "Alarm state change history. alarm_name is OPTIONAL — omit it to list all alarm "
        "transitions in the time range (e.g. 'son 24 saatte hangi alarmlar tetiklendi'). "
        "Infer time range from the user message."
    ),
    "get_metric_data": (
        "Fetch metric datapoints. Infer namespace, metric_name, statistic, period, dimensions, "
        "and time range from context (e.g. 'son 1 saat CPU' → namespace/stat + last 1h)."
    ),
    "get_metric_metadata": (
        "Discover metrics in a namespace. Infer namespace and optional metric name filter from the user."
    ),
    "get_recommended_metric_alarms": (
        "Suggested alarm thresholds for a metric. Infer namespace and metric_name from the user."
    ),
    "analyze_metric": (
        "Summarize metric trend (min/max/avg). Infer namespace, metric_name, statistic, and time range."
    ),
}

ALLOWED_TOOL_NAMES = set(CLOUDWATCH_TOOLS.keys())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""
You are a CloudWatch SRE assistant. YOU decide which tools to call from the user's message.

## Step 1 — What does the user want?

| Intent | User signals (TR/EN) | Tool |
|--------|----------------------|------|
| Log group NAMES (catalog) | "log grupları listele", "hangi log grouplar var", "tüm log grupları" | describe_log_groups |
| Log CONTENT — group unknown | "400 hataları", "5XX var mı", "errorleri listele", "farketmez", "bilmem sen bul" | search_logs_across_groups |
| Log CONTENT — group known | user names a log group path | query_log_group |
| Alarms — currently in ALARM | "aktif alarm" | get_active_alarms |
| Alarms — history / triggered | "tetiklendi", "alarm geçmişi", "son 24 saatte hangi alarmlar" | get_alarm_history (alarm_name optional — omit for all) |
| Metrics | CPU, memory, ECS metrik | analyze_metric / get_metric_data |

CRITICAL rules:
- HTTP status searches (400, 4xx, 5XX) = log LINE search, NOT log group catalog.
- If user does NOT name a log group → call search_logs_across_groups. NEVER ask "hangi log grubu".
- "farketmez / bilmem / sen bul / hangisinde varsa" → search_logs_across_groups with the same filter as the prior question.
- Named log group path in message (e.g. `/aws/apigateway/.../access-logs`) → query_log_group immediately.
- Alarm history without a specific name → get_alarm_history with start/end only (no alarm_name). NEVER ask user for alarm names.

## Query examples for search_logs_across_groups

400 errors:
fields @timestamp, @log, @message
| filter @message like /(?i)(\\b400\\b|status.?code.?400|HTTP\\/1\\.[01] 400)/
| sort @timestamp desc | limit 50

5XX errors:
fields @timestamp, @log, @message
| filter @message like /(?i)(\\b5\\d{{2}}\\b|status.?code.?5\\d{{2}}|HTTP\\/1\\.[01] 5\\d{{2}})/
| sort @timestamp desc | limit 50

Generic errors:
fields @timestamp, @log, @message
| filter @message like /(?i)(error|exception|fail|fatal)/
| sort @timestamp desc | limit 50

## Other examples
- "tüm log gruplarımı listele" → describe_log_groups(max_items=1000)
- "codebuild ve lambda log grupları" → describe_log_groups(log_group_name_keywords=[...])
- Times: "son 1 saat" → hours=1; default 1h if unspecified.

## Rules
- Latest user message defines intent. Prior turn clarifies filter (400 vs 5XX), not log group name.
- Pass user terms as-is; fuzzy AWS matching is server-side.
- Ask at most ONE clarifying question, and only if the request is truly impossible to run.
- Allowed tools: {", ".join(sorted(ALLOWED_TOOL_NAMES))}
- Default region: {AWS_REGION}

## Language
Reply in the same language as the user's latest message.

## Response format
Search/log-line answers are formatted server-side from real tool data.
NEVER invent log group names, counts, or placeholder rows (no "other-log-group", no fake tables).
If you need more detail, call search_logs_across_groups again with a higher max_result_lines.
"""

# ---------------------------------------------------------------------------
# MCP app
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "aws-cloudwatch",
    host=HOST,
    port=PORT,
    instructions=(
        "AWS CloudWatch MCP Server. "
        "Use these tools to query CloudWatch Logs, Metrics, Alarms, and Dashboards. "
        "All timestamps are in ISO-8601 UTC format. "
        "IRSA is used for authentication — no explicit credentials required."
    ),
)


@mcp.prompt()
def investigation_prompt() -> str:
    """CloudWatch incident investigation system prompt."""
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# AWS clients (lazy-initialised per region)
# ---------------------------------------------------------------------------

_clients: dict[str, dict[str, Any]] = {}


def get_client(service: str, region: Optional[str] = None):
    region = region or AWS_REGION
    if region not in _clients:
        _clients[region] = {}
    if service not in _clients[region]:
        _clients[region][service] = boto3.client(service, region_name=region)
    return _clients[region][service]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Optional[str], default: Optional[datetime] = None) -> datetime:
    if value is None:
        if default is None:
            raise ValueError("Timestamp is required")
        return default
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_epoch_seconds(value: datetime) -> int:
    return int(value.timestamp())


def matches_log_group_prefix(name: str, prefix: Optional[str]) -> bool:
    if not prefix:
        return True
    return name == prefix or name.startswith(f"{prefix}/") or name.startswith(prefix)


def filter_log_groups(log_groups: list[dict[str, Any]], prefix: Optional[str]) -> list[dict[str, Any]]:
    if not prefix:
        return log_groups
    return [lg for lg in log_groups if matches_log_group_prefix(lg.get("logGroupName", ""), prefix)]


def filter_log_groups_contains(log_groups: list[dict[str, Any]], query: Optional[str]) -> list[dict[str, Any]]:
    if not query:
        return log_groups
    q = query.lower()
    return [lg for lg in log_groups if q in lg.get("logGroupName", "").lower()]


def normalize_log_group_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


KNOWN_AWS_LOG_SEGMENTS: dict[str, str] = {
    "containerinsight": "containerinsights",
    "containerinsights": "containerinsights",
    "codebuild": "codebuild",
    "apigateway": "apigateway",
    "opensearch": "opensearch",
    "opensearchservice": "opensearchservice",
    "lambda": "lambda",
    "eks": "eks",
    "rds": "rds",
    "glue": "glue",
    "dynamodb": "dynamodb",
    "cognito": "cognito",
    "msk": "msk",
    "waf": "waf",
    "ec2": "ec2",
    "ecs": "ecs",
    "vpc": "vpc",
    "cloudtrail": "cloudtrail",
}


def fuzzy_token_match(query_token: str, candidate_token: str) -> bool:
    if not query_token or not candidate_token:
        return False
    if query_token == candidate_token:
        return True
    if query_token in candidate_token or candidate_token in query_token:
        return True
    if query_token + "s" == candidate_token or candidate_token + "s" == query_token:
        return True
    if len(query_token) >= 8 and candidate_token.startswith(query_token):
        return True
    if len(candidate_token) >= 8 and query_token.startswith(candidate_token):
        return True
    return False


def resolve_log_group_search_keyword(query: str) -> str:
    token = normalize_log_group_token(query)
    for key, canonical in KNOWN_AWS_LOG_SEGMENTS.items():
        key_token = normalize_log_group_token(key)
        if fuzzy_token_match(token, key_token):
            return canonical
    return token


def matches_log_group_fuzzy(name: str, query: str) -> bool:
    if not query:
        return True
    if query.lower() in name.lower():
        return True

    query_token = normalize_log_group_token(query)
    if not query_token:
        return True

    resolved = resolve_log_group_search_keyword(query)
    resolved_token = normalize_log_group_token(resolved)
    name_norm = normalize_log_group_token(name)

    if resolved_token and resolved_token in name_norm:
        return True
    if query_token in name_norm:
        return True

    for part in name.split("/"):
        part_token = normalize_log_group_token(part)
        if fuzzy_token_match(query_token, part_token):
            return True
        if fuzzy_token_match(resolved_token, part_token):
            return True

    return fuzzy_token_match(query_token, name_norm)


def filter_log_groups_fuzzy_any(
    log_groups: list[dict[str, Any]], queries: list[str]
) -> list[dict[str, Any]]:
    if not queries:
        return log_groups
    return [
        lg
        for lg in log_groups
        if any(matches_log_group_fuzzy(lg.get("logGroupName", ""), query) for query in queries)
    ]


def filter_log_groups_fuzzy(log_groups: list[dict[str, Any]], query: Optional[str]) -> list[dict[str, Any]]:
    if not query:
        return log_groups
    return [lg for lg in log_groups if matches_log_group_fuzzy(lg.get("logGroupName", ""), query)]


def normalize_log_group_prefix_arg(prefix: Optional[str]) -> Optional[str]:
    if not prefix:
        return None
    stripped = prefix.strip()
    if not stripped:
        return None
    if "/" in stripped:
        return stripped
    return resolve_log_group_search_keyword(stripped)


def dedupe_log_groups(log_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for log_group in log_groups:
        name = log_group.get("logGroupName", "")
        if name and name not in seen:
            seen.add(name)
            unique.append(log_group)
    return unique


def log_group_search_prefixes(prefix: str) -> list[str]:
    """Build AWS prefix candidates for a user search term."""
    normalized = prefix.strip()
    if not normalized:
        return []

    if is_keyword_log_group_search(normalized):
        keyword = resolve_log_group_search_keyword(normalized)
        candidates = [f"/aws/{keyword}", keyword]
        raw_token = normalize_log_group_token(normalized)
        if raw_token and raw_token != keyword:
            candidates.extend([f"/aws/{raw_token}", raw_token])
        return list(dict.fromkeys(candidates))

    return [normalized]


def is_keyword_log_group_search(prefix: str) -> bool:
    return "/" not in prefix.strip()


def fetch_log_groups(
    logs_client, *, prefix: Optional[str] = None, max_items: int = 50
) -> list[dict[str, Any]]:
    paginator = logs_client.get_paginator("describe_log_groups")
    kwargs: dict[str, Any] = {"PaginationConfig": {"MaxItems": max_items}}
    if prefix:
        kwargs["logGroupNamePrefix"] = prefix
    groups: list[dict[str, Any]] = []
    for page in paginator.paginate(**kwargs):
        groups.extend(page.get("logGroups", []))
    return groups


def slim_log_group(log_group: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields useful for LLM listing to save context window."""
    slim: dict[str, Any] = {"logGroupName": log_group.get("logGroupName", "")}
    if log_group.get("retentionInDays") is not None:
        slim["retentionInDays"] = log_group["retentionInDays"]
    if log_group.get("storedBytes"):
        slim["storedBytes"] = log_group["storedBytes"]
    return slim


def format_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    if num < 1024**2:
        return f"{num / 1024:.1f} KB"
    if num < 1024**3:
        return f"{num / 1024**2:.1f} MB"
    return f"{num / 1024**3:.1f} GB"


def parse_log_groups_tool_result(tool_result: str) -> dict[str, Any] | None:
    try:
        data = json.loads(tool_result)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "log_groups" not in data:
        return None
    return data


def format_log_groups_list(data: dict[str, Any]) -> str:
    groups = data.get("log_groups") or []
    region = data.get("region", AWS_REGION)
    count = data.get("count", len(groups))
    lines = [f"**{count} log grubu** ({region})", ""]
    if data.get("message"):
        lines.append(str(data["message"]))
        lines.append("")
    for index, group in enumerate(groups, 1):
        name = group.get("logGroupName", "")
        details: list[str] = []
        if group.get("retentionInDays") is not None:
            details.append(f"{group['retentionInDays']} gün saklama")
        if group.get("storedBytes"):
            details.append(format_bytes(int(group["storedBytes"])))
        suffix = f" – {', '.join(details)}" if details else ""
        lines.append(f"{index}. `{name}`{suffix}")
    return "\n".join(lines)


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

LOG_SEARCH_QUERY_PRESETS = {
    "errors": (
        "fields @timestamp, @log, @message\n"
        "| filter @message like /(?i)(\"Level\":\"Error\"|ACCESS_DENIED|errorResponseType|"
        "Unhandled exception|exception|fatal|stack trace)/\n"
        "| filter @message not like /\"errorMessage\":\"-\"|\"Level\":\"Information\"/\n"
        "| sort @timestamp desc\n"
        "| limit 50"
    ),
    "http_400": (
        "fields @timestamp, @log, @message\n"
        "| filter @message like /(?i)(statusCode[\"']?\\s*[:=]\\s*\"?400\"?|"
        "status.?code.?400|HTTP\\/1\\.[01] 400\\b)/\n"
        "| sort @timestamp desc\n"
        "| limit 50"
    ),
    "http_500": (
        "fields @timestamp, @log, @message\n"
        "| filter @message like /(?i)(statusCode[\"']?\\s*[:=]\\s*\"?500\"?|"
        "status.?code.?500|HTTP\\/1\\.[01] 500\\b|\\\"StatusCode\\\":500)/\n"
        "| sort @timestamp desc\n"
        "| limit 50"
    ),
    "http_5xx": (
        "fields @timestamp, @log, @message\n"
        "| filter @message like /(?i)(statusCode[\"']?\\s*[:=]\\s*\"?5\\d{2}\"?|"
        "status.?code.?5\\d{2}|HTTP\\/1\\.[01] 5\\d{2}\\b)/\n"
        "| sort @timestamp desc\n"
        "| limit 50"
    ),
}

LogSearchFilter = Literal["errors", "http_400", "http_500", "http_5xx"]

SHOW_LOG_GROUPS_AGAIN = re.compile(
    r"göstersene|göster\s*abi|listeyi\s*göster|tekrar\s*listele|hepsini\s*göster|göster\s*bana",
    re.I,
)

INSIGHTS_MAX_LOG_GROUPS = 50
DEFAULT_LOG_SEARCH_KEYWORDS = [
    "containerinsights",
    "api-gateway",
    "alb",
    "lambda",
    "cloudfront",
]


def normalize_log_group_path(raw: str) -> str:
    path = (raw or "").strip()
    if path.startswith("aws/"):
        return "/" + path
    if not path.startswith("/"):
        return "/" + path.lstrip("/")
    return path


def extract_log_group_paths(message: str) -> list[str]:
    paths: list[str] = []
    for match in LOG_GROUP_PATH_PATTERN.finditer(message):
        paths.append(normalize_log_group_path(match.group("path")))
    return list(dict.fromkeys(paths))


def parse_hours_from_message(message: str) -> int:
    lowered = message.lower()
    match = re.search(r"son\s+(\d+)\s*saat", lowered)
    if match:
        return min(24, max(1, int(match.group(1))))
    if "24 saat" in lowered or "dün" in lowered:
        return 24
    return 1


def infer_log_search_filter(message: str) -> LogSearchFilter:
    lowered = message.lower()
    if re.search(r"\b500\b|500\s*l", lowered):
        return "http_500"
    if re.search(r"5\s*xx", lowered):
        return "http_5xx"
    if re.search(r"4\s*xx|\b400\b", lowered):
        return "http_400"
    return "errors"


def normalize_insights_log_group(name: str) -> str:
    if not name:
        return name
    if re.match(r"^\d+:", name):
        return name.split(":", 1)[1]
    return name


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
    return bool(re.search(r"4\s*xx|\b400\b|5\s*xx|\b500\b|500\s*l", message, re.I))


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
            r"var\s*mı|içerik|satır|timeout",
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
            ts = iso_utc(ts.astimezone(timezone.utc))
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


def is_ecs_metric_request(message: str) -> bool:
    lowered = message.lower()
    if not re.search(r"\becs\b", lowered):
        return False
    return bool(re.search(r"cpu|memory|mem|metrik|metric|analiz", lowered))


def ecs_metrics_from_message(message: str) -> list[tuple[str, str]]:
    lowered = message.lower()
    metrics: list[tuple[str, str]] = []
    if re.search(r"cpu", lowered):
        metrics.append(("AWS/ECS", "CPUUtilization"))
    if re.search(r"memory|mem", lowered):
        metrics.append(("AWS/ECS", "MemoryUtilization"))
    if not metrics:
        metrics.append(("AWS/ECS", "CPUUtilization"))
    return metrics


def format_log_search_results(data: dict[str, Any]) -> str:
    """Format Insights search hits with real log group names only."""
    if data.get("error"):
        return f"Log araması başarısız: {data['error']}"

    hours = data.get("hours", 1)
    match_count = int(data.get("match_count") or 0)
    groups_searched = int(data.get("log_groups_searched") or 0)
    status = data.get("status", "Unknown")

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
                    "message": str(item.get("message", ""))[:300],
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
                    "message": fields.get("@message", "")[:300],
                }
            )

    all_groups = data.get("all_log_group_names") or data.get("log_group_names") or []
    if data.get("log_group_name"):
        all_groups = [str(data["log_group_name"])] + [g for g in all_groups if g != data["log_group_name"]]
    if not rows:
        sample = ", ".join(f"`{name}`" for name in all_groups[:12])
        suffix = f"\n\nLog group: {sample}" if sample else ""
        if data.get("log_group_name"):
            return (
                f"`{data['log_group_name']}` — son {hours} saat içinde eşleşen satır bulunamadı "
                f"(durum: {status}).{suffix}"
            )
        return (
            f"Son {hours} saat içinde {groups_searched} log grubunda arama yapıldı — "
            f"eşleşen satır bulunamadı (durum: {status}).{suffix}"
        )

    by_group: dict[str, int] = {}
    for row in rows:
        log_group = (row.get("log_group") or "").strip() or "log-group-bilinmiyor"
        by_group[log_group] = by_group.get(log_group, 0) + 1

    lines = []
    if data.get("log_group_name"):
        lines.append(
            f"**`{data['log_group_name']}` — son {hours} saat — {match_count} eşleşen satır** "
            f"(durum: {status})"
        )
    else:
        lines.append(
            f"**Son {hours} saat — {match_count} eşleşen satır** "
            f"({groups_searched} log grubunda arandı, durum: {status})"
        )
    lines.extend(["", "**Log gruplarına göre:**"])
    for log_group, count in sorted(by_group.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{normalize_insights_log_group(log_group)}`: {count} satır")

    lines.extend(["", "**Detay:**", ""])
    display_limit = 80
    for index, row in enumerate(rows[:display_limit], 1):
        log_group = normalize_insights_log_group(row.get("log_group", ""))
        timestamp = row.get("timestamp", "")
        message = str(row.get("message", "")).replace("\n", " ")
        lines.append(f"{index}. `{log_group}` — {timestamp}\n   {message}")

    if data.get("truncated") or match_count > len(rows[:display_limit]):
        remaining = max(0, match_count - min(len(rows), display_limit))
        if remaining:
            lines.append(f"\n... ve {remaining} satır daha (sorgu limiti içinde).")

    if all_groups:
        lines.append(f"\n**Aranan log grupları ({len(all_groups)}):**")
        for name in all_groups[:40]:
            lines.append(f"- `{normalize_insights_log_group(name)}`")
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

    data = parse_log_groups_tool_result(tool_result)
    if data is None:
        data = parsed if isinstance(parsed, dict) else None
    if not data:
        return None
    if tool_name == "describe_log_groups" and "log_groups" in data:
        return format_log_groups_list(data)
    if tool_name == "get_active_alarms":
        return format_active_alarms_list(data)
    if tool_name == "get_alarm_history":
        return format_alarm_history(data)
    if tool_name == "analyze_metric":
        return format_analyze_metric(data)
    return None


def adjust_describe_log_groups_args(
    _user_message: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Clamp limits only; intent/keywords come from the LLM via prompt + tool schema."""
    args = dict(arguments)
    requested = int(args.get("max_items") or 50)
    args["max_items"] = min(MAX_LOG_GROUPS_LIST, max(1, requested))
    raw_keywords = args.get("log_group_name_keywords")
    if isinstance(raw_keywords, list):
        args["log_group_name_keywords"] = dedupe_search_keywords(raw_keywords)
    return args


def dedupe_search_keywords(keywords: list[str]) -> list[str]:
    """Merge synonyms like 'container insights' + 'containerinsight' into one term."""
    seen: set[str] = set()
    unique: list[str] = []
    for keyword in keywords:
        term = (keyword or "").strip()
        if not term:
            continue
        canonical = (
            resolve_log_group_search_keyword(term)
            if is_keyword_log_group_search(term)
            else normalize_log_group_token(term)
        )
        if canonical in seen:
            continue
        seen.add(canonical)
        unique.append(term)
    return unique


def summarize_message_for_history(content: str) -> str:
    """Shrink huge list replies so the next turn is not polluted."""
    if re.search(r"^\*\*\d+\s+log grubu\*\*", content, re.M):
        count = re.search(r"\*\*(\d+)\s+log grubu\*\*", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} log grubu listelendi. Sonraki soru bağımsız değerlendir.]"
    if re.search(r"^\*\*\d+\s+aktif alarm\*\*", content, re.M):
        count = re.search(r"\*\*(\d+)\s+aktif alarm\*\*", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} aktif alarm listelendi.]"
    if re.search(r"^\*\*Son \d+ saat — \d+ eşleşen satır\*\*", content, re.M):
        count = re.search(r"— (\d+) eşleşen satır", content)
        n = count.group(1) if count else "?"
        return f"[Önceki: {n} log satırı arandı ve gerçek gruplarla listelendi.]"
    return trim_message_content(content)


def collect_log_group_names_for_search(
    logs_client,
    keywords: list[str],
    *,
    max_groups: int = INSIGHTS_MAX_LOG_GROUPS,
) -> list[str]:
    names: list[str] = []
    per_keyword = max(5, max_groups // max(1, len(keywords)))
    for keyword in keywords:
        prefix = normalize_log_group_prefix_arg(keyword) or keyword
        filtered, _ = _search_log_groups_by_keyword(
            logs_client,
            search_label=keyword,
            prefix=prefix,
            max_items=per_keyword,
        )
        for item in filtered:
            name = item.get("logGroupName")
            if name:
                names.append(name)
        if len(names) >= max_groups:
            break
    return list(dict.fromkeys(names))[:max_groups]


def format_insights_rows(results: list[Any], *, limit: int = 30) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in results[:limit]:
        if not isinstance(row, list):
            continue
        fields = {
            item.get("field", ""): str(item.get("value", ""))
            for item in row
            if isinstance(item, dict)
        }
        rows.append(
            {
                "timestamp": fields.get("@timestamp", ""),
                "log_group": normalize_insights_log_group(
                    fields.get("@log", fields.get("@logGroup", ""))
                ),
                "message": fields.get("@message", "")[:240],
            }
        )
    return rows


def format_insights_rows_with_group(
    results: list[Any], *, log_group_name: str, limit: int = 50
) -> list[dict[str, str]]:
    rows = format_insights_rows(results, limit=limit)
    for row in rows:
        if not row.get("log_group"):
            row["log_group"] = log_group_name
        else:
            row["log_group"] = normalize_insights_log_group(row["log_group"])
    return rows


def compact_tool_result_for_context(tool_name: str, tool_result: str) -> str:
    """Keep LLM context small after tool calls (mechanical, not intent-based)."""
    if tool_name == "describe_log_groups":
        data = parse_log_groups_tool_result(tool_result)
        if data:
            names = [
                group.get("logGroupName", "")
                for group in data.get("log_groups", [])
                if group.get("logGroupName")
            ]
            payload = {
                "count": len(names),
                "log_group_names": names[:INSIGHTS_MAX_LOG_GROUPS],
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
                "results": format_insights_rows(results, limit=20),
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


def wait_for_logs_query(logs_client, query_id: str) -> dict[str, Any]:
    deadline = time.time() + LOGS_QUERY_MAX_WAIT
    while time.time() < deadline:
        response = logs_client.get_query_results(queryId=query_id)
        status = response.get("status")
        if status in {"Complete", "Failed", "Cancelled", "Timeout", "Unknown"}:
            return response
        time.sleep(LOGS_QUERY_POLL_INTERVAL)
    return logs_client.get_query_results(queryId=query_id)


# ---------------------------------------------------------------------------
# CloudWatch MCP tools (boto3)
# ---------------------------------------------------------------------------


def _search_log_groups_by_keyword(
    logs_client,
    *,
    search_label: str,
    prefix: str,
    max_items: int,
) -> tuple[list[dict[str, Any]], bool]:
    collected: list[dict[str, Any]] = []
    for candidate_prefix in log_group_search_prefixes(prefix):
        collected.extend(
            fetch_log_groups(logs_client, prefix=candidate_prefix, max_items=max_items)
        )
    collected = dedupe_log_groups(collected)

    fallback_applied = False
    if is_keyword_log_group_search(search_label):
        filtered = filter_log_groups_fuzzy(collected, search_label)
        need_broad_scan = len(filtered) < max_items and len(collected) < max_items
        if need_broad_scan:
            broad_groups = fetch_log_groups(logs_client, max_items=MAX_LOG_GROUPS_LIST)
            broad_filtered = filter_log_groups_fuzzy(broad_groups, search_label)
            merged = dedupe_log_groups(filtered + broad_filtered)
            fallback_applied = len(merged) > len(filtered)
            filtered = merged[:max_items]
    else:
        filtered = filter_log_groups(collected, prefix)[:max_items]
        if not filtered:
            fallback_applied = True
            broad_groups = fetch_log_groups(
                logs_client, max_items=min(max_items * 5, MAX_LOG_GROUPS_LIST)
            )
            filtered = filter_log_groups_fuzzy(broad_groups, search_label)[:max_items]

    return filtered, fallback_applied


@mcp.tool(name="describe_log_groups", description=CLOUDWATCH_TOOLS["describe_log_groups"])
def describe_log_groups(
    log_group_name_prefix: Annotated[
        Optional[str],
        Field(
            description=(
                "Single filter: pass the user's term exactly as they wrote it "
                "(e.g. 'container insights', 'codebuild', '/aws/lambda/'). "
                "Leave null only when listing all log groups. Fuzzy matching is automatic."
            ),
        ),
    ] = None,
    log_group_name_keywords: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "Multiple filters OR'd together when the user asks for several types in one "
                "message. Example: ['container insights', 'lambda', 'codebuild']. "
                "Use this instead of log_group_name_prefix when there are 2+ distinct topics."
            ),
        ),
    ] = None,
    max_items: Annotated[
        int,
        Field(description="Max log groups to return. Use 1000 for full/tüm/hepsi requests."),
    ] = 50,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    max_items = min(MAX_LOG_GROUPS_LIST, max(1, max_items))

    keywords = dedupe_search_keywords([k.strip() for k in (log_group_name_keywords or []) if k and k.strip()])
    if not keywords and log_group_name_prefix:
        keywords = [log_group_name_prefix.strip()]

    if not keywords:
        log_groups = fetch_log_groups(logs_client, max_items=max_items)
        slim_groups = [slim_log_group(lg) for lg in log_groups]
        return {
            "region": region or AWS_REGION,
            "count": len(slim_groups),
            "log_groups": slim_groups,
            "fallback_contains_used": False,
            "message": (
                f"Returned first {len(slim_groups)} log groups. "
                "Use a prefix/keyword if you need a filtered list."
                if len(slim_groups) >= max_items
                else None
            ),
        }

    if len(keywords) == 1:
        search_label = keywords[0]
        prefix = normalize_log_group_prefix_arg(search_label) or search_label
        filtered, fallback_applied = _search_log_groups_by_keyword(
            logs_client,
            search_label=search_label,
            prefix=prefix,
            max_items=max_items,
        )
        message = None
        if not filtered:
            message = (
                f"'{search_label}' için log group bulunamadı. "
                "Yazım varyasyonları ve /aws/ prefix'leri denendi."
            )
        elif fallback_applied or is_keyword_log_group_search(search_label):
            message = (
                f"'{search_label}' ile eşleşen log grupları listelendi "
                "(yaklaşık/eksik yazım desteklenir)."
            )
        return {
            "region": region or AWS_REGION,
            "count": len(filtered),
            "log_groups": [slim_log_group(lg) for lg in filtered],
            "fallback_contains_used": fallback_applied,
            "message": message,
        }

    merged: list[dict[str, Any]] = []
    fallback_applied = False
    per_keyword_limit = max(max_items, MAX_LOG_GROUPS_LIST)
    for keyword in keywords:
        prefix = normalize_log_group_prefix_arg(keyword) or keyword
        filtered, keyword_fallback = _search_log_groups_by_keyword(
            logs_client,
            search_label=keyword,
            prefix=prefix,
            max_items=per_keyword_limit,
        )
        fallback_applied = fallback_applied or keyword_fallback
        merged = dedupe_log_groups(merged + filtered)

    merged.sort(key=lambda group: group.get("logGroupName", ""))
    label = "', '".join(keywords)
    message = (
        f"'{label}' ile eşleşen log grupları listelendi "
        f"(çoklu filtre, {len(keywords)} kategori)."
    )
    if not merged:
        message = (
            f"'{label}' için log group bulunamadı. "
            "Yazım varyasyonları ve /aws/ prefix'leri denendi."
        )

    return {
        "region": region or AWS_REGION,
        "count": len(merged),
        "log_groups": [slim_log_group(lg) for lg in merged],
        "fallback_contains_used": fallback_applied,
        "message": message,
        "matched_keywords": keywords,
    }


@mcp.tool(name="query_log_group", description=CLOUDWATCH_TOOLS["query_log_group"])
def query_log_group(
    log_group_name: Annotated[
        str,
        Field(description="Full CloudWatch log group path, e.g. /aws/apigateway/my-api/access-logs"),
    ],
    hours: Annotated[int, Field(description="Lookback window in hours. Default 1.")] = 1,
    search_filter: Annotated[
        LogSearchFilter,
        Field(description="Preset Insights filter for errors or HTTP status codes."),
    ] = "errors",
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    log_group_name = normalize_log_group_path(log_group_name)
    hours = min(24, max(1, int(hours)))
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    query_string = LOG_SEARCH_QUERY_PRESETS[search_filter]

    try:
        response = logs_client.start_query(
            logGroupName=log_group_name,
            startTime=to_epoch_seconds(start_dt),
            endTime=to_epoch_seconds(end_dt),
            queryString=query_string,
        )
    except ClientError as exc:
        return {
            "error": str(exc),
            "log_group_name": log_group_name,
            "hours": hours,
            "search_filter": search_filter,
            "match_count": 0,
            "results": [],
        }

    completed = wait_for_logs_query(logs_client, response["queryId"])
    raw_results = completed.get("results", [])
    formatted = format_insights_rows_with_group(
        raw_results, log_group_name=log_group_name, limit=50
    )
    return {
        "status": completed.get("status"),
        "region": region or AWS_REGION,
        "log_group_name": log_group_name,
        "hours": hours,
        "search_filter": search_filter,
        "log_groups_searched": 1,
        "all_log_group_names": [log_group_name],
        "match_count": len(raw_results),
        "results": formatted,
        "truncated": len(raw_results) > 50,
    }


@mcp.tool(name="analyze_log_group", description=CLOUDWATCH_TOOLS["analyze_log_group"])
def analyze_log_group(
    log_group_name: Annotated[str, Field(description="CloudWatch log group name.")],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[str, Field(description="ISO-8601 UTC end time.")],
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    start = to_epoch_seconds(parse_time(start_time))
    end = to_epoch_seconds(parse_time(end_time))

    error_query = """
fields @timestamp, @message
| filter @message like /(?i)(error|exception|fail|timeout)/
| stats count() as error_count by bin(5m)
| sort @timestamp desc
| limit 20
"""
    error_response = logs_client.start_query(
        logGroupName=log_group_name,
        startTime=start,
        endTime=end,
        queryString=error_query,
    )
    error_results = wait_for_logs_query(logs_client, error_response["queryId"])

    top_query = """
fields @message
| stats count() as occurrences by @message
| sort occurrences desc
| limit 10
"""
    top_response = logs_client.start_query(
        logGroupName=log_group_name,
        startTime=start,
        endTime=end,
        queryString=top_query,
    )
    top_results = wait_for_logs_query(logs_client, top_response["queryId"])

    return {
        "log_group_name": log_group_name,
        "region": region or AWS_REGION,
        "error_patterns": error_results.get("results", []),
        "top_messages": top_results.get("results", []),
    }


@mcp.tool(name="execute_log_insights_query", description=CLOUDWATCH_TOOLS["execute_log_insights_query"])
def execute_log_insights_query(
    log_group_names: Annotated[
        list[str],
        Field(description="1–50 log group names. Get via describe_log_groups if user did not specify."),
    ],
    query_string: Annotated[
        str,
        Field(
            description=(
                "Logs Insights query. Error search example: "
                "fields @timestamp, @log, @message | filter @message like /(?i)(error|exception|fail)/ "
                "| sort @timestamp desc | limit 50"
            )
        ),
    ],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[str, Field(description="ISO-8601 UTC end time.")],
    wait_for_completion: Annotated[bool, Field(description="Wait until query completes.")] = True,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    start = to_epoch_seconds(parse_time(start_time))
    end = to_epoch_seconds(parse_time(end_time))

    try:
        response = logs_client.start_query(
            logGroupNames=log_group_names,
            startTime=start,
            endTime=end,
            queryString=query_string,
        )
    except ClientError as exc:
        return {
            "error": str(exc),
            "log_group_names": log_group_names,
            "query_string": query_string,
        }

    query_id = response["queryId"]
    result: dict[str, Any] = {"query_id": query_id, "status": "Running"}

    if wait_for_completion:
        completed = wait_for_logs_query(logs_client, query_id)
        raw_results = completed.get("results", [])
        result.update(
            {
                "status": completed.get("status"),
                "results": format_insights_rows(raw_results, limit=200),
                "statistics": completed.get("statistics", {}),
                "log_groups_searched": len(log_group_names),
                "all_log_group_names": log_group_names,
                "match_count": len(raw_results),
                "truncated": len(raw_results) > 200,
            }
        )
    return result


@mcp.tool(name="search_logs_across_groups", description=CLOUDWATCH_TOOLS["search_logs_across_groups"])
def search_logs_across_groups(
    query_string: Annotated[
        str,
        Field(
            description=(
                "CloudWatch Logs Insights query. Examples: "
                "400 → filter @message like /(?i)(\\b400\\b|status.?code.?400)/; "
                "5XX → filter @message like /(?i)(\\b5\\d{2}\\b|status.?code.?5\\d{2})/"
            )
        ),
    ],
    hours: Annotated[int, Field(description="Lookback window in hours. Default 1.")] = 1,
    max_result_lines: Annotated[
        int,
        Field(description="Max matching log lines to return. Use 100–200 for detailed requests."),
    ] = 50,
    log_group_name_keywords: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "Optional log group filters. If omitted, searches containerinsights, "
                "api-gateway, alb, lambda, cloudfront."
            )
        ),
    ] = None,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    keywords = dedupe_search_keywords(
        [k.strip() for k in (log_group_name_keywords or DEFAULT_LOG_SEARCH_KEYWORDS) if k and k.strip()]
    )
    group_names = collect_log_group_names_for_search(logs_client, keywords)
    if not group_names:
        return {
            "error": "Arama için uygun log group bulunamadı.",
            "keywords_tried": keywords,
            "match_count": 0,
            "results": [],
        }

    hours = min(24, max(1, int(hours)))
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    start = to_epoch_seconds(start_dt)
    end = to_epoch_seconds(end_dt)

    try:
        response = logs_client.start_query(
            logGroupNames=group_names,
            startTime=start,
            endTime=end,
            queryString=query_string,
        )
    except ClientError as exc:
        return {
            "error": str(exc),
            "log_groups_searched": len(group_names),
            "log_group_names": group_names[:10],
            "keywords_tried": keywords,
            "match_count": 0,
            "results": [],
        }

    completed = wait_for_logs_query(logs_client, response["queryId"])
    results = completed.get("results", [])
    max_lines = min(200, max(1, int(max_result_lines)))
    formatted = format_insights_rows(results, limit=max_lines)
    return {
        "status": completed.get("status"),
        "region": region or AWS_REGION,
        "hours": hours,
        "log_groups_searched": len(group_names),
        "all_log_group_names": group_names,
        "log_group_names": group_names[:15],
        "keywords_tried": keywords,
        "query_string": query_string,
        "match_count": len(results),
        "results": formatted,
        "truncated": len(results) > max_lines,
    }


@mcp.tool(name="get_logs_insight_query_results", description=CLOUDWATCH_TOOLS["get_logs_insight_query_results"])
def get_logs_insight_query_results(
    query_id: Annotated[str, Field(description="Logs Insights query ID.")],
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    response = logs_client.get_query_results(queryId=query_id)
    return {
        "query_id": query_id,
        "status": response.get("status"),
        "results": response.get("results", []),
        "statistics": response.get("statistics", {}),
    }


@mcp.tool(name="cancel_logs_insight_query", description=CLOUDWATCH_TOOLS["cancel_logs_insight_query"])
def cancel_logs_insight_query(
    query_id: Annotated[str, Field(description="Logs Insights query ID to cancel.")],
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    logs_client.stop_query(queryId=query_id)
    return {"query_id": query_id, "status": "Cancelled"}


@mcp.tool(name="get_active_alarms", description=CLOUDWATCH_TOOLS["get_active_alarms"])
def get_active_alarms(
    max_items: Annotated[int, Field(description="Maximum alarms to return.")] = 50,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    cloudwatch = get_client("cloudwatch", region)
    paginator = cloudwatch.get_paginator("describe_alarms")
    metric_alarms: list[dict[str, Any]] = []
    composite_alarms: list[dict[str, Any]] = []

    for page in paginator.paginate(
        StateValue="ALARM",
        AlarmTypes=["MetricAlarm", "CompositeAlarm"],
        PaginationConfig={"MaxItems": max_items},
    ):
        metric_alarms.extend(page.get("MetricAlarms", []))
        composite_alarms.extend(page.get("CompositeAlarms", []))

    return {
        "region": region or AWS_REGION,
        "metric_alarms": metric_alarms[:max_items],
        "composite_alarms": composite_alarms[:max_items],
        "count": min(len(metric_alarms) + len(composite_alarms), max_items),
    }


@mcp.tool(name="get_alarm_history", description=CLOUDWATCH_TOOLS["get_alarm_history"])
def get_alarm_history(
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[str, Field(description="ISO-8601 UTC end time.")],
    alarm_name: Annotated[
        Optional[str],
        Field(description="Optional alarm name. Omit to list all alarm transitions in range."),
    ] = None,
    max_records: Annotated[int, Field(description="Maximum history records.")] = 100,
    history_item_type: Annotated[
        Literal["ConfigurationUpdate", "StateUpdate", "Action"],
        Field(description="History item type filter."),
    ] = "StateUpdate",
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    cloudwatch = get_client("cloudwatch", region)
    start_dt = parse_time(start_time)
    end_dt = parse_time(end_time)
    hours = max(1, int((end_dt - start_dt).total_seconds() // 3600))

    kwargs: dict[str, Any] = {
        "StartDate": start_dt,
        "EndDate": end_dt,
        "MaxRecords": min(500, max(1, max_records)),
        "HistoryItemType": history_item_type,
    }
    if alarm_name and alarm_name.strip():
        kwargs["AlarmName"] = alarm_name.strip()

    try:
        response = cloudwatch.describe_alarm_history(**kwargs)
    except ClientError as exc:
        return {
            "error": str(exc),
            "alarm_name": alarm_name,
            "region": region or AWS_REGION,
            "hours": hours,
            "history_items": [],
        }

    return {
        "alarm_name": alarm_name,
        "region": region or AWS_REGION,
        "hours": hours,
        "history_items": response.get("AlarmHistoryItems", []),
    }


@mcp.tool(name="get_metric_data", description=CLOUDWATCH_TOOLS["get_metric_data"])
def get_metric_data(
    namespace: Annotated[str, Field(description="CloudWatch metric namespace.")],
    metric_name: Annotated[str, Field(description="CloudWatch metric name.")],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[Optional[str], Field(description="ISO-8601 UTC end time.")] = None,
    statistic: Annotated[
        Literal["Average", "Sum", "Maximum", "Minimum", "SampleCount"],
        Field(description="Metric statistic."),
    ] = "Average",
    period: Annotated[int, Field(description="Period in seconds.")] = 300,
    dimensions: Annotated[
        Optional[list[dict[str, str]]],
        Field(description='Dimensions, e.g. [{"Name": "InstanceId", "Value": "i-123"}].'),
    ] = None,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    cloudwatch = get_client("cloudwatch", region)
    end_dt = parse_time(end_time, default=utc_now())
    start_dt = parse_time(start_time)

    metric_query = {
        "Id": "m1",
        "MetricStat": {
            "Metric": {
                "Namespace": namespace,
                "MetricName": metric_name,
                "Dimensions": dimensions or [],
            },
            "Period": period,
            "Stat": statistic,
        },
    }

    response = cloudwatch.get_metric_data(
        MetricDataQueries=[metric_query],
        StartTime=start_dt,
        EndTime=end_dt,
    )
    return {
        "region": region or AWS_REGION,
        "namespace": namespace,
        "metric_name": metric_name,
        "statistic": statistic,
        "results": response.get("MetricDataResults", []),
    }


@mcp.tool(name="get_metric_metadata", description=CLOUDWATCH_TOOLS["get_metric_metadata"])
def get_metric_metadata(
    namespace: Annotated[Optional[str], Field(description="Metric namespace filter.")] = None,
    metric_name: Annotated[Optional[str], Field(description="Metric name filter.")] = None,
    max_items: Annotated[int, Field(description="Maximum metrics to return.")] = 100,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    cloudwatch = get_client("cloudwatch", region)
    paginator = cloudwatch.get_paginator("list_metrics")
    kwargs: dict[str, Any] = {"PaginationConfig": {"MaxItems": max_items}}
    if namespace:
        kwargs["Namespace"] = namespace
    if metric_name:
        kwargs["MetricName"] = metric_name

    metrics: list[dict[str, Any]] = []
    for page in paginator.paginate(**kwargs):
        metrics.extend(page.get("Metrics", []))

    return {
        "region": region or AWS_REGION,
        "count": len(metrics),
        "metrics": metrics,
    }


@mcp.tool(name="get_recommended_metric_alarms", description=CLOUDWATCH_TOOLS["get_recommended_metric_alarms"])
def get_recommended_metric_alarms(
    namespace: Annotated[str, Field(description="CloudWatch metric namespace.")],
    metric_name: Annotated[str, Field(description="CloudWatch metric name.")],
    statistic: Annotated[
        Literal["Average", "Sum", "Maximum", "Minimum", "SampleCount"],
        Field(description="Metric statistic."),
    ] = "Average",
    period: Annotated[int, Field(description="Period in seconds.")] = 300,
    dimensions: Annotated[Optional[list[dict[str, str]]], Field(description="Metric dimensions.")] = None,
    lookback_hours: Annotated[int, Field(description="Hours of history for recommendation.")] = 24,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=lookback_hours)
    metric_data = get_metric_data(
        namespace=namespace,
        metric_name=metric_name,
        start_time=iso_utc(start_dt),
        end_time=iso_utc(end_dt),
        statistic=statistic,
        period=period,
        dimensions=dimensions,
        region=region,
    )

    values: list[float] = []
    for result in metric_data.get("results", []):
        for point in result.get("Values", []):
            values.append(float(point))

    if not values:
        return {
            "namespace": namespace,
            "metric_name": metric_name,
            "message": "Not enough data to recommend alarm thresholds.",
        }

    average = sum(values) / len(values)
    maximum = max(values)
    p95 = sorted(values)[int(len(values) * 0.95) - 1]

    return {
        "namespace": namespace,
        "metric_name": metric_name,
        "statistic": statistic,
        "sample_size": len(values),
        "recommendations": {
            "warning_threshold": round(p95, 4),
            "critical_threshold": round(maximum, 4),
            "baseline_average": round(average, 4),
            "evaluation_periods": 2,
            "datapoints_to_alarm": 2,
        },
    }


@mcp.tool(name="analyze_metric", description=CLOUDWATCH_TOOLS["analyze_metric"])
def analyze_metric(
    namespace: Annotated[str, Field(description="CloudWatch metric namespace.")],
    metric_name: Annotated[str, Field(description="CloudWatch metric name.")],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[Optional[str], Field(description="ISO-8601 UTC end time.")] = None,
    statistic: Annotated[
        Literal["Average", "Sum", "Maximum", "Minimum", "SampleCount"],
        Field(description="Metric statistic."),
    ] = "Average",
    period: Annotated[int, Field(description="Period in seconds.")] = 300,
    dimensions: Annotated[Optional[list[dict[str, str]]], Field(description="Metric dimensions.")] = None,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    metric_data = get_metric_data(
        namespace=namespace,
        metric_name=metric_name,
        start_time=start_time,
        end_time=end_time,
        statistic=statistic,
        period=period,
        dimensions=dimensions,
        region=region,
    )

    values: list[float] = []
    timestamps: list[str] = []
    for result in metric_data.get("results", []):
        values.extend(float(v) for v in result.get("Values", []))
        timestamps.extend(result.get("Timestamps", []))

    if not values:
        return {
            "namespace": namespace,
            "metric_name": metric_name,
            "message": "No metric datapoints found for the given time range.",
        }

    first_half = values[: len(values) // 2] or values
    second_half = values[len(values) // 2 :] or values
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    trend = "stable"
    if second_avg > first_avg * 1.1:
        trend = "increasing"
    elif second_avg < first_avg * 0.9:
        trend = "decreasing"

    return {
        "namespace": namespace,
        "metric_name": metric_name,
        "statistic": statistic,
        "datapoint_count": len(values),
        "min": min(values),
        "max": max(values),
        "average": sum(values) / len(values),
        "trend": trend,
        "first_timestamp": iso_utc(timestamps[0]) if timestamps else None,
        "last_timestamp": iso_utc(timestamps[-1]) if timestamps else None,
    }


logger.info("CloudWatch MCP tools registered (boto3)")

# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

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
            "description": CLOUDWATCH_TOOLS.get(tool.name, tool.description or ""),
            "parameters": tool.inputSchema,
        },
    }


async def get_openai_tools() -> list[dict[str, Any]]:
    global _openai_tools_cache
    if _openai_tools_cache is None:
        tools = await mcp.list_tools()
        _openai_tools_cache = [
            _mcp_tool_to_openai(tool)
            for tool in tools
            if tool.name in ALLOWED_TOOL_NAMES
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


class CloudWatchMcpSession:
    def __init__(self) -> None:
        self._session_cm = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "CloudWatchMcpSession":
        self._session_cm = create_connected_server_and_client_session(mcp._mcp_server)
        self.session = await self._session_cm.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        self.session = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self.session is None:
            raise RuntimeError("MCP session is not initialized")
        if name not in ALLOWED_TOOL_NAMES:
            raise ValueError(f"Tool '{name}' is not allowed. Use one of: {sorted(ALLOWED_TOOL_NAMES)}")

        logger.info("Calling tool %s", name)
        result = await self.session.call_tool(name, arguments)
        return serialize_tool_result(result)


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
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "search_logs_across_groups", "arguments": tool_args}],
        "iterations": 1,
    }


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
    return {
        "response": direct or tool_result,
        "tool_calls": [{"name": "query_log_group", "arguments": tool_args}],
        "iterations": 1,
    }


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
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    tool_args = {
        "start_time": iso_utc(start_dt),
        "end_time": iso_utc(end_dt),
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
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    metrics = ecs_metrics_from_message(message)
    tool_calls: list[dict[str, Any]] = []
    sections: list[str] = []

    for namespace, metric_name in metrics:
        tool_args = {
            "namespace": namespace,
            "metric_name": metric_name,
            "start_time": iso_utc(start_dt),
            "end_time": iso_utc(end_dt),
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
    tools = await get_openai_tools()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    trimmed_history = [] if _retry_without_history else trim_conversation_history(conversation_history)
    if trimmed_history:
        messages.extend(trimmed_history)

    messages.append({"role": "user", "content": message})
    tool_calls_made: list[dict[str, Any]] = []
    last_direct_format: tuple[str, str] | None = None

    async with CloudWatchMcpSession() as mcp_session:
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

        ecs_metrics = await run_ecs_metric_analysis(message, mcp_session)
        if ecs_metrics is not None:
            return ecs_metrics

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
                if (
                    not _retry_without_history
                    and ("context length" in err_text or "maximum context" in err_text)
                ):
                    logger.warning("Context length exceeded; retrying without history")
                    return await run_agent(
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
                if not content:
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
                            return {
                                "response": direct,
                                "tool_calls": tool_calls_made,
                                "iterations": iteration + 1,
                            }
                return {
                    "response": humanize_response(assistant_message.content or ""),
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

            messages.append(assistant_message.model_dump(exclude_none=True))

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}

                if tool_name == "describe_log_groups":
                    arguments = adjust_describe_log_groups_args(message, arguments)
                    if arguments.get("log_group_name_prefix"):
                        arguments["log_group_name_prefix"] = normalize_log_group_prefix_arg(
                            arguments["log_group_name_prefix"]
                        )

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

            if (
                len(assistant_message.tool_calls) == 1
                and assistant_message.tool_calls[0].function.name in IMMEDIATE_FORMAT_TOOLS
                and last_direct_format is not None
            ):
                direct_name, direct_result = last_direct_format
                direct = try_direct_tool_response(direct_name, direct_result)
                if direct is not None:
                    return {
                        "response": direct,
                        "tool_calls": tool_calls_made,
                        "iterations": iteration + 1,
                    }

    return {
        "response": "Maximum tool iterations reached. Partial investigation completed.",
        "tool_calls": tool_calls_made,
        "iterations": MAX_TOOL_ITERATIONS,
    }


# ---------------------------------------------------------------------------
# Dashboard API (direct CloudWatch — no LLM)
# ---------------------------------------------------------------------------

DASHBOARD_ERROR_FILTER = (
    '| filter @message like /(?i)("Level":"Error"|ACCESS_DENIED|errorResponseType|statusCode":"5|'
    'HTTP\\/1\\.[01] 5)/\n'
    '| filter @message not like /"errorMessage":"-"|"Level":"Information"/\n'
)

_dashboard_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _insights_fields(row: list[Any]) -> dict[str, str]:
    return {
        str(item.get("field", "")): str(item.get("value", ""))
        for item in row
        if isinstance(item, dict)
    }


def _parse_timeline_results(results: list[Any]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, list):
            continue
        fields = _insights_fields(row)
        timestamp = fields.get("@timestamp", "")
        if not timestamp:
            for key, value in fields.items():
                if key.startswith("bin("):
                    timestamp = value
                    break
        try:
            count = int(float(fields.get("errors", 0) or 0))
        except ValueError:
            count = 0
        if timestamp:
            timeline.append({"timestamp": timestamp, "count": count})
    return timeline


def _parse_errors_by_log_group(results: list[Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, list):
            continue
        fields = _insights_fields(row)
        log_group = normalize_insights_log_group(
            fields.get("@log", fields.get("@logGroup", ""))
        )
        if not log_group:
            continue
        try:
            count = int(float(fields.get("errors", 0) or 0))
        except ValueError:
            count = 0
        groups.append({"log_group": log_group, "count": count})
    return groups


def _run_insights_query(
    logs_client,
    *,
    log_group_names: list[str],
    query_string: str,
    start: int,
    end: int,
) -> list[Any]:
    if not log_group_names:
        return []
    response = logs_client.start_query(
        logGroupNames=log_group_names,
        startTime=start,
        endTime=end,
        queryString=query_string,
    )
    completed = wait_for_logs_query(logs_client, response["queryId"])
    if completed.get("status") != "Complete":
        return []
    return completed.get("results") or []


def get_dashboard_overview(hours: int = 24) -> dict[str, Any]:
    hours = min(72, max(1, int(hours)))
    cache_key = f"overview:{hours}"
    cached = _dashboard_cache.get(cache_key)
    if cached and time.time() - cached[0] < DASHBOARD_CACHE_TTL:
        return cached[1]

    logs_client = get_client("logs")
    group_names = collect_log_group_names_for_search(logs_client, DEFAULT_LOG_SEARCH_KEYWORDS)
    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    start = to_epoch_seconds(start_dt)
    end = to_epoch_seconds(end_dt)
    bin_width = "15m" if hours <= 6 else ("1h" if hours <= 24 else "3h")

    timeline_query = (
        "fields @timestamp\n"
        f"{DASHBOARD_ERROR_FILTER}"
        f"| stats count() as errors by bin({bin_width})\n"
        "| sort @timestamp asc"
    )
    by_group_query = (
        "fields @log\n"
        f"{DASHBOARD_ERROR_FILTER}"
        "| stats count() as errors by @log\n"
        "| sort errors desc\n"
        "| limit 12"
    )

    errors_timeline: list[dict[str, Any]] = []
    errors_by_log_group: list[dict[str, Any]] = []
    insights_error: str | None = None

    if group_names:
        try:
            timeline_results = _run_insights_query(
                logs_client,
                log_group_names=group_names,
                query_string=timeline_query,
                start=start,
                end=end,
            )
            errors_timeline = _parse_timeline_results(timeline_results)
            by_group_results = _run_insights_query(
                logs_client,
                log_group_names=group_names,
                query_string=by_group_query,
                start=start,
                end=end,
            )
            errors_by_log_group = _parse_errors_by_log_group(by_group_results)
        except ClientError as exc:
            insights_error = str(exc)

    alarms_data = get_active_alarms(max_items=100)
    metric_alarms = alarms_data.get("metric_alarms") or []
    composite_alarms = alarms_data.get("composite_alarms") or []
    alarm_items = [
        {
            "name": alarm.get("AlarmName", ""),
            "type": "metric",
            "metric": alarm.get("MetricName", ""),
        }
        for alarm in metric_alarms[:20]
    ] + [
        {"name": alarm.get("AlarmName", ""), "type": "composite", "metric": ""}
        for alarm in composite_alarms[:20]
    ]

    total_errors = sum(point["count"] for point in errors_timeline)

    payload: dict[str, Any] = {
        "region": AWS_REGION,
        "hours": hours,
        "generated_at": iso_utc(end_dt),
        "log_groups_searched": len(group_names),
        "active_alarms": {
            "count": len(metric_alarms) + len(composite_alarms),
            "items": alarm_items,
        },
        "errors_timeline": errors_timeline,
        "errors_by_log_group": errors_by_log_group,
        "summary": {
            "total_errors": total_errors,
            "top_log_group": errors_by_log_group[0]["log_group"] if errors_by_log_group else None,
        },
    }
    if insights_error:
        payload["insights_error"] = insights_error

    _dashboard_cache[cache_key] = (time.time(), payload)
    return payload


def get_dashboard_error_details(
    hours: int = 24,
    *,
    limit: int | None = None,
    log_group: str | None = None,
) -> dict[str, Any]:
    hours = min(72, max(1, int(hours)))
    row_limit = min(500, max(10, int(limit or DASHBOARD_ERROR_DETAILS_LIMIT)))
    log_group_filter = (log_group or "").strip() or None
    cache_key = f"errors:{hours}:{row_limit}:{log_group_filter or 'all'}"
    cached = _dashboard_cache.get(cache_key)
    if cached and time.time() - cached[0] < DASHBOARD_CACHE_TTL:
        return cached[1]

    logs_client = get_client("logs")
    if log_group_filter:
        group_names = [log_group_filter]
    else:
        group_names = collect_log_group_names_for_search(
            logs_client, DEFAULT_LOG_SEARCH_KEYWORDS
        )

    end_dt = utc_now()
    start_dt = end_dt - timedelta(hours=hours)
    start = to_epoch_seconds(start_dt)
    end = to_epoch_seconds(end_dt)

    details_query = (
        "fields @timestamp, @log, @message\n"
        f"{DASHBOARD_ERROR_FILTER}"
        "| sort @timestamp desc\n"
        f"| limit {row_limit}"
    )

    items: list[dict[str, str]] = []
    insights_error: str | None = None

    if group_names:
        try:
            detail_results = _run_insights_query(
                logs_client,
                log_group_names=group_names,
                query_string=details_query,
                start=start,
                end=end,
            )
            items = format_insights_rows(detail_results, limit=row_limit)
            for row in items:
                row["message"] = row.get("message", "")[:800]
        except ClientError as exc:
            insights_error = str(exc)

    payload: dict[str, Any] = {
        "region": AWS_REGION,
        "hours": hours,
        "limit": row_limit,
        "log_group": log_group_filter,
        "log_groups_searched": len(group_names),
        "count": len(items),
        "items": items,
        "generated_at": iso_utc(end_dt),
    }
    if insights_error:
        payload["insights_error"] = insights_error

    _dashboard_cache[cache_key] = (time.time(), payload)
    return payload


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] | None = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]]
    iterations: int


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Warming up CloudWatch MCP tools...")
    tools = await get_openai_tools()
    logger.info("Ready with %d tools. vLLM: %s", len(tools), VLLM_BASE_URL)
    yield


app = FastAPI(
    title="CloudWatch Agent",
    description="LLM-powered CloudWatch observability agent",
    version="0.1.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/mcp", mcp.streamable_http_app())


@app.get("/")
async def ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    tools = await get_openai_tools()
    tool_catalog = [
        {"name": name, "description": CLOUDWATCH_TOOLS[name]}
        for name in sorted(ALLOWED_TOOL_NAMES)
    ]
    return {
        "status": "ready",
        "tools_loaded": len(tools),
        "allowed_tools": sorted(ALLOWED_TOOL_NAMES),
        "tools": tool_catalog,
        "model": MODEL_NAME,
        "vllm_base_url": VLLM_BASE_URL,
        "aws_region": AWS_REGION,
    }


@app.get("/api/dashboard/overview")
async def dashboard_overview(hours: int = 24) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(get_dashboard_overview, hours)
    except (ClientError, NoCredentialsError) as exc:
        logger.exception("Dashboard overview failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dashboard overview failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/dashboard/errors")
async def dashboard_errors(
    hours: int = 24,
    limit: int = DASHBOARD_ERROR_DETAILS_LIMIT,
    log_group: str | None = None,
) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            get_dashboard_error_details,
            hours,
            limit=limit,
            log_group=log_group,
        )
    except (ClientError, NoCredentialsError) as exc:
        logger.exception("Dashboard error details failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Dashboard error details failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    history = None
    if request.history:
        history = [message.model_dump() for message in request.history]

    try:
        result = await run_agent(request.message, history)
        return ChatResponse(**result)
    except (ClientError, NoCredentialsError) as exc:
        logger.exception("AWS error during chat")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run() -> None:
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run()
