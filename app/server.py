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

from app.config import *
from app.agent import find_api_connection_error, get_openai_tools, get_system_prompt, run_agent
from app.queries import LOG_SEARCH_QUERY_PRESETS, LogResponseMode, LogSearchFilter, resolve_log_search_query

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools registry
# ---------------------------------------------------------------------------

CLOUDWATCH_TOOLS = {
    "describe_log_groups": (
        "List CloudWatch log group NAMES (catalog) only — discover or filter groups by vague/partial "
        "terms (e.g. 'api gateway logları', 'lambda grupları'). "
        "Do NOT use when the user already gave a full log group path (with '/') — use query_log_group instead. "
        "Do NOT use for error/HTTP counts, 500 searches, rankings, or log line content — "
        "use search_logs_across_groups or query_log_group. "
        "Single filter: log_group_name_prefix. Multiple types: log_group_name_keywords (OR). "
        "Fuzzy matching is automatic."
    ),
    "analyze_log_group": (
        "Legacy error analysis for one log group. Prefer query_log_group when user gives a log group path."
    ),
    "query_log_group": (
        "Query log LINES in exactly ONE log group. **Use this whenever the user gives a full log group path** "
        "(contains '/', e.g. /aws/apigateway/era-api-gateway-dev/access-logs or aws/apigateway/...). "
        "Do NOT use describe_log_groups or search_logs_across_groups for that case. "
        "Params: log_group_name (exact path); time via hours (MUST match user: 'son 12 saat' → hours=12) "
        "OR start_time+end_time (ISO-8601 UTC); search_filter presets: errors, http_2xx, "
        "http_400, http_401, http_403, http_404, http_500, http_500_backend (backend 500, excludes "
        "AUTHORIZER_FAILURE/ACCESS_DENIED), http_502, http_503, http_504, http_5xx; "
        "tenant_filter for customer/tenant slug in access logs (tenantDomain, tenantId, path); "
        "response_mode: summary | detail | analysis (analysis = breakdown + likely causes from real data). "
        "If results need refinement, recall this same tool — do not switch tools."
    ),
    "execute_log_insights_query": (
        "Search log LINES in SPECIFIC named log groups (you must already have names). "
        "For HTTP codes (400, 5XX) or errors when user did NOT name a group, prefer "
        "search_logs_across_groups instead — do NOT ask which log group."
    ),
    "search_logs_across_groups": (
        "Search logs across multiple log groups or system-wide. "
        "**Do NOT use when the user named one exact full log group path** — use query_log_group instead. "
        "Use here for: system-wide counts/rankings, multiple groups, or fuzzy keywords "
        "('api gateway logları' without a full path). "
        "Scope: omit log_group_names for system-wide default groups; "
        "pass log_group_names for specific groups (A, B, E, F…). "
        "response_mode: summary=counts per group (system-wide 500/errors), "
        "count_only=counts for named groups only (no log lines), "
        "detail=full log lines (use max_result_lines=150–200). "
        "Set hours from the user's time phrase ('son 12 saat' → hours=12). "
        "Ask the user ONE clarifying question only when scope (all vs specific) or "
        "output (count vs detail) is genuinely unclear."
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

DEFAULT_LOG_SEARCH_KEYWORDS = [
    "containerinsights",
    "api-gateway",
    "alb",
    "lambda",
    "cloudfront",
]

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
    return get_system_prompt()


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


def resolve_query_time_window(
    *,
    hours: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[datetime, datetime, int]:
    if start_time and end_time:
        start_dt = parse_time(start_time)
        end_dt = parse_time(end_time)
    else:
        hours_val = min(168, max(1, int(hours or 1)))
        end_dt = utc_now()
        start_dt = end_dt - timedelta(hours=hours_val)
    if end_dt <= start_dt:
        raise ValueError("end_time must be after start_time")
    hours_out = max(1, int((end_dt - start_dt).total_seconds() // 3600) or 1)
    return start_dt, end_dt, hours_out


def normalize_log_group_path(raw: str) -> str:
    path = (raw or "").strip()
    if path.startswith("aws/"):
        return "/" + path
    if path and not path.startswith("/"):
        return "/" + path.lstrip("/")
    return path


def normalize_insights_log_group(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if ":log-group:" in text:
        return text.split(":log-group:", 1)[-1]
    # CloudWatch Insights @log field: 123456789012:/aws/lambda/my-fn
    if re.match(r"^\d+:(/.+)", text):
        return text.split(":", 1)[1]
    return text


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


def dedupe_search_keywords(keywords: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw in keywords:
        label = (raw or "").strip()
        if not label:
            continue
        normalized = resolve_log_group_search_keyword(label)
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            unique.append(normalized)
    return unique


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


def resolve_explicit_log_group_names(
    logs_client,
    names: list[str],
    *,
    max_groups: int = INSIGHTS_MAX_LOG_GROUPS,
) -> list[str]:
    """Resolve user-supplied log group labels to full CloudWatch log group paths."""
    resolved: list[str] = []
    for raw in names:
        label = (raw or "").strip()
        if not label:
            continue
        path = normalize_log_group_path(label)
        if path.count("/") >= 3:
            resolved.append(path)
            continue
        prefix = normalize_log_group_prefix_arg(label) or label
        filtered, _ = _search_log_groups_by_keyword(
            logs_client,
            search_label=label,
            prefix=prefix,
            max_items=10,
        )
        exact = [
            group["logGroupName"]
            for group in filtered
            if label.lower() in group.get("logGroupName", "").lower()
        ]
        if len(exact) == 1:
            resolved.append(exact[0])
            continue
        if len(filtered) == 1:
            resolved.append(filtered[0]["logGroupName"])
            continue
        for group in filtered[:3]:
            name = group.get("logGroupName")
            if name:
                resolved.append(name)
    return list(dict.fromkeys(resolved))[:max_groups]


def resolve_search_log_groups(
    logs_client,
    *,
    log_group_names: list[str] | None = None,
    log_group_name_keywords: list[str] | None = None,
    max_groups: int = INSIGHTS_MAX_LOG_GROUPS,
) -> tuple[list[str], list[str], list[str]]:
    """Return (group_names, keywords_tried, explicit_labels)."""
    explicit = [n.strip() for n in (log_group_names or []) if n and n.strip()]
    if explicit:
        resolved = resolve_explicit_log_group_names(
            logs_client, explicit, max_groups=max_groups
        )
        return resolved, explicit, explicit

    keywords = dedupe_search_keywords(
        [k.strip() for k in (log_group_name_keywords or DEFAULT_LOG_SEARCH_KEYWORDS) if k and k.strip()]
    )
    groups = collect_log_group_names_for_search(logs_client, keywords, max_groups=max_groups)
    return groups, keywords, []


def format_insights_timestamp(raw: str) -> str:
    if not raw:
        return ""
    try:
        ms = int(float(raw))
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
    except (ValueError, OSError, OverflowError):
        return raw


def format_insights_rows(
    results: list[Any],
    *,
    limit: int = 30,
    max_message_chars: int | None = 240,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in results[:limit]:
        if not isinstance(row, list):
            continue
        fields = {
            item.get("field", ""): str(item.get("value", ""))
            for item in row
            if isinstance(item, dict)
        }
        message = fields.get("@message", "")
        if max_message_chars is not None:
            message = message[:max_message_chars]
        rows.append(
            {
                "timestamp": format_insights_timestamp(fields.get("@timestamp", "")),
                "log_group": normalize_insights_log_group(
                    fields.get("@log", fields.get("@logGroup", ""))
                ),
                "log_stream": fields.get("@logStream", ""),
                "message": message,
                "matches": fields.get("matches", ""),
            }
        )
    return rows


def format_insights_rows_with_group(
    results: list[Any],
    *,
    log_group_name: str,
    limit: int = 50,
    max_message_chars: int | None = 240,
) -> list[dict[str, str]]:
    rows = format_insights_rows(
        results, limit=limit, max_message_chars=max_message_chars
    )
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
            if status != "Complete":
                logger.warning(
                    "CloudWatch Insights query_id=%s finished with status=%s stats=%s",
                    query_id,
                    status,
                    response.get("statistics", {}),
                )
            return response
        time.sleep(LOGS_QUERY_POLL_INTERVAL)
    response = logs_client.get_query_results(queryId=query_id)
    logger.warning(
        "CloudWatch Insights query_id=%s timed out after %ss (last status=%s)",
        query_id,
        LOGS_QUERY_MAX_WAIT,
        response.get("status"),
    )
    return response


def insights_log_group_arn(log_group_name: str, region: str | None = None) -> str:
    return (
        f"arn:aws:logs:{region or AWS_REGION}:{AWS_ACCOUNT_ID}:log-group:{log_group_name}"
    )


def log_insights_query_start(
    *,
    log_group_names: list[str],
    query_string: str,
    start: int,
    end: int,
    context: str,
) -> None:
    window_sec = max(0, end - start)
    start_dt = datetime.fromtimestamp(start, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end, tz=timezone.utc)
    for group in log_group_names:
        logger.info(
            "CloudWatch Insights START [%s]\n"
            'SOURCE "%s" START=-%ds END=0s | utc %s -> %s\n%s',
            context,
            insights_log_group_arn(group),
            window_sec,
            iso_utc(start_dt),
            iso_utc(end_dt),
            query_string.strip(),
        )


def log_insights_query_result(
    *,
    context: str,
    query_id: str,
    status: str,
    record_count: int,
    elapsed_s: float,
    statistics: dict[str, Any] | None = None,
    results: list[Any] | None = None,
) -> None:
    stats = statistics or {}
    sample = ""
    if results:
        sample = json.dumps(results[:3], default=str, ensure_ascii=False)[:800]
    logger.info(
        "CloudWatch Insights RESULT [%s] query_id=%s status=%s records=%d "
        "elapsed=%.2fs bytes_scanned=%s records_matched=%s sample=%s",
        context,
        query_id,
        status,
        record_count,
        elapsed_s,
        stats.get("bytesScanned", "?"),
        stats.get("recordsMatched", "?"),
        sample or "<empty>",
    )


def execute_insights_query(
    logs_client,
    *,
    log_group_names: list[str] | None = None,
    log_group_name: str | None = None,
    query_string: str,
    start: int,
    end: int,
    context: str = "agent",
) -> dict[str, Any]:
    groups = list(log_group_names or [])
    if log_group_name:
        groups = [log_group_name]
    if not groups:
        return {"status": "Failed", "results": [], "statistics": {}}

    log_insights_query_start(
        log_group_names=groups,
        query_string=query_string,
        start=start,
        end=end,
        context=context,
    )
    started = time.time()
    kwargs: dict[str, Any] = {
        "startTime": start,
        "endTime": end,
        "queryString": query_string,
    }
    if len(groups) == 1:
        kwargs["logGroupName"] = groups[0]
    else:
        kwargs["logGroupNames"] = groups

    try:
        response = logs_client.start_query(**kwargs)
    except ClientError as exc:
        logger.exception(
            "CloudWatch Insights ERROR [%s] groups=%s query=%s",
            context,
            groups,
            query_string.strip(),
        )
        raise

    query_id = response["queryId"]
    completed = wait_for_logs_query(logs_client, query_id)
    results = completed.get("results") or []
    log_insights_query_result(
        context=context,
        query_id=query_id,
        status=str(completed.get("status", "Unknown")),
        record_count=len(results),
        elapsed_s=time.time() - started,
        statistics=completed.get("statistics"),
        results=results,
    )
    completed["query_id"] = query_id
    return completed


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

    logger.info(
        "describe_log_groups START keywords=%s prefix=%r max_items=%d region=%s",
        keywords or ["<all>"],
        log_group_name_prefix,
        max_items,
        region or AWS_REGION,
    )

    if not keywords:
        log_groups = fetch_log_groups(logs_client, max_items=max_items)
        slim_groups = [slim_log_group(lg) for lg in log_groups]
        result = {
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
        logger.info(
            "describe_log_groups RESULT count=%d sample=%s",
            result["count"],
            [g.get("logGroupName") for g in slim_groups[:10]],
        )
        return result

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
        slim = [slim_log_group(lg) for lg in filtered]
        result = {
            "region": region or AWS_REGION,
            "count": len(filtered),
            "log_groups": slim,
            "fallback_contains_used": fallback_applied,
            "message": message,
        }
        logger.info(
            "describe_log_groups RESULT keyword=%r count=%d fallback=%s sample=%s",
            search_label,
            result["count"],
            fallback_applied,
            [g.get("logGroupName") for g in slim[:10]],
        )
        return result

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

    slim_merged = [slim_log_group(lg) for lg in merged]
    result = {
        "region": region or AWS_REGION,
        "count": len(merged),
        "log_groups": slim_merged,
        "fallback_contains_used": fallback_applied,
        "message": message,
        "matched_keywords": keywords,
    }
    logger.info(
        "describe_log_groups RESULT keywords=%s count=%d fallback=%s sample=%s",
        keywords,
        result["count"],
        fallback_applied,
        [g.get("logGroupName") for g in slim_merged[:10]],
    )
    return result


@mcp.tool(name="query_log_group", description=CLOUDWATCH_TOOLS["query_log_group"])
def query_log_group(
    log_group_name: Annotated[
        str,
        Field(description="Full CloudWatch log group path, e.g. /aws/apigateway/my-api/access-logs"),
    ],
    hours: Annotated[
        int,
        Field(
            description=(
                "Lookback window in hours when start/end omitted. Default 1 only if the user "
                "did not mention a time range. MUST match user phrasing: 'son 12 saat' → 12, "
                "'son 3 gün' → 72, 'dün' → 24."
            ),
        ),
    ] = 1,
    start_time: Annotated[
        Optional[str],
        Field(description="ISO-8601 UTC start. Use with end_time for comparisons or exact windows."),
    ] = None,
    end_time: Annotated[
        Optional[str],
        Field(description="ISO-8601 UTC end. Use with start_time."),
    ] = None,
    search_filter: Annotated[
        LogSearchFilter,
        Field(description="Preset Insights filter for errors, 2xx, or HTTP status codes."),
    ] = "errors",
    tenant_filter: Annotated[
        Optional[str],
        Field(
            description=(
                "Scope to one tenant/customer slug in access logs "
                "(matches tenantDomain, tenantId, or path segment — e.g. bozkurteradev)."
            ),
        ),
    ] = None,
    response_mode: Annotated[
        LogResponseMode,
        Field(
            description=(
                "summary=count only; detail=CloudWatch-style log lines; "
                "analysis=aggregated breakdown and likely causes from real log data."
            ),
        ),
    ] = "detail",
    max_result_lines: Annotated[int, Field(description="Max log lines when response_mode=detail.")] = 100,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    log_group_name = normalize_log_group_path(log_group_name)
    try:
        start_dt, end_dt, hours_out = resolve_query_time_window(
            hours=hours,
            start_time=start_time,
            end_time=end_time,
        )
    except ValueError as exc:
        return {
            "error": str(exc),
            "log_group_name": log_group_name,
            "match_count": 0,
            "results": [],
        }

    query_string = resolve_log_search_query(
        search_filter,
        rank_by_log_group=(response_mode in {"summary", "count_only"}),
        line_limit=min(200, max(1, int(max_result_lines))),
        rank_limit=1,
        tenant=(tenant_filter or "").strip() or None,
    )
    start = to_epoch_seconds(start_dt)
    end = to_epoch_seconds(end_dt)

    try:
        completed = execute_insights_query(
            logs_client,
            log_group_name=log_group_name,
            query_string=query_string,
            start=start,
            end=end,
            context=f"query_log_group:{search_filter}",
        )
    except ClientError as exc:
        return {
            "error": str(exc),
            "log_group_name": log_group_name,
            "hours": hours_out,
            "start_time": iso_utc(start_dt),
            "end_time": iso_utc(end_dt),
            "search_filter": search_filter,
            "query_string": query_string,
            "match_count": 0,
            "results": [],
        }

    raw_results = completed.get("results", [])
    rank_results = response_mode in {"summary", "count_only"}
    limit = 1 if rank_results else min(200, max(1, int(max_result_lines)))
    if response_mode == "analysis":
        limit = min(200, max(50, int(max_result_lines)))
    formatted = format_insights_rows_with_group(
        raw_results,
        log_group_name=log_group_name,
        limit=limit,
        max_message_chars=None if response_mode in {"detail", "analysis"} else 240,
    )
    if rank_results and formatted:
        try:
            match_total = int(float(formatted[0].get("matches") or len(raw_results)))
        except ValueError:
            match_total = len(raw_results)
    else:
        match_total = len(raw_results)
    return {
        "status": completed.get("status"),
        "region": region or AWS_REGION,
        "log_group_name": log_group_name,
        "hours": hours_out,
        "start_time": iso_utc(start_dt),
        "end_time": iso_utc(end_dt),
        "search_filter": search_filter,
        "tenant_filter": (tenant_filter or "").strip() or None,
        "response_mode": response_mode,
        "query_string": query_string,
        "query_id": completed.get("query_id"),
        "log_groups_searched": 1,
        "all_log_group_names": [log_group_name],
        "match_count": match_total,
        "total_matches": match_total,
        "max_result_lines": limit,
        "results": formatted,
        "truncated": len(raw_results) > limit,
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
    search_filter: Annotated[
        Optional[LogSearchFilter],
        Field(
            description=(
                "Preset filter: errors, http_2xx, http_400, http_401, http_403, http_404, "
                "http_500, http_500_backend, http_502, http_503, http_504, http_5xx. "
                "Use http_500_backend for integration/backend 500 only (excludes authorizer denials)."
            ),
        ),
    ] = None,
    query_string: Annotated[
        Optional[str],
        Field(
            description=(
                "Custom Insights query when search_filter is omitted. "
                "For 2xx/success use HTTP 2xx patterns — NOT the errors preset."
            )
        ),
    ] = None,
    hours: Annotated[
        int,
        Field(
            description=(
                "Lookback hours when start_time/end_time omitted. Default 1 only if the user "
                "did not mention a time range. MUST match user phrasing: 'son 12 saat' → 12, "
                "'son 3 gün' → 72, 'dün' → 24."
            ),
        ),
    ] = 1,
    start_time: Annotated[
        Optional[str],
        Field(description="ISO-8601 UTC start for exact windows and comparisons."),
    ] = None,
    end_time: Annotated[
        Optional[str],
        Field(description="ISO-8601 UTC end for exact windows and comparisons."),
    ] = None,
    period_label: Annotated[
        Optional[str],
        Field(description="Human label for this window in comparison replies, e.g. 'Today' / 'Dün'."),
    ] = None,
    response_mode: Annotated[
        LogResponseMode,
        Field(
            description=(
                "summary=count per log group system-wide (e.g. all 500s); "
                "count_only=counts for selected groups only, no log lines; "
                "detail=full timestamp/message lines (use for A+B detailed); "
                "analysis=breakdown and likely causes from matched lines."
            ),
        ),
    ] = "detail",
    rank_by_log_group: Annotated[
        bool,
        Field(
            description=(
                "Deprecated — use response_mode=summary or count_only. "
                "When true, returns stats count() by @log."
            ),
        ),
    ] = False,
    max_result_lines: Annotated[
        int,
        Field(description="Max log lines (detail) or max groups (summary/count_only)."),
    ] = 50,
    log_group_names: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "Specific log groups when user named multiple exact paths or labels. "
                "Do NOT use keywords here when user gave one full path — use query_log_group instead. "
                "Examples: ['/aws/apigateway/x/access', '/aws/lambda/y']. "
                "Omit for system-wide default search."
            ),
        ),
    ] = None,
    log_group_name_keywords: Annotated[
        Optional[list[str]],
        Field(
            description=(
                "Fuzzy filters for vague scope only (e.g. 'api-gateway', 'lambda'). "
                "Never use when user already provided the exact full log group path with '/'."
            )
        ),
    ] = None,
    tenant_filter: Annotated[
        Optional[str],
        Field(description="Tenant/customer slug filter for access logs (tenantDomain, path, tenantId)."),
    ] = None,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    mode: LogResponseMode = response_mode
    if rank_by_log_group and mode == "detail":
        mode = "summary"

    group_names, keywords, explicit_labels = resolve_search_log_groups(
        logs_client,
        log_group_names=log_group_names,
        log_group_name_keywords=log_group_name_keywords,
    )
    if not group_names:
        return {
            "error": "Arama için uygun log group bulunamadı.",
            "keywords_tried": keywords,
            "requested_log_group_names": explicit_labels,
            "match_count": 0,
            "results": [],
        }

    rank_results = mode in {"summary", "count_only"}
    if search_filter:
        resolved_query = resolve_log_search_query(
            search_filter,
            rank_by_log_group=rank_results,
            line_limit=min(200, max(1, int(max_result_lines))),
            rank_limit=min(50, max(len(group_names), int(max_result_lines))),
            tenant=(tenant_filter or "").strip() or None,
        )
    elif query_string and query_string.strip():
        resolved_query = query_string.strip()
        if rank_results and "stats count()" not in resolved_query.lower():
            filter_body = resolved_query
            if filter_body.startswith("fields "):
                filter_body = "\n".join(filter_body.split("\n")[1:])
            resolved_query = (
                f"fields @log\n{filter_body.rstrip()}\n"
                f"| stats count() as matches by @log\n"
                f"| sort matches desc\n"
                f"| limit {min(50, max(1, int(max_result_lines)))}"
            )
    else:
        return {
            "error": "search_filter veya query_string gerekli.",
            "match_count": 0,
            "results": [],
        }

    try:
        start_dt, end_dt, hours_out = resolve_query_time_window(
            hours=hours,
            start_time=start_time,
            end_time=end_time,
        )
    except ValueError as exc:
        return {
            "error": str(exc),
            "keywords_tried": keywords,
            "requested_log_group_names": explicit_labels,
            "match_count": 0,
            "results": [],
        }

    start = to_epoch_seconds(start_dt)
    end = to_epoch_seconds(end_dt)

    try:
        completed = execute_insights_query(
            logs_client,
            log_group_names=group_names,
            query_string=resolved_query,
            start=start,
            end=end,
            context=f"search_logs_across_groups:{mode}",
        )
    except ClientError as exc:
        return {
            "error": str(exc),
            "log_groups_searched": len(group_names),
            "log_group_names": group_names[:10],
            "keywords_tried": keywords,
            "requested_log_group_names": explicit_labels,
            "search_filter": search_filter,
            "query_string": resolved_query,
            "response_mode": mode,
            "rank_by_log_group": rank_results,
            "period_label": period_label,
            "match_count": 0,
            "results": [],
        }

    results = completed.get("results", [])
    max_lines = min(200, max(1, int(max_result_lines)))
    formatted = format_insights_rows(
        results,
        limit=max_lines if not rank_results else 50,
        max_message_chars=None
        if mode in {"detail", "analysis"} and not rank_results
        else 240,
    )
    total_matches = 0
    if rank_results:
        for row in formatted:
            try:
                total_matches += int(float(row.get("matches") or 0))
            except ValueError:
                continue
    else:
        total_matches = len(results)

    return {
        "status": completed.get("status"),
        "region": region or AWS_REGION,
        "hours": hours_out,
        "start_time": iso_utc(start_dt),
        "end_time": iso_utc(end_dt),
        "period_label": period_label,
        "response_mode": mode,
        "rank_by_log_group": rank_results,
        "log_groups_searched": len(group_names),
        "all_log_group_names": group_names,
        "log_group_names": group_names[:15],
        "requested_log_group_names": explicit_labels,
        "keywords_tried": keywords,
        "search_filter": search_filter,
        "tenant_filter": (tenant_filter or "").strip() or None,
        "query_string": resolved_query,
        "query_id": completed.get("query_id"),
        "match_count": total_matches if rank_results else len(results),
        "total_matches": total_matches,
        "max_result_lines": max_lines,
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
    completed = execute_insights_query(
        logs_client,
        log_group_names=log_group_names,
        query_string=query_string,
        start=start,
        end=end,
        context="dashboard",
    )
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
    log_context: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[ChatMessage] | None = None


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]]
    iterations: int
    log_context: dict[str, Any] | None = None


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
        if find_api_connection_error(exc) is not None:
            logger.exception("vLLM connection failed during chat")
            raise HTTPException(
                status_code=502,
                detail=(
                    "Model servisine (vLLM) bağlanılamadı. "
                    "vllm-gptoss pod'unun ayakta ve hazır olduğunu kontrol edip tekrar dene."
                ),
            ) from exc
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
