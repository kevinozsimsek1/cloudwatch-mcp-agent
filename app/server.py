import json
import logging
import os
import re
import time
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

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tools registry
# ---------------------------------------------------------------------------

CLOUDWATCH_TOOLS = {
    "describe_log_groups": "Finds metadata about CloudWatch log groups",
    "analyze_log_group": "Analyzes CloudWatch logs",
    "execute_log_insights_query": "Run Logs Insights query",
    "get_logs_insight_query_results": "Fetch query results",
    "cancel_logs_insight_query": "Cancel query",
    "get_active_alarms": "List active alarms",
    "get_alarm_history": "Alarm history",
    "get_metric_data": "Metric data",
    "get_metric_metadata": "Metric metadata",
    "get_recommended_metric_alarms": "Recommended alarms",
    "analyze_metric": "Metric analysis",
}

ALLOWED_TOOL_NAMES = set(CLOUDWATCH_TOOLS.keys())

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""
You are a CloudWatch SRE assistant. Help engineers investigate logs, metrics, and alarms.

Rules:
- Use the available tools when you need real CloudWatch data. Do not guess AWS data.
- Allowed tools: {", ".join(sorted(ALLOWED_TOOL_NAMES))}
- Default AWS region: {AWS_REGION}
- If the user specifies a log group name, keyword, or prefix, query only matching log groups.
  Keywords like "codebuild" should match /aws/codebuild/* log groups.
  Users may write imperfect keywords (spaces, hyphens, singular/plural). The tool handles fuzzy
  matching — pass the user's term as-is, do not require exact AWS naming.
- For describe_log_groups: set max_items to 1000 when the user asks for all/tüm/tamamını/hepsi.

Language (critical):
- ALWAYS reply in the same language the user used in their latest message.
- Turkish message → Turkish reply. English message → English reply.
- This includes refusals, apologies, errors, and off-topic requests — never switch to English
  just because you are declining or cannot help.

Response format:
- Your final answer to the user must be plain, readable text — NOT raw JSON.
- Summarize findings clearly: bullet lists, short paragraphs, key numbers.
- If nothing is found, say so clearly and suggest a better search term if useful.
- Tool calls are handled separately; do not wrap your final answer in {{}} or JSON.
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
        candidates = [keyword, f"/aws/{keyword}"]
        raw_token = normalize_log_group_token(normalized)
        if raw_token and raw_token != keyword:
            candidates.append(raw_token)
            candidates.append(f"/aws/{raw_token}")
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


def adjust_describe_log_groups_args(
    user_message: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    args = dict(arguments)
    lowered = user_message.lower()
    wants_all = any(
        keyword in lowered
        for keyword in ("tüm", "tamam", "tamamını", "hepsi", "hepsini", "all")
    )
    requested = int(args.get("max_items") or 50)
    if wants_all:
        args["max_items"] = min(MAX_LOG_GROUPS_LIST, max(requested, 1000))
    else:
        args["max_items"] = min(MAX_LOG_GROUPS_LIST, requested)
    return args


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
            "content": trim_message_content(message.get("content", "")),
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


@mcp.tool(name="describe_log_groups", description=CLOUDWATCH_TOOLS["describe_log_groups"])
def describe_log_groups(
    log_group_name_prefix: Annotated[
        Optional[str],
        Field(
            description=(
                "Filter log groups. Keywords are fuzzy: 'container insights', "
                "'container-insights', and 'containerinsight' all match /aws/containerinsights/*."
            ),
        ),
    ] = None,
    max_items: Annotated[int, Field(description="Maximum number of log groups to return.")] = 50,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    prefix = normalize_log_group_prefix_arg(log_group_name_prefix)
    max_items = min(MAX_LOG_GROUPS_LIST, max(1, max_items))
    search_label = (log_group_name_prefix or "").strip() or prefix or ""

    if not prefix:
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

    collected: list[dict[str, Any]] = []
    for candidate_prefix in log_group_search_prefixes(prefix):
        collected.extend(fetch_log_groups(logs_client, prefix=candidate_prefix, max_items=max_items))
    collected = dedupe_log_groups(collected)

    fallback_applied = False
    if is_keyword_log_group_search(search_label):
        filtered = filter_log_groups_fuzzy(collected, search_label)
        if len(filtered) < max_items:
            broad_groups = fetch_log_groups(logs_client, max_items=MAX_LOG_GROUPS_LIST)
            broad_filtered = filter_log_groups_fuzzy(broad_groups, search_label)
            merged = dedupe_log_groups(filtered + broad_filtered)
            fallback_applied = len(merged) > len(filtered)
            filtered = merged[:max_items]
    else:
        filtered = filter_log_groups(collected, prefix)[:max_items]
        if not filtered:
            fallback_applied = True
            broad_groups = fetch_log_groups(logs_client, max_items=min(max_items * 5, MAX_LOG_GROUPS_LIST))
            filtered = filter_log_groups_fuzzy(broad_groups, search_label)[:max_items]

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
    log_group_names: Annotated[list[str], Field(description="Log group names to query.")],
    query_string: Annotated[str, Field(description="CloudWatch Logs Insights query.")],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[str, Field(description="ISO-8601 UTC end time.")],
    wait_for_completion: Annotated[bool, Field(description="Wait until query completes.")] = True,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    logs_client = get_client("logs", region)
    start = to_epoch_seconds(parse_time(start_time))
    end = to_epoch_seconds(parse_time(end_time))

    response = logs_client.start_query(
        logGroupNames=log_group_names,
        startTime=start,
        endTime=end,
        queryString=query_string,
    )
    query_id = response["queryId"]
    result: dict[str, Any] = {"query_id": query_id, "status": "Running"}

    if wait_for_completion:
        completed = wait_for_logs_query(logs_client, query_id)
        result.update(
            {
                "status": completed.get("status"),
                "results": completed.get("results", []),
                "statistics": completed.get("statistics", {}),
            }
        )
    return result


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
    alarm_name: Annotated[str, Field(description="CloudWatch alarm name.")],
    start_time: Annotated[str, Field(description="ISO-8601 UTC start time.")],
    end_time: Annotated[str, Field(description="ISO-8601 UTC end time.")],
    max_records: Annotated[int, Field(description="Maximum history records.")] = 50,
    region: Annotated[Optional[str], Field(description="AWS region.")] = None,
) -> dict[str, Any]:
    cloudwatch = get_client("cloudwatch", region)
    response = cloudwatch.describe_alarm_history(
        AlarmName=alarm_name,
        StartDate=parse_time(start_time),
        EndDate=parse_time(end_time),
        MaxRecords=max_records,
    )
    return {
        "alarm_name": alarm_name,
        "region": region or AWS_REGION,
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


async def run_agent(
    message: str, conversation_history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    tools = await get_openai_tools()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    trimmed_history = trim_conversation_history(conversation_history)
    if trimmed_history:
        messages.extend(trimmed_history)

    messages.append({"role": "user", "content": message})
    tool_calls_made: list[dict[str, Any]] = []

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
                if "context length" in err_text or "maximum context" in err_text:
                    return {
                        "response": (
                            "Sohbet geçmişi veya son sorgu çok uzun olduğu için model yanıt veremedi. "
                            "Sol panelden **Sıfırla** veya **Yeni** ile yeni sohbet başlatıp tekrar dene."
                        ),
                        "tool_calls": tool_calls_made,
                        "iterations": iteration + 1,
                    }
                raise
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                return {
                    "response": humanize_response(assistant_message.content or ""),
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

            messages.append(assistant_message.model_dump(exclude_none=True))

            last_tool_result: str | None = None
            single_describe_log_groups = (
                len(assistant_message.tool_calls) == 1
                and assistant_message.tool_calls[0].function.name == "describe_log_groups"
            )

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

                tool_result = await mcp_session.call_tool(tool_name, arguments)
                last_tool_result = tool_result
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
                        "content": tool_result,
                    }
                )

            if single_describe_log_groups and last_tool_result:
                parsed = parse_log_groups_tool_result(last_tool_result)
                if parsed is not None:
                    return {
                        "response": format_log_groups_list(parsed),
                        "tool_calls": tool_calls_made,
                        "iterations": iteration + 1,
                    }

    return {
        "response": "Maximum tool iterations reached. Partial investigation completed.",
        "tool_calls": tool_calls_made,
        "iterations": MAX_TOOL_ITERATIONS,
    }


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
