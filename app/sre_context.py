"""SRE reference data and natural-language time parsing (fallback + prompt hints)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import DISPLAY_TIMEZONE

METRIC_NAMESPACE_TABLE = """
| Service / signal | Namespace | Common metrics | Typical dimensions |
|------------------|-----------|----------------|--------------------|
| ECS | AWS/ECS | CPUUtilization, MemoryUtilization | ClusterName, ServiceName |
| Lambda | AWS/Lambda | Duration, Errors, Invocations, Throttles, ConcurrentExecutions | FunctionName |
| ALB | AWS/ApplicationELB | TargetResponseTime, HTTPCode_Target_5XX_Count, RequestCount | LoadBalancer, TargetGroup |
| NLB | AWS/NetworkELB | ActiveFlowCount, ProcessedBytes | LoadBalancer |
| API Gateway | AWS/ApiGateway | Count, 4XXError, 5XXError, Latency, IntegrationLatency | ApiName, Stage |
| RDS | AWS/RDS | CPUUtilization, DatabaseConnections, FreeableMemory, ReadLatency | DBInstanceIdentifier |
| EC2 | AWS/EC2 | CPUUtilization, NetworkIn, NetworkOut, StatusCheckFailed | InstanceId |
| SQS | AWS/SQS | ApproximateAgeOfOldestMessage, NumberOfMessagesSent, ApproximateNumberOfMessagesVisible | QueueName |
| SNS | AWS/SNS | NumberOfMessagesPublished, NumberOfNotificationsFailed | TopicName |
| DynamoDB | AWS/DynamoDB | ConsumedReadCapacityUnits, ThrottledRequests, UserErrors | TableName |
| CloudFront | AWS/CloudFront | 4xxErrorRate, 5xxErrorRate, Requests | DistributionId, Region |
| EKS / containers | ContainerInsights | node_cpu_utilization, pod_memory_utilization, cluster_failed_node_count | ClusterName, Namespace, PodName |
| CodeBuild | AWS/CodeBuild | Builds, FailedBuilds, Duration | ProjectName |
| ElastiCache | AWS/ElastiCache | CPUUtilization, CurrConnections, Evictions | CacheClusterId |
| NAT Gateway | AWS/NATGateway | BytesOutToDestination, ActiveConnectionCount | NatGatewayId |
""".strip()

_AMBIGUOUS_TIME = re.compile(
    r"\b(recently|just\s+now|a\s+while\s+ago|az\s+önce|yakın\s+zamanda|"
    r"biraz\s+önce|şimdi|şu\s+an|henüz)\b",
    re.I,
)

# Number + unit, tolerant of Turkish suffixes ("6 saatte", "2 günde", "15 dakikada", "24 saatlik").
_NUMBER = r"(?<![\w.,/-])(\d+)"
_MINUTES_RE = re.compile(_NUMBER + r"(?:\s*(?:dakika|dk|minute|min)\w*|m\b)", re.I)
_HOURS_RE = re.compile(_NUMBER + r"(?:\s*(?:saat|hour|hr)\w*|h\b)", re.I)
_DAYS_RE = re.compile(_NUMBER + r"(?:\s*(?:gün|gun|day)\w*|d\b)", re.I)
_WEEKS_RE = re.compile(_NUMBER + r"(?:\s*(?:hafta|week)\w*|w\b)", re.I)

_HALF_HOUR_RE = re.compile(r"\b(yarım\s+saat|half\s+an?\s+hour)", re.I)
_QUARTER_HOUR_RE = re.compile(r"\b(çeyrek\s+saat|quarter\s+of\s+an\s+hour)", re.I)

_TR_NUMBER_WORDS = {
    "bir": 1, "iki": 2, "üç": 3, "dört": 4, "beş": 5, "altı": 6, "yedi": 7,
    "sekiz": 8, "dokuz": 9, "on": 10, "on iki": 12, "yirmi": 20, "otuz": 30,
    "kırk beş": 45, "kırk": 40, "elli": 50, "altmış": 60,
}
_TR_NUMBER_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in _TR_NUMBER_WORDS), key=len, reverse=True)) + r")"
    r"\s+(?=(?:saat|gün|dakika|hafta))",
    re.I,
)

_TODAY_RE = re.compile(r"\b(bugün|bugun|today)\b", re.I)
_YESTERDAY_RE = re.compile(r"\b(dün|dun|yesterday)\b", re.I)
_THIS_WEEK_RE = re.compile(r"\b(bu\s+hafta|this\s+week)\b", re.I)
_LAST_WEEK_RE = re.compile(r"\b(geçen\s+hafta|last\s+week)\b", re.I)

MAX_LOOKBACK_MINUTES = 168 * 60


def display_timezone() -> tzinfo:
    try:
        return ZoneInfo(DISPLAY_TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        # Turkey has no DST since 2016; fixed offset is a safe fallback without tzdata.
        return timezone(timedelta(hours=3))


@dataclass(frozen=True)
class TimeWindow:
    """Resolved lookback: either rolling (minutes) or a fixed calendar range (start/end, UTC)."""

    minutes: int
    label: str
    start: datetime | None = None
    end: datetime | None = None

    @property
    def hours(self) -> int:
        return max(1, -(-self.minutes // 60))

    @property
    def is_fixed(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def is_sub_hour_precision(self) -> bool:
        return not self.is_fixed and self.minutes % 60 != 0


def _normalize_number_words(text: str) -> str:
    return _TR_NUMBER_WORD_RE.sub(lambda m: f"{_TR_NUMBER_WORDS[m.group(1).lower()]} ", text)


def _rolling(minutes: int) -> TimeWindow:
    minutes = min(MAX_LOOKBACK_MINUTES, max(1, minutes))
    if minutes % 60:
        return TimeWindow(minutes=minutes, label=f"son {minutes} dakika")
    return TimeWindow(minutes=minutes, label=f"son {minutes // 60} saat")


def _fixed(start_local: datetime, end_local: datetime, label: str) -> TimeWindow:
    start = start_local.astimezone(timezone.utc)
    end = end_local.astimezone(timezone.utc)
    minutes = max(1, int((end - start).total_seconds() // 60))
    return TimeWindow(minutes=minutes, label=label, start=start, end=end)


def calendar_window(name: str, now: datetime | None = None) -> TimeWindow:
    """Calendar periods in the display timezone (Europe/Istanbul by default)."""
    tz = display_timezone()
    local_now = (now or datetime.now(timezone.utc)).astimezone(tz)
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = midnight - timedelta(days=midnight.weekday())
    if name == "today":
        return _fixed(midnight, local_now, f"bugün ({midnight:%d.%m.%Y} 00:00 → şimdi, TR saati)")
    if name == "yesterday":
        start = midnight - timedelta(days=1)
        return _fixed(start, midnight, f"dün ({start:%d.%m.%Y} 00:00–24:00, TR saati)")
    if name == "this_week":
        return _fixed(week_start, local_now, f"bu hafta ({week_start:%d.%m.%Y} → şimdi, TR saati)")
    if name == "last_week":
        start = week_start - timedelta(days=7)
        return _fixed(
            start,
            week_start,
            f"geçen hafta ({start:%d.%m} – {week_start - timedelta(days=1):%d.%m.%Y}, TR saati)",
        )
    raise ValueError(f"unknown calendar window: {name}")


def parse_time_window(message: str, now: datetime | None = None) -> TimeWindow | None:
    """Explicit time phrase → TimeWindow; None when the user named no time range."""
    lowered = _normalize_number_words(message.lower())

    if _QUARTER_HOUR_RE.search(lowered):
        return _rolling(15)
    if _HALF_HOUR_RE.search(lowered):
        return _rolling(30)

    match = _MINUTES_RE.search(lowered)
    if match:
        return _rolling(int(match.group(1)))
    match = _HOURS_RE.search(lowered)
    if match:
        return _rolling(int(match.group(1)) * 60)
    match = _DAYS_RE.search(lowered)
    if match:
        return _rolling(int(match.group(1)) * 24 * 60)
    match = _WEEKS_RE.search(lowered)
    if match:
        return _rolling(int(match.group(1)) * 7 * 24 * 60)

    if _YESTERDAY_RE.search(lowered):
        return calendar_window("yesterday", now)
    if _TODAY_RE.search(lowered):
        return calendar_window("today", now)
    if _LAST_WEEK_RE.search(lowered):
        return calendar_window("last_week", now)
    if _THIS_WEEK_RE.search(lowered):
        return calendar_window("this_week", now)
    return None


def has_explicit_time_window(message: str) -> bool:
    """True when the user named a concrete lookback (not default 1h)."""
    return parse_time_window(message) is not None


def parse_hours_from_message(message: str) -> int:
    """Convert common TR/EN time phrases to a lookback in hours (1–168)."""
    window = parse_time_window(message)
    return window.hours if window else 1


def is_ambiguous_time_reference(message: str) -> bool:
    if has_explicit_time_window(message):
        return False
    return bool(_AMBIGUOUS_TIME.search(message.lower()))


def infer_metric_targets(message: str) -> list[tuple[str, str]]:
    """Map natural language to CloudWatch namespace + metric (mechanical fallback helper)."""
    lowered = message.lower()
    targets: list[tuple[str, str]] = []

    def add(namespace: str, metric: str) -> None:
        pair = (namespace, metric)
        if pair not in targets:
            targets.append(pair)

    if re.search(r"\becs\b", lowered):
        if re.search(r"cpu", lowered):
            add("AWS/ECS", "CPUUtilization")
        if re.search(r"memory|mem", lowered):
            add("AWS/ECS", "MemoryUtilization")
        if not targets:
            add("AWS/ECS", "CPUUtilization")
    if re.search(r"\blambda\b", lowered):
        if re.search(r"error|hata", lowered):
            add("AWS/Lambda", "Errors")
        elif re.search(r"throttl", lowered):
            add("AWS/Lambda", "Throttles")
        elif re.search(r"duration|latency|süre", lowered):
            add("AWS/Lambda", "Duration")
        else:
            add("AWS/Lambda", "Invocations")
    if re.search(r"\b(rds|database|veritaban)", lowered):
        if re.search(r"connection|bağlant", lowered):
            add("AWS/RDS", "DatabaseConnections")
        else:
            add("AWS/RDS", "CPUUtilization")
    if re.search(r"\b(alb|application\s+load\s+balancer)\b", lowered):
        if re.search(r"5\s*xx|error", lowered):
            add("AWS/ApplicationELB", "HTTPCode_Target_5XX_Count")
        else:
            add("AWS/ApplicationELB", "TargetResponseTime")
    if re.search(r"\bapi\s*gateway\b", lowered):
        if re.search(r"5\s*xx|\b5\d\d\b|error|hata", lowered):
            add("AWS/ApiGateway", "5XXError")
        elif re.search(r"4\s*xx|\b4\d\d\b", lowered):
            add("AWS/ApiGateway", "4XXError")
        elif re.search(r"latency|gecikme", lowered):
            add("AWS/ApiGateway", "Latency")
        else:
            add("AWS/ApiGateway", "Count")
    if re.search(r"\bec2\b", lowered):
        add("AWS/EC2", "CPUUtilization")
    if re.search(r"\bsqs\b", lowered):
        add("AWS/SQS", "ApproximateNumberOfMessagesVisible")
    if re.search(r"\bcloudfront\b", lowered):
        add("AWS/CloudFront", "5xxErrorRate")
    if re.search(r"\bdynamo", lowered):
        add("AWS/DynamoDB", "ConsumedReadCapacityUnits")
    if re.search(r"container\s*insights|eks\b|kubernetes|k8s", lowered):
        if re.search(r"memory|mem", lowered):
            add("ContainerInsights", "pod_memory_utilization")
        else:
            add("ContainerInsights", "node_cpu_utilization")

    if not targets and re.search(r"cpu|memory|mem|metrik|metric|latency|gecikme", lowered):
        add("AWS/ECS", "CPUUtilization")
    return targets


def comparison_period_hints() -> str:
    return """
Comparison windows (call the tool TWICE with start_time + end_time, same filter):
- Calendar days/weeks are in Europe/Istanbul time (UTC+3). Use the [Current time] system note for "now".
- Today vs yesterday: period_label="Bugün" and period_label="Dün" — the server fills exact TR-calendar
  boundaries from period_label, so always set it.
- This week vs last week: period_label="Bu hafta" / "Geçen hafta".
- Last N hours vs prior N hours: two equal windows ending at now and at N hours ago (ISO-8601 UTC).
Always label each result (period_label) and never infer one period from the other.
""".strip()
