"""SRE reference data and natural-language time parsing (fallback + prompt hints)."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

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

_MINUTES_RE = re.compile(
    r"(?:son\s+)?(\d+)\s*(?:dakika|dk|minute?s?)\b|"
    r"(?:last\s+)?(\d+)\s*(?:minute?s?|mins?)\b",
    re.I,
)
_HOURS_RE = re.compile(
    r"(?:son\s+)?(\d+)\s*(?:saat|hr?s?)(?:de|dır|dir|da|ta)?\b|"
    r"(?:last\s+)?(\d+)\s*(?:hour?s?|hrs?)\b",
    re.I,
)
_DAYS_RE = re.compile(
    r"(?:son\s+)?(\d+)\s*(?:gün|day?s?)\b|"
    r"(?:last\s+)?(\d+)\s*days?\b",
    re.I,
)


def has_explicit_time_window(message: str) -> bool:
    """True when the user named a concrete lookback (not default 1h)."""
    lowered = message.lower()
    if _HOURS_RE.search(lowered) or _DAYS_RE.search(lowered) or _MINUTES_RE.search(lowered):
        return True
    if re.search(r"\b(bugün|today|dün|yesterday|bu\s+hafta|this\s+week|geçen\s+hafta|last\s+week)\b", lowered):
        return True
    return "24 saat" in lowered


def parse_hours_from_message(message: str) -> int:
    """Convert common TR/EN time phrases to a lookback in hours (1–168)."""
    lowered = message.lower()

    minutes_match = _MINUTES_RE.search(lowered)
    if minutes_match:
        minutes = int(next(g for g in minutes_match.groups() if g))
        return min(168, max(1, (minutes + 59) // 60 or 1))

    hours_match = _HOURS_RE.search(lowered)
    if hours_match:
        return min(168, max(1, int(next(g for g in hours_match.groups() if g))))

    days_match = _DAYS_RE.search(lowered)
    if days_match:
        days = int(next(g for g in days_match.groups() if g))
        return min(168, max(1, days * 24))

    if re.search(r"\b(bugün|today)\b", lowered):
        return min(168, max(1, datetime.now(timezone.utc).hour or 1))
    if re.search(r"\b(dün|yesterday)\b", lowered):
        return 24
    if re.search(r"\b(bu\s+hafta|this\s+week)\b", lowered):
        return min(168, max(24, datetime.now(timezone.utc).weekday() * 24 + 24))
    if re.search(r"\b(geçen\s+hafta|last\s+week)\b", lowered):
        return 168
    if "24 saat" in lowered:
        return 24

    if _AMBIGUOUS_TIME.search(lowered):
        return 1

    return 1


def is_ambiguous_time_reference(message: str) -> bool:
    lowered = message.lower()
    if _HOURS_RE.search(lowered) or _DAYS_RE.search(lowered) or _MINUTES_RE.search(lowered):
        return False
    if re.search(r"\b(bugün|dün|today|yesterday|this\s+week|last\s+week)\b", lowered):
        return False
    return bool(_AMBIGUOUS_TIME.search(lowered))


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
        if re.search(r"error", lowered):
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
        if re.search(r"5\s*xx", lowered):
            add("AWS/ApiGateway", "5XXError")
        elif re.search(r"4\s*xx", lowered):
            add("AWS/ApiGateway", "4XXError")
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
- Today vs yesterday: today 00:00 UTC → now, and yesterday 00:00 → 23:59 UTC
- This week vs last week: use ISO-8601 UTC boundaries for each 7-day block
- Last N hours vs prior N hours: two equal windows ending at now and at N hours ago
Always label each result (period_label) and never infer one period from the other.
""".strip()
