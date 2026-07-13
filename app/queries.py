"""CloudWatch Logs Insights query presets shared by MCP tools and the agent."""

from __future__ import annotations

import re
from typing import Literal

LogSearchFilter = Literal[
    "errors",
    "http_2xx",
    "http_400",
    "http_401",
    "http_403",
    "http_404",
    "http_500",
    "http_500_backend",
    "http_502",
    "http_503",
    "http_504",
    "http_5xx",
]

LogResponseMode = Literal["summary", "detail", "count_only", "analysis"]

_HTTP_5XX_FILTER = (
    "| filter (status >= 500 and status < 600) or @message like /\"status\":\"5[0-9]{2}\"/ "
    "or @message like /(?i)(statusCode[\"']?\\s*[:=]\\s*\"?5\\d{2}\"?|"
    "status.?code.?5\\d{2}|HTTP\\/1\\.[01] 5\\d{2}\\b)/\n"
)

_HTTP_2XX_FILTER = (
    "| filter (status >= 200 and status < 300) or @message like /\"status\":\"2[0-9]{2}\"/ "
    "or @message like /(?i)(statusCode[\"']?\\s*[:=]\\s*\"?2\\d{2}\"?|"
    "status.?code.?2\\d{2}|HTTP\\/1\\.[01] 2\\d{2}\\b)/\n"
)

_ERRORS_FILTER = (
    "| filter @message like /(?i)(\"Level\":\"Error\"|ACCESS_DENIED|errorResponseType|"
    "Unhandled exception|exception|fatal|stack trace)/\n"
    "| filter @message not like /\"errorMessage\":\"-\"|\"Level\":\"Information\"/\n"
)


def _http_code_filter(code: str) -> str:
    """Match API Gateway / ALB access-log JSON (status as number or string)."""
    return (
        f"| filter status = {code} or status = \"{code}\" "
        f"or @message like /\"status\"\\s*:\\s*\"?{code}\"?/"
        f" or @message like /\"integrationStatus\"\\s*:\\s*\"?{code}\"?/"
        f" or @message like /HTTP\\/1\\.[01] {code}\\b/\n"
    )


_HTTP_500_BACKEND_FILTER = (
    "| filter status = 500 or status = \"500\"\n"
    "| filter @message not like /errorResponseType\":\"(ACCESS_DENIED|AUTHORIZER_FAILURE)/\n"
)


def _line_query(filter_body: str, *, limit: int) -> str:
    if filter_body.startswith("fields "):
        base = filter_body
    else:
        base = f"fields @timestamp, @log, @message\n{filter_body}"
    return f"{base.rstrip()}\n| sort @timestamp desc\n| limit {limit}"


def _ranked_query(filter_body: str, *, limit: int) -> str:
    if filter_body.startswith("fields "):
        filter_body = filter_body.split("\n", 1)[1]
    return (
        f"fields @log\n{filter_body.rstrip()}\n"
        f"| stats count() as matches by @log\n"
        f"| sort matches desc\n"
        f"| limit {limit}"
    )


_FILTER_CLAUSES: dict[LogSearchFilter, str] = {
    "errors": _ERRORS_FILTER,
    "http_2xx": _HTTP_2XX_FILTER,
    "http_400": _http_code_filter("400").strip(),
    "http_401": _http_code_filter("401").strip(),
    "http_403": _http_code_filter("403").strip(),
    "http_404": _http_code_filter("404").strip(),
    "http_500": _http_code_filter("500").strip(),
    "http_500_backend": _HTTP_500_BACKEND_FILTER.strip(),
    "http_502": _http_code_filter("502").strip(),
    "http_503": _http_code_filter("503").strip(),
    "http_504": _http_code_filter("504").strip(),
    "http_5xx": _HTTP_5XX_FILTER,
}

LOG_SEARCH_QUERY_PRESETS: dict[LogSearchFilter, str] = {
    key: _line_query(clause, limit=100 if key != "errors" else 50)
    for key, clause in _FILTER_CLAUSES.items()
}

LOG_SEARCH_RANKING_PRESETS: dict[LogSearchFilter, str] = {
    key: _ranked_query(clause, limit=12)
    for key, clause in _FILTER_CLAUSES.items()
}


def inject_query_filters(query_string: str, *, tenant: str | None = None) -> str:
    """Append tenant scoping to an Insights query (path, tenantId, tenantDomain)."""
    if not tenant:
        return query_string
    safe = re.escape(tenant.strip())
    filter_line = (
        "| filter @message like /(?i)({0}|\"tenantDomain\"\\s*:\\s*\"[^\"]*{0}|"
        "\"tenantId\"\\s*:\\s*\"[^\"]*{0}|\"path\"\\s*:\\s*\"[^\"]*/{0})/"
    ).format(safe)
    lowered = query_string.lower()
    insert_pos = len(query_string)
    for marker in ("| sort ", "| stats ", "| limit "):
        idx = lowered.find(marker)
        if idx != -1:
            insert_pos = min(insert_pos, idx)
    if insert_pos < len(query_string):
        return f"{query_string[:insert_pos].rstrip()}\n{filter_line}\n{query_string[insert_pos:].lstrip()}"
    return f"{query_string.rstrip()}\n{filter_line}\n"


def resolve_log_search_query(
    search_filter: LogSearchFilter,
    *,
    rank_by_log_group: bool = False,
    line_limit: int = 100,
    rank_limit: int = 12,
    tenant: str | None = None,
) -> str:
    if rank_by_log_group:
        query = _ranked_query(_FILTER_CLAUSES[search_filter], limit=rank_limit)
    elif search_filter in LOG_SEARCH_QUERY_PRESETS and line_limit in {50, 100}:
        preset = LOG_SEARCH_QUERY_PRESETS[search_filter]
        if line_limit != (50 if search_filter == "errors" else 100):
            query = _line_query(_FILTER_CLAUSES[search_filter], limit=line_limit)
        else:
            query = preset
    else:
        query = _line_query(_FILTER_CLAUSES[search_filter], limit=line_limit)
    return inject_query_filters(query, tenant=tenant)


SEARCH_FILTER_ALIASES: dict[str, LogSearchFilter] = {
    "2xx": "http_2xx",
    "success": "http_2xx",
    "401": "http_401",
    "403": "http_403",
    "404": "http_404",
    "502": "http_502",
    "503": "http_503",
    "504": "http_504",
}
