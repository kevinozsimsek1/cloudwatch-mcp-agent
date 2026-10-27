import json

import app.server  # noqa: F401
from app.agent import (
    build_log_context_from_tool_data,
    compact_tool_result_for_context,
    format_analyze_metric,
    format_log_search_results,
)


def _detail_payload(**extra):
    payload = {
        "status": "Complete",
        "log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs",
        "hours": 24,
        "search_filter": "http_2xx",
        "response_mode": "detail",
        "match_count": 15483,
        "total_matches": 15483,
        "max_result_lines": 15,
        "results": [
            {"timestamp": "2026-10-27 10:00:00.000", "log_group": "/aws/apigateway/era-api-gateway-dev/access-logs",
             "message": '{"status":"200","path":"/x"}'}
            for _ in range(15)
        ],
        "truncated": True,
    }
    payload.update(extra)
    return payload


def test_detail_header_reports_real_total_not_preview_size():
    text = format_log_search_results(_detail_payload())
    assert "**15483** kayıt" in text
    assert "en yeni **15** satır getirildi" in text
    assert "en fazla 500 satır" in text


def test_summary_shows_system_scope_and_hidden_groups():
    text = format_log_search_results(
        {
            "status": "Complete",
            "hours": 6,
            "search_filter": "errors",
            "response_mode": "summary",
            "scope": "system",
            "excluded_prefixes": ["/aws/codebuild/"],
            "log_groups_searched": 195,
            "groups_with_matches": 15,
            "total_matches": 6089,
            "match_count": 6089,
            "results": [
                {"log_group": "/aws/msk-debezium-dev", "matches": "3178"},
                {"log_group": "/aws/containerinsights/era-eks-dev/plt-comm-catalog", "matches": "748"},
            ],
        }
    )
    assert "**15 log grubunda**" in text
    assert "**Toplam: 6089**" in text
    assert "tüm sistem** — 195 log grubu" in text
    assert "13 grupta" in text


def test_single_group_errors_summary_includes_status_breakdown():
    text = format_log_search_results(
        {
            "status": "Complete",
            "log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs",
            "hours": 12,
            "search_filter": "errors",
            "response_mode": "summary",
            "match_count": 560,
            "total_matches": 560,
            "results": [{"log_group": "/aws/apigateway/era-api-gateway-dev/access-logs", "matches": "560"}],
            "status_breakdown": [{"status": "403", "count": 3000}, {"status": "500", "count": 16}],
        }
    )
    assert "Evet — son **12 saat** içinde" in text
    assert "**403**: 3000" in text
    assert "**500**: 16" in text


def test_window_label_replaces_rolling_hours():
    text = format_log_search_results(
        {
            "status": "Complete",
            "log_group_name": "/aws/x",
            "hours": 24,
            "window_label": "dün (26.10.2026 00:00–24:00, TR saati)",
            "period_label": "dün (26.10.2026 00:00–24:00, TR saati)",
            "search_filter": "http_500",
            "response_mode": "summary",
            "match_count": 490,
            "results": [{"log_group": "/aws/x", "matches": "490"}],
        }
    )
    assert "**dün (26.10.2026 00:00–24:00, TR saati)** içinde" in text
    assert "son **24 saat**" not in text


def test_fixed_window_is_carried_into_followup_context():
    ctx = build_log_context_from_tool_data(
        {
            "log_group_name": "/aws/x",
            "hours": 24,
            "search_filter": "http_500",
            "window_label": "dün",
            "start_time": "2026-10-25T21:00:00Z",
            "end_time": "2026-10-26T21:00:00Z",
            "match_count": 490,
        }
    )
    assert ctx["start_time"] == "2026-10-25T21:00:00Z"
    assert ctx["window_label"] == "dün"


def test_llm_context_keeps_formatted_rows():
    compact = json.loads(
        compact_tool_result_for_context("query_log_group", json.dumps(_detail_payload()))
    )
    assert len(compact["results"]) == 15
    assert compact["match_count"] == 15483
    assert compact["truncated"] is True


def test_metric_formatter_shows_sum_and_direction():
    text = format_analyze_metric(
        {
            "namespace": "AWS/ApiGateway",
            "metric_name": "5XXError",
            "statistic": "Sum",
            "aggregation": "account-wide",
            "datapoint_count": 24,
            "min": 0,
            "max": 120,
            "average": 50.5,
            "sum": 1214,
            "trend": "increasing",
            "first_half_avg": 40,
            "second_half_avg": 60,
            "first_timestamp": "2026-10-26T11:00:00Z",
            "last_timestamp": "2026-10-27T10:00:00Z",
        }
    )
    assert "**Toplam: 1.214**" in text
    assert "artıyor" in text and "+50%" in text
    assert "hesap geneli" in text


def test_ranking_card_and_suggestions():
    from app.agent import build_response_card, build_suggestions

    card = build_response_card(
        "search_logs_across_groups",
        {
            "status": "Complete",
            "hours": 6,
            "search_filter": "errors",
            "response_mode": "summary",
            "scope": "system",
            "log_groups_searched": 195,
            "groups_with_matches": 15,
            "total_matches": 6089,
            "bytes_scanned": 3_500_000_000,
            "insights_batches": 5,
            "results": [
                {"log_group": "/aws/containerinsights/era-eks-dev/plt-comm-catalog", "matches": "748"},
                {"log_group": "/aws/msk-debezium-dev", "matches": "3178"},
            ],
        },
    )
    assert card["kind"] == "ranking"
    assert card["top"][0] == {"group": "/aws/msk-debezium-dev", "count": 3178}
    assert card["total"] == 6089 and card["groups_with_matches"] == 15
    assert build_suggestions(card)[0] == "1 numarayı detaylı göster"


def test_count_card_carries_status_breakdown():
    from app.agent import build_response_card, build_suggestions

    card = build_response_card(
        "query_log_group",
        {
            "log_group_name": "/aws/x",
            "hours": 12,
            "search_filter": "errors",
            "response_mode": "summary",
            "match_count": 560,
            "results": [{"log_group": "/aws/x", "matches": "560"}],
            "status_breakdown": [{"status": "403", "count": 3000}],
        },
    )
    assert card["kind"] == "count"
    assert card["status_breakdown"][0]["status"] == "403"
    assert build_suggestions(card) == ["detaylı göster", "logları analiz et", "son 24 saate bak"]


def test_metric_card_has_series_and_change():
    from app.agent import build_response_card

    card = build_response_card(
        "analyze_metric",
        {
            "namespace": "AWS/Lambda",
            "metric_name": "Errors",
            "statistic": "Sum",
            "datapoint_count": 2,
            "sum": 30,
            "first_half_avg": 10,
            "second_half_avg": 20,
            "trend": "increasing",
            "series": [["2026-10-27T00:00:00Z", 10], ["2026-10-27T01:00:00Z", 20]],
        },
    )
    assert card["kind"] == "metric"
    assert card["change_pct"] == 100.0
    assert len(card["series"]) == 2
