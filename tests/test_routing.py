import pytest

import app.server  # noqa: F401  (agent resolves server lazily)
from app.agent import (
    adjust_search_logs_args,
    apply_time_window_args,
    extract_tenant_filter,
    infer_keywords_from_message,
    infer_log_search_filter,
    is_metric_analysis_request,
)
from app.sre_context import infer_metric_targets


@pytest.mark.parametrize(
    "message",
    [
        "api gateway access loglarında son 6 saatte error var mı",
        "How many 500 errors did the API gateway access logs have in the last 24 hours?",
        "Lambda fonksiyonlarında son 24 saatte hata var mı?",
        "Son 6 saatte hangi log grubunda en çok error var?",
    ],
)
def test_log_questions_are_not_hijacked_by_metric_handler(message):
    assert not is_metric_analysis_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "Lambda error metriği son 24 saatte artıyor mu?",
        "ECS CPU son 1 saat",
        "RDS bağlantı sayısı son 3 saat",
        "api gateway 5xx metriği",
    ],
)
def test_explicit_metric_questions_route_to_metrics(message):
    assert is_metric_analysis_request(message)


def test_api_gateway_error_metric_maps_to_5xx():
    assert ("AWS/ApiGateway", "5XXError") in infer_metric_targets("api gateway 500 metriği")
    assert ("AWS/Lambda", "Errors") in infer_metric_targets("lambda hata metriği")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("plt-comm-order servisinde son 6 saatte error var mı?", ["plt-comm-order"]),
        ("debezium loglarında son 6 saatte hata var mı", ["debezium"]),
        ("nginx loglarında 502 var mı", ["nginx"]),
        ("Sistemde son 6 saatte en çok hata veren servis hangisi?", []),
        ("tüm log gruplarında error", []),
        ("son 24 saatte 500 hataları", []),
    ],
)
def test_named_service_becomes_search_keyword(message, expected):
    assert infer_keywords_from_message(message) == expected


def test_full_path_segments_are_not_keywords():
    keywords = infer_keywords_from_message(
        "/aws/containerinsights/era-eks-dev/plt-comm-order son 1 saat hata"
    )
    assert "era-eks-dev" not in keywords
    assert "plt-comm-order" not in keywords


def test_yesterday_uses_calendar_window_not_rolling_24h():
    args = apply_time_window_args("Dün kaç tane 500 oldu", {"hours": 24})
    assert args["start_time"].endswith("T21:00:00Z")
    assert args["end_time"].endswith("T21:00:00Z")
    assert args["period_label"].startswith("dün")


def test_minutes_are_passed_through():
    args = apply_time_window_args("son 30 dakikada 500 var mı", {"hours": 1})
    assert args == {"hours": 1, "minutes": 30}


def test_user_hours_override_llm_guess():
    args = apply_time_window_args("son 6 saatte error", {"hours": 1, "start_time": "x", "end_time": "y"})
    assert args == {"hours": 6}


def test_comparison_leg_gets_tr_calendar_from_period_label():
    args = apply_time_window_args(
        "bugün vs dün 500 karşılaştır",
        {"period_label": "Dün", "start_time": "2026-10-26T00:00:00Z", "end_time": "2026-10-26T23:59:59Z"},
    )
    assert args["period_label"] == "Dün"
    assert args["start_time"].endswith("T21:00:00Z")
    assert args["hours"] == 24


def test_comparison_without_label_keeps_llm_window():
    original = {"start_time": "2026-10-20T00:00:00Z", "end_time": "2026-10-21T00:00:00Z"}
    assert apply_time_window_args("son 6 saat vs önceki 6 saat", dict(original)) == original


@pytest.mark.parametrize(
    ("message", "tenant"),
    [
        ("Sadece api-gateway ve lambda log gruplarında son 1 saatte 500 sayısını getir", None),
        ("sadece lambda loglarında hata", None),
        ("sadece 500 olanları göster", None),
        ("sadece bozkurteradev status 500 olanları göster", "bozkurteradev"),
        ("bozkurteradev tenantına ait 500ler", "bozkurteradev"),
    ],
)
def test_tenant_extraction_ignores_log_group_scope(message, tenant):
    assert extract_tenant_filter(message) == tenant


def test_scoped_count_only_status_search():
    message = "Sadece api-gateway ve lambda log gruplarında son 1 saatte 500 sayısını getir, detay istemiyorum"
    args = adjust_search_logs_args(message, {"search_filter": infer_log_search_filter(message)})
    assert args["response_mode"] == "count_only"
    assert args["log_group_name_keywords"] == ["api-gateway", "lambda"]
    assert "tenant_filter" not in args
