from datetime import datetime, timedelta, timezone

import pytest

import app.server as srv
from app.queries import _FILTER_CLAUSES, resolve_count_query


def _row(**fields):
    return [{"field": k, "value": str(v)} for k, v in fields.items()]


CATALOG = (
    [{"logGroupName": f"/aws/containerinsights/era-eks-dev/svc-{i:02d}", "logGroupClass": "STANDARD"} for i in range(33)]
    + [{"logGroupName": f"/aws/lambda/fn-{i:03d}", "logGroupClass": "STANDARD"} for i in range(110)]
    + [{"logGroupName": f"/aws/codebuild/build-{i:03d}", "logGroupClass": "STANDARD"} for i in range(40)]
    + [
        {"logGroupName": "/aws/msk-debezium-dev", "logGroupClass": "STANDARD"},
        {"logGroupName": "/aws/containerinsights/era-eks-dev/plt-comm-order", "logGroupClass": "STANDARD"},
        {"logGroupName": "/aws/archive/cold", "logGroupClass": "INFREQUENT_ACCESS"},
    ]
)


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(srv, "get_log_group_catalog", lambda _client: CATALOG)


def test_system_scope_covers_every_runtime_group(catalog):
    names = srv.collect_log_group_names_for_search(object(), None)
    assert len(names) == len(CATALOG) - 40
    assert "/aws/msk-debezium-dev" in names
    assert not any(n.startswith("/aws/codebuild/") for n in names)


def test_keyword_scope_returns_all_matches_not_first_ten(catalog):
    names = srv.collect_log_group_names_for_search(object(), ["lambda"])
    assert len(names) == 110


def test_codebuild_included_when_user_asks_for_it(catalog):
    names = srv.collect_log_group_names_for_search(object(), ["codebuild"])
    assert len(names) == 40


def test_batches_respect_50_limit_and_log_class(catalog):
    names = srv.collect_log_group_names_for_search(object(), None)
    batches = srv.batch_log_groups_for_insights(object(), names)
    assert all(len(b) <= 50 for b in batches)
    assert sum(len(b) for b in batches) == len(names)
    cold = [b for b in batches if "/aws/archive/cold" in b]
    assert cold == [["/aws/archive/cold"]]


def test_merge_ranked_rows_sums_and_sorts():
    merged = srv.merge_insights_results(
        [_row(**{"@log": "a", "matches": 5}), _row(**{"@log": "b", "matches": 9}), _row(**{"@log": "a", "matches": 1})],
        sum_field="matches",
    )
    assert [srv._row_fields(r) for r in merged] == [{"@log": "b", "matches": "9"}, {"@log": "a", "matches": "6"}]


def test_merge_keeps_rows_without_sum_field():
    rows = [_row(status="500", n=3)]
    assert srv.merge_insights_results(rows, sum_field="matches") == rows


def test_merge_lines_newest_first():
    merged = srv.merge_insights_results(
        [_row(**{"@timestamp": "2026-10-27 10:00:00.000"}), _row(**{"@timestamp": "2026-10-27 11:00:00.000"})]
    )
    assert srv._row_fields(merged[0])["@timestamp"].startswith("2026-10-27 11")


def test_errors_filter_counts_http_5xx():
    assert "status >= 500" in _FILTER_CLAUSES["errors"]


def test_count_query_has_no_limit():
    query = resolve_count_query("http_2xx")
    assert "stats count() as matches by @log" in query
    assert "limit" not in query


@pytest.mark.parametrize(
    ("metric", "stat"),
    [("Errors", "Sum"), ("Count", "Sum"), ("5XXError", "Sum"), ("CPUUtilization", "Average"), ("Latency", "Average")],
)
def test_default_statistic(metric, stat):
    assert srv.default_statistic_for_metric(metric) == stat


def test_metrics_insights_uses_top_level_schema():
    assert (
        srv.metrics_insights_expression("AWS/ApiGateway", "Count", "Sum")
        == 'SELECT SUM("Count") FROM SCHEMA("AWS/ApiGateway", ApiName)'
    )
    assert srv.metrics_insights_expression("Custom/App", "Foo", "Sum") is None


class FakeCloudWatch:
    """Returns datapoints newest-first (CloudWatch's default order) unless ScanBy says otherwise."""

    def __init__(self):
        self.calls = []

    def get_metric_data(self, **kwargs):
        self.calls.append(kwargs)
        base = datetime(2026, 10, 27, tzinfo=timezone.utc)
        timestamps = [base + timedelta(hours=h) for h in range(6)]
        values = [1, 1, 1, 10, 10, 10]  # rising over time
        return {
            "MetricDataResults": [
                {"Timestamps": list(reversed(timestamps)), "Values": list(reversed(values))}
            ]
        }


def test_analyze_metric_trend_uses_chronological_order(monkeypatch):
    fake = FakeCloudWatch()
    monkeypatch.setattr(srv, "get_client", lambda service, region=None: fake)
    result = srv.analyze_metric(
        namespace="AWS/Lambda", metric_name="Errors", start_time="2026-10-27T00:00:00Z"
    )
    assert result["trend"] == "increasing"
    assert result["statistic"] == "Sum"
    assert result["sum"] == 33
    assert result["first_timestamp"] < result["last_timestamp"]
    assert fake.calls[0]["ScanBy"] == "TimestampAscending"
    assert "Expression" in fake.calls[0]["MetricDataQueries"][0]


def test_sub_hour_window(monkeypatch):
    now = datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(srv, "utc_now", lambda: now)
    start, end, hours = srv.resolve_query_time_window(minutes=30)
    assert end - start == timedelta(minutes=30)
    assert hours == 1
    assert srv.describe_time_window(minutes=30, start_time=None, end_time=None, period_label=None) == "son 30 dakika"
