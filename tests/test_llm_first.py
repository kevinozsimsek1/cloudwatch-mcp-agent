import json

import app.server  # noqa: F401
from app.agent import compose_llm_first_answer, normalize_tool_call, verify_insight_numbers

TOOL_RESULT = json.dumps(
    {
        "status": "Complete",
        "log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs",
        "hours": 24,
        "search_filter": "http_5xx",
        "response_mode": "summary",
        "match_count": 1209,
        "total_matches": 1209,
        "results": [{"log_group": "/aws/apigateway/era-api-gateway-dev/access-logs", "matches": "1209"}],
    }
)


def test_insight_keeps_numbers_from_tool_results():
    text = "Son 24 saatte 1.209 adet 5xx var. Çoğu entegrasyon zaman aşımı gibi görünüyor."
    assert verify_insight_numbers(text, [TOOL_RESULT]) == text


def test_insight_drops_invented_numbers_only():
    text = "Toplam 1209 kayıt var. Bunların 873 tanesi swagger endpoint'inden geliyor. Detaya bakmanı öneririm."
    assert verify_insight_numbers(text, [TOOL_RESULT]) == "Toplam 1209 kayıt var. Detaya bakmanı öneririm."


def test_insight_drops_computed_percentages_and_dangling_follow_up():
    text = "Bu grupta 1209 kayıt var. Toplamın %44'ü burada. Bu da dikkat çekici. 2 API etkileniyor."
    assert verify_insight_numbers(text, [TOOL_RESULT]) == "Bu grupta 1209 kayıt var. 2 API etkileniyor."


def test_clean_insight_removes_tables_and_placeholders():
    from app.agent import clean_insight

    text = "En sık endpoint'ler şunlar:\n\n| Endpoint | Sayı |\n|---|---|\n| … | … |\n\nÇoğu swagger kaynaklı.\nŞunlara bakabilirsin:"
    assert clean_insight(text) == "En sık endpoint'ler şunlar:\n\nÇoğu swagger kaynaklı."


def test_scope_guard_replaces_invented_group_with_search():
    from app.agent import align_scope_with_context

    name, args = align_scope_with_context(
        "peki 4xx?",
        "query_log_group",
        {"log_group_name": "/aws/apigateway/era-api-gateway-dev/access-logs", "hours": 1, "search_filter": "http_4xx"},
        {"search_filter": "http_500", "hours": 1},
        [{"role": "user", "content": "son 1 saatte sistemde 500 var mı"}],
    )
    assert name == "search_logs_across_groups"
    assert "log_group_name" not in args and args["search_filter"] == "http_4xx"


def test_scope_guard_keeps_follow_up_on_previous_group():
    from app.agent import align_scope_with_context

    group = "/aws/apigateway/era-api-gateway-dev/access-logs"
    name, args = align_scope_with_context(
        "bunların kaçı 403?",
        "search_logs_across_groups",
        {"search_filter": "http_403", "hours": 12, "response_mode": "summary"},
        {"log_group_name": group, "hours": 12, "search_filter": "errors"},
        [],
    )
    assert name == "query_log_group"
    assert args["log_group_name"] == group and args["hours"] == 12 and args["search_filter"] == "http_403"


def test_metric_raw_call_becomes_summary():
    from app.agent import align_scope_with_context

    name, args = align_scope_with_context(
        "API Gateway 5xx metriği", "get_metric_data",
        {"namespace": "AWS/ApiGateway", "metric_name": "5XXError", "start_time": "x", "bogus": 1}, {}, [],
    )
    assert name == "analyze_metric" and "bogus" not in args


def test_compose_puts_model_text_in_insight_and_data_in_response():
    tool_calls = [{"name": "query_log_group", "arguments": {}}]
    result = compose_llm_first_answer(
        "api gateway 5xx",
        "1209 adet 5xx var; çoğu entegrasyon zaman aşımı.",
        [("query_log_group", {}, TOOL_RESULT)],
        tool_calls,
        2,
    )
    assert result["insight"].startswith("1209 adet 5xx")
    assert "**1209** adet" in result["response"]
    assert result["card"]["kind"] == "count"
    assert result["log_context"]["log_group_name"].endswith("access-logs")


def test_compose_without_tools_returns_model_text():
    result = compose_llm_first_answer("500 var mı?", "Tüm sistemde mi bakayım?", [], [], 1)
    assert result == {"response": "Tüm sistemde mi bakayım?", "tool_calls": [], "iterations": 1}


def test_llm_first_keeps_group_resolved_from_context():
    name, args = normalize_tool_call(
        "2. grubun detayını göster",
        "query_log_group",
        {"log_group_name": "/aws/containerinsights/era-eks-dev/plt-comm-catalog", "hours": 6, "search_filter": "errors"},
        keep_llm_args=True,
    )
    assert name == "query_log_group"
    assert args["log_group_name"] == "/aws/containerinsights/era-eks-dev/plt-comm-catalog"
    assert args["hours"] == 6
    assert args["search_filter"] == "errors"


def test_finalize_insight_drops_reasoning_leak_and_jargon():
    from app.agent import finalize_insight

    assert finalize_insight("son 24 saatte tetiklenenler?", "The tool returned a large JSON. We need to summarize.") is None
    text = "En çok hata debezium'da.\nDetay için `response_mode=detail` ile sorgulayabiliriz.\nŞunlardı:\n1.\n2."
    assert finalize_insight("en çok hata nerede", text) == "En çok hata debezium'da."


def test_followup_keeps_system_scope_for_new_filter():
    from app.agent import align_scope_with_context

    group = "/aws/apigateway/era-api-gateway-dev/access-logs"
    name, args = align_scope_with_context(
        "peki 4xx?",
        "query_log_group",
        {"log_group_name": group, "hours": 1, "search_filter": "http_4xx"},
        {"scope": "system", "log_group_name": group, "search_filter": "http_500", "hours": 1},
        [],
    )
    assert name == "search_logs_across_groups"
    assert "log_group_name" not in args and args["response_mode"] == "summary"


def test_drilldown_on_single_result_group_stays_on_group():
    from app.agent import align_scope_with_context

    group = "/aws/apigateway/era-api-gateway-dev/access-logs"
    name, args = align_scope_with_context(
        "detaylı göster",
        "query_log_group",
        {"log_group_name": group, "hours": 1, "search_filter": "http_500"},
        {"scope": "system", "log_group_name": group, "search_filter": "http_500", "hours": 1},
        [],
    )
    assert name == "query_log_group" and args["log_group_name"] == group


def test_single_explicit_group_search_becomes_group_query():
    from app.agent import align_scope_with_context

    name, args = align_scope_with_context(
        "ikinci servisin son 1 saatine bak",
        "search_logs_across_groups",
        {"search_filter": "errors", "hours": 1, "log_group_names": ["/aws/otlp/era-eks-dev/pos-backend"]},
        {"ranked_groups": [("/aws/msk-debezium-dev", 9), ("/aws/otlp/era-eks-dev/pos-backend", 5)]},
        [],
    )
    assert name == "query_log_group" and args["log_group_name"] == "/aws/otlp/era-eks-dev/pos-backend"


def test_clean_insight_drops_group_rows_already_on_card():
    from app.agent import clean_insight, finalize_insight

    text = (
        "Son 6 saatte en çok hata üreten log grupları:\n\n"
        "3. **/aws/containerinsights/era-eks-dev/plt-comm-catalog** – 688 hata\n"
        "4. **API‑Gateway‑Execution‑Logs_rmfixdooo6/dev** – 523 hata\n\n"
        "Hataların çoğu pos-backend istemci bağlantılarından geliyor."
    )
    assert finalize_insight("en çok hata", clean_insight(text)) == "Hataların çoğu pos-backend istemci bağlantılarından geliyor."
