import os

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
AWS_ACCOUNT_ID = os.getenv("AWS_ACCOUNT_ID", "212845026981")
INSIGHTS_MAX_LOG_GROUPS = int(os.getenv("INSIGHTS_MAX_LOG_GROUPS", "50"))
ENABLE_MECHANICAL_FALLBACKS = os.getenv("ENABLE_MECHANICAL_FALLBACKS", "false").lower() in {
    "1",
    "true",
    "yes",
}
