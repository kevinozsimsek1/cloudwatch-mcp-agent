# CloudWatch MCP Agent

| Dosya              | Açıklama                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| `config.py`        | Ortam değişkenlerini ve uygulama yapılandırmasını yönetir.                   |
| `queries.py`       | CloudWatch Logs Insights sorgu şablonlarını içerir.                          |
| `sre_context.py`   | Doğal dil zaman ifadelerini yorumlar ve SRE referans verilerini sağlar.      |
| `agent.py`         | LLM promptları, istek yönlendirme ve yanıt formatlamasını gerçekleştirir. |
| `server.py`        | MCP araçlarını, AWS Boto3 entegrasyonunu ve HTTP API'yi sunar.               |
| `index.html`       | Chat arayüzünü, dashboard'u ve log kartlarını içerir.                        |
| `deployment.yaml`  | Uygulamanın EKS ortamına dağıtım manifestidir.                               |
| `.env.example`     | Lokal geliştirme ortamı için örnek ortam değişkenlerini içerir.              |
| `sre_chat_test.py` | Otomatik regresyon testlerini ve senaryo doğrulamalarını içerir.             |

VLLM + AGENT CloudWatch log, metric ve alarm sorularını yanıtlayan LLM agent. Tek bir FastAPI servisi olarak çalışır; CloudWatch tool'ları in-process MCP ile sunar ve cevapları cluster içindeki vLLM modelinden üretir.

```
Kullanıcı → Chat UI / API → Agent loop → vLLM → MCP tools (boto3) → AWS CloudWatch (IRSA)
```

## Gereksinimler

- Kubernetes cluster (EKS)
- `kubectl` ve cluster erişimi
- Cluster içinde çalışan vLLM OpenAI-compatible endpoint
- CloudWatch okuma yetkisi olan IAM role (IRSA ile ServiceAccount'a bağlı)

## Kubernetes'e deploy

Manifest'leri kendi ortamına göre düzenle (`k8s/deployment.yaml` içindeki ECR image, IAM role ARN, vLLM URL):

```bash
kubectl apply -f k8s/deployment.yaml
```

Deploy sonrası kontrol:

```bash
kubectl get pods -n mcp-llm-agent
kubectl get svc -n mcp-llm-agent
```


## UI'ye erişim (port-forward)

Service tipi `ClusterIP` olduğu için dışarıdan doğrudan erişilemez. Chat arayüzünü açmak için **her oturumda** (veya ihtiyaç olduğunda) port-forward çalıştır:

```bash
kubectl port-forward -n mcp-llm-agent svc/cloudwatch-agent 8080:80
```

Terminal açık kalsın. Tarayıcıda:

**http://localhost:8080**

Port-forward'u arka planda çalıştırmak istersen:

```bash
kubectl port-forward -n mcp-llm-agent svc/cloudwatch-agent 8080:80 &
```


### HealthCheck kontrolleri

Port-forward aktifken:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/ready
```

`/ready` yanıtında kayıtlı CloudWatch tool listesi de döner.

## Lokal geliştirme

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # değerleri düzenle
uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload
```

Lokal çalışırken port-forward gerekmez; doğrudan http://localhost:8080 açılır.

## Docker image

```bash
docker build --platform linux/amd64 -t cloudwatch-agent:latest .
docker run --rm -p 8080:8080 --env-file .env cloudwatch-agent:latest
```

## API

| Endpoint | Açıklama |
|----------|----------|
| `GET /` | Chat UI |
| `GET /health` | Liveness |
| `GET /ready` | Readiness + tool kataloğu |
| `POST /chat` | Agent sohbet API |
| `GET /mcp` | MCP endpoint (FastMCP) |

Örnek chat isteği:

```bash
curl -s http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Aktif CloudWatch alarmları var mı?","history":[]}'
```

## CloudWatch tool'ları
mouse ile üzerine geline tool listesi ui'da gözüküyor

- `describe_log_groups` — log group listeleme / arama
- `analyze_log_group` — log group analizi
- `execute_log_insights_query` — Logs Insights sorgusu başlat
- `get_logs_insight_query_results` — sorgu sonuçları
- `cancel_logs_insight_query` — sorguyu iptal
- `get_active_alarms` — aktif alarmlar
- `get_alarm_history` — alarm geçmişi
- `get_metric_data` — metric verisi
- `get_metric_metadata` — metric metadata
- `get_recommended_metric_alarms` — önerilen alarmlar
- `analyze_metric` — metric analizi

## env

| Değişken | Açıklama | Örnek |
|----------|----------|-------|
| `VLLM_BASE_URL` | vLLM OpenAI API base URL | `http://vllm-gptoss.llm-model.svc.cluster.local:8080/v1` |
| `MODEL_NAME` | Model adı | `openai/gpt-oss-20b` |
| `AWS_REGION` | AWS bölgesi | `eu-central-1` |
| `MAX_TOOL_ITERATIONS` | Agent tool döngü limiti | `10` |
| `MAX_HISTORY_MESSAGES` | Sohbet geçmişi mesaj limiti | `6` |
| `MAX_HISTORY_MESSAGE_CHARS` | Mesaj başına karakter limiti | `2500` |
| `MAX_TOOL_RESULT_CHARS` | Tool çıktısı truncate limiti | `8000` |
| `MAX_LOG_GROUPS_LIST` | Listelenecek max log group | `1000` |
| `LLM_MAX_TOKENS` | LLM max token | `700` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` |
| `LOG_LEVEL` | Log seviyesi | `INFO` |
