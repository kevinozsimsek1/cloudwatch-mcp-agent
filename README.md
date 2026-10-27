# CloudWatch MCP Agent

| Dosya              | Açıklama                                                                     |
| ------------------ | ---------------------------------------------------------------------------- |
| `config.py`        | Ortam değişkenlerini ve uygulama yapılandırmasını yönetir.                   |
| `queries.py`       | CloudWatch Logs Insights sorgu şablonlarını içerir.                          |
| `sre_context.py`   | LLM zaman ifadelerini yorumlar ve SRE referans verilerini sağlar.      |
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
| `MAX_HISTORY_MESSAGES` | LLM'in gördüğü geçmiş mesaj sayısı | `12` |
| `MAX_HISTORY_MESSAGE_CHARS` | Mesaj başına karakter limiti | `2500` |
| `MAX_TOOL_RESULT_CHARS` | Tool çıktısı truncate limiti | `8000` |
| `MAX_LOG_GROUPS_LIST` | Listelenecek max log group | `1000` |
| `LLM_MAX_TOKENS` | LLM max token (gpt-oss reasoning dahil) | `2000` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` |
| `LOG_LEVEL` | Log seviyesi | `INFO` |
| `DISPLAY_TIMEZONE` | "bugün / dün / bu hafta" takvim sınırları | `Europe/Istanbul` |
| `SEARCH_MAX_LOG_GROUPS` | Tek aramada taranacak max log group (50'lik batch'lere bölünür) | `500` |
| `SYSTEM_SCAN_EXCLUDE_PREFIXES` | Sistem çapı aramadan hariç tutulan prefix'ler (virgülle) | `/aws/codebuild/` |
| `LOG_GROUP_CATALOG_TTL` | Log group kataloğu cache süresi (sn) | `300` |
| `INSIGHTS_MAX_CONCURRENCY` | Paralel Insights batch sayısı | `8` |

## Agent nasıl çalışır

Her mesajı LLM (gpt-oss-20b) karşılar; takip soruları, atıflar ("2. grubun detayı",
"bunların kaçı 403?", "peki dün?") ve "neden" soruları önceki tool sonuçları ve oturum bağlamı üzerinden
çözülür. Kod tarafı doğruluğu garanti eder:

- Tool argümanları kullanıcının sözüne göre düzeltilir (zaman penceresi, kapsam, tenant).
- Konuşmanın kapsamı korunur: modelin uydurduğu log grubu sistem aramasına çevrilir, tek gruplu bir
  cevabın ardından gelen "bunların…" sorusu aynı grupta kalır.
- Cevap = gerçek veriden çizilen kart + modelin 2–4 cümlelik yorumu. Yorumda tool sonuçlarında geçmeyen
  sayı içeren cümleler, tablolar ve teknik parametre adları sunucu tarafından silinir.

## Cevap doğruluğu — nasıl çalışır

- **Sistem çapı arama** ("sistemde", "hangi log grubunda en çok…") hesaptaki **tüm runtime log
  gruplarını** tarar (CI/codebuild hariç). Insights tek sorguda 50 grup kabul ettiği için gruplar
  log class'a göre 50'lik batch'lere bölünür ve paralel çalıştırılır; sonuçlar birleştirilir.
  Cevapta "Kapsam: tüm sistem — N log grubu tarandı" satırı görünür.
- **Servis adı** ("plt-comm-order servisinde", "debezium loglarında") aramayı o gruplarla sınırlar.
- **Detay modunda** gösterilen satırlar bir önizlemedir; başlıktaki sayı ayrı bir `stats count()`
  sorgusundan gelen **gerçek toplamdır**.
- **errors** filtresi = uygulama hata kayıtları + tüm HTTP 5xx yanıtları. Access log'larda 4xx/5xx
  dağılımı ayrıca gösterilir.
- **Zaman:** "son 6 saatte", "son 2 günde", "son 30 dakikada" gibi Türkçe ekli ifadeler doğru
  ayrıştırılır; dakika hassasiyeti korunur. "dün" / "bugün" / "geçen hafta" İstanbul takvimine
  göredir (son 24 saat değil).
- **Metrikler:** sayaçlar (Errors, Count, 5XXError…) için Sum, gauge'lar (CPU, latency) için
  Average kullanılır. Boyut verilmezse hesap geneli toplam, çift sayım olmadan
  (`SCHEMA(namespace, üst-boyut)`) hesaplanır. Trend kronolojik sıradan hesaplanır.

## Testler

```bash
pip install -r requirements-dev.txt
python -m pytest -q                                   # unit testler (AWS/LLM gerekmez)

# Canlı servis üzerinde (port-forward açıkken):
python scripts/sre_chat_test.py http://localhost:8080/chat        # tool seçimi regresyonu
python scripts/e2e_user_persona_test.py http://localhost:8080/chat # multi-turn persona akışları

# Sohbet: takip/atıf/"neden" sorularında doğru grup, filtre ve sürenin seçildiğini ölçer.
python scripts/conversation_eval.py --chat-url http://localhost:8080/chat --runs 2

# Doğruluk: agent cevaplarını doğrudan CloudWatch sorgularıyla karşılaştırır,
# her soruyu N kez sorup tutarlılığı ölçer (CloudWatch okuma yetkisi gerekir).
AWS_PROFILE=<profil> python scripts/ground_truth_check.py --chat-url http://localhost:8080/chat --runs 3
```
