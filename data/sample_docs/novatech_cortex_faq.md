# NovaTech Cortex Platform — Frequently Asked Questions

## General

### What is NovaTech Cortex?

NovaTech Cortex is an enterprise AI platform that provides end-to-end capabilities for building, deploying, and managing AI applications. It includes pre-built models for natural language processing, computer vision, and predictive analytics, along with tools for custom model training and fine-tuning.

### What makes Cortex different from other AI platforms?

Three key differentiators set Cortex apart:

1. **Hybrid Search Architecture**: Cortex combines semantic vector search with traditional keyword search (BM25) and uses a cross-encoder re-ranker to ensure the most relevant results. This approach consistently outperforms pure vector search by 15-20% on standard benchmarks.

2. **Agentic Routing**: Rather than sending every query through the same pipeline, Cortex uses an intelligent LLM agent to analyze each query and route it to the most appropriate data source — whether that's a vector database, SQL database, or real-time web search.

3. **Enterprise-Grade Security**: SOC 2 Type II certified, HIPAA compliant, with fine-grained RBAC and complete audit logging.

### Which LLMs does Cortex support?

Cortex supports multiple LLM providers out of the box:
- Google Gemini 1.5 Pro and Flash
- Anthropic Claude 3.5 Sonnet and Opus
- Meta Llama 3.1 (via Groq for inference)
- OpenAI GPT-4o and o1-preview
- Custom fine-tuned models via NVIDIA NIM

Customers can switch between providers without changing their application code, thanks to our unified API layer.

## Deployment & Infrastructure

### How is Cortex deployed?

Cortex offers three deployment options:

- **SaaS (Multi-tenant)**: Fully managed by NovaTech. Data is encrypted at rest and in transit. Available in US-East, US-West, EU-West, and AP-Southeast regions.
- **Dedicated Cloud**: Single-tenant deployment on your cloud provider (AWS, GCP, Azure). NovaTech manages the infrastructure.
- **On-Premise**: Full deployment within your data center. Requires Kubernetes (v1.28+) and NVIDIA GPUs for inference.

### What are the hardware requirements for on-premise deployment?

Minimum requirements for a production deployment:
- 8 CPU cores, 64GB RAM for the control plane
- NVIDIA A100 or H100 GPUs (minimum 2) for model inference
- 500GB NVMe SSD for vector database storage
- 10Gbps network connectivity between nodes

### What is the uptime SLA?

SaaS: 99.99% uptime (approximately 4 minutes of downtime per month)
Dedicated Cloud: 99.95% uptime
On-Premise: Dependent on customer infrastructure; NovaTech provides 24/7 support

## Data & Privacy

### Where is my data stored?

For SaaS deployments, data is stored in the region you select during onboarding. Data never leaves your selected region. We use AES-256 encryption at rest and TLS 1.3 in transit.

### Does NovaTech use customer data to train models?

Absolutely not. Customer data is never used to train, fine-tune, or improve NovaTech's base models. Each customer's data is completely isolated. This commitment is enshrined in our Data Processing Agreement (DPA) and independently verified through our SOC 2 audit.

### Can I bring my own encryption keys?

Yes. Cortex supports customer-managed encryption keys (CMEK) through AWS KMS, Google Cloud KMS, or Azure Key Vault. This gives you full control over data encryption and the ability to revoke access at any time.

## Pricing

### How is Cortex priced?

Cortex uses a consumption-based pricing model with three tiers:

| Tier         | Monthly Base | Included Queries | Overage per 1K Queries |
|--------------|-------------|------------------|------------------------|
| Starter      | $2,500      | 100,000          | $15                    |
| Professional | $8,000      | 500,000          | $10                    |
| Enterprise   | Custom      | Unlimited        | N/A                    |

All tiers include: platform access, standard models, vector database hosting, and email support. Professional and Enterprise tiers add: custom model fine-tuning, dedicated support, and SLA guarantees.

### Is there a free trial?

Yes! We offer a 14-day free trial with full access to the Professional tier. No credit card required. Sign up at https://cortex.novatech.ai/trial.

## Integration

### What APIs does Cortex expose?

Cortex provides both REST and gRPC APIs. Key endpoints include:

- `/v2/ingest` — Document ingestion and processing
- `/v2/query` — Unified query interface (supports RAG, SQL, and web search)
- `/v2/models` — Model management (deploy, fine-tune, evaluate)
- `/v2/admin` — User management, usage analytics, audit logs

All APIs are versioned and backward-compatible. We maintain N-1 version support.

### Does Cortex integrate with existing tools?

Yes. Pre-built integrations are available for:
- **Data Sources**: Snowflake, BigQuery, PostgreSQL, MongoDB, S3, GCS
- **Communication**: Slack, Microsoft Teams, email
- **Workflow**: Zapier, Make, Airflow, Prefect
- **Observability**: Datadog, Grafana, New Relic
- **Identity**: Okta, Auth0, Azure AD

Custom integrations can be built using our SDK (Python, JavaScript, Go).

## Support

### What support options are available?

| Support Tier | Response Time (P1) | Channels              | Availability |
|-------------|--------------------|-----------------------|-------------|
| Standard    | 4 hours            | Email, Portal         | Business hrs |
| Premium     | 1 hour             | Email, Portal, Phone  | 24/5         |
| Enterprise  | 15 minutes         | All + Slack Channel   | 24/7/365     |

Enterprise customers are assigned a dedicated Technical Account Manager (TAM) and receive quarterly business reviews.
