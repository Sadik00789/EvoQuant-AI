# ⚡ EvoQuant-AI: Evolutionary Swarm Analytics Engine

> **Event-driven, multi-agent quantitative trading framework running convex risk parity allocations, news sentiment RAG, and LLM-driven strategy evolution over a TimescaleDB time-series backbone.**

---

## 📐 System Architecture

EvoQuant-AI operates as a containerized microservice architecture using a decoupled **Producer-Consumer** pattern over Redis pub/sub messaging.

```text
                 ┌────────────────────────┐
                 │  Alpaca Market Streams │
                 └───────────┬────────────┘
                             │ WebSockets
                             ▼
                    ┌──────────────────┐
                    │ data_producer.py │
                    │(Calculates RSI/MA)│
                    └────────┬─────────┘
                             │
                             │ Redis Pub/Sub
                             ▼
                    ┌──────────────────┐
                    │  evoquant_redis  │
                    └────────┬─────────┘
                             │
                             │ Market Ticks
                             ▼
                    ┌──────────────────┐
                    │ swarm_consumer.py│
                    │ ┌──────────────┐ │
                    │ │ Risk Engine  │ │
                    │ │ Sentiment RAG│ │
                    │ │ Gemma 4-31B  │
                    │ │ Debate Loop  │ │
                    │ └──────────────┘ │
                    └────────┬─────────┘
                             │
                             │ ACID State & Telemetry
                             ▼
                 ┌────────────────────────┐
                 │  evoquant_timescaledb  │
                 │ (PostgreSQL + Hypert)  │
                 └───────────┬────────────┘
                             │
                             │ SQLAlchemy ORM
                             ▼
                 ┌────────────────────────┐
                 │      dashboard.py      │
                 │  (Streamlit UI :8501)  │
                 └────────────────────────┘
```

---

## 🛠️ Microservices Breakdown

| Service | Container Name | Description |
|---|---|---|
| Market Data Producer | `evoquant_producer` | Establishes WebSocket channels to Alpaca, calculates real-time technical indicators (RSI, MACD, ATR, relative strength), and publishes 15-minute bar matrices to Redis. |
| Message Broker | `evoquant_redis` | In-memory Redis instance serving as the asynchronous pub/sub pipeline between the data engine and trade execution layer. |
| Swarm Orchestrator | `evoquant_consumer` | Consumes bar events via a resilient auto-reconnect loop, evaluates macro news sentiment, runs an Adversarial Debate Loop where bull/bear LLM agents contest each trade signal before consensus, triggers parallel asynchronous multi-provider Gemma 4-31B fallback chains (Groq, OpenRouter, SambaNova, GitHub Models), executes risk parity scaling with the SPY 200 SMA Macro Trend Guard, manages long and short positions with dividend-aware cover execution, and dispatches live paper orders to Alpaca. |
| Time-Series Storage | `evoquant_timescaledb` | PostgreSQL 16 database powered by TimescaleDB hypertables for persistent storage of trade history, agent equity telemetry, and macro regime snapshots. |
| Analytics Dashboard | `evoquant_dashboard` | Dark-themed Streamlit analytics terminal providing real-time portfolio heatmaps, Darwinian agent leaderboards, and execution risk audit trails. |

---

## 💡 Key Trading Features

- **Short Selling & Cover Execution:** Agents can open short positions when the swarm consensus and macro trend guard signal a bearish regime, with dedicated cover-order logic to close out shorts on reversal signals, stop-loss triggers, or risk-parity rebalancing.
- **Dividend-Aware Trading:** The engine tracks each holding's ex-dividend date and applies dividend-adjustment rules automatically — flattening or hedging long positions ahead of ex-dividend dates where relevant, and applying a borrow-cost / dividend-liability check before opening or holding a short position through an ex-dividend date.
- **Adversarial Debate Loop:** Bull/bear LLM agents contest each trade signal before consensus is reached (see Swarm Orchestrator above).

---

## 🧰 Tech Stack

- **Language & Runtime:** Python 3.12-slim (pinned, non-root container user)
- **Containerization:** Docker & Docker Compose V2
- **Storage Layer:** PostgreSQL 16 / TimescaleDB (psycopg3, SQLAlchemy)
- **In-Memory Messaging:** Redis (alpine)
- **Quantitative Engine:** Pandas, NumPy, SciPy, Alpaca-Py, Requests — includes short/cover execution logic and ex-dividend date tracking for dividend-aware position management
- **LLM Orchestration:** Gemma 4-31B across Groq, OpenRouter, SambaNova, and GitHub Models APIs, with an Adversarial Debate Loop (bull/bear agent contestation) for signal validation
- **Visualization:** Streamlit, Plotly

---

## 📂 Directory Structure

```text
EvoQuant-AI/
├── .streamlit/
│   └── config.toml             # Streamlit dark theme settings
├── tests/
│   ├── test_swarm.py           # Core unit & integration test suite
│   └── test_hardening.py       # Aggregation, allocation, broker, dividend, payload tests
├── plans/
│   └── evoquant-critical-fixes-plan.md  # Audit findings & remediation roadmap
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusions (blocks .env and cache)
├── Dockerfile                   # Pinned, non-root container build definition
├── docker-compose.yml           # Orchestration spec with Redis auth + healthchecks
├── requirements.txt             # Python dependency manifests (pruned)
├── config.py                    # Centralized environment-driven settings
├── metrics.py                   # Lightweight operational metrics registry
├── data_producer.py             # Market ingestion with true 15m bar aggregation
├── swarm_consumer.py            # Streams consumer, risk overlay, allocation & execution
├── engine.py                    # Portfolio manager, evolution schemas & debate engine
├── evolution_engine.py          # Darwinian strategy evolution & tournament logic
├── risk_engine.py               # Directional stops, cooldowns, session breaker, regime scaler
├── portfolio_risk.py            # Canonical allocator: risk parity + gross/net/sector/CVaR caps
├── broker.py                    # Async, idempotent Alpaca bridge + per-agent sub-accounts
├── dividend_guard.py            # Ex-dividend, short liability & borrow-cost checks
├── news_fetcher.py              # Per-ticker + macro headline enrichment
├── sentiment_agent.py           # Batched bull/bear/arbiter debate & sentiment cache
├── backtest.py                  # Point-in-time long/short backtester with walk-forward
├── db_manager.py                # Compatibility shim to the canonical DB manager
├── dashboard.py                 # Streamlit frontend terminal
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

Ensure you have the following installed on your machine or cloud server:

- Docker Desktop (or Docker Engine on Linux) with Docker Compose V2
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EvoQuant-AI.git
cd EvoQuant-AI
```

### 3. Configure Environment Variables

Copy the template file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your API credentials:

```env
# Primary LLM API Key (Google AI Studio)
GEMINI_API_KEY=your_google_ai_studio_api_key_here

# PostgreSQL / TimescaleDB (password now REQUIRED)
POSTGRES_HOST=timescaledb
POSTGRES_PORT=5432
POSTGRES_DB=evoquant_db
POSTGRES_USER=evoquant
POSTGRES_PASSWORD=change_me_to_a_strong_password

# Redis (password now REQUIRED; datastore is not publicly exposed)
REDIS_PASSWORD=change_me_to_a_strong_password

# Broker Execution (Alpaca paper)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Per-agent paper sub-accounts (JSON) so each agent trades an isolated book.
# ALPACA_SUBACCOUNTS={"Agent_Alpha":{"key":"K1","secret":"S1"},"Agent_Beta":{"key":"K2","secret":"S2"}}

# Safety / behavior flags
SHADOW_MODE=false
MAX_GROSS_EXPOSURE=1.00
MAX_NET_EXPOSURE=0.60
MAX_SECTOR_EXPOSURE=0.30
CVAR_BUDGET=0.04
```


### 4. Build and Launch the Stack

Run Docker Compose in detached mode:

```bash
docker compose up --build -d
```

### 5. Access the Dashboard

Open your web browser and navigate to:

```text
http://localhost:8501
```

*(If deployed on a remote cloud instance, replace `localhost` with your server's public IP address.)*

---

## 📊 Infrastructure Management

**View Container Health & Status**

```bash
docker compose ps
```

**Monitor Live Logs**

```bash
# View all swarm trade execution logs
docker compose logs -f consumer

# View market data feed logs
docker compose logs -f producer

# View dashboard UI logs
docker compose logs -f dashboard
```

**Restart Services**

```bash
docker compose restart
```

**Stop the System**

Stop containers while preserving database volume data:

```bash
docker compose down
```

Reset the entire database volume and start fresh:

```bash
docker compose down -v
```

---

## 🛡️ License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📸 Screenshots

<img width="1916" height="787" alt="EvoQuant-AI Dashboard Screenshot 1" src="https://github.com/user-attachments/assets/a59cf0bd-f515-4bcc-acba-4f8ae8edc69e" />

<img width="1522" height="652" alt="EvoQuant-AI Dashboard Screenshot 2" src="https://github.com/user-attachments/assets/d244ba83-6b72-4307-81c3-70ea9a1a72b3" />

<img width="1535" height="781" alt="EvoQuant-AI Dashboard Screenshot 3" src="https://github.com/user-attachments/assets/d4724d55-5348-4841-ba7d-cf577c922ebe" />
