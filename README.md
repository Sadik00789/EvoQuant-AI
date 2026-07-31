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

- **Language & Runtime:** Python 3.14-slim
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
│   └── test_swarm.py           # Unit & integration test suite
├── .env.example                 # Environment variables template
├── .gitignore                   # Git exclusions (blocks .env and cache)
├── Dockerfile                   # Multi-stage container build definition
├── docker-compose.yml           # Orchestration spec for all 5 services
├── requirements.txt             # Python dependency manifests
├── data_producer.py             # Market WebSocket stream ingestion
├── swarm_consumer.py            # Swarm trade decision, Adversarial Debate Loop & logging worker
├── engine.py                    # Cross-asset portfolio manager, short/cover execution, dividend rules & Alpaca bridge & DB connector
├── evolution_engine.py          # Darwinian strategy evolution & tournament logic
├── risk_engine.py               # Convex risk parity, downside semi-variance, short-side exposure & macro guard
├── sentiment_agent.py           # RSS/News RAG sentiment evaluation agent
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
# Database Credentials
POSTGRES_USER=evoquant
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=evoquant_db

# Market Data API & Alpaca Paper Trading Bridge
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# LLM Swarm API Keys (Multi-Provider Llama 3.3 70B Fallback)
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
SAMBANOVA_API_KEY=your_sambanova_api_key
GITHUB_TOKEN=your_github_token
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
