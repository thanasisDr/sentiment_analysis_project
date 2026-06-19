# Enterprise Infrastructure Plan

A phased plan to evolve the sentiment-analysis service from a single-process
FastAPI app into a production, enterprise-grade deployment.

**Targets:** Docker Compose (local / demo, enterprise-shaped) → Azure Kubernetes
Service (AKS, production).

**Serving:** both an in-process predictor *and* a dedicated model server, kept
side by side behind one swappable abstraction.

This is a design document only. Each phase is intended to be implemented on its
own branch off `develop`, merged with `--no-ff` per the project convention.

---

## Context: what already exists

- **Fix #1:** model loads from the MLflow Model Registry via
  `models:/<name>@<alias>`, decoupled from any training run id.
- **Fix #2:** `/predict` is a synchronous `def` so FastAPI dispatches it to its
  threadpool; the event loop no longer blocks under concurrent load. A
  concurrent load test (`test/load_test.py`) proves p99 ≈ 44ms against the
  300ms budget at concurrency 50.

Fix #2 solves concurrency *within a single process*. Enterprise infra mostly
moves the concurrency problem outward — scale with processes and pods instead of
threads, decouple the model into its own scalable service, make deploys safe and
elastic, and wrap everything in resilience + observability so the p99 SLA holds
under real traffic.

---

## Guiding architecture decision (spans all phases)

To support both serving modes without forking the codebase, introduce a thin
abstraction in the app:

```
Predictor (interface)
  ├─ InProcessPredictor   → loads the sklearn model from the MLflow registry (today's behavior)
  └─ RemotePredictor      → calls the dedicated model server over gRPC/HTTP
```

Selected by env var (`PREDICTOR_BACKEND=inprocess|remote`). This single seam is
what lets Compose and AKS each run either mode, and lets you A/B the two.
Everything below leans on it.

---

## Phase 0 — Foundation & hygiene
**Branch:** `feat/prod-foundation` · folds in existing review items #4, #5

- Pin all deps consistently / add a lockfile (`pip-tools` or `uv`); resolve the
  mlflow 3.13 vs 2.18 mismatch between `requirements.txt` and
  `requirements_backend.txt`.
- Multi-stage `Dockerfile`, non-root user, `.dockerignore` (stop `COPY .`
  dragging in `mlruns/`, `.git`, data).
- Run under `gunicorn -k uvicorn.workers.UvicornWorker -w N` for process-level
  concurrency on top of the threadpool fix.
- Add `/health/live` + `/health/ready` endpoints (ready flips true only after
  the model loads) — the contract K8s probes need later.
- Structured JSON logging + graceful shutdown on SIGTERM (drain in-flight
  requests via the `lifespan` handler).

**Deliverable:** a hardened, config-driven image that runs identically
everywhere.

## Phase 1 — Remote MLflow + artifact store
**Branch:** `feat/remote-mlflow`

- Stand up an MLflow **tracking server** backed by **Azure Database for
  PostgreSQL** (backend store) + **Azure Blob Storage** (artifact store),
  replacing the local `sqlite:///mlflow.db`.
- App resolves `models:/<name>@champion` from the remote registry (Fix #1
  already uses this URI — just repoint `MLFLOW_TRACKING_URI`).
- Training pipeline pushes artifacts to Blob.

**Deliverable:** model store decoupled from the serving host; any
replica/container can load the same model.

## Phase 2 — Docker Compose environment
**Branch:** `feat/docker-compose` · first enterprise-shaped target

`docker-compose.yml` with profiles so either serving mode runs locally:

```
services:
  api            # gunicorn+uvicorn, PREDICTOR_BACKEND configurable
  model-server   # (profile: remote) dedicated server, see Phase 3
  mlflow         # tracking server
  postgres       # mlflow backend
  prometheus     # scrapes /metrics
  grafana        # dashboards + p99 panels
```

- `--profile inprocess` vs `--profile remote` toggles the two serving modes.
- Wire `test/load_test.py` to run against the compose stack to keep proving p99.

**Deliverable:** one `docker compose up` brings up a full mini-stack; reviewers
see the whole system locally.

## Phase 3 — Dedicated model server
**Branch:** `feat/model-server`

- Package the sklearn model behind **BentoML** (simplest for sklearn; Triton
  with the FIL/Python backend is the heavyweight alternative).
- Implement `RemotePredictor` to call it over gRPC/HTTP; enable **dynamic
  batching** (coalesces concurrent requests into one vectorized `predict` — a
  bigger throughput win than threading for CPU-bound models).
- Both backends now live; pick via `PREDICTOR_BACKEND`. Extend the load test to
  compare in-process vs remote p99/throughput.

**Deliverable:** an independently scalable inference service + a real batching
benchmark.

## Phase 4 — AKS infrastructure (IaC)
**Branch:** `feat/aks-infra`

- **Terraform** (or Bicep) provisioning: AKS cluster, **ACR** (registry),
  **Key Vault**, Blob, PostgreSQL, **Azure Monitor managed Prometheus + Azure
  Managed Grafana**.
- **Helm chart** for the app (and a sub-chart for the model server): Deployment,
  Service, HPA, ConfigMap, probes (reusing the Phase 0 health endpoints).
- Ingress via **AGIC** (Application Gateway Ingress Controller) or NGINX; TLS via
  **cert-manager**.
- Secrets via **Azure Key Vault + Secrets Store CSI driver** (no secrets in
  env/images).

**Deliverable:** a reproducible cluster + deployable charts; `helm install` runs
the same app on AKS.

## Phase 5 — Observability & resilience
**Branch:** `feat/observability`

- **OpenTelemetry** tracing across ingress → API → model server.
- Logs to **Azure Monitor / Log Analytics**; metrics to managed Prometheus.
- **SLO alerting** on p99 < 300ms (error-budget burn) + Grafana dashboards.
- Resilience: gateway timeouts/retries, **rate limiting / load shedding** (429
  over latency blowup), `PodDisruptionBudget`, autoscaling on a **custom latency
  metric via KEDA** (native AKS add-on) rather than CPU alone.

**Deliverable:** the p99 SLA is observed and enforced, not just measured once.

## Phase 6 — CI/CD + deployment safety
**Branch:** `feat/cicd` · folds in existing review item #7

- **GitHub Actions**: lint (black/isort) + tests + security scan (Trivy/Bandit)
  → build → push to **ACR** → deploy to AKS.
- **Canary / blue-green** rollout; gate promotion on live p99/error-rate.
- Model promotion = flip the MLflow `champion` alias; optional **canary
  serving** of `challenger` to a traffic slice for an online A/B.

**Deliverable:** a safe, automated path from PR to production.

## Phase 7 (stretch) — ML-specific monitoring
**Branch:** `feat/ml-monitoring`

- Data/prediction **drift** detection (Evidently), prediction-distribution
  dashboards, feedback capture. The failure mode unique to ML systems is silent
  model decay.

---

## Suggested sequencing

Phases 0 → 1 → 2 yield a complete, locally-runnable enterprise-shaped system
quickly. Phase 3 adds the second serving mode. Phases 4 → 6 are the cloud/AKS
lift. Phase 7 is optional polish.

| Phase | Branch | Target / theme |
|-------|--------|----------------|
| 0 | `feat/prod-foundation` | Hygiene, image, health, gunicorn |
| 1 | `feat/remote-mlflow` | Remote MLflow + Blob + Postgres |
| 2 | `feat/docker-compose` | Local enterprise-shaped stack |
| 3 | `feat/model-server` | Dedicated model server (BentoML/Triton) |
| 4 | `feat/aks-infra` | AKS + ACR + IaC + Helm |
| 5 | `feat/observability` | Tracing, SLO alerts, resilience, KEDA |
| 6 | `feat/cicd` | GitHub Actions → ACR → AKS, canary |
| 7 | `feat/ml-monitoring` | Drift / online A/B (stretch) |
