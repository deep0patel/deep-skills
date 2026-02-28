---
name: agent-forge
description: Use when building AI agents, agent systems, or asking how to implement LLM routing, MCP integration, agent memory, multi-agent orchestration, or production deployment of AI agents
---

# Agent Forge

Build exactly the agent the user needs — no more, no less.

## Step 1: Gather Requirements

**Before writing any code**, ask the user these questions. You need all answers before proceeding.

### Project
- What should your agent do? (use case / problem to solve)
- Preferred language and runtime? (TypeScript/Node · Python · Go · Rust · other)

### Features (ask as a checklist — each is optional)

| Feature | Options |
|---------|---------|
| **Memory** | None · Session-only (in-memory) · Persistent (choose backend) · With vector/semantic search |
| **Memory backend** | SQLite (local) · PostgreSQL + pgvector (production) · Redis (distributed) |
| **Tools/MCP** | None · Use existing MCP servers · Build custom MCP server · Both |
| **Which MCP servers** | filesystem · github · brave-search · custom (describe) |
| **Multi-agent** | Single agent · Swarm with coordinator (Queen + workers) · Task queue for async work |
| **LLM routing** | Single provider · Multi-provider (cost-optimized · performance · balanced) |
| **Deployment** | Local only · Docker · Serverless (Lambda/Functions) · Kubernetes · Edge |
| **Production hardening** | None · Retries + circuit breakers · Full observability (logs/metrics/traces) · Secrets management |
| **Testing** | Skip · Unit tests · Benchmarks (SWE-Bench, HumanEval) · Both |

### Constraints
- Max cost per run? (e.g., $0.10 · $1.00 · unlimited)
- Latency target? (100ms · 1s · 10s · none)
- Existing infrastructure to integrate?

---

## Step 2: Route to References

Based on answers, read **only** the relevant reference files before generating code.

| User needs | Read this file |
|-----------|---------------|
| Core agent loop (always required) | `phase-1-foundation.md` |
| Any MCP tools or custom tool server | `phase-2-mcp.md` |
| Any form of memory (even session-only) | `phase-3-memory.md` |
| Multi-agent, swarm, or task queue | `phase-4-multi-agent.md` |
| Retries, circuit breakers, observability, or secrets | `phase-5-production.md` |
| Docker, Kubernetes, Terraform, or cloud deployment | `phase-6-deployment.md` |
| Tests, benchmarks, or validation checklists | `phase-7-testing.md` |

**Do not read files for features the user did not request.**

---

## Step 3: Generate

Combine only the pieces relevant to the user's requirements.

**Scoping rules:**
- Simple agent (no memory, no tools) → Phase 1 only
- Agent with tools → Phase 1 + Phase 2
- Agent with memory → Phase 1 + Phase 3 (only the backend they chose)
- Full production agent → Phases 1–5 + relevant deployment

**Generation checklist:**
- [ ] Project structure scoped to chosen features (omit unused directories)
- [ ] Config file includes only configured providers/backends
- [ ] Only chosen memory backend implemented (not all three)
- [ ] MCP servers listed only if requested
- [ ] Multi-agent code only if swarm was requested
- [ ] Deployment config only for chosen target
- [ ] README explains quick start for this specific configuration

**Anti-patterns to avoid:**
- Generating all memory backends when user chose one
- Adding multi-agent scaffolding when single agent was requested
- Including Kubernetes config when user only needs Docker
- Adding observability stack when user said "skip production hardening"
