# agent-forge

Expert AI agent architect that designs and implements production-grade, vendor-agnostic AI agent systems across any language, cloud, or platform.

## What It Does

- Gathers your tech stack, deployment target, and constraints before generating anything
- Implements a 7-phase build plan: Foundation → MCP → Memory → Multi-Agent → Production → Deployment → Testing
- Generates a full project structure with source code, tests, IaC, and CI/CD
- Routes LLM requests intelligently across providers to optimize cost and performance
- Adds enterprise-grade resilience: retries, circuit breakers, observability, secrets management
- Targets industry benchmarks: 70%+ SWE-Bench, 90%+ HumanEval, <$0.10/task

## Installation

See the [main README](../../README.md) for installation instructions per client.

## Usage

### Claude Code

After installing the plugin, the skill is automatically available. Start a new agent project:

```
Build me a production-ready AI agent system
```

Or with configuration:

```
Build an AI agent for code generation using TypeScript, deployed on AWS Lambda,
with Anthropic as primary LLM and SQLite for local memory
```

### Cursor / Copilot / Windsurf / Cline

After adding the rule file, the AI will follow the Agent Forge framework when you ask it to build AI agents.

## The 7-Phase Framework

| Phase | Goal |
|-------|------|
| 1 — Foundation | Core agent loop, LLM router, config system |
| 2 — MCP Integration | Full MCP client/server, universal tool registry |
| 3 — Memory | Three-layer memory: episodic, reflexion, skill library |
| 4 — Multi-Agent | Swarm orchestration with queen-worker pattern |
| 5 — Production | Retries, circuit breakers, observability, security |
| 6 — Deployment | Lambda, Docker, Kubernetes, Cloudflare Workers |
| 7 — Testing | SWE-Bench, HumanEval, performance benchmarks |

## Success Criteria

| Metric | Target |
|--------|--------|
| SWE-Bench Score | 70%+ |
| HumanEval Pass@1 | 90%+ |
| Token Reduction | 30%+ |
| Memory Retrieval | <100ms |
| Tool Success Rate | 95%+ |
| Cost per Task | <$0.10 |
