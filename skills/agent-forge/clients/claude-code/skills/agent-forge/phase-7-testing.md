# Phase 7: Testing & Benchmarking

**Goal:** Validate performance against industry standards.

## Benchmark Suite

```typescript
class BenchmarkSuite {
  private agent: Agent;

  async runAll(): Promise<BenchmarkResults> {
    return {
      sweBench:         await this.runSWEBench(),
      humanEval:        await this.runHumanEval(),
      customMetrics:    await this.runCustomMetrics(),
      performanceTests: await this.runPerformanceTests()
    };
  }

  async runSWEBench(): Promise<SWEBenchResult> {
    const testCases = await loadSWEBenchData();
    let resolved = 0;

    for (const testCase of testCases) {
      const result = await this.agent.run(testCase.problem);
      if (await this.validateSolution(result, testCase.tests)) resolved++;
    }

    return {
      total: testCases.length,
      resolved,
      percentage: (resolved / testCases.length) * 100,
      target: 70 // Industry competitive: 70%+
    };
  }

  async runCustomMetrics(): Promise<CustomMetrics> {
    return {
      tokenEfficiency:        await this.measureTokenEfficiency(),
      parallelizationSpeedup: await this.measureParallelization(),
      costSavings:            await this.measureCostOptimization(),
      memoryRetrieval:        await this.measureMemorySpeed()
    };
  }

  private async measureTokenEfficiency(): Promise<number> {
    const baseline  = await this.runWithoutOptimizations();
    const optimized = await this.runWithOptimizations();
    return ((baseline.tokens - optimized.tokens) / baseline.tokens) * 100;
  }

  private async measureMemorySpeed(): Promise<number> {
    const queries = generateTestQueries(100);
    const start = Date.now();
    for (const query of queries) await this.agent.memory.search(query);
    return (Date.now() - start) / queries.length; // target: <100ms avg
  }
}
```

## Deliverables Checklist

Generate these artifacts for each complete agent system:

### Codebase
```
universal-agent/
├── src/                    # All source code (phases 1-5)
├── tests/                  # Unit + integration tests (>80% coverage)
├── config/                 # Configuration templates
├── terraform/              # IaC for AWS, GCP, Azure
├── k8s/                    # Kubernetes manifests
├── docker/                 # Dockerfiles
├── .github/workflows/      # CI/CD pipelines
├── examples/               # Usage examples
└── benchmarks/             # Performance tests
```

### Documentation
- `README.md` — Quick start guide
- `ARCHITECTURE.md` — System design
- `DEPLOYMENT.md` — Deployment options
- `CONFIGURATION.md` — All config options
- `EXTENDING.md` — How to add tools, providers, backends
- `TROUBLESHOOTING.md` — Common issues
- `ADRs/` — Architecture decision records

## Final Validation Checklist

### Core Functionality
- [ ] Agent loop executes without hitting max iterations on normal tasks
- [ ] LLM routing works for all configured providers
- [ ] MCP client connects to all configured servers
- [ ] Tools execute and return expected shapes
- [ ] Memory stores, retrieves, and returns similarity scores
- [ ] Error handling catches failures and retries appropriately

### Production Readiness
- [ ] Logs output structured JSON
- [ ] Metrics are collected and exported
- [ ] Secrets are externalized (no hardcoded keys)
- [ ] Tests pass with >80% coverage
- [ ] Security scan shows no critical issues
- [ ] Memory retrieval P95 < 100ms
- [ ] Tool success rate >95% under load

### Deployment
- [ ] Docker image builds and runs
- [ ] Local docker-compose starts all services cleanly
- [ ] IaC templates are valid (`terraform validate` / `kubectl dry-run`)
- [ ] CI/CD pipeline executes end-to-end
- [ ] Deployment to target platform succeeds with smoke test passing

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "Circuit breaker is open" | Too many LLM request failures | Check API keys and provider status |
| "Memory retrieval timeout" | Vector search too slow | Add DB indexes, reduce limit, upgrade backend |
| "MCP server connection failed" | Server process crashed | Check server logs, verify command/args in config |
| "Max iterations reached" | Agent stuck in tool loop | Add iteration logging, check tool responses for errors |
| High token usage | No memory retrieval | Enable episodic memory to avoid re-summarizing context |
