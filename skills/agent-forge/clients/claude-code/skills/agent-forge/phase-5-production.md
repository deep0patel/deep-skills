# Phase 5: Production Hardening

**Goal:** Add enterprise-grade reliability and observability.

## 5.1 Resilience — Retry, Circuit Breaker, Fallback

```typescript
class ResilienceManager {
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();

  async withRetry<T>(fn: () => Promise<T>, options: RetryOptions = {}): Promise<T> {
    const { maxRetries = 3, backoff = "exponential", initialDelay = 1000, maxDelay = 30000, timeout = 60000 } = options;
    let lastError: Error;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.withTimeout(fn, timeout);
      } catch (error) {
        lastError = error;
        if (attempt < maxRetries) {
          let delay = backoff === "exponential" ? initialDelay * Math.pow(2, attempt)
                    : backoff === "linear"      ? initialDelay * (attempt + 1)
                    : initialDelay;
          delay = Math.min(delay * (0.5 + Math.random() * 0.5), maxDelay); // jitter
          await this.sleep(delay);
        }
      }
    }
    throw lastError!;
  }

  async withCircuitBreaker<T>(key: string, fn: () => Promise<T>, options: CircuitBreakerOptions = {}): Promise<T> {
    if (!this.circuitBreakers.has(key)) this.circuitBreakers.set(key, new CircuitBreaker(options));
    return await this.circuitBreakers.get(key)!.execute(fn);
  }

  async withFallback<T>(primary: () => Promise<T>, fallback: () => Promise<T>): Promise<T> {
    try { return await primary(); }
    catch (error) { console.warn("Primary failed, using fallback:", error); return await fallback(); }
  }
}

class CircuitBreaker {
  private state: "closed" | "open" | "half-open" = "closed";
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;

  constructor(private options: CircuitBreakerOptions) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === "open") {
      if (Date.now() - this.lastFailureTime > this.options.resetTimeout) this.state = "half-open";
      else throw new Error("Circuit breaker is open");
    }
    try {
      const result = await fn();
      this.failureCount = 0;
      if (this.state === "half-open" && ++this.successCount >= this.options.successThreshold) {
        this.state = "closed"; this.successCount = 0;
      }
      return result;
    } catch (error) {
      this.lastFailureTime = Date.now();
      if (++this.failureCount >= this.options.failureThreshold) this.state = "open";
      throw error;
    }
  }
}
```

## 5.2 Observability — Logging, Metrics, Tracing

```typescript
class ObservabilityManager {
  private logger: Logger;
  private metrics: MetricsCollector;
  private tracer: Tracer;

  constructor(config: ObservabilityConfig) {
    this.logger  = this.createLogger(config.logging);
    this.metrics = this.createMetrics(config.metrics);
    this.tracer  = this.createTracer(config.tracing);
  }

  private createLogger(config: LoggingConfig): Logger {
    const transports = config.outputs.map(output => {
      switch (output.type) {
        case "console":    return new ConsoleTransport(output);
        case "file":       return new FileTransport(output);
        case "cloudwatch": return new CloudWatchTransport(output);
        case "datadog":    return new DataDogTransport(output);
      }
    });
    return new Logger({ level: config.level, format: config.format, transports });
  }

  private createMetrics(config: MetricsConfig): MetricsCollector {
    switch (config.backend) {
      case "prometheus": return new PrometheusCollector(config);
      case "cloudwatch": return new CloudWatchMetrics(config);
      case "datadog":    return new DataDogMetrics(config);
      default:           return new InMemoryMetrics();
    }
  }

  private createTracer(config: TracingConfig): Tracer {
    if (!config.enabled) return new NoOpTracer();
    switch (config.backend) {
      case "jaeger":  return new JaegerTracer(config);
      case "zipkin":  return new ZipkinTracer(config);
      case "xray":    return new XRayTracer(config);
      default:        return new OpenTelemetryTracer(config);
    }
  }

  log(level: string, message: string, context?: any): void {
    this.logger.log(level, message, { ...context, timestamp: new Date().toISOString(), service: "universal-agent" });
  }
  recordMetric(name: string, value: number, tags?: Record<string, string>): void {
    this.metrics.record({ name, value, tags, timestamp: Date.now() });
  }
  startSpan(name: string, attributes?: any): Span { return this.tracer.startSpan(name, attributes); }
}

// Instrumented agent wrapper
class ObservableAgent extends Agent {
  async run(message: string): Promise<string> {
    const span = this.observability.startSpan("agent.run", { message: message.substring(0, 100) });
    const startTime = Date.now();
    try {
      this.observability.log("info", "Starting agent execution", { messageLength: message.length });
      const result = await super.run(message);
      this.observability.recordMetric("agent.execution.duration", Date.now() - startTime, { success: "true" });
      span.setStatus("ok");
      return result;
    } catch (error) {
      this.observability.log("error", "Agent execution failed", { error: error.message });
      this.observability.recordMetric("agent.execution.errors", 1);
      span.setStatus("error"); span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }
}
```

**Key metrics to track:**
- `agent.execution.duration` — time per request
- `agent.execution.tokens` — token usage
- `agent.tool.calls` — tool invocation count
- `agent.memory.retrieval.time` — memory query latency

## 5.3 Security & Secrets Management

```typescript
class SecurityManager {
  private secretsProvider: SecretsProvider;
  private encryption: EncryptionService;
  private secretsCache: Map<string, { value: string; expiresAt: number }> = new Map();

  constructor(config: SecurityConfig) {
    this.encryption = new EncryptionService(config.encryptionKey);
    switch (config.provider) {
      case "env":                   this.secretsProvider = new EnvironmentSecretsProvider(); break;
      case "aws-secrets-manager":   this.secretsProvider = new AWSSecretsManager(config.region); break;
      case "gcp-secret-manager":    this.secretsProvider = new GCPSecretManager(config.projectId); break;
      case "azure-key-vault":       this.secretsProvider = new AzureKeyVault(config.vaultUrl); break;
      case "hashicorp-vault":       this.secretsProvider = new HashiCorpVault(config.vaultAddr); break;
      default:                      this.secretsProvider = new EnvironmentSecretsProvider();
    }
  }

  async getSecret(key: string): Promise<string> {
    const cached = this.secretsCache.get(key);
    if (cached && cached.expiresAt > Date.now()) return cached.value;
    const value = await this.secretsProvider.get(key);
    this.secretsCache.set(key, { value, expiresAt: Date.now() + 3_600_000 });
    return value;
  }

  encrypt(plaintext: string): string { return this.encryption.encrypt(plaintext); }
  decrypt(ciphertext: string): string { return this.encryption.decrypt(ciphertext); }

  sanitizeInput(input: string, type: "sql" | "shell" | "path"): string {
    switch (type) {
      case "sql":   return input.replace(/['";\\]/g, "");
      case "shell": return input.replace(/[;&|`$()]/g, "");
      case "path":  return input.replace(/\.\./g, "").replace(/[<>:"|?*]/g, "");
    }
  }

  async checkRateLimit(key: string, limit: number, window: number): Promise<boolean> {
    const count = await this.rateLimiter.increment(key, window);
    return count <= limit;
  }
}
```

**Security checklist:**
- Never commit API keys — use secrets manager
- Sanitize all user input at system boundaries
- Restrict file access to allowed directories
- Implement rate limiting to prevent abuse
- Audit log all operations
