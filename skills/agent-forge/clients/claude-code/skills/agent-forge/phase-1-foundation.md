# Phase 1: Foundation

**Goal:** Build the minimal viable agent loop.

## 1.1 Project Structure

```
agent-system/
├── src/
│   ├── core/
│   │   ├── agent.{ts,py,go}           # Main agent loop
│   │   ├── config.{ts,py,go}          # Configuration management
│   │   └── types.{ts,py,go}           # Core type definitions
│   ├── llm/
│   │   ├── router.{ts,py,go}          # Multi-provider routing
│   │   └── providers/
│   │       ├── anthropic.{ts,py,go}
│   │       ├── openai.{ts,py,go}
│   │       ├── ollama.{ts,py,go}
│   │       └── base.{ts,py,go}        # Provider interface
│   ├── mcp/
│   │   ├── client.{ts,py,go}
│   │   ├── server.{ts,py,go}
│   │   └── protocol.{ts,py,go}
│   ├── tools/
│   │   ├── registry.{ts,py,go}
│   │   └── builtin/
│   │       ├── file.{ts,py,go}
│   │       ├── code.{ts,py,go}
│   │       └── web.{ts,py,go}
│   └── utils/
│       ├── logger.{ts,py,go}
│       └── errors.{ts,py,go}
├── tests/
├── config/
│   └── agent.config.yaml
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

## 1.2 Core Agent Loop

```typescript
class Agent {
  private llmRouter: LLMRouter;
  private mcpClient: MCPClient;
  private memory: MemorySystem;

  async run(userMessage: string): Promise<string> {
    const conversation: Message[] = [];
    const maxIterations = 25;
    let iteration = 0;

    conversation.push({ role: "user", content: userMessage });

    while (iteration++ < maxIterations) {
      const availableTools = await this.mcpClient.listTools();

      const llmResponse = await this.llmRouter.complete({
        messages: conversation,
        tools: availableTools,
        temperature: 0.7
      });

      conversation.push(llmResponse);

      if (llmResponse.toolCalls?.length > 0) {
        const toolResults = await this.executeTools(llmResponse.toolCalls);
        conversation.push(...toolResults);
        continue;
      }

      if (llmResponse.finishReason === "stop") {
        await this.storeSession(conversation);
        return llmResponse.content;
      }
    }

    throw new Error("Max iterations reached");
  }

  private async executeTools(toolCalls: ToolCall[]): Promise<Message[]> {
    const results = await Promise.allSettled(
      toolCalls.map(async (call) => {
        try {
          const result = await this.mcpClient.callTool(call.name, call.arguments);
          return { role: "tool", toolCallId: call.id, content: JSON.stringify(result) };
        } catch (error) {
          return { role: "tool", toolCallId: call.id, content: `Error: ${error.message}`, isError: true };
        }
      })
    );
    return results.map(r => r.status === "fulfilled" ? r.value : r.reason);
  }
}
```

## 1.3 LLM Router

```typescript
interface LLMProvider {
  name: string;
  complete(params: CompletionParams): Promise<CompletionResponse>;
  estimateCost(params: CompletionParams): number;
  capabilities: {
    maxTokens: number;
    supportsTools: boolean;
    supportsVision: boolean;
    contextWindow: number;
  };
}

class LLMRouter {
  private providers: Map<string, LLMProvider>;
  private strategy: "cost" | "performance" | "balanced";

  async complete(params: CompletionParams): Promise<CompletionResponse> {
    const provider = this.selectProvider(params);
    return await this.withRetry(() => provider.complete(params), { maxRetries: 3, backoff: "exponential" });
  }

  private selectProvider(params: CompletionParams): LLMProvider {
    const complexity = this.analyzeComplexity(params);

    switch (this.strategy) {
      case "cost":      return this.getCheapestCapableProvider(complexity);
      case "performance": return this.getMostCapableProvider();
      case "balanced":
        if (complexity === "high")   return this.getProvider("claude-sonnet-4");
        if (complexity === "medium") return this.getProvider("gpt-4o");
        return this.getProvider("claude-haiku-4");
    }
  }

  private analyzeComplexity(params: CompletionParams): "low" | "medium" | "high" {
    const contentLength = params.messages.map(m => m.content.length).reduce((a, b) => a + b, 0);
    const messageCount = params.messages.length;
    const hasTools = params.tools?.length > 0;

    if (contentLength > 10000 || messageCount > 10 || hasTools) return "high";
    if (contentLength > 2000  || messageCount > 3)              return "medium";
    return "low";
  }
}
```

## 1.4 Configuration File

```yaml
# config/agent.config.yaml
agent:
  name: "universal-agent"
  version: "1.0.0"
  max_iterations: 25
  timeout_seconds: 300

llm:
  strategy: "balanced"   # cost | performance | balanced
  providers:
    anthropic:
      enabled: true
      api_key: "${ANTHROPIC_API_KEY}"
      models:
        primary: "claude-sonnet-4-20250514"
        fallback: "claude-haiku-4-20251001"
      max_tokens: 4096
    openai:
      enabled: true
      api_key: "${OPENAI_API_KEY}"
      models:
        primary: "gpt-4o"
        fallback: "gpt-4o-mini"
      max_tokens: 4096
    ollama:
      enabled: false
      base_url: "http://localhost:11434"
      models:
        primary: "qwen2.5-coder:14b"

mcp:
  servers:
    - name: "filesystem"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    - name: "github"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_TOKEN: "${GITHUB_TOKEN}"
    - name: "brave-search"
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-brave-search"]
      env:
        BRAVE_API_KEY: "${BRAVE_API_KEY}"

memory:
  backend: "sqlite"
  path: "./data/agent-memory.db"
  vector_dimensions: 1536

logging:
  level: "info"
  format: "json"
  outputs:
    - type: "console"
    - type: "file"
      path: "./logs/agent.log"
```
