# Universal Agent System Master Prompt (2025)
## The Complete Blueprint for Production-Grade AI Agents

You are an expert AI agent architect tasked with designing and implementing a **production-ready, vendor-agnostic agent system** that can run anywhere, connect to anything, and scale infinitely.

---

## INITIAL CONFIGURATION

Before starting implementation, gather these minimal inputs:

```yaml
# User Configuration Template
project:
  name: "my-agent-system"
  use_case: "code generation | data analysis | automation | custom"
  
tech_stack:
  language: "typescript | python | go | rust | java"
  runtime: "node | deno | python3.11+ | go1.21+ | rustc"
  
deployment:
  target: "local | serverless | container | kubernetes | edge"
  cloud: "aws | gcp | azure | cloudflare | none"
  infrastructure: "terraform | pulumi | cdk | docker-compose | manual"
  
llm_strategy:
  primary_provider: "anthropic | openai | google | azure | ollama"
  fallback_providers: ["provider1", "provider2"]
  routing: "cost-optimized | performance | balanced"
  
memory:
  backend: "sqlite | postgres | redis | qdrant | weaviate"
  vector_db: "local-embeddings | openai | cohere | none"
  
constraints:
  max_cost_per_run: "$0.10 | $1.00 | unlimited"
  latency_target: "100ms | 1s | 10s | none"
  existing_infrastructure: "describe any existing systems to integrate"
```

---

## ARCHITECTURE OVERVIEW

### Core Philosophy: "Water-Like Adaptability"

Your agent system should be like water - taking the shape of whatever container (infrastructure) it's poured into while maintaining its essential properties (capabilities).

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│          (CLI, API, Web UI, IDE Plugin)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   AGENT ORCHESTRATOR                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Task       │  │   Memory     │  │   Learning   │ │
│  │   Router     │  │   System     │  │   Engine     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                    MCP CLIENT LAYER                      │
│  (Model Context Protocol - Universal Tool Interface)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  EXECUTION ENGINES                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   LLM    │ │  Tools   │ │  Swarm   │ │ External │  │
│  │  Router  │ │ Executor │ │  Coord   │ │   APIs   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## PHASE 1: FOUNDATION

### Goal: Build the minimal viable agent loop

#### 1.1 Project Structure (Universal Template)

```
agent-system/
├── src/
│   ├── core/
│   │   ├── agent.{ts,py,go}           # Main agent loop
│   │   ├── config.{ts,py,go}          # Configuration management
│   │   └── types.{ts,py,go}           # Core type definitions
│   ├── llm/
│   │   ├── router.{ts,py,go}          # Multi-provider routing
│   │   ├── providers/
│   │   │   ├── anthropic.{ts,py,go}
│   │   │   ├── openai.{ts,py,go}
│   │   │   ├── ollama.{ts,py,go}
│   │   │   └── base.{ts,py,go}        # Provider interface
│   ├── mcp/
│   │   ├── client.{ts,py,go}          # MCP client implementation
│   │   ├── server.{ts,py,go}          # MCP server (for exposing tools)
│   │   └── protocol.{ts,py,go}        # MCP protocol types
│   ├── tools/
│   │   ├── registry.{ts,py,go}        # Tool registration system
│   │   └── builtin/                    # Built-in tools
│   │       ├── file.{ts,py,go}
│   │       ├── code.{ts,py,go}
│   │       └── web.{ts,py,go}
│   └── utils/
│       ├── logger.{ts,py,go}
│       └── errors.{ts,py,go}
├── tests/
├── docs/
├── config/
│   └── agent.config.yaml
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

#### 1.2 Core Agent Loop (Language-Agnostic Pseudocode)

```typescript
// The heart of your agent - keep it simple initially
class Agent {
  private llmRouter: LLMRouter;
  private mcpClient: MCPClient;
  private memory: MemorySystem;
  
  async run(userMessage: string): Promise<string> {
    const conversation: Message[] = [];
    const maxIterations = 25; // Prevent infinite loops
    let iteration = 0;
    
    // Add initial user message
    conversation.push({
      role: "user",
      content: userMessage
    });
    
    while (iteration++ < maxIterations) {
      // 1. Get available tools from MCP servers
      const availableTools = await this.mcpClient.listTools();
      
      // 2. Route to appropriate LLM based on task complexity
      const llmResponse = await this.llmRouter.complete({
        messages: conversation,
        tools: availableTools,
        temperature: 0.7
      });
      
      conversation.push(llmResponse);
      
      // 3. Handle tool calls if present
      if (llmResponse.toolCalls?.length > 0) {
        const toolResults = await this.executeTools(llmResponse.toolCalls);
        conversation.push(...toolResults);
        continue; // Loop back for next LLM turn
      }
      
      // 4. If no tool calls, task is complete
      if (llmResponse.finishReason === "stop") {
        await this.storeSession(conversation); // Learn from session
        return llmResponse.content;
      }
    }
    
    throw new Error("Max iterations reached");
  }
  
  private async executeTools(toolCalls: ToolCall[]): Promise<Message[]> {
    // Parallel execution when safe
    const results = await Promise.allSettled(
      toolCalls.map(async (call) => {
        try {
          const result = await this.mcpClient.callTool(call.name, call.arguments);
          return {
            role: "tool",
            toolCallId: call.id,
            content: JSON.stringify(result)
          };
        } catch (error) {
          return {
            role: "tool",
            toolCallId: call.id,
            content: `Error: ${error.message}`,
            isError: true
          };
        }
      })
    );
    
    return results.map(r => r.status === "fulfilled" ? r.value : r.reason);
  }
}
```

#### 1.3 LLM Router (Multi-Provider Strategy)

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
    
    return await this.withRetry(
      () => provider.complete(params),
      { maxRetries: 3, backoff: "exponential" }
    );
  }
  
  private selectProvider(params: CompletionParams): LLMProvider {
    const complexity = this.analyzeComplexity(params);
    
    switch (this.strategy) {
      case "cost":
        // Use cheapest provider that meets requirements
        return this.getCheapestCapableProvider(complexity);
        
      case "performance":
        // Use most capable provider regardless of cost
        return this.getMostCapableProvider();
        
      case "balanced":
        // Balance cost vs capability
        if (complexity === "high") {
          return this.getProvider("claude-sonnet-4");
        } else if (complexity === "medium") {
          return this.getProvider("gpt-4o");
        } else {
          return this.getProvider("claude-haiku-4");
        }
    }
  }
  
  private analyzeComplexity(params: CompletionParams): "low" | "medium" | "high" {
    // Heuristics for task complexity
    const messageCount = params.messages.length;
    const hasTools = params.tools && params.tools.length > 0;
    const contentLength = params.messages
      .map(m => m.content.length)
      .reduce((a, b) => a + b, 0);
    
    if (contentLength > 10000 || messageCount > 10 || hasTools) {
      return "high";
    } else if (contentLength > 2000 || messageCount > 3) {
      return "medium";
    }
    return "low";
  }
}
```

#### 1.4 Configuration System (Provider-Agnostic)

```yaml
# config/agent.config.yaml
agent:
  name: "universal-agent"
  version: "1.0.0"
  max_iterations: 25
  timeout_seconds: 300

llm:
  strategy: "balanced"  # cost | performance | balanced
  
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
  backend: "sqlite"  # sqlite | postgres | redis
  path: "./data/agent-memory.db"
  vector_dimensions: 1536  # OpenAI ada-002 / Sentence transformers

logging:
  level: "info"  # debug | info | warn | error
  format: "json"  # json | pretty
  outputs:
    - type: "console"
    - type: "file"
      path: "./logs/agent.log"
```

---

## PHASE 2: MCP INTEGRATION

### Goal: Implement full MCP client/server capabilities

#### 2.1 MCP Client Implementation

```typescript
/**
 * MCP Client - Connects to MCP servers and manages tool execution
 * 
 * Reference: https://modelcontextprotocol.io/docs/concepts/architecture
 */
class MCPClient {
  private servers: Map<string, MCPServerConnection>;
  
  async initialize(config: MCPServerConfig[]): Promise<void> {
    for (const serverConfig of config) {
      const connection = await this.connectToServer(serverConfig);
      this.servers.set(serverConfig.name, connection);
    }
  }
  
  private async connectToServer(config: MCPServerConfig): Promise<MCPServerConnection> {
    // Spawn MCP server process
    const process = spawn(config.command, config.args, {
      env: { ...process.env, ...config.env },
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Create JSON-RPC transport
    const transport = new StdioTransport(process.stdin, process.stdout);
    
    // Initialize connection
    const client = new Client({
      name: "universal-agent",
      version: "1.0.0"
    }, {
      capabilities: {
        tools: {},
        prompts: {},
        resources: {}
      }
    });
    
    await client.connect(transport);
    
    // Discover server capabilities
    const capabilities = await client.listTools();
    
    return {
      process,
      client,
      capabilities,
      name: config.name
    };
  }
  
  async listTools(): Promise<Tool[]> {
    const allTools: Tool[] = [];
    
    for (const [serverName, connection] of this.servers) {
      const tools = await connection.client.listTools();
      
      // Namespace tools by server
      allTools.push(...tools.tools.map(tool => ({
        ...tool,
        name: `${serverName}/${tool.name}`,
        _server: serverName
      })));
    }
    
    return allTools;
  }
  
  async callTool(toolName: string, args: Record<string, unknown>): Promise<unknown> {
    // Parse server namespace
    const [serverName, actualToolName] = toolName.includes('/') 
      ? toolName.split('/', 2)
      : [this.servers.keys().next().value, toolName];
    
    const connection = this.servers.get(serverName);
    if (!connection) {
      throw new Error(`MCP server '${serverName}' not found`);
    }
    
    // Call tool via MCP protocol
    const result = await connection.client.callTool({
      name: actualToolName,
      arguments: args
    });
    
    return result.content;
  }
  
  async cleanup(): Promise<void> {
    for (const connection of this.servers.values()) {
      await connection.client.close();
      connection.process.kill();
    }
  }
}
```

#### 2.2 MCP Server (Expose Your Agent's Capabilities)

```typescript
/**
 * MCP Server - Expose your agent as an MCP server for other tools
 */
class MCPServer {
  private server: Server;
  private tools: Map<string, MCPTool>;
  
  constructor() {
    this.server = new Server({
      name: "universal-agent",
      version: "1.0.0"
    }, {
      capabilities: {
        tools: {}
      }
    });
    
    this.tools = new Map();
    this.registerDefaultTools();
  }
  
  private registerDefaultTools(): void {
    // Register built-in tools
    this.registerTool({
      name: "execute_code",
      description: "Execute code in a sandboxed environment",
      inputSchema: {
        type: "object",
        properties: {
          language: { type: "string", enum: ["python", "javascript", "bash"] },
          code: { type: "string" },
          timeout: { type: "number", default: 30 }
        },
        required: ["language", "code"]
      },
      handler: async (args) => {
        return await this.executeCode(args);
      }
    });
    
    this.registerTool({
      name: "analyze_repository",
      description: "Analyze a code repository structure",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          depth: { type: "number", default: 3 }
        },
        required: ["path"]
      },
      handler: async (args) => {
        return await this.analyzeRepo(args);
      }
    });
  }
  
  registerTool(tool: MCPToolDefinition): void {
    this.tools.set(tool.name, tool);
  }
  
  async start(transport: Transport): Promise<void> {
    // Handle tool list requests
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: Array.from(this.tools.values()).map(t => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema
      }))
    }));
    
    // Handle tool call requests
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const tool = this.tools.get(request.params.name);
      if (!tool) {
        throw new Error(`Tool not found: ${request.params.name}`);
      }
      
      const result = await tool.handler(request.params.arguments);
      
      return {
        content: [{
          type: "text",
          text: JSON.stringify(result, null, 2)
        }]
      };
    });
    
    await this.server.connect(transport);
  }
}
```

#### 2.3 Tool Registry (Universal Tool Management)

```typescript
/**
 * Universal tool registry that works with MCP and non-MCP tools
 */
class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  private mcpClient: MCPClient;
  
  async initialize(mcpClient: MCPClient): Promise<void> {
    this.mcpClient = mcpClient;
    
    // Register built-in tools
    this.registerBuiltinTools();
    
    // Sync MCP tools
    await this.syncMCPTools();
  }
  
  private registerBuiltinTools(): void {
    // File operations
    this.register({
      name: "read_file",
      description: "Read contents of a file",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string", description: "File path to read" }
        },
        required: ["path"]
      },
      execute: async (args) => {
        const content = await fs.readFile(args.path, 'utf-8');
        return { content, size: content.length };
      }
    });
    
    this.register({
      name: "write_file",
      description: "Write content to a file",
      inputSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          content: { type: "string" }
        },
        required: ["path", "content"]
      },
      execute: async (args) => {
        await fs.writeFile(args.path, args.content, 'utf-8');
        return { success: true, path: args.path };
      }
    });
    
    // Add 20+ more built-in tools: list_files, execute_bash, 
    // run_tests, git_commit, search_code, etc.
  }
  
  private async syncMCPTools(): Promise<void> {
    const mcpTools = await this.mcpClient.listTools();
    
    for (const tool of mcpTools) {
      this.register({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
        execute: async (args) => {
          return await this.mcpClient.callTool(tool.name, args);
        },
        source: "mcp"
      });
    }
  }
  
  register(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }
  
  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }
  
  list(): Tool[] {
    return Array.from(this.tools.values());
  }
  
  async execute(name: string, args: Record<string, unknown>): Promise<unknown> {
    const tool = this.get(name);
    if (!tool) {
      throw new Error(`Tool not found: ${name}`);
    }
    
    // Validate arguments against schema
    this.validateArgs(args, tool.inputSchema);
    
    // Execute with timeout and retry
    return await this.withRetry(
      () => tool.execute(args),
      { maxRetries: 2, timeout: 30000 }
    );
  }
}
```

---

## PHASE 3: MEMORY SYSTEM

### Goal: Implement persistent, searchable memory

#### 3.1 Memory Architecture (Three-Layer Design)

```typescript
/**
 * Three-layer memory system:
 * 1. Episodic Memory - Conversation history with semantic search
 * 2. Reflexion Memory - Success/failure patterns for learning
 * 3. Skill Library - Consolidated, reusable patterns
 */
class MemorySystem {
  private episodic: EpisodicMemory;
  private reflexion: ReflexionMemory;
  private skills: SkillLibrary;
  
  constructor(config: MemoryConfig) {
    const backend = this.createBackend(config);
    const embedder = this.createEmbedder(config);
    
    this.episodic = new EpisodicMemory(backend, embedder);
    this.reflexion = new ReflexionMemory(backend, embedder);
    this.skills = new SkillLibrary(backend, embedder);
  }
  
  private createBackend(config: MemoryConfig): MemoryBackend {
    switch (config.backend) {
      case "sqlite":
        return new SQLiteBackend(config.path);
      case "postgres":
        return new PostgresBackend(config.connectionString);
      case "redis":
        return new RedisBackend(config.url);
      default:
        throw new Error(`Unsupported backend: ${config.backend}`);
    }
  }
  
  private createEmbedder(config: MemoryConfig): Embedder {
    if (config.vectorProvider === "openai") {
      return new OpenAIEmbedder(config.apiKey);
    } else if (config.vectorProvider === "local") {
      return new LocalEmbedder(); // Uses transformers.js
    }
    throw new Error(`Unsupported embedder: ${config.vectorProvider}`);
  }
}

// Layer 1: Episodic Memory
class EpisodicMemory {
  async store(episode: {
    sessionId: string;
    messages: Message[];
    metadata: Record<string, unknown>;
  }): Promise<void> {
    // Store conversation with embeddings
    const embedding = await this.embedder.embed(
      this.summarizeConversation(episode.messages)
    );
    
    await this.backend.insert('episodes', {
      id: generateId(),
      session_id: episode.sessionId,
      messages: JSON.stringify(episode.messages),
      embedding: embedding,
      metadata: JSON.stringify(episode.metadata),
      created_at: new Date()
    });
  }
  
  async search(query: string, limit = 10): Promise<Episode[]> {
    const queryEmbedding = await this.embedder.embed(query);
    
    // Cosine similarity search
    const results = await this.backend.vectorSearch({
      table: 'episodes',
      vector: queryEmbedding,
      limit,
      minSimilarity: 0.7
    });
    
    return results.map(r => ({
      sessionId: r.session_id,
      messages: JSON.parse(r.messages),
      metadata: JSON.parse(r.metadata),
      similarity: r.similarity
    }));
  }
}

// Layer 2: Reflexion Memory
class ReflexionMemory {
  async storeEpisode(episode: {
    task: string;
    solution: string;
    success: boolean;
    reward: number;
    critique: string;
    context: any;
  }): Promise<void> {
    const embedding = await this.embedder.embed(episode.task);
    
    await this.backend.insert('reflexion_episodes', {
      id: generateId(),
      task: episode.task,
      solution: episode.solution,
      success: episode.success,
      reward: episode.reward,
      critique: episode.critique,
      context: JSON.stringify(episode.context),
      embedding: embedding,
      created_at: new Date()
    });
    
    // Trigger learning if we have enough data
    const episodeCount = await this.backend.count('reflexion_episodes');
    if (episodeCount % 100 === 0) {
      await this.consolidatePatterns();
    }
  }
  
  async retrieveRelevant(task: string, minReward = 0.7): Promise<ReflexionEpisode[]> {
    const embedding = await this.embedder.embed(task);
    
    const results = await this.backend.query(`
      SELECT * FROM reflexion_episodes
      WHERE success = 1 
        AND reward >= ?
        AND cosine_similarity(embedding, ?) > 0.7
      ORDER BY reward DESC, created_at DESC
      LIMIT 10
    `, [minReward, embedding]);
    
    return results;
  }
  
  private async consolidatePatterns(): Promise<void> {
    // Cluster similar successful episodes
    const episodes = await this.backend.query(`
      SELECT * FROM reflexion_episodes 
      WHERE success = 1 AND reward > 0.8
    `);
    
    const clusters = this.clusterEpisodes(episodes);
    
    // Extract patterns and store as skills
    for (const cluster of clusters) {
      if (cluster.length >= 3) {
        const skill = this.extractSkill(cluster);
        await this.skills.register(skill);
      }
    }
  }
}

// Layer 3: Skill Library
class SkillLibrary {
  async register(skill: {
    name: string;
    description: string;
    trigger: string;
    implementation: string;
    metadata: Record<string, unknown>;
  }): Promise<void> {
    const embedding = await this.embedder.embed(
      `${skill.name} ${skill.description} ${skill.trigger}`
    );
    
    await this.backend.insert('skills', {
      id: generateId(),
      name: skill.name,
      description: skill.description,
      trigger: skill.trigger,
      implementation: skill.implementation,
      metadata: JSON.stringify(skill.metadata),
      embedding: embedding,
      usage_count: 0,
      success_rate: 0.0,
      created_at: new Date()
    });
  }
  
  async search(query: string, limit = 5): Promise<Skill[]> {
    const embedding = await this.embedder.embed(query);
    
    const results = await this.backend.vectorSearch({
      table: 'skills',
      vector: embedding,
      limit,
      minSimilarity: 0.75
    });
    
    return results;
  }
  
  async recordUsage(skillId: string, success: boolean): Promise<void> {
    await this.backend.execute(`
      UPDATE skills 
      SET usage_count = usage_count + 1,
          success_rate = (success_rate * usage_count + ?) / (usage_count + 1)
      WHERE id = ?
    `, [success ? 1.0 : 0.0, skillId]);
  }
}
```

#### 3.2 Backend Adapters (Multi-Database Support)

```typescript
// SQLite Backend (best for local/small-scale)
class SQLiteBackend implements MemoryBackend {
  private db: Database;
  
  constructor(path: string) {
    this.db = new Database(path);
    this.initialize();
  }
  
  private initialize(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        messages TEXT,
        embedding BLOB,
        metadata TEXT,
        created_at DATETIME
      );
      
      CREATE TABLE IF NOT EXISTS reflexion_episodes (
        id TEXT PRIMARY KEY,
        task TEXT,
        solution TEXT,
        success INTEGER,
        reward REAL,
        critique TEXT,
        context TEXT,
        embedding BLOB,
        created_at DATETIME
      );
      
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        trigger TEXT,
        implementation TEXT,
        metadata TEXT,
        embedding BLOB,
        usage_count INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        created_at DATETIME
      );
      
      CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
      CREATE INDEX IF NOT EXISTS idx_reflexion_success ON reflexion_episodes(success, reward);
      CREATE INDEX IF NOT EXISTS idx_skills_usage ON skills(usage_count, success_rate);
    `);
  }
  
  async vectorSearch(params: VectorSearchParams): Promise<any[]> {
    // For SQLite, we do brute-force cosine similarity
    // For production, use extension like sqlite-vss or external vector DB
    const all = await this.query(`SELECT * FROM ${params.table}`);
    
    const withScores = all.map(row => ({
      ...row,
      similarity: this.cosineSimilarity(params.vector, row.embedding)
    }));
    
    return withScores
      .filter(r => r.similarity >= params.minSimilarity)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, params.limit);
  }
}

// PostgreSQL Backend (best for production/scale)
class PostgresBackend implements MemoryBackend {
  private pool: Pool;
  
  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
    this.initialize();
  }
  
  private async initialize(): Promise<void> {
    // Enable pgvector extension for efficient vector search
    await this.execute('CREATE EXTENSION IF NOT EXISTS vector');
    
    await this.execute(`
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        messages TEXT,
        embedding vector(1536),
        metadata JSONB,
        created_at TIMESTAMP DEFAULT NOW()
      );
      
      CREATE TABLE IF NOT EXISTS reflexion_episodes (
        id TEXT PRIMARY KEY,
        task TEXT,
        solution TEXT,
        success BOOLEAN,
        reward REAL,
        critique TEXT,
        context JSONB,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT NOW()
      );
      
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        name TEXT,
        description TEXT,
        trigger TEXT,
        implementation TEXT,
        metadata JSONB,
        embedding vector(1536),
        usage_count INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT NOW()
      );
      
      -- Create HNSW index for fast vector search
      CREATE INDEX IF NOT EXISTS episodes_embedding_idx 
        ON episodes USING hnsw (embedding vector_cosine_ops);
      CREATE INDEX IF NOT EXISTS reflexion_embedding_idx 
        ON reflexion_episodes USING hnsw (embedding vector_cosine_ops);
      CREATE INDEX IF NOT EXISTS skills_embedding_idx 
        ON skills USING hnsw (embedding vector_cosine_ops);
    `);
  }
  
  async vectorSearch(params: VectorSearchParams): Promise<any[]> {
    const result = await this.query(`
      SELECT *, (embedding <=> $1::vector) AS distance
      FROM ${params.table}
      WHERE (embedding <=> $1::vector) < $2
      ORDER BY embedding <=> $1::vector
      LIMIT $3
    `, [
      `[${params.vector.join(',')}]`,
      1 - params.minSimilarity, // Convert similarity to distance
      params.limit
    ]);
    
    return result.rows.map(row => ({
      ...row,
      similarity: 1 - row.distance
    }));
  }
}
```

---

## PHASE 4: MULTI-AGENT ORCHESTRATION

### Goal: Enable swarm intelligence and parallel task execution

#### 4.1 Swarm Orchestrator (Hive-Mind Pattern)

```typescript
/**
 * Swarm Orchestrator - Coordinate multiple specialized agents
 * Inspired by claude-flow's hive-mind architecture
 */
class SwarmOrchestrator {
  private queen: Agent; // Coordinator/strategist
  private workers: Map<string, Agent> = new Map();
  private taskQueue: TaskQueue;
  private sharedMemory: SharedMemory;
  
  constructor(config: SwarmConfig) {
    this.queen = new Agent({
      name: "queen",
      role: "coordinator",
      systemPrompt: this.getQueenPrompt(),
      llmRouter: config.llmRouter,
      memory: config.sharedMemory
    });
    
    this.taskQueue = new TaskQueue(config.queueBackend);
    this.sharedMemory = config.sharedMemory;
  }
  
  async execute(task: string): Promise<SwarmResult> {
    // 1. Queen analyzes and creates execution plan
    const plan = await this.queen.plan(task);
    
    // 2. Spawn specialized workers for each subtask
    const workers = await this.spawnWorkers(plan);
    
    // 3. Execute plan (parallel when possible, sequential when needed)
    const results = await this.executeSwarm(workers, plan);
    
    // 4. Queen synthesizes results
    const finalResult = await this.queen.synthesize({
      task,
      plan,
      workerResults: results
    });
    
    // 5. Cleanup workers
    await this.cleanupWorkers(workers);
    
    return finalResult;
  }
  
  private async spawnWorkers(plan: ExecutionPlan): Promise<Agent[]> {
    const workers: Agent[] = [];
    
    for (const subtask of plan.subtasks) {
      const worker = new Agent({
        name: `worker-${subtask.id}`,
        role: subtask.specialization, // e.g., "coder", "tester", "researcher"
        systemPrompt: this.getWorkerPrompt(subtask.specialization),
        llmRouter: this.selectOptimalLLM(subtask),
        memory: this.sharedMemory,
        tools: this.filterToolsForRole(subtask.specialization)
      });
      
      workers.push(worker);
      this.workers.set(worker.name, worker);
    }
    
    return workers;
  }
  
  private async executeSwarm(
    workers: Agent[], 
    plan: ExecutionPlan
  ): Promise<Map<string, any>> {
    const results = new Map<string, any>();
    const dependencyGraph = this.buildDependencyGraph(plan);
    
    // Execute tasks respecting dependencies
    const executionOrder = this.topologicalSort(dependencyGraph);
    
    for (const taskBatch of executionOrder) {
      // Execute independent tasks in parallel
      const batchResults = await Promise.all(
        taskBatch.map(async (taskId) => {
          const subtask = plan.subtasks.find(t => t.id === taskId);
          const worker = workers.find(w => w.name.includes(taskId));
          
          // Provide context from dependencies
          const context = this.gatherDependencyResults(
            subtask.dependencies, 
            results
          );
          
          const result = await worker.run(subtask.description, context);
          
          // Store in shared memory for other workers
          await this.sharedMemory.set(taskId, result);
          
          return { taskId, result };
        })
      );
      
      // Store results for next batch
      batchResults.forEach(({ taskId, result }) => {
        results.set(taskId, result);
      });
    }
    
    return results;
  }
  
  private getQueenPrompt(): string {
    return `You are the Queen agent - a strategic coordinator for a swarm of specialized AI agents.

Your responsibilities:
1. Analyze complex tasks and break them into subtasks
2. Determine optimal execution strategy (parallel vs sequential)
3. Identify dependencies between subtasks
4. Assign appropriate specializations to each subtask
5. Synthesize results from worker agents into coherent final output

When planning:
- Maximize parallelization where possible
- Minimize inter-agent communication overhead
- Choose appropriate specializations (coder, tester, researcher, analyst, etc.)
- Consider resource constraints and task complexity

Output your plan as JSON:
{
  "subtasks": [
    {
      "id": "task-1",
      "description": "...",
      "specialization": "coder",
      "dependencies": [],
      "estimatedComplexity": "low|medium|high"
    }
  ],
  "executionStrategy": "parallel|sequential|hybrid"
}`;
  }
  
  private getWorkerPrompt(specialization: string): string {
    const prompts = {
      coder: "You are an expert software engineer. Write clean, efficient, well-tested code.",
      tester: "You are a QA specialist. Write comprehensive tests and find edge cases.",
      researcher: "You are a research analyst. Gather information and provide insights.",
      analyst: "You are a data analyst. Process data and extract meaningful patterns.",
      reviewer: "You are a code reviewer. Identify issues and suggest improvements."
    };
    
    return prompts[specialization] || "You are a general-purpose AI assistant.";
  }
  
  private selectOptimalLLM(subtask: Subtask): LLMRouter {
    // Route based on subtask complexity and specialization
    if (subtask.estimatedComplexity === "high") {
      return new LLMRouter({ strategy: "performance" });
    } else if (subtask.estimatedComplexity === "low") {
      return new LLMRouter({ strategy: "cost" });
    }
    return new LLMRouter({ strategy: "balanced" });
  }
}

// Shared Memory for Worker Communication
class SharedMemory {
  private backend: MemoryBackend;
  private cache: Map<string, any> = new Map();
  
  async set(key: string, value: any, ttl?: number): Promise<void> {
    this.cache.set(key, value);
    
    await this.backend.insert('shared_memory', {
      key,
      value: JSON.stringify(value),
      expires_at: ttl ? new Date(Date.now() + ttl * 1000) : null,
      created_at: new Date()
    });
  }
  
  async get(key: string): Promise<any> {
    // Check cache first
    if (this.cache.has(key)) {
      return this.cache.get(key);
    }
    
    // Fallback to backend
    const result = await this.backend.query(
      'SELECT value FROM shared_memory WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)',
      [key, new Date()]
    );
    
    if (result.length > 0) {
      const value = JSON.parse(result[0].value);
      this.cache.set(key, value);
      return value;
    }
    
    return null;
  }
  
  async broadcast(message: BroadcastMessage): Promise<void> {
    // Notify all workers of important events
    await this.backend.insert('broadcasts', {
      id: generateId(),
      from: message.from,
      type: message.type,
      payload: JSON.stringify(message.payload),
      created_at: new Date()
    });
  }
}
```

#### 4.2 Task Queue (For Async Work Distribution)

```typescript
/**
 * Task Queue - Distribute work across agents asynchronously
 * Supports Redis, RabbitMQ, AWS SQS, or in-memory queue
 */
class TaskQueue {
  private backend: QueueBackend;
  
  constructor(config: QueueConfig) {
    this.backend = this.createBackend(config);
  }
  
  private createBackend(config: QueueConfig): QueueBackend {
    switch (config.type) {
      case "redis":
        return new RedisQueue(config.url);
      case "rabbitmq":
        return new RabbitMQQueue(config.url);
      case "sqs":
        return new SQSQueue(config.region, config.queueUrl);
      case "memory":
        return new InMemoryQueue();
      default:
        throw new Error(`Unsupported queue type: ${config.type}`);
    }
  }
  
  async enqueue(task: Task, priority = 0): Promise<string> {
    const taskId = generateId();
    
    await this.backend.push({
      id: taskId,
      type: task.type,
      payload: task.payload,
      priority,
      retries: 0,
      maxRetries: task.maxRetries || 3,
      createdAt: Date.now()
    });
    
    return taskId;
  }
  
  async dequeue(workerType?: string): Promise<Task | null> {
    return await this.backend.pop(workerType);
  }
  
  async ack(taskId: string): Promise<void> {
    await this.backend.acknowledge(taskId);
  }
  
  async nack(taskId: string, requeue = true): Promise<void> {
    await this.backend.reject(taskId, requeue);
  }
}

// Redis-based queue implementation
class RedisQueue implements QueueBackend {
  private client: Redis;
  
  constructor(url: string) {
    this.client = new Redis(url);
  }
  
  async push(task: QueueTask): Promise<void> {
    // Use sorted set for priority queue
    await this.client.zadd(
      'task-queue',
      task.priority,
      JSON.stringify(task)
    );
  }
  
  async pop(workerType?: string): Promise<QueueTask | null> {
    // Atomic pop with ZPOPMAX
    const result = await this.client.zpopmax('task-queue');
    
    if (!result || result.length === 0) {
      return null;
    }
    
    const task = JSON.parse(result[0]);
    
    // Move to processing set
    await this.client.setex(
      `processing:${task.id}`,
      300, // 5 minute timeout
      JSON.stringify(task)
    );
    
    return task;
  }
  
  async acknowledge(taskId: string): Promise<void> {
    await this.client.del(`processing:${taskId}`);
  }
  
  async reject(taskId: string, requeue: boolean): Promise<void> {
    const taskData = await this.client.get(`processing:${taskId}`);
    
    if (!taskData) return;
    
    const task = JSON.parse(taskData);
    
    if (requeue && task.retries < task.maxRetries) {
      task.retries++;
      await this.push(task);
    }
    
    await this.client.del(`processing:${taskId}`);
  }
}
```

---

## PHASE 5: PRODUCTION HARDENING

### Goal: Add enterprise-grade reliability and observability

#### 5.1 Error Handling & Resilience

```typescript
/**
 * Comprehensive error handling with retries, circuit breakers, and fallbacks
 */
class ResilienceManager {
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  
  async withRetry<T>(
    fn: () => Promise<T>,
    options: RetryOptions = {}
  ): Promise<T> {
    const {
      maxRetries = 3,
      backoff = "exponential",
      initialDelay = 1000,
      maxDelay = 30000,
      timeout = 60000
    } = options;
    
    let lastError: Error;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await this.withTimeout(fn, timeout);
      } catch (error) {
        lastError = error;
        
        if (attempt < maxRetries) {
          const delay = this.calculateDelay(attempt, backoff, initialDelay, maxDelay);
          await this.sleep(delay);
        }
      }
    }
    
    throw lastError;
  }
  
  async withCircuitBreaker<T>(
    key: string,
    fn: () => Promise<T>,
    options: CircuitBreakerOptions = {}
  ): Promise<T> {
    let breaker = this.circuitBreakers.get(key);
    
    if (!breaker) {
      breaker = new CircuitBreaker(options);
      this.circuitBreakers.set(key, breaker);
    }
    
    return await breaker.execute(fn);
  }
  
  async withFallback<T>(
    primary: () => Promise<T>,
    fallback: () => Promise<T>
  ): Promise<T> {
    try {
      return await primary();
    } catch (error) {
      console.warn('Primary operation failed, using fallback:', error);
      return await fallback();
    }
  }
  
  private calculateDelay(
    attempt: number,
    strategy: string,
    initial: number,
    max: number
  ): number {
    let delay: number;
    
    switch (strategy) {
      case "exponential":
        delay = initial * Math.pow(2, attempt);
        break;
      case "linear":
        delay = initial * (attempt + 1);
        break;
      case "constant":
        delay = initial;
        break;
      default:
        delay = initial;
    }
    
    // Add jitter to prevent thundering herd
    delay = delay * (0.5 + Math.random() * 0.5);
    
    return Math.min(delay, max);
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
      if (Date.now() - this.lastFailureTime > this.options.resetTimeout) {
        this.state = "half-open";
      } else {
        throw new Error("Circuit breaker is open");
      }
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }
  
  private onSuccess(): void {
    this.failureCount = 0;
    
    if (this.state === "half-open") {
      this.successCount++;
      
      if (this.successCount >= this.options.successThreshold) {
        this.state = "closed";
        this.successCount = 0;
      }
    }
  }
  
  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    
    if (this.failureCount >= this.options.failureThreshold) {
      this.state = "open";
    }
  }
}
```

#### 5.2 Observability (Logging, Metrics, Tracing)

```typescript
/**
 * Comprehensive observability stack
 * Supports: Console, File, Syslog, CloudWatch, DataDog, Grafana
 */
class ObservabilityManager {
  private logger: Logger;
  private metrics: MetricsCollector;
  private tracer: Tracer;
  
  constructor(config: ObservabilityConfig) {
    this.logger = this.createLogger(config.logging);
    this.metrics = this.createMetrics(config.metrics);
    this.tracer = this.createTracer(config.tracing);
  }
  
  private createLogger(config: LoggingConfig): Logger {
    const transports: any[] = [];
    
    for (const output of config.outputs) {
      switch (output.type) {
        case "console":
          transports.push(new ConsoleTransport(output));
          break;
        case "file":
          transports.push(new FileTransport(output));
          break;
        case "cloudwatch":
          transports.push(new CloudWatchTransport(output));
          break;
        case "datadog":
          transports.push(new DataDogTransport(output));
          break;
      }
    }
    
    return new Logger({
      level: config.level,
      format: config.format,
      transports
    });
  }
  
  private createMetrics(config: MetricsConfig): MetricsCollector {
    switch (config.backend) {
      case "prometheus":
        return new PrometheusCollector(config);
      case "cloudwatch":
        return new CloudWatchMetrics(config);
      case "datadog":
        return new DataDogMetrics(config);
      default:
        return new InMemoryMetrics();
    }
  }
  
  private createTracer(config: TracingConfig): Tracer {
    if (!config.enabled) {
      return new NoOpTracer();
    }
    
    switch (config.backend) {
      case "jaeger":
        return new JaegerTracer(config);
      case "zipkin":
        return new ZipkinTracer(config);
      case "xray":
        return new XRayTracer(config);
      default:
        return new OpenTelemetryTracer(config);
    }
  }
  
  // Structured logging
  log(level: string, message: string, context?: any): void {
    this.logger.log(level, message, {
      ...context,
      timestamp: new Date().toISOString(),
      service: "universal-agent"
    });
  }
  
  // Metrics recording
  recordMetric(name: string, value: number, tags?: Record<string, string>): void {
    this.metrics.record({
      name,
      value,
      tags,
      timestamp: Date.now()
    });
  }
  
  // Distributed tracing
  startSpan(name: string, attributes?: any): Span {
    return this.tracer.startSpan(name, attributes);
  }
}

// Example usage in agent
class ObservableAgent extends Agent {
  async run(message: string): Promise<string> {
    const span = this.observability.startSpan("agent.run", {
      message: message.substring(0, 100)
    });
    
    try {
      this.observability.log("info", "Starting agent execution", {
        messageLength: message.length
      });
      
      const startTime = Date.now();
      const result = await super.run(message);
      const duration = Date.now() - startTime;
      
      this.observability.recordMetric("agent.execution.duration", duration, {
        success: "true"
      });
      
      span.setStatus("ok");
      return result;
    } catch (error) {
      this.observability.log("error", "Agent execution failed", {
        error: error.message,
        stack: error.stack
      });
      
      this.observability.recordMetric("agent.execution.errors", 1);
      
      span.setStatus("error");
      span.recordException(error);
      throw error;
    } finally {
      span.end();
    }
  }
}
```

#### 5.3 Security & Secrets Management

```typescript
/**
 * Security manager for API keys, credentials, and sensitive data
 */
class SecurityManager {
  private secretsProvider: SecretsProvider;
  private encryption: EncryptionService;
  
  constructor(config: SecurityConfig) {
    this.secretsProvider = this.createSecretsProvider(config);
    this.encryption = new EncryptionService(config.encryptionKey);
  }
  
  private createSecretsProvider(config: SecurityConfig): SecretsProvider {
    switch (config.provider) {
      case "env":
        return new EnvironmentSecretsProvider();
      case "aws-secrets-manager":
        return new AWSSecretsManager(config.region);
      case "gcp-secret-manager":
        return new GCPSecretManager(config.projectId);
      case "azure-key-vault":
        return new AzureKeyVault(config.vaultUrl);
      case "hashicorp-vault":
        return new HashiCorpVault(config.vaultAddr);
      default:
        return new EnvironmentSecretsProvider();
    }
  }
  
  async getSecret(key: string): Promise<string> {
    // Check cache first
    const cached = this.secretsCache.get(key);
    if (cached && !this.isExpired(cached)) {
      return cached.value;
    }
    
    // Fetch from provider
    const value = await this.secretsProvider.get(key);
    
    // Cache with TTL
    this.secretsCache.set(key, {
      value,
      expiresAt: Date.now() + 3600000 // 1 hour
    });
    
    return value;
  }
  
  // Encrypt sensitive data before storage
  encrypt(plaintext: string): string {
    return this.encryption.encrypt(plaintext);
  }
  
  decrypt(ciphertext: string): string {
    return this.encryption.decrypt(ciphertext);
  }
  
  // Validate and sanitize user input
  sanitizeInput(input: string, type: "sql" | "shell" | "path"): string {
    switch (type) {
      case "sql":
        return input.replace(/['";\\]/g, "");
      case "shell":
        return input.replace(/[;&|`$()]/g, "");
      case "path":
        return input.replace(/\.\./g, "").replace(/[<>:"|?*]/g, "");
      default:
        return input;
    }
  }
  
  // Rate limiting
  async checkRateLimit(key: string, limit: number, window: number): Promise<boolean> {
    const count = await this.rateLimiter.increment(key, window);
    return count <= limit;
  }
}
```

---

## PHASE 6: DEPLOYMENT & INFRASTRUCTURE

### Goal: Deploy anywhere with one command

#### 6.1 Universal Deployment System

```typescript
/**
 * Deploy your agent to any platform with minimal configuration
 */
class DeploymentManager {
  async deploy(target: DeploymentTarget, config: DeploymentConfig): Promise<void> {
    // 1. Build application
    await this.build(config);
    
    // 2. Run tests
    await this.test(config);
    
    // 3. Package for target platform
    const artifact = await this.package(target, config);
    
    // 4. Deploy to target
    await this.deployToTarget(target, artifact, config);
    
    // 5. Run smoke tests
    await this.smokeTest(target, config);
  }
  
  private async deployToTarget(
    target: DeploymentTarget,
    artifact: Artifact,
    config: DeploymentConfig
  ): Promise<void> {
    switch (target.type) {
      case "aws-lambda":
        await this.deployToLambda(artifact, config);
        break;
      case "gcp-cloud-functions":
        await this.deployToCloudFunctions(artifact, config);
        break;
      case "azure-functions":
        await this.deployToAzureFunctions(artifact, config);
        break;
      case "docker":
        await this.deployDocker(artifact, config);
        break;
      case "kubernetes":
        await this.deployK8s(artifact, config);
        break;
      case "cloudflare-workers":
        await this.deployCloudflareWorkers(artifact, config);
        break;
      case "vercel":
        await this.deployVercel(artifact, config);
        break;
      default:
        throw new Error(`Unsupported deployment target: ${target.type}`);
    }
  }
}
```

#### 6.2 Infrastructure as Code Templates

```hcl
# Terraform: Deploy to AWS Lambda + DynamoDB + API Gateway
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Lambda function
resource "aws_lambda_function" "agent" {
  filename      = "agent.zip"
  function_name = "universal-agent"
  role          = aws_iam_role.agent_role.arn
  handler       = "index.handler"
  runtime       = "nodejs20.x"
  timeout       = 300
  memory_size   = 1024
  
  environment {
    variables = {
      MEMORY_TABLE = aws_dynamodb_table.memory.name
      ANTHROPIC_API_KEY = var.anthropic_api_key
    }
  }
}

# DynamoDB for memory
resource "aws_dynamodb_table" "memory" {
  name           = "agent-memory"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "id"
  
  attribute {
    name = "id"
    type = "S"
  }
  
  global_secondary_index {
    name            = "session-index"
    hash_key        = "session_id"
    projection_type = "ALL"
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "agent_api" {
  name          = "agent-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id           = aws_apigatewayv2_api.agent_api.id
  integration_type = "AWS_PROXY"
  integration_uri  = aws_lambda_function.agent.invoke_arn
}
```

```yaml
# Docker Compose: Run locally with all services
version: '3.8'

services:
  agent:
    build: .
    ports:
      - "3000:3000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/agent
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./data:/app/data
  
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=agent
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  # Optional: Vector database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

```yaml
# Kubernetes: Production deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: universal-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent
  template:
    metadata:
      labels:
        app: agent
    spec:
      containers:
      - name: agent
        image: your-registry/universal-agent:latest
        ports:
        - containerPort: 3000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: anthropic-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: agent-service
spec:
  selector:
    app: agent
  ports:
  - port: 80
    targetPort: 3000
  type: LoadBalancer
```

---

## PHASE 7: TESTING & BENCHMARKING

### Goal: Validate performance against industry standards

```typescript
/**
 * Comprehensive benchmark suite
 */
class BenchmarkSuite {
  private agent: Agent;
  
  async runAll(): Promise<BenchmarkResults> {
    return {
      sweБench: await this.runSWEBench(),
      humanEval: await this.runHumanEval(),
      customMetrics: await this.runCustomMetrics(),
      performanceTests: await this.runPerformanceTests()
    };
  }
  
  async runSWEBench(): Promise<SWEBenchResult> {
    // Test against SWE-Bench verified dataset
    const testCases = await loadSWEBenchData();
    let resolved = 0;
    
    for (const testCase of testCases) {
      const result = await this.agent.run(testCase.problem);
      
      if (await this.validateSolution(result, testCase.tests)) {
        resolved++;
      }
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
      tokenEfficiency: await this.measureTokenEfficiency(),
      parallelizationSpeedup: await this.measureParallelization(),
      costSavings: await this.measureCostOptimization(),
      memoryRetrieval: await this.measureMemorySpeed()
    };
  }
  
  private async measureTokenEfficiency(): Promise<number> {
    // Compare token usage with/without memory and routing
    const baseline = await this.runWithoutOptimizations();
    const optimized = await this.runWithOptimizations();
    
    return ((baseline.tokens - optimized.tokens) / baseline.tokens) * 100;
  }
  
  private async measureMemorySpeed(): Promise<number> {
    const queries = generateTestQueries(100);
    const startTime = Date.now();
    
    for (const query of queries) {
      await this.agent.memory.search(query);
    }
    
    const avgTime = (Date.now() - startTime) / queries.length;
    return avgTime; // Target: <100ms
  }
}
```

---

## AUTO-GENERATED DOCUMENTATION

### The coding assistant must generate comprehensive documentation:

#### 7.1 Implementation Guide

```markdown
# Universal Agent Implementation Guide

## Quick Start

### 1. Installation

\`\`\`bash
# Clone your generated repository
git clone <your-repo>
cd universal-agent

# Install dependencies
npm install  # or: pip install -r requirements.txt, go mod download, etc.

# Configure environment
cp .env.example .env
# Edit .env with your API keys
\`\`\`

### 2. Configuration

Edit `config/agent.config.yaml`:

\`\`\`yaml
llm:
  strategy: "balanced"
  providers:
    anthropic:
      enabled: true
      api_key: "${ANTHROPIC_API_KEY}"
\`\`\`

### 3. Run Your First Agent

\`\`\`bash
# Start the agent
npm start

# Or run a specific task
npm run task "Create a Python script that analyzes CSV files"
\`\`\`

## Architecture Overview

[Generated architecture diagram]

Your agent follows a modular architecture:

1. **Core Agent Loop** - Manages conversation and tool execution
2. **LLM Router** - Intelligently routes to optimal models
3. **MCP Client** - Connects to MCP servers for tools
4. **Memory System** - Stores and retrieves past experiences
5. **Swarm Orchestrator** - Coordinates multiple agents (if enabled)

## Configuration Reference

### LLM Providers

#### Anthropic (Claude)
- **Best for**: Complex reasoning, code generation
- **Models**: claude-sonnet-4, claude-haiku-4
- **Cost**: $3-15 per million tokens

[Full configuration options for all providers...]

### MCP Servers

Your agent can connect to any MCP server. Included servers:

- **filesystem**: File operations in allowed directories
- **github**: GitHub API operations
- **brave-search**: Web search capabilities

To add more servers, edit `mcp.servers` in config.

### Memory Backends

Choose based on your scale:

- **SQLite**: Local development, single-instance
- **PostgreSQL**: Production, vector search with pgvector
- **Redis**: Distributed, multi-agent coordination

## Extension Guide

### Adding a New Tool

\`\`\`typescript
// src/tools/custom/my-tool.ts
export const myCustomTool: Tool = {
  name: "my_custom_tool",
  description: "Does something amazing",
  inputSchema: {
    type: "object",
    properties: {
      input: { type: "string" }
    }
  },
  execute: async (args) => {
    // Your implementation
    return { result: "success" };
  }
};

// Register in src/tools/registry.ts
toolRegistry.register(myCustomTool);
\`\`\`

### Adding a New LLM Provider

\`\`\`typescript
// src/llm/providers/custom-provider.ts
export class CustomProvider implements LLMProvider {
  name = "custom-provider";
  
  async complete(params: CompletionParams): Promise<CompletionResponse> {
    // Implement your provider logic
  }
  
  estimateCost(params: CompletionParams): number {
    // Calculate cost
  }
  
  capabilities = {
    maxTokens: 4096,
    supportsTools: true,
    supportsVision: false,
    contextWindow: 128000
  };
}
\`\`\`

### Creating a New MCP Server

\`\`\`yaml
# Add to config/agent.config.yaml
mcp:
  servers:
    - name: "my-server"
      command: "node"
      args: ["./servers/my-server.js"]
      env:
        MY_API_KEY: "${MY_API_KEY}"
\`\`\`

## Deployment Guide

### Option 1: Docker

\`\`\`bash
docker build -t universal-agent .
docker run -p 3000:3000 --env-file .env universal-agent
\`\`\`

### Option 2: AWS Lambda

\`\`\`bash
# Package for Lambda
npm run package:lambda

# Deploy with Terraform
cd terraform/aws
terraform init
terraform apply
\`\`\`

### Option 3: Kubernetes

\`\`\`bash
# Build and push image
docker build -t your-registry/agent:latest .
docker push your-registry/agent:latest

# Deploy
kubectl apply -f k8s/
\`\`\`

## Monitoring & Observability

### Logs

Logs are output in JSON format to stdout and optionally to files:

\`\`\`json
{
  "level": "info",
  "timestamp": "2025-01-17T12:00:00Z",
  "message": "Agent execution completed",
  "duration": 1234,
  "tokens": 567
}
\`\`\`

### Metrics

Key metrics tracked:
- `agent.execution.duration` - Time per request
- `agent.execution.tokens` - Token usage
- `agent.tool.calls` - Tool invocation count
- `agent.memory.retrieval.time` - Memory query latency

### Tracing

Distributed traces show the full execution path through your agent.

## Troubleshooting

### Common Issues

**Issue**: "Circuit breaker is open"
- **Cause**: Too many failed LLM requests
- **Solution**: Check API keys, verify provider status

**Issue**: "Memory retrieval timeout"
- **Cause**: Vector search taking too long
- **Solution**: Add indexes, reduce search limit, upgrade database

**Issue**: "MCP server connection failed"
- **Cause**: Server process crashed or not accessible
- **Solution**: Check server logs, verify command/args in config

## Performance Tuning

### Optimize Token Usage
- Enable memory retrieval to avoid repeating context
- Use cheaper models for simple tasks
- Implement smart routing based on complexity

### Optimize Latency
- Enable parallel tool execution
- Use local embeddings instead of API calls
- Add caching layer for repeated queries

### Optimize Cost
- Route low-complexity tasks to cheaper models
- Batch similar requests
- Implement aggressive caching

## Security Best Practices

1. **API Keys**: Never commit to git, use secrets manager
2. **Input Validation**: Sanitize all user input
3. **File Access**: Restrict to allowed directories
4. **Rate Limiting**: Prevent abuse
5. **Audit Logging**: Track all operations

## Support & Resources

- Documentation: [Link to your docs]
- GitHub Issues: [Link to issues]
- Community Discord: [Link if applicable]
- Email: support@your-domain.com
\`\`\`

#### 7.2 Architecture Decision Records (ADRs)

```markdown
# Architecture Decision Record: LLM Routing Strategy

**Status**: Accepted
**Date**: 2025-01-17
**Decision Makers**: [Your name/team]

## Context

The agent needs to support multiple LLM providers for cost optimization, reliability, and flexibility. We need to decide how to route requests to different models.

## Decision

We will implement a three-strategy routing system:

1. **Cost-Optimized**: Always choose cheapest capable model
2. **Performance**: Always choose most capable model
3. **Balanced**: Analyze task complexity and route accordingly

## Rationale

- Different tasks have different requirements
- Cost varies 100x between models (Haiku vs Opus)
- User should control the tradeoff
- Complexity analysis enables automatic optimization

## Consequences

**Positive**:
- 30-60% cost reduction on average
- Maintained quality for complex tasks
- User control and transparency

**Negative**:
- Added complexity in routing logic
- Requires complexity heuristics
- More provider management overhead

## Alternatives Considered

1. **Single Provider**: Simpler but no cost optimization
2. **Manual Selection**: User chooses per request (too much friction)
3. **Random**: No optimization benefits

---

# Architecture Decision Record: Memory Backend

**Status**: Accepted
**Date**: 2025-01-17

## Context

The agent needs persistent memory to learn from past interactions. We need to choose a storage backend that balances simplicity, performance, and scalability.

## Decision

Support multiple backends through adapter pattern:
- **SQLite**: Default for local/development
- **PostgreSQL + pgvector**: Production recommendation
- **Redis**: For distributed multi-agent systems

## Rationale

- SQLite: Zero-config, perfect for getting started
- PostgreSQL: Battle-tested, excellent vector search
- Redis: Fast, distributed, good for coordination
- Adapter pattern: Users can add any backend

## Consequences

**Positive**:
- Easy local development (SQLite)
- Production-ready option (Postgres)
- No vendor lock-in
- Extensible to any database

**Negative**:
- More code to maintain (adapters)
- Different features per backend
- Migration complexity

## Implementation Notes

All backends must implement:
- `insert(table, data)`
- `query(sql, params)`
- `vectorSearch(params)`

Vector search falls back to brute-force on backends without native support.
```

---

## SUCCESS CRITERIA & KPIs

Your implementation should achieve:

### Performance Benchmarks

| Metric | Target | Industry Standard |
|--------|--------|-------------------|
| SWE-Bench Score | 70%+ | 72.7% (Claude 4) |
| HumanEval Pass@1 | 90%+ | 92% (GPT-4) |
| Token Reduction | 30%+ | N/A |
| Memory Retrieval | <100ms | N/A |
| Tool Success Rate | 95%+ | N/A |
| Parallel Speedup | 3-5x | N/A |
| Cost per Task | <$0.10 | N/A |

### Quality Metrics

- ✅ **Zero Vendor Lock-in**: Runs on any platform
- ✅ **Production Ready**: Error handling, logging, monitoring
- ✅ **Extensible**: Easy to add tools, providers, backends
- ✅ **Well Documented**: Comprehensive guides and examples
- ✅ **Tested**: >80% code coverage
- ✅ **Secure**: Secrets management, input validation
- ✅ **Observable**: Logs, metrics, traces

### Learning Metrics

- Session-over-session improvement on repeated tasks
- Skill library growth rate
- Memory retrieval relevance scores
- Reflexion accuracy (success prediction)

---

## CRITICAL ANTI-PATTERNS TO AVOID

### Don't Do This

1. **Hard-coding provider names**: Use abstraction layers
2. **Ignoring errors**: Every operation can fail
3. **No timeout**: Always set reasonable timeouts
4. **Synchronous tools**: Parallelize when safe
5. **No memory**: Stateless agents don't learn
6. **Over-engineering**: Start simple, add complexity as needed
7. **Skipping tests**: You can't improve what you don't measure
8. **Poor secrets management**: Never commit API keys
9. **No observability**: You can't debug what you can't see
10. **Vendor lock-in**: Abstract cloud services behind interfaces

### Do This Instead

1. **Provider abstraction**: `interface LLMProvider { ... }`
2. **Comprehensive error handling**: Try-catch with retries
3. **Timeout everything**: `withTimeout(fn, 30000)`
4. **Parallel execution**: `Promise.all()` when safe
5. **Persistent memory**: Store episodes, learn patterns
6. **Incremental complexity**: Build in phases
7. **Benchmark regularly**: Track improvements
8. **Secrets manager**: AWS Secrets, Vault, env vars
9. **Structured logging**: JSON logs with context
10. **Adapter pattern**: Swap implementations easily

---

## DELIVERABLES

When you execute this prompt, your coding assistant should generate:

### 1. Complete Codebase
```
universal-agent/
├── src/                    # All source code
├── tests/                  # Comprehensive tests
├── docs/                   # Full documentation
├── config/                 # Configuration templates
├── terraform/              # IaC for AWS, GCP, Azure
├── k8s/                    # Kubernetes manifests
├── docker/                 # Dockerfiles
├── .github/workflows/      # CI/CD pipelines
├── examples/               # Usage examples
└── benchmarks/             # Performance tests
```

### 2. Documentation Suite
- `README.md` - Quick start guide
- `ARCHITECTURE.md` - System design
- `DEPLOYMENT.md` - Deployment options
- `CONFIGURATION.md` - All config options
- `EXTENDING.md` - How to add features
- `TROUBLESHOOTING.md` - Common issues
- `ADRs/` - Architecture decisions

### 3. Configuration Files
- `agent.config.yaml` - Main configuration
- `.env.example` - Environment template
- `docker-compose.yaml` - Local development
- `terraform/*.tf` - Cloud deployment
- `k8s/*.yaml` - Kubernetes deployment

### 4. Testing & Benchmarking
- Unit tests (>80% coverage)
- Integration tests
- SWE-Bench evaluation
- Performance benchmarks
- Load testing scripts

### 5. CI/CD Pipeline
- Automated testing
- Linting and formatting
- Security scanning
- Docker image builds
- Deployment automation

---

## LEARNING RESOURCES

### Essential Reading

1. **Agent Design**:
   - https://ampcode.com/how-to-build-an-agent
   - Anthropic Prompt Engineering Guide
   
2. **MCP Protocol**:
   - https://modelcontextprotocol.io/docs
   - MCP GitHub Examples
   
3. **Memory Systems**:
   - https://github.com/ruvnet/agentic-flow
   - ReasoningBank Paper
   
4. **Multi-Agent**:
   - AutoGen Framework
   - CrewAI Documentation

### Reference Implementations

- **agentic-flow**: Memory + SONA learning
- **claude-flow**: Hive-mind architecture
- **Amp Code**: Simple agent loop
- **Continue**: IDE integration
- **Cursor**: Agent mode

---

## EXECUTION INSTRUCTIONS

### For the Coding Assistant

When a user provides this prompt with their configuration:

1. **Analyze Requirements**
   - Parse user's tech stack preferences
   - Identify deployment target
   - Note any constraints

2. **Generate Phase 1** (Foundation)
   - Create project structure
   - Implement core agent loop
   - Add 3 basic tools
   - Generate README

3. **Generate Phase 2** (MCP Integration)
   - Implement MCP client
   - Connect to configured servers
   - Add tool registry
   - Update docs

4. **Generate Phase 3** (Memory)
   - Implement chosen backend
   - Add vector embeddings
   - Create skill library
   - Add examples

5. **Generate Phase 4** (Multi-Agent - if requested)
   - Implement swarm orchestrator
   - Add task queue
   - Create worker agents
   - Add coordination examples

6. **Generate Phase 5** (Production)
   - Add error handling
   - Implement observability
   - Add security features
   - Create deployment configs

7. **Generate Phase 6** (Deploy)
   - Create IaC templates
   - Generate Dockerfiles
   - Add CI/CD pipeline
   - Write deployment guide

8. **Generate Phase 7** (Testing)
   - Create test suite
   - Add benchmarks
   - Generate test data
   - Create evaluation scripts

9. **Generate Documentation**
   - Implementation guide
   - API reference
   - Extension guide
   - ADRs

10. **Final Validation**
    - Verify all files compile/run
    - Check documentation completeness
    - Validate against success criteria
    - Generate checklist for user

---

## 📋 FINAL CHECKLIST

Before considering the agent system complete, verify:

### Core Functionality
- [ ] Agent loop executes successfully
- [ ] LLM routing works for all configured providers
- [ ] MCP client connects to servers
- [ ] Tools execute correctly
- [ ] Memory stores and retrieves data
- [ ] Error handling catches failures gracefully

### Production Readiness
- [ ] Logging outputs structured data
- [ ] Metrics are collected
- [ ] Secrets are externalized
- [ ] Tests pass with >80% coverage
- [ ] Security scanning shows no critical issues
- [ ] Performance meets targets

### Documentation
- [ ] README explains quick start
- [ ] Architecture is documented
- [ ] All config options explained
- [ ] Extension guide shows how to add features
- [ ] Troubleshooting covers common issues
- [ ] ADRs explain key decisions

### Deployment
- [ ] Docker image builds
- [ ] Runs locally via docker-compose
- [ ] IaC templates are valid
- [ ] CI/CD pipeline executes
- [ ] Deployment to target platform succeeds

### Quality
- [ ] Code follows best practices
- [ ] No vendor lock-in
- [ ] Extensible architecture
- [ ] Benchmarks validate performance
- [ ] Users can swap any component

---

## 💡 FINAL NOTES

This is a **comprehensive blueprint**, not a rigid prescription. Adapt based on:

- **Your use case**: Code generation needs different tools than data analysis
- **Your scale**: Local hobby project vs enterprise deployment
- **Your constraints**: Budget, latency, compliance requirements
- **Your expertise**: Start simple, add complexity incrementally

### The Three Commandments

1. **Start Simple**: Build Phase 1 first, validate it works, then expand
2. **Stay Agnostic**: Never lock into one vendor, cloud, or model
3. **Measure Everything**: Benchmark, test, iterate based on data

### Remember

> "The best agent is the one that ships. A simple agent that solves real problems today beats a perfect agent that never launches."

Start with the basic loop, add memory, integrate tools, then scale up. Each phase should be a working system, not a stepping stone.

---
