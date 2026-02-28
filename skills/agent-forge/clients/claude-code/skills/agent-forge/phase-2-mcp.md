# Phase 2: MCP Integration

**Goal:** Implement full MCP client/server capabilities.

Reference: https://modelcontextprotocol.io/docs/concepts/architecture

## 2.1 MCP Client

```typescript
class MCPClient {
  private servers: Map<string, MCPServerConnection>;

  async initialize(config: MCPServerConfig[]): Promise<void> {
    for (const serverConfig of config) {
      const connection = await this.connectToServer(serverConfig);
      this.servers.set(serverConfig.name, connection);
    }
  }

  private async connectToServer(config: MCPServerConfig): Promise<MCPServerConnection> {
    const proc = spawn(config.command, config.args, {
      env: { ...process.env, ...config.env },
      stdio: ["pipe", "pipe", "pipe"]
    });

    const transport = new StdioTransport(proc.stdin, proc.stdout);
    const client = new Client({ name: "universal-agent", version: "1.0.0" }, {
      capabilities: { tools: {}, prompts: {}, resources: {} }
    });

    await client.connect(transport);
    const capabilities = await client.listTools();
    return { process: proc, client, capabilities, name: config.name };
  }

  async listTools(): Promise<Tool[]> {
    const allTools: Tool[] = [];
    for (const [serverName, connection] of this.servers) {
      const tools = await connection.client.listTools();
      // Namespace tools by server to avoid collisions
      allTools.push(...tools.tools.map(tool => ({
        ...tool,
        name: `${serverName}/${tool.name}`,
        _server: serverName
      })));
    }
    return allTools;
  }

  async callTool(toolName: string, args: Record<string, unknown>): Promise<unknown> {
    const [serverName, actualToolName] = toolName.includes("/")
      ? toolName.split("/", 2)
      : [this.servers.keys().next().value, toolName];

    const connection = this.servers.get(serverName);
    if (!connection) throw new Error(`MCP server '${serverName}' not found`);

    const result = await connection.client.callTool({ name: actualToolName, arguments: args });
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

## 2.2 MCP Server (Expose Agent Capabilities)

```typescript
class MCPServer {
  private server: Server;
  private tools: Map<string, MCPTool>;

  constructor() {
    this.server = new Server({ name: "universal-agent", version: "1.0.0" }, {
      capabilities: { tools: {} }
    });
    this.tools = new Map();
    this.registerDefaultTools();
  }

  private registerDefaultTools(): void {
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
      handler: async (args) => await this.executeCode(args)
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
      handler: async (args) => await this.analyzeRepo(args)
    });
  }

  registerTool(tool: MCPToolDefinition): void {
    this.tools.set(tool.name, tool);
  }

  async start(transport: Transport): Promise<void> {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: Array.from(this.tools.values()).map(t => ({
        name: t.name,
        description: t.description,
        inputSchema: t.inputSchema
      }))
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const tool = this.tools.get(request.params.name);
      if (!tool) throw new Error(`Tool not found: ${request.params.name}`);
      const result = await tool.handler(request.params.arguments);
      return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
    });

    await this.server.connect(transport);
  }
}
```

## 2.3 Tool Registry

```typescript
class ToolRegistry {
  private tools: Map<string, Tool> = new Map();
  private mcpClient: MCPClient;

  async initialize(mcpClient: MCPClient): Promise<void> {
    this.mcpClient = mcpClient;
    this.registerBuiltinTools();
    await this.syncMCPTools();
  }

  private registerBuiltinTools(): void {
    this.register({
      name: "read_file",
      description: "Read contents of a file",
      inputSchema: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"]
      },
      execute: async (args) => {
        const content = await fs.readFile(args.path, "utf-8");
        return { content, size: content.length };
      }
    });

    this.register({
      name: "write_file",
      description: "Write content to a file",
      inputSchema: {
        type: "object",
        properties: { path: { type: "string" }, content: { type: "string" } },
        required: ["path", "content"]
      },
      execute: async (args) => {
        await fs.writeFile(args.path, args.content, "utf-8");
        return { success: true, path: args.path };
      }
    });
    // Add more: list_files, execute_bash, run_tests, git_commit, search_code, etc.
  }

  private async syncMCPTools(): Promise<void> {
    const mcpTools = await this.mcpClient.listTools();
    for (const tool of mcpTools) {
      this.register({
        name: tool.name,
        description: tool.description,
        inputSchema: tool.inputSchema,
        execute: async (args) => await this.mcpClient.callTool(tool.name, args),
        source: "mcp"
      });
    }
  }

  register(tool: Tool): void { this.tools.set(tool.name, tool); }
  get(name: string): Tool | undefined { return this.tools.get(name); }
  list(): Tool[] { return Array.from(this.tools.values()); }

  async execute(name: string, args: Record<string, unknown>): Promise<unknown> {
    const tool = this.get(name);
    if (!tool) throw new Error(`Tool not found: ${name}`);
    this.validateArgs(args, tool.inputSchema);
    return await this.withRetry(() => tool.execute(args), { maxRetries: 2, timeout: 30000 });
  }
}
```
