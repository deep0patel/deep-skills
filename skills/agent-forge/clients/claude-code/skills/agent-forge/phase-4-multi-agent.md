# Phase 4: Multi-Agent Orchestration

**Goal:** Enable swarm intelligence and parallel task execution.

## Architecture: Hive-Mind Pattern

- **Queen agent**: coordinator/strategist — plans and synthesizes
- **Worker agents**: specialists (coder, tester, researcher, analyst, reviewer)
- **Shared memory**: inter-agent communication
- **Task queue**: async work distribution

## 4.1 Swarm Orchestrator

```typescript
class SwarmOrchestrator {
  private queen: Agent;
  private workers: Map<string, Agent> = new Map();
  private taskQueue: TaskQueue;
  private sharedMemory: SharedMemory;

  constructor(config: SwarmConfig) {
    this.queen = new Agent({
      name: "queen", role: "coordinator",
      systemPrompt: QUEEN_PROMPT,
      llmRouter: config.llmRouter,
      memory: config.sharedMemory
    });
    this.taskQueue = new TaskQueue(config.queueBackend);
    this.sharedMemory = config.sharedMemory;
  }

  async execute(task: string): Promise<SwarmResult> {
    const plan = await this.queen.plan(task);             // 1. Plan
    const workers = await this.spawnWorkers(plan);        // 2. Spawn
    const results = await this.executeSwarm(workers, plan); // 3. Execute
    const final = await this.queen.synthesize({ task, plan, workerResults: results }); // 4. Synthesize
    await this.cleanupWorkers(workers);
    return final;
  }

  private async spawnWorkers(plan: ExecutionPlan): Promise<Agent[]> {
    return plan.subtasks.map(subtask => {
      const worker = new Agent({
        name: `worker-${subtask.id}`,
        role: subtask.specialization,
        systemPrompt: WORKER_PROMPTS[subtask.specialization] ?? "You are a general-purpose AI assistant.",
        llmRouter: this.selectOptimalLLM(subtask),
        memory: this.sharedMemory,
        tools: this.filterToolsForRole(subtask.specialization)
      });
      this.workers.set(worker.name, worker);
      return worker;
    });
  }

  private async executeSwarm(workers: Agent[], plan: ExecutionPlan): Promise<Map<string, any>> {
    const results = new Map<string, any>();
    const executionOrder = this.topologicalSort(this.buildDependencyGraph(plan));

    for (const taskBatch of executionOrder) {
      // Execute independent tasks in parallel
      const batchResults = await Promise.all(
        taskBatch.map(async (taskId) => {
          const subtask = plan.subtasks.find(t => t.id === taskId)!;
          const worker = workers.find(w => w.name.includes(taskId))!;
          const context = this.gatherDependencyResults(subtask.dependencies, results);
          const result = await worker.run(subtask.description, context);
          await this.sharedMemory.set(taskId, result);
          return { taskId, result };
        })
      );
      batchResults.forEach(({ taskId, result }) => results.set(taskId, result));
    }
    return results;
  }

  private selectOptimalLLM(subtask: Subtask): LLMRouter {
    if (subtask.estimatedComplexity === "high") return new LLMRouter({ strategy: "performance" });
    if (subtask.estimatedComplexity === "low")  return new LLMRouter({ strategy: "cost" });
    return new LLMRouter({ strategy: "balanced" });
  }
}

const QUEEN_PROMPT = `You are the Queen agent — strategic coordinator for a swarm of specialized AI agents.
Responsibilities: decompose complex tasks, determine parallel/sequential execution, assign specializations, synthesize results.
Output your plan as JSON:
{
  "subtasks": [{ "id": "task-1", "description": "...", "specialization": "coder", "dependencies": [], "estimatedComplexity": "low|medium|high" }],
  "executionStrategy": "parallel|sequential|hybrid"
}`;

const WORKER_PROMPTS: Record<string, string> = {
  coder:    "You are an expert software engineer. Write clean, efficient, well-tested code.",
  tester:   "You are a QA specialist. Write comprehensive tests and find edge cases.",
  researcher: "You are a research analyst. Gather information and provide insights.",
  analyst:  "You are a data analyst. Process data and extract meaningful patterns.",
  reviewer: "You are a code reviewer. Identify issues and suggest improvements."
};
```

## 4.2 Shared Memory

```typescript
class SharedMemory {
  private backend: MemoryBackend;
  private cache: Map<string, any> = new Map();

  async set(key: string, value: any, ttl?: number): Promise<void> {
    this.cache.set(key, value);
    await this.backend.insert("shared_memory", {
      key, value: JSON.stringify(value),
      expires_at: ttl ? new Date(Date.now() + ttl * 1000) : null,
      created_at: new Date()
    });
  }

  async get(key: string): Promise<any> {
    if (this.cache.has(key)) return this.cache.get(key);
    const result = await this.backend.query(
      "SELECT value FROM shared_memory WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
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
    await this.backend.insert("broadcasts", {
      id: generateId(), from: message.from, type: message.type,
      payload: JSON.stringify(message.payload), created_at: new Date()
    });
  }
}
```

## 4.3 Task Queue

```typescript
class TaskQueue {
  private backend: QueueBackend;

  constructor(config: QueueConfig) {
    switch (config.type) {
      case "redis":    this.backend = new RedisQueue(config.url); break;
      case "rabbitmq": this.backend = new RabbitMQQueue(config.url); break;
      case "sqs":      this.backend = new SQSQueue(config.region, config.queueUrl); break;
      default:         this.backend = new InMemoryQueue();
    }
  }

  async enqueue(task: Task, priority = 0): Promise<string> {
    const taskId = generateId();
    await this.backend.push({ id: taskId, ...task, priority, retries: 0, maxRetries: task.maxRetries ?? 3, createdAt: Date.now() });
    return taskId;
  }

  async dequeue(workerType?: string): Promise<Task | null> { return await this.backend.pop(workerType); }
  async ack(taskId: string): Promise<void>  { await this.backend.acknowledge(taskId); }
  async nack(taskId: string, requeue = true): Promise<void> { await this.backend.reject(taskId, requeue); }
}

class RedisQueue implements QueueBackend {
  private client: Redis;
  constructor(url: string) { this.client = new Redis(url); }

  async push(task: QueueTask): Promise<void> {
    await this.client.zadd("task-queue", task.priority, JSON.stringify(task));
  }

  async pop(): Promise<QueueTask | null> {
    const result = await this.client.zpopmax("task-queue");
    if (!result?.length) return null;
    const task = JSON.parse(result[0]);
    await this.client.setex(`processing:${task.id}`, 300, JSON.stringify(task));
    return task;
  }

  async acknowledge(taskId: string): Promise<void> { await this.client.del(`processing:${taskId}`); }

  async reject(taskId: string, requeue: boolean): Promise<void> {
    const taskData = await this.client.get(`processing:${taskId}`);
    if (!taskData) return;
    const task = JSON.parse(taskData);
    if (requeue && task.retries < task.maxRetries) { task.retries++; await this.push(task); }
    await this.client.del(`processing:${taskId}`);
  }
}
```
