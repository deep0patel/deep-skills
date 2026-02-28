# Phase 3: Memory System

**Goal:** Implement persistent, searchable memory across sessions.

## Architecture: Three Layers

| Layer | Class | Purpose |
|-------|-------|---------|
| 1 | `EpisodicMemory` | Conversation history with semantic search |
| 2 | `ReflexionMemory` | Success/failure patterns for learning |
| 3 | `SkillLibrary` | Consolidated, reusable patterns |

## 3.1 Memory System Bootstrap

```typescript
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
      case "sqlite":   return new SQLiteBackend(config.path);
      case "postgres": return new PostgresBackend(config.connectionString);
      case "redis":    return new RedisBackend(config.url);
      default: throw new Error(`Unsupported backend: ${config.backend}`);
    }
  }

  private createEmbedder(config: MemoryConfig): Embedder {
    if (config.vectorProvider === "openai") return new OpenAIEmbedder(config.apiKey);
    if (config.vectorProvider === "local")  return new LocalEmbedder(); // transformers.js
    throw new Error(`Unsupported embedder: ${config.vectorProvider}`);
  }
}
```

## 3.2 Layer 1 — Episodic Memory

```typescript
class EpisodicMemory {
  async store(episode: { sessionId: string; messages: Message[]; metadata: Record<string, unknown> }): Promise<void> {
    const embedding = await this.embedder.embed(this.summarizeConversation(episode.messages));
    await this.backend.insert("episodes", {
      id: generateId(),
      session_id: episode.sessionId,
      messages: JSON.stringify(episode.messages),
      embedding,
      metadata: JSON.stringify(episode.metadata),
      created_at: new Date()
    });
  }

  async search(query: string, limit = 10): Promise<Episode[]> {
    const queryEmbedding = await this.embedder.embed(query);
    const results = await this.backend.vectorSearch({
      table: "episodes", vector: queryEmbedding, limit, minSimilarity: 0.7
    });
    return results.map(r => ({
      sessionId: r.session_id,
      messages: JSON.parse(r.messages),
      metadata: JSON.parse(r.metadata),
      similarity: r.similarity
    }));
  }
}
```

## 3.3 Layer 2 — Reflexion Memory

```typescript
class ReflexionMemory {
  async storeEpisode(episode: {
    task: string; solution: string; success: boolean; reward: number; critique: string; context: any;
  }): Promise<void> {
    const embedding = await this.embedder.embed(episode.task);
    await this.backend.insert("reflexion_episodes", { id: generateId(), ...episode, embedding, created_at: new Date() });

    const count = await this.backend.count("reflexion_episodes");
    if (count % 100 === 0) await this.consolidatePatterns(); // periodic learning
  }

  async retrieveRelevant(task: string, minReward = 0.7): Promise<ReflexionEpisode[]> {
    const embedding = await this.embedder.embed(task);
    return await this.backend.query(`
      SELECT * FROM reflexion_episodes
      WHERE success = 1 AND reward >= ? AND cosine_similarity(embedding, ?) > 0.7
      ORDER BY reward DESC, created_at DESC LIMIT 10
    `, [minReward, embedding]);
  }

  private async consolidatePatterns(): Promise<void> {
    const episodes = await this.backend.query(`
      SELECT * FROM reflexion_episodes WHERE success = 1 AND reward > 0.8
    `);
    const clusters = this.clusterEpisodes(episodes);
    for (const cluster of clusters) {
      if (cluster.length >= 3) await this.skills.register(this.extractSkill(cluster));
    }
  }
}
```

## 3.4 Layer 3 — Skill Library

```typescript
class SkillLibrary {
  async register(skill: { name: string; description: string; trigger: string; implementation: string; metadata: Record<string, unknown> }): Promise<void> {
    const embedding = await this.embedder.embed(`${skill.name} ${skill.description} ${skill.trigger}`);
    await this.backend.insert("skills", { id: generateId(), ...skill, embedding, usage_count: 0, success_rate: 0.0, created_at: new Date() });
  }

  async search(query: string, limit = 5): Promise<Skill[]> {
    const embedding = await this.embedder.embed(query);
    return await this.backend.vectorSearch({ table: "skills", vector: embedding, limit, minSimilarity: 0.75 });
  }

  async recordUsage(skillId: string, success: boolean): Promise<void> {
    await this.backend.execute(`
      UPDATE skills SET usage_count = usage_count + 1,
        success_rate = (success_rate * usage_count + ?) / (usage_count + 1)
      WHERE id = ?
    `, [success ? 1.0 : 0.0, skillId]);
  }
}
```

## 3.5 SQLite Backend (Local/Dev)

```typescript
class SQLiteBackend implements MemoryBackend {
  private db: Database;

  constructor(path: string) {
    this.db = new Database(path);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY, session_id TEXT, messages TEXT,
        embedding BLOB, metadata TEXT, created_at DATETIME
      );
      CREATE TABLE IF NOT EXISTS reflexion_episodes (
        id TEXT PRIMARY KEY, task TEXT, solution TEXT, success INTEGER,
        reward REAL, critique TEXT, context TEXT, embedding BLOB, created_at DATETIME
      );
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY, name TEXT, description TEXT, trigger TEXT,
        implementation TEXT, metadata TEXT, embedding BLOB,
        usage_count INTEGER DEFAULT 0, success_rate REAL DEFAULT 0.0, created_at DATETIME
      );
      CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
      CREATE INDEX IF NOT EXISTS idx_reflexion_success ON reflexion_episodes(success, reward);
    `);
  }

  async vectorSearch(params: VectorSearchParams): Promise<any[]> {
    // Brute-force cosine similarity — use sqlite-vss or pgvector for production
    const all = await this.query(`SELECT * FROM ${params.table}`);
    return all
      .map(row => ({ ...row, similarity: this.cosineSimilarity(params.vector, row.embedding) }))
      .filter(r => r.similarity >= params.minSimilarity)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, params.limit);
  }
}
```

## 3.6 PostgreSQL Backend (Production)

```typescript
class PostgresBackend implements MemoryBackend {
  private pool: Pool;

  constructor(connectionString: string) {
    this.pool = new Pool({ connectionString });
    this.initialize();
  }

  private async initialize(): Promise<void> {
    await this.execute("CREATE EXTENSION IF NOT EXISTS vector");
    await this.execute(`
      CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY, session_id TEXT, messages TEXT,
        embedding vector(1536), metadata JSONB, created_at TIMESTAMP DEFAULT NOW()
      );
      CREATE TABLE IF NOT EXISTS reflexion_episodes (
        id TEXT PRIMARY KEY, task TEXT, solution TEXT, success BOOLEAN,
        reward REAL, critique TEXT, context JSONB,
        embedding vector(1536), created_at TIMESTAMP DEFAULT NOW()
      );
      CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY, name TEXT, description TEXT, trigger TEXT,
        implementation TEXT, metadata JSONB, embedding vector(1536),
        usage_count INTEGER DEFAULT 0, success_rate REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT NOW()
      );
      -- HNSW index for fast ANN search
      CREATE INDEX IF NOT EXISTS episodes_embedding_idx ON episodes USING hnsw (embedding vector_cosine_ops);
      CREATE INDEX IF NOT EXISTS skills_embedding_idx ON skills USING hnsw (embedding vector_cosine_ops);
    `);
  }

  async vectorSearch(params: VectorSearchParams): Promise<any[]> {
    const result = await this.query(`
      SELECT *, (embedding <=> $1::vector) AS distance
      FROM ${params.table}
      WHERE (embedding <=> $1::vector) < $2
      ORDER BY embedding <=> $1::vector
      LIMIT $3
    `, [`[${params.vector.join(",")}]`, 1 - params.minSimilarity, params.limit]);

    return result.rows.map(row => ({ ...row, similarity: 1 - row.distance }));
  }
}
```
