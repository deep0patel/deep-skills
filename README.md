# deep-skills

Curated AI coding skills by [Deep Patel](https://github.com/deep0patel). One skill, every tool.

Write once, use everywhere — Claude Code, Cursor, GitHub Copilot, Windsurf, Cline.

## Installation

### Claude Code (Plugin)

```bash
/plugin marketplace add deep0patel/deep-skills
/plugin install <skill-name>@deep-skills
```

### Cursor

Copy the `.mdc` file from any skill's `clients/cursor/` into your project's `.cursor/rules/`:

```bash
curl -o .cursor/rules/agent-forge.mdc https://raw.githubusercontent.com/deep0patel/deep-skills/main/skills/agent-forge/clients/cursor/agent-forge.mdc
```

### GitHub Copilot

Copy from `clients/copilot/`:

```bash
curl -o .github/copilot-instructions.md https://raw.githubusercontent.com/deep0patel/deep-skills/main/skills/agent-forge/clients/copilot/.github/copilot-instructions.md
```

### Windsurf

Copy from `clients/windsurf/`:

```bash
curl -o .windsurf/rules/agent-forge.md https://raw.githubusercontent.com/deep0patel/deep-skills/main/skills/agent-forge/clients/windsurf/.windsurf/rules/agent-forge.md
```

### Cline

Copy from `clients/cline/`:

```bash
curl -o .cline/rules/agent-forge.md https://raw.githubusercontent.com/deep0patel/deep-skills/main/skills/agent-forge/clients/cline/.cline/rules/agent-forge.md
```

## Available Skills

| Skill | Description | Version |
|-------|-------------|---------|
| [agent-forge](./skills/agent-forge/) | Expert AI agent architect for building production-grade, vendor-agnostic AI agent systems | 1.0.0 |

## Structure

```
skills/<skill-name>/
├── skill.yaml          # Metadata (name, version, tags, description)
├── prompt.md           # Core prompt — single source of truth
├── README.md           # Usage docs and examples
└── clients/            # Tool-specific adaptations
    ├── claude-code/    # Claude Code plugin format
    ├── cursor/         # Cursor .mdc rule
    ├── copilot/        # GitHub Copilot instructions
    ├── windsurf/       # Windsurf rule
    └── cline/          # Cline rule
```

## License

MIT - see [LICENSE](./LICENSE)
