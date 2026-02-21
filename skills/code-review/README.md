# code-review

Expert code reviewer that catches bugs, security issues, performance problems, and enforces best practices.

## What It Does

- Identifies bugs, logic errors, and edge cases
- Flags security vulnerabilities (OWASP top 10)
- Spots performance bottlenecks
- Enforces code quality and naming conventions
- Provides actionable fix suggestions with severity levels

## Installation

See the [main README](../../README.md) for installation instructions per client.

## Usage

### Claude Code

After installing the plugin, the skill is automatically available. Ask Claude to review your code:

```
Review this file for bugs and security issues
```

### Cursor / Copilot / Windsurf / Cline

After adding the rule file, the AI will apply code review guidelines when you ask it to review code.

## Review Output Format

Issues are reported with:
- **Location** — file and line
- **Severity** — Critical / Warning / Suggestion
- **Issue** — what's wrong
- **Fix** — concrete code fix
