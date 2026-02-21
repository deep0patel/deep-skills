#!/usr/bin/env bash
set -euo pipefail

# Validate all skills in the marketplace
# Checks: skill.yaml exists, prompt.md exists, required client dirs exist

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
ERRORS=0

echo "Validating deep-skills marketplace..."
echo ""

for skill_dir in "$SKILLS_DIR"/*/; do
  skill_name=$(basename "$skill_dir")
  echo "--- $skill_name ---"

  # Check skill.yaml
  if [ ! -f "$skill_dir/skill.yaml" ]; then
    echo "  ERROR: Missing skill.yaml"
    ERRORS=$((ERRORS + 1))
  else
    echo "  OK: skill.yaml"
  fi

  # Check prompt.md
  if [ ! -f "$skill_dir/prompt.md" ]; then
    echo "  ERROR: Missing prompt.md"
    ERRORS=$((ERRORS + 1))
  else
    echo "  OK: prompt.md"
  fi

  # Check README.md
  if [ ! -f "$skill_dir/README.md" ]; then
    echo "  WARN: Missing README.md"
  else
    echo "  OK: README.md"
  fi

  # Check clients directory
  if [ ! -d "$skill_dir/clients" ]; then
    echo "  ERROR: Missing clients/ directory"
    ERRORS=$((ERRORS + 1))
  else
    # Check Claude Code client
    if [ -d "$skill_dir/clients/claude-code" ]; then
      if [ ! -f "$skill_dir/clients/claude-code/.claude-plugin/plugin.json" ]; then
        echo "  ERROR: claude-code missing .claude-plugin/plugin.json"
        ERRORS=$((ERRORS + 1))
      else
        echo "  OK: claude-code"
      fi
    fi

    # Check Cursor client
    if [ -d "$skill_dir/clients/cursor" ]; then
      mdc_count=$(find "$skill_dir/clients/cursor" -name "*.mdc" | wc -l | tr -d ' ')
      if [ "$mdc_count" -eq 0 ]; then
        echo "  ERROR: cursor missing .mdc file"
        ERRORS=$((ERRORS + 1))
      else
        echo "  OK: cursor"
      fi
    fi

    # Check Copilot client
    if [ -d "$skill_dir/clients/copilot" ]; then
      if [ ! -f "$skill_dir/clients/copilot/.github/copilot-instructions.md" ]; then
        echo "  ERROR: copilot missing .github/copilot-instructions.md"
        ERRORS=$((ERRORS + 1))
      else
        echo "  OK: copilot"
      fi
    fi

    # Check Windsurf client
    if [ -d "$skill_dir/clients/windsurf" ]; then
      ws_count=$(find "$skill_dir/clients/windsurf/.windsurf/rules" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$ws_count" -eq 0 ]; then
        echo "  ERROR: windsurf missing .windsurf/rules/*.md"
        ERRORS=$((ERRORS + 1))
      else
        echo "  OK: windsurf"
      fi
    fi

    # Check Cline client
    if [ -d "$skill_dir/clients/cline" ]; then
      cl_count=$(find "$skill_dir/clients/cline/.cline/rules" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
      if [ "$cl_count" -eq 0 ]; then
        echo "  ERROR: cline missing .cline/rules/*.md"
        ERRORS=$((ERRORS + 1))
      else
        echo "  OK: cline"
      fi
    fi
  fi

  echo ""
done

# Check marketplace.json
echo "--- marketplace.json ---"
if [ ! -f "$REPO_ROOT/.claude-plugin/marketplace.json" ]; then
  echo "  ERROR: Missing .claude-plugin/marketplace.json"
  ERRORS=$((ERRORS + 1))
else
  echo "  OK: marketplace.json"
fi

echo ""
if [ $ERRORS -gt 0 ]; then
  echo "FAILED: $ERRORS error(s) found"
  exit 1
else
  echo "PASSED: All checks passed"
fi
