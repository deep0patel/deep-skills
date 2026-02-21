#!/usr/bin/env bash
set -euo pipefail

# Auto-generates marketplace.json from skill.yaml files
# Run this after adding/updating skills

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILLS_DIR="$REPO_ROOT/skills"
MARKETPLACE="$REPO_ROOT/.claude-plugin/marketplace.json"

echo "Syncing marketplace.json..."

# Build plugins array by reading each skill.yaml
plugins="["
first=true

for skill_dir in "$SKILLS_DIR"/*/; do
  skill_name=$(basename "$skill_dir")
  yaml_file="$skill_dir/skill.yaml"

  if [ ! -f "$yaml_file" ]; then
    echo "  SKIP: $skill_name (no skill.yaml)"
    continue
  fi

  # Parse fields from skill.yaml (simple grep-based, no yq dependency)
  name=$(grep '^name:' "$yaml_file" | head -1 | sed 's/name: *//')
  description=$(grep '^description:' "$yaml_file" | head -1 | sed 's/description: *//')

  if [ "$first" = true ]; then
    first=false
  else
    plugins+=","
  fi

  plugins+="{\"name\":\"$name\",\"source\":\"./skills/$skill_name/clients/claude-code\",\"description\":\"$description\"}"

  echo "  ADD: $skill_name"
done

plugins+="]"

# Write marketplace.json
cat > "$MARKETPLACE" << JSONEOF
{
  "name": "deep-skills",
  "description": "Curated AI coding skills by Deep Patel",
  "author": {
    "name": "Deep Patel",
    "github": "deep0patel"
  },
  "version": "1.0.0",
  "plugins": $plugins
}
JSONEOF

echo "Done. Updated $MARKETPLACE"
