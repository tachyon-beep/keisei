# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Keisei project.

## Active Workflows

### Claude Code (`claude.yml`)
- **Triggers**: When `@claude` is mentioned in issues, PR comments, or reviews
- **Purpose**: Interactive AI assistance for development tasks

### Claude Code Review (`claude-code-review.yml`)
- **Triggers**: When PRs are opened/updated with Python code changes
- **Purpose**: Automated code review

## Disabled Workflows

### CI Pipeline (`ci.yml.disabled`)
- **Status**: Disabled (rename to `ci.yml` to re-enable)
- **Contains**: Lint, type check, test suite with coverage, integration tests, security scans

## Setup Requirements

The Claude workflows require:
- `CLAUDE_CODE_OAUTH_TOKEN` secret configured in repository settings
