# Distributing agent-eval via Azure Artifacts

This document describes how `agent-eval` is published from a personal GitHub repository to the company's Azure DevOps Artifacts feed, so that company developers can install it with `pip install agent-eval`.

## Architecture

```
Personal GitHub (source of truth)
        |
        |-- GitHub Actions (CI) --> build wheel --> push to Azure Artifacts
        |
        +-- (optional) mirror to Azure DevOps Repos

Azure Artifacts (private PyPI feed)
        |
        +-- Company developers: pip install agent-eval
```

## Configuration

- **Azure DevOps org:** `storaenso-data-services`
- **Azure DevOps project:** `Data Science Products and Projects`
- **Feed name:** `python-internal` (project-scoped)
- **Feed URL (upload):** `https://pkgs.dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_packaging/python-internal/pypi/upload/`
- **Feed URL (install):** `https://pkgs.dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_packaging/python-internal/pypi/simple/`
- **GitHub secret:** `AZURE_ARTIFACTS_PAT` (PAT with Packaging Read & Write scope)

## One-time setup (already completed)

### 1. Azure Artifacts feed

Created `python-internal` as a project-scoped feed under **Data Science Products and Projects** at:
https://dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_artifacts

### 2. Personal Access Token (PAT)

Created in Azure DevOps → User Settings → Personal Access Tokens with **Packaging → Read & Write** scope, scoped to the `storaenso-data-services` organization.

### 3. GitHub secret

Added `AZURE_ARTIFACTS_PAT` in GitHub repo → Settings → Secrets and variables → Actions.

## Publishing workflow

Publishing is automated via GitHub Actions (see `.github/workflows/publish-azure-artifacts.yml`).

The workflow triggers when you push a version tag:

```bash
# 1. Bump version in pyproject.toml
# 2. Commit the change
# 3. Tag and push
git tag v0.2.0
git push origin v0.2.0
```

GitHub Actions will then:
1. Build the wheel and sdist.
2. Upload them to the Azure Artifacts feed via `twine`.

## Consumer setup (company developers)

### pip configuration

Developers need to configure pip to use the Azure Artifacts feed as an extra index.

**Option A: pip config (recommended for local dev)**

```bash
pip config set global.extra-index-url https://az:{PAT}@pkgs.dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_packaging/python-internal/pypi/simple/
```

**Option B: per-project requirements.txt**

```
--extra-index-url https://pkgs.dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_packaging/python-internal/pypi/simple/
agent-eval==0.1.0
```

Authentication is handled via `keyring`, environment variables, or inline PAT.

### Installing the package

```bash
# Base install
pip install agent-eval

# With extras
pip install agent-eval[server]
pip install agent-eval[rl]
pip install agent-eval[all]
```

### Docker usage

```dockerfile
FROM python:3.10-slim

ARG FEED_PAT
RUN pip install agent-eval[server] \
    --extra-index-url https://az:${FEED_PAT}@pkgs.dev.azure.com/storaenso-data-services/Data%20Science%20Products%20and%20Projects/_packaging/python-internal/pypi/simple/

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
```

Pass the PAT at build time:

```bash
docker build --build-arg FEED_PAT=<your-pat> .
```

In `docker-compose.yml`:

```yaml
services:
  app:
    build:
      context: .
      args:
        FEED_PAT: ${AZURE_ARTIFACTS_PAT}
```

### Azure DevOps Pipelines (CI/CD)

In company pipelines, use the built-in feed authentication:

```yaml
steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.10'

  - task: PipAuthenticate@1
    inputs:
      artifactFeeds: 'Data Science Products and Projects/python-internal'

  - script: pip install agent-eval[server]
```

## Optional: Mirror repo to Azure DevOps

To make the source code visible in Azure DevOps (for auditing, code search, etc.), the GitHub Actions workflow includes an optional mirror step. This keeps Azure DevOps as a **read-only mirror** while your personal GitHub remains the source of truth.

To enable it, uncomment the mirror job in `.github/workflows/publish-azure-artifacts.yml` and update the project name in the Azure DevOps repo URL.

## Versioning

Follow semver. The version is defined in `pyproject.toml`:

```toml
[project]
version = "0.1.0"
```

Bump it before tagging a release. The tag name must match the pattern `v*` (e.g., `v0.1.0`, `v0.2.0`).
