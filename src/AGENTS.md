<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# src

## Purpose

에이전트 소스 코드의 최상위 컨테이너 디렉토리. 실제 구현은 `react_agent/` 패키지에 위치한다.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `react_agent/` | ReAct 에이전트 핵심 구현 패키지 (see `react_agent/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- 이 디렉토리 자체에는 파일이 없음. 모든 코드는 `react_agent/` 하위에 존재
- `pyproject.toml`에서 듀얼 매핑: `"langgraph.templates.react_agent" = "src/react_agent"` + `"react_agent" = "src/react_agent"`
- `make lint`의 mypy strict 대상: `src/` 전체 (`PYTHON_FILES=src/`)
- 새 패키지를 추가할 경우 `pyproject.toml`의 `[tool.setuptools]` `packages`와 `package-dir` 모두 업데이트 필요

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
