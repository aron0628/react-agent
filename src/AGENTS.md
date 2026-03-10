<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

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
- `pyproject.toml`에서 `packages = ["react_agent"]`로 매핑됨

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
