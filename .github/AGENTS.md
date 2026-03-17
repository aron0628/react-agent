<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# .github

## Purpose

GitHub Actions CI/CD 워크플로우 설정 디렉토리. 코드 품질(린트, 타입 체크, 맞춤법)과 테스트(단위, 통합)를 자동화한다.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `workflows/` | GitHub Actions 워크플로우 YAML 정의 (see `workflows/AGENTS.md`) |

## For AI Agents

### Working In This Directory

- 워크플로우 수정 시 YAML 문법 주의 (들여쓰기 기반)
- 새 시크릿 추가 시 GitHub 리포지토리 Settings > Secrets에서 등록 필요
- 워크플로우 테스트는 `workflow_dispatch` 트리거로 수동 실행 가능

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
