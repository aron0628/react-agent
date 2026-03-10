<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

# tests

## Purpose

에이전트의 단위 테스트와 통합 테스트를 포함하는 디렉토리. pytest 프레임워크를 사용하며, 통합 테스트는 LangSmith의 `@unit` 데코레이터와 VCR 카세트(cassette)를 통한 응답 캐싱을 지원한다.

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `unit_tests/` | 개별 모듈의 단위 테스트 (see `unit_tests/AGENTS.md`) |
| `integration_tests/` | 전체 그래프 실행 통합 테스트 (see `integration_tests/AGENTS.md`) |
| `cassettes/` | VCR 카세트 파일 — API 응답 녹화본으로 통합 테스트 재현성 보장 |

## For AI Agents

### Working In This Directory

- 단위 테스트: `make test` 또는 `python -m pytest tests/unit_tests/`
- 통합 테스트: `python -m pytest tests/integration_tests/` (실제 API 키 필요)
- 테스트 파일에서는 ruff의 `D`(docstring), `UP`(pyupgrade) 규칙 비활성화
- 통합 테스트는 `pytest-asyncio`와 `vcrpy` 추가 의존성 필요

### Testing Requirements

- 새 기능 추가 시 `unit_tests/`에 대응하는 테스트 파일 작성
- 그래프 동작 변경 시 `integration_tests/test_graph.py` 업데이트 및 카세트 재녹화 필요

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
