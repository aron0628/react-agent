<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

# unit_tests

## Purpose

에이전트 개별 모듈의 단위 테스트. API 키 없이 실행 가능하며, CI에서 모든 push/PR에 대해 자동 실행된다.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 테스트 패키지 초기화 |
| `test_configuration.py` | `Configuration.from_context()` 기본 생성 테스트 — 빈 컨텍스트에서 기본값으로 정상 생성되는지 확인 |

## For AI Agents

### Working In This Directory

- `make test` 또는 `python -m pytest tests/unit_tests/`로 실행
- 새 소스 모듈 추가 시 대응하는 `test_*.py` 파일 작성
- API 호출이 필요한 테스트는 mock 사용 또는 `integration_tests/`에 배치
- ruff의 docstring(`D`) 및 pyupgrade(`UP`) 규칙 비활성화됨

### Testing Requirements

- 외부 의존성 없이 실행 가능해야 함
- `pytest` 프레임워크 사용

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
