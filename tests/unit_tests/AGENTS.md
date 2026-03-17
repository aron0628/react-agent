<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# unit_tests

## Purpose

에이전트 개별 모듈의 단위 테스트. API 키 없이 실행 가능하며, CI에서 모든 push/PR에 대해 Python 3.11/3.12 매트릭스로 자동 실행된다.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 테스트 패키지 초기화 (빈 파일) |
| `test_configuration.py` | `Configuration.from_context()` 테스트 — 빈 컨텍스트(LangGraph 런타임 외부)에서 기본값으로 정상 생성되는지 확인. `RuntimeError` catch 경로와 기본값 할당 검증 |

## Test Coverage Status

| 소스 모듈 | 테스트 파일 | 커버리지 |
|----------|-----------|---------|
| `configuration.py` | `test_configuration.py` | 기본 생성만 (from_context 경로) |
| `graph.py` | 없음 | 통합 테스트에서 커버 |
| `state.py` | 없음 | 미테스트 |
| `tools.py` | 없음 | 통합 테스트에서 커버 |
| `prompts.py` | 없음 | 상수만 포함, 테스트 불필요 |
| `utils.py` | 없음 | 미테스트 (현재 미사용 모듈) |

## For AI Agents

### Working In This Directory

- `make test` 또는 `python -m pytest tests/unit_tests/`로 실행
- 새 소스 모듈 추가 시 대응하는 `test_*.py` 파일 작성
- API 호출이 필요한 테스트는 mock 사용 또는 `integration_tests/`에 배치
- ruff의 docstring(`D`) 및 pyupgrade(`UP`) 규칙 비활성화됨
- 테스트 함수명: `test_` 접두사 필수 (pytest convention)

### Testing Requirements

- 외부 의존성(API 키, 네트워크) 없이 실행 가능해야 함
- `pytest` 프레임워크 사용
- 타입 힌트 권장 (`-> None` 반환 타입)

### Suggested Additions

- `test_state.py` — `InputState`, `State` 생성 및 `add_messages` 리듀서 동작 검증
- `test_utils.py` — `get_message_text()` 다양한 입력(str, dict, list) 처리 검증
- `test_graph.py` — mock LLM으로 `call_model`, `route_model_output` 단위 테스트
- `test_configuration.py` 확장 — `from_runnable_config()` 경로, 커스텀 configurable 값 전달 테스트

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
