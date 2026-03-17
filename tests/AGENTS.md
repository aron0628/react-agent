<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# tests

## Purpose

에이전트의 단위 테스트와 통합 테스트를 포함하는 디렉토리. pytest 프레임워크를 사용하며, 통합 테스트는 LangSmith의 `@unit` 데코레이터와 VCR 카세트(cassette)를 통한 응답 캐싱을 지원한다.

## Test Strategy

```
tests/
├── unit_tests/          ← 빠른 검증, API 키 불필요, CI에서 모든 push/PR 실행
│   └── test_configuration.py
├── integration_tests/   ← 전체 그래프 E2E, API 키 필요, 일일 스케줄 실행
│   └── test_graph.py
└── cassettes/           ← VCR 카세트로 API 응답 캐싱
    └── *.yaml
```

| 테스트 유형 | 실행 방법 | API 키 | CI 트리거 | 소요 시간 |
|------------|----------|--------|----------|----------|
| 단위 테스트 | `make test` | 불필요 | push/PR | 수 초 |
| 통합 테스트 | `pytest tests/integration_tests/` | 필수 | 매일 14:37 UTC | 수십 초 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `unit_tests/` | 개별 모듈의 단위 테스트 — mock 기반, 외부 의존성 없음 (see `unit_tests/AGENTS.md`) |
| `integration_tests/` | 전체 그래프 실행 통합 테스트 — 실제 API 호출 (see `integration_tests/AGENTS.md`) |
| `cassettes/` | VCR 카세트 파일 — LangSmith `LANGSMITH_TEST_CACHE` 경로로 API 응답 녹화본 저장. 통합 테스트 재현성 보장. YAML 형식, UUID 기반 파일명 |

## For AI Agents

### Working In This Directory

- 단위 테스트: `make test` 또는 `python -m pytest tests/unit_tests/`
- 통합 테스트: `python -m pytest tests/integration_tests/` (실제 API 키 필요)
- 테스트 파일에서는 ruff의 `D`(docstring), `UP`(pyupgrade) 규칙 비활성화 (`pyproject.toml` `[tool.ruff.lint.per-file-ignores]` 참조)
- 통합 테스트 실행 시 추가 의존성: `pytest-asyncio`, `vcrpy` (`uv pip install -U pytest-asyncio vcrpy`)

### Testing Requirements

- 새 소스 모듈 추가 시 `unit_tests/`에 대응하는 `test_*.py` 파일 작성
- 그래프 동작 변경 시 `integration_tests/test_graph.py` 업데이트 및 카세트 재녹화 필요
- 카세트 파일이 변경되면 커밋에 포함
- 단위 테스트는 외부 API 호출 없이 실행 가능해야 함 (mock 사용)

### Common Patterns

- `@pytest.mark.asyncio` — 비동기 테스트 함수 표시 (통합 테스트)
- `@unit` (LangSmith) — 테스트 결과를 LangSmith에 자동 추적
- `graph.ainvoke({"messages": [...]}, {"configurable": {...}})` — 그래프 실행 패턴
- 응답 검증: `res["messages"][-1].content`에서 키워드 존재 확인

## Dependencies

### Internal

- `src/react_agent/` — 테스트 대상 패키지
- `src/react_agent/configuration.py` — `Configuration.from_context()` 단위 테스트
- `src/react_agent/graph.py:graph` — 통합 테스트에서 `graph.ainvoke()` 호출

### External

| Package | Purpose |
|---------|---------|
| `pytest` | 테스트 프레임워크 |
| `pytest-asyncio` | 비동기 테스트 지원 (통합 테스트) |
| `vcrpy` | API 응답 카세트 녹화/재생 (통합 테스트) |
| `langsmith` | `@unit` 데코레이터로 테스트 추적 (통합 테스트) |

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
