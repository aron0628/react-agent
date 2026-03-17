<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# workflows

## Purpose

GitHub Actions CI/CD 워크플로우 정의. 단위 테스트(push/PR 트리거)와 통합 테스트(일일 스케줄) 두 가지 파이프라인을 운영한다.

## Key Files

| File | Description |
|------|-------------|
| `unit-tests.yml` | **CI 파이프라인** (`CI`). push(main)/PR/수동 트리거. Python 3.11/3.12 매트릭스 × ubuntu-latest. 5단계: uv 설치+의존성 → ruff check(전체) → mypy strict(`src/`) → codespell(README.md + `src/`) → pytest(`tests/unit_tests`). concurrency 그룹으로 동일 브랜치 중복 실행 자동 취소 |
| `integration-tests.yml` | **통합 테스트** (`Integration Tests`). 매일 14:37 UTC(한국 23:37)/수동 트리거. Python 3.11/3.12 매트릭스. 추가 의존성 `pytest-asyncio`, `vcrpy` 설치. 환경변수: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`(secrets), `LANGSMITH_TRACING=true`, `LANGSMITH_TEST_CACHE=tests/cassettes` |

## Pipeline Comparison

| 항목 | unit-tests.yml | integration-tests.yml |
|-----|---------------|----------------------|
| 이름 | CI | Integration Tests |
| 트리거 | push(main), PR, 수동 | 매일 14:37 UTC, 수동 |
| Python | 3.11, 3.12 | 3.11, 3.12 |
| 린트 | ruff, mypy, codespell | 없음 |
| 테스트 | `tests/unit_tests` | `tests/integration_tests` |
| API 키 | 불필요 | `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY` |
| 추가 의존성 | ruff, mypy | pytest-asyncio, vcrpy |
| concurrency | 워크플로우+ref 그룹, cancel-in-progress | 워크플로우+ref 그룹, cancel-in-progress |

## For AI Agents

### Working In This Directory

- 두 워크플로우 모두 `concurrency` 설정으로 동일 브랜치의 중복 실행 자동 취소 (`cancel-in-progress: true`)
- 패키지 설치는 `uv`를 사용 (`curl -LsSf https://astral.sh/uv/install.sh | sh` → `uv venv` → `uv pip install`)
- 통합 테스트에 필요한 시크릿: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`
- `LANGSMITH_TEST_CACHE=tests/cassettes`로 VCR 캐싱 경로 지정
- `actions/checkout@v4`, `actions/setup-python@v4` 사용

### Common Patterns

- `strategy.matrix`로 Python 버전 매트릭스 테스트 (3.11, 3.12)
- `cancel-in-progress: true`로 리소스 절약
- `workflow_dispatch`로 수동 트리거 지원 (디버깅, 긴급 실행용)
- 린트는 단위 테스트 워크플로우에서만 실행 (통합 테스트는 기능 검증에 집중)

### Note

- 통합 테스트 CI에서 `ANTHROPIC_API_KEY`를 주입하지만, 현재 소스 코드는 `ChatOpenAI` (OpenAI API)만 사용. 모델을 Anthropic으로 전환하거나 `langchain-anthropic` 추가 시 활용 가능. 현재는 템플릿 잔여 설정으로 보임

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
