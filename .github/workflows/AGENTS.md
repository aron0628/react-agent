<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

# workflows

## Purpose

GitHub Actions CI/CD 워크플로우 정의. 단위 테스트(push/PR 트리거)와 통합 테스트(일일 스케줄) 두 가지 파이프라인을 운영한다.

## Key Files

| File | Description |
|------|-------------|
| `unit-tests.yml` | **CI 파이프라인**: push/PR 시 실행. Python 3.11/3.12 매트릭스. uv로 의존성 설치 → ruff 린트 → mypy strict 타입 체크 → codespell 맞춤법 검사 → pytest 단위 테스트 |
| `integration-tests.yml` | **통합 테스트**: 매일 14:37 UTC(한국 23:37) 스케줄 실행. ANTHROPIC/TAVILY/LANGSMITH API 키를 secrets에서 주입. pytest-asyncio + vcrpy 사용 |

## For AI Agents

### Working In This Directory

- 두 워크플로우 모두 `concurrency` 설정으로 동일 브랜치의 중복 실행 자동 취소
- 패키지 설치는 `uv`를 사용 (`curl`로 설치 후 `uv pip install`)
- 통합 테스트에 필요한 시크릿: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`
- `LANGSMITH_TEST_CACHE=tests/cassettes`로 VCR 캐싱 경로 지정

### Common Patterns

- `strategy.matrix`로 Python 버전 매트릭스 테스트 (3.11, 3.12)
- `cancel-in-progress: true`로 리소스 절약

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
