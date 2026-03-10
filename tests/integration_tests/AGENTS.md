<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

# integration_tests

## Purpose

전체 ReAct 에이전트 그래프의 엔드투엔드 통합 테스트. 실제 LLM API와 Tavily 검색을 호출하여 에이전트의 전체 추론-행동 사이클을 검증한다. LangSmith의 `@unit` 데코레이터를 통해 테스트 결과를 추적한다.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 테스트 패키지 초기화 |
| `test_graph.py` | `graph.ainvoke()`로 전체 에이전트 실행 테스트. "Who is the founder of LangChain?" 질문으로 에이전트가 "harrison"을 포함한 답변 생성 확인 |

## For AI Agents

### Working In This Directory

- 실행에 API 키 필요: `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`
- `pytest-asyncio` 필수 (`@pytest.mark.asyncio` 사용)
- `vcrpy`로 API 응답 캐싱 지원 — 카세트 파일은 `tests/cassettes/`에 저장
- LangSmith `@unit` 데코레이터로 테스트 결과 자동 추적
- CI에서는 매일 스케줄로 실행 (수동 트리거도 가능)

### Testing Requirements

- 새 통합 테스트 추가 시 `@pytest.mark.asyncio`와 `@unit` 데코레이터 적용
- 테스트가 실제 API를 호출하므로 비용 인지 필요
- 카세트 파일 변경 시 커밋에 포함

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
