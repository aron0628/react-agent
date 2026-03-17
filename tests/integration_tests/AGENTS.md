<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# integration_tests

## Purpose

전체 ReAct 에이전트 그래프의 엔드투엔드 통합 테스트. 실제 LLM API와 Tavily 검색을 호출하여 에이전트의 전체 추론-행동 사이클을 검증한다. LangSmith의 `@unit` 데코레이터를 통해 테스트 결과를 추적한다.

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 테스트 패키지 초기화 (빈 파일) |
| `test_graph.py` | `graph.ainvoke()` E2E 테스트. "Who is the founder of LangChain?" 질문 → 에이전트가 Tavily 검색 실행 → "harrison" 키워드 포함 답변 생성 확인. `configurable`로 `system_prompt` 오버라이드 예시 포함 |

## Test Execution Flow

```
test_react_agent_simple_passthrough()
  │
  ├─ graph.ainvoke({"messages": [("user", "Who is the founder of LangChain?")]})
  │    │
  │    ├─ call_model → ChatOpenAI(gpt-4.1) → tool_calls: [search("LangChain founder")]
  │    ├─ tools → TavilySearch → 검색 결과
  │    ├─ call_model → 최종 응답 생성
  │    └─ return {"messages": [..., AIMessage("Harrison Chase...")]}
  │
  └─ assert "harrison" in res["messages"][-1].content.lower()
```

## For AI Agents

### Working In This Directory

- 실행에 API 키 필요: `OPENAI_API_KEY` (ChatOpenAI), `TAVILY_API_KEY` (웹 검색), `LANGSMITH_API_KEY` (결과 추적)
- CI에서는 `ANTHROPIC_API_KEY`도 secrets에 설정되어 있으나, 현재 코드는 OpenAI만 사용
- 추가 의존성: `pytest-asyncio` (`@pytest.mark.asyncio`), `vcrpy` (카세트 캐싱)
- LangSmith `@unit` 데코레이터로 테스트 결과 자동 추적 — `LANGSMITH_TRACING=true` 필요
- VCR 카세트 캐싱: `LANGSMITH_TEST_CACHE=tests/cassettes` 환경변수로 경로 지정
- CI에서는 매일 14:37 UTC 스케줄로 실행 (수동 `workflow_dispatch`도 가능)

### Testing Requirements

- 새 통합 테스트 추가 시 `@pytest.mark.asyncio`와 `@unit` 데코레이터 적용
- 테스트가 실제 API를 호출하므로 비용 인지 필요
- 카세트 파일 변경 시 커밋에 포함
- 검증은 키워드 존재 확인 패턴: `assert "keyword" in str(res["messages"][-1].content).lower()`

### Common Patterns

- `graph.ainvoke(input, config)` — input은 `{"messages": [...]}`, config는 `{"configurable": {...}}`
- `("user", "message text")` — LangChain 메시지 튜플 축약형 (`HumanMessage` 대체)
- `@unit` — LangSmith 테스트 단위 등록 (실행 이력, 비교, 회귀 감지)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
