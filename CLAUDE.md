# Project: react-agent

LangGraph ReAct Agent 템플릿. 추론-행동(Reasoning-Action) 루프를 통해 사용자 질의에 도구를 활용하여 답변하는 AI 에이전트.

## Quick Reference

- **Language**: Python >= 3.11
- **Package Manager**: uv
- **Framework**: LangGraph + LangChain
- **Default Model**: gpt-4.1 (ChatOpenAI)
- **Entry Point**: `src/react_agent/graph.py:graph`

## Commands

| Command | Description |
|---------|-------------|
| `make test` | 단위 테스트 실행 (`pytest tests/unit_tests/`) |
| `make lint` | 린트 (ruff check + ruff format --diff + mypy --strict) |
| `make format` | 코드 포맷팅 (ruff format + isort) |
| `python -m pytest tests/integration_tests/` | 통합 테스트 (API 키 필요) |

## Architecture

```
User Query → [call_model] → tool_calls? ─Yes→ [ToolNode] → [call_model] (loop)
                                         ─No→  __end__
```

### Key Files

| File | Role |
|------|------|
| `src/react_agent/graph.py` | 그래프 정의, `call_model` 노드, 조건부 라우팅, 컴파일 |
| `src/react_agent/configuration.py` | `Configuration` dataclass (model, system_prompt, max_search_results) |
| `src/react_agent/state.py` | `InputState`(messages), `State`(+is_last_step) |
| `src/react_agent/tools.py` | 도구 등록 (`TOOLS` 리스트). 현재 Tavily 검색만 포함 |
| `src/react_agent/prompts.py` | 시스템 프롬프트 상수 (`{system_time}` 포맷 변수) |
| `src/react_agent/utils.py` | `get_message_text()`, `load_chat_model()` 유틸리티 |
| `langgraph.json` | LangGraph 배포 설정 (그래프 진입점 지정) |

## Code Conventions

- **Docstring**: Google style (ruff pydocstyle `convention = "google"`)
- **Type checking**: mypy strict 모드 적용
- **Lint rules**: ruff — pycodestyle(E), pyflakes(F), isort(I), pydocstyle(D), no-print(T201), pyupgrade(UP)
- **Async**: 모든 노드 함수와 도구는 `async def`로 정의
- **Configuration pattern**: `Configuration.from_runnable_config(config)` 또는 `Configuration.from_context()` (도구 내부)

## Environment Variables

```bash
# Required
OPENAI_API_KEY=...          # LLM API 키
TAVILY_API_KEY=...          # 웹 검색 도구

# Optional (LangSmith tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ReAct-Agent-Template
```

## Adding New Tools

1. `src/react_agent/tools.py`에 `async def` 함수 작성 (docstring 필수 — LLM에 설명으로 전달됨)
2. `TOOLS` 리스트에 추가
3. 필요 시 `Configuration`에 관련 설정 필드 추가

## Adding New Nodes

1. `src/react_agent/graph.py`에 노드 함수 정의 (`State`, `RunnableConfig` 파라미터)
2. `builder.add_node()` → `builder.add_edge()` 또는 `builder.add_conditional_edges()`로 연결

## Testing

- 단위 테스트: `tests/unit_tests/` — API 키 불필요, mock 사용
- 통합 테스트: `tests/integration_tests/` — 실제 API 호출, `@pytest.mark.asyncio` + LangSmith `@unit`
- VCR 카세트: `tests/cassettes/` — API 응답 캐싱으로 재현성 보장
- 테스트 파일에서는 docstring(D), pyupgrade(UP) 린트 규칙 비활성화

## CI/CD

- **Unit tests** (`unit-tests.yml`): push/PR 시 자동 실행. Python 3.11/3.12 매트릭스, ruff + mypy + codespell + pytest
- **Integration tests** (`integration-tests.yml`): 매일 14:37 UTC 스케줄. GitHub Secrets에서 API 키 주입
