# Project: react-agent

LangGraph ReAct Agent 템플릿. 추론-행동(Reasoning-Action) 루프를 통해 사용자 질의에 도구를 활용하여 답변하는 AI 에이전트.

## Quick Reference

- **Language**: Python >= 3.11
- **Package Manager**: uv
- **Framework**: LangGraph >= 0.6.10 + LangChain >= 0.3.27
- **Default Model**: gpt-4.1 (ChatOpenAI, temperature=0.1)
- **Entry Point**: `src/react_agent/graph.py:graph`
- **Deployment Config**: `langgraph.json` (env: `.env`)

## Commands

| Command | Description |
|---------|-------------|
| `make test` | 단위 테스트 실행 (`pytest tests/unit_tests/`) |
| `make lint` | 린트 (ruff check + ruff format --diff + mypy --strict) |
| `make format` | 코드 포맷팅 (ruff format + isort) |
| `make spell_check` | codespell 맞춤법 검사 |
| `python -m pytest tests/integration_tests/` | 통합 테스트 (API 키 필요) |

## Architecture

```
User Query → [summarize_conversation] → [call_model] → tool_calls? ─Yes→ [ToolNode] → [call_model] (loop)
                                                                   ─No→  __end__
```

### Execution Flow

1. `summarize_conversation`: checks message count and summarizes if needed
2. `call_model`: Configuration 로드 → `ChatOpenAI.bind_tools(TOOLS)` → 시스템 프롬프트(`{system_time}` 포맷) + 메시지 → LLM 호출
3. `route_model_output`: `AIMessage.tool_calls` 존재 여부로 분기
4. `tools` (ToolNode): tool_calls 병렬 실행, `ToolMessage`로 반환
5. `is_last_step` 가드: 재귀 한계 도달 시 도구 호출 대신 사과 메시지 반환

### Module Dependency

```
graph.py → configuration.py → prompts.py
graph.py → state.py
graph.py → tools.py → configuration.py
utils.py  (독립 — 현재 미사용, 다중 프로바이더 확장용)
```

### Key Files

| File | Role |
|------|------|
| `src/react_agent/graph.py` | **핵심**. StateGraph 정의, `call_model` 노드, `route_model_output` 조건부 라우터, 그래프 컴파일 |
| `src/react_agent/configuration.py` | `Configuration` dataclass — `model`(gpt-4.1), `system_prompt`, `max_search_results`(10). 팩토리: `from_runnable_config()` (노드), `from_context()` (도구) |
| `src/react_agent/state.py` | `InputState`(messages + add_messages 리듀서), `State`(+is_last_step managed 변수) |
| `src/react_agent/tools.py` | 도구 등록 (`TOOLS` 리스트). 현재 Tavily 검색(`search`)만 포함. `Configuration.from_context()`로 설정 접근 |
| `src/react_agent/prompts.py` | 시스템 프롬프트 상수 (`SYSTEM_PROMPT`, `{system_time}` 포맷 변수) |
| `src/react_agent/utils.py` | `get_message_text()`, `load_chat_model()` 유틸리티 (현재 미사용) |
| `langgraph.json` | LangGraph 배포 설정 — 그래프 진입점 `./src/react_agent/graph.py:graph`, env `.env` |

## Code Conventions

- **Docstring**: Google style (ruff pydocstyle `convention = "google"`, `D417` 제외)
- **Type checking**: mypy strict 모드 적용 — 모든 함수에 타입 힌트 필수
- **Lint rules**: ruff — pycodestyle(E), pyflakes(F), isort(I), pydocstyle(D), no-print(T201), pyupgrade(UP). `UP006`, `UP007`, `UP035` 무시
- **Async**: 모든 노드 함수와 도구는 `async def`로 정의
- **Configuration pattern**: `Configuration.from_runnable_config(config)` (노드 함수) 또는 `Configuration.from_context()` (도구 함수 — LangGraph 컨텍스트 자동 해석)
- **Dataclass + field metadata**: `metadata={"description": ...}`로 LangGraph Studio UI 설명 표시
- **Annotated 템플릿 메타데이터**: `Annotated[str, {"__template_metadata__": {"kind": "llm"}}]`로 LangGraph 모델 선택 UI 힌트
- **cast() 패턴**: `model.ainvoke()` 반환값을 `cast(AIMessage, ...)` — mypy strict 호환
- **InputState/State 분리**: InputState는 외부 노출 인터페이스(messages만), State는 내부 확장(is_last_step 등)
- **듀얼 패키지 매핑**: `pyproject.toml`에서 `langgraph.templates.react_agent` + `react_agent` 동일 소스 매핑

## Environment Variables

```bash
# Required
OPENAI_API_KEY=...          # LLM API 키 (ChatOpenAI)
TAVILY_API_KEY=...          # 웹 검색 도구 (TavilySearch)

# Optional (LangSmith tracing)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=ReAct-Agent-Template
```

> `python-dotenv`는 의존성에 포함되나 소스에서 직접 `load_dotenv()` 호출 없음 — `langgraph.json`의 `"env": ".env"` 설정을 통해 LangGraph 런타임이 자동 로딩

## Adding New Tools

1. `src/react_agent/tools.py`에 `async def` 함수 작성 (docstring 필수 — LLM에 설명으로 전달됨)
2. `TOOLS` 리스트에 추가
3. 도구 내에서 설정 접근 시 `Configuration.from_context()` 사용
4. 필요 시 `Configuration`에 관련 설정 필드 추가 (`field(metadata={"description": ...})`)

## Adding New Nodes

1. `src/react_agent/graph.py`에 `async def node(state: State, config: RunnableConfig)` 정의
2. `builder.add_node(node)` → `builder.add_edge()` 또는 `builder.add_conditional_edges()`로 연결
3. 조건부 라우터는 `Literal["__end__", "next_node"]` 반환 타입으로 정의

## Extending State

1. `src/react_agent/state.py`의 `State` 클래스에 필드 추가
2. 주석 예시 참조: `retrieved_documents: List[Document]`, `extracted_entities: Dict[str, Any]`
3. `InputState`는 외부에 노출하는 필드만, `State`에 내부 처리용 필드 추가

## Testing

- 단위 테스트: `tests/unit_tests/` — API 키 불필요, mock 사용
- 통합 테스트: `tests/integration_tests/` — 실제 API 호출, `@pytest.mark.asyncio` + LangSmith `@unit`
- VCR 카세트: `tests/cassettes/` — `LANGSMITH_TEST_CACHE` 경로로 API 응답 캐싱, 재현성 보장
- 테스트 파일에서는 docstring(D), pyupgrade(UP) 린트 규칙 비활성화
- 그래프 실행 패턴: `graph.ainvoke({"messages": [("user", "질문")]}, {"configurable": {...}})`

### Test Coverage Gaps

| 모듈 | 테스트 상태 |
|------|-----------|
| `configuration.py` | `from_context()` 기본 생성만 테스트됨. `from_runnable_config()`, 커스텀 값 전달 미테스트 |
| `state.py` | 미테스트 |
| `utils.py` | 미테스트 (현재 미사용 모듈) |
| `graph.py` | 통합 테스트에서만 커버 (단위 테스트 없음) |

## CI/CD

- **Unit tests** (`unit-tests.yml`): push(main)/PR/수동 트리거. Python 3.11/3.12 매트릭스. uv → ruff check → mypy strict → codespell → pytest
- **Integration tests** (`integration-tests.yml`): 매일 14:37 UTC(한국 23:37)/수동 트리거. GitHub Secrets에서 API 키 주입. pytest-asyncio + vcrpy 추가 설치
- 두 워크플로우 모두 `concurrency` + `cancel-in-progress: true`로 중복 실행 자동 취소

## Known Issues

- `graph.py:6` — `import os` 미사용 (제거 가능)
- `utils.py` — `get_message_text()`와 `load_chat_model()` 모두 코드베이스에서 호출되지 않음. 다중 프로바이더 지원 시 `load_chat_model()` 활용 가능
- `graph.py:42` — `temperature=0.1` 하드코딩. `Configuration`에 설정 필드로 분리 권장
- CI `integration-tests.yml`에서 `ANTHROPIC_API_KEY`를 secrets로 주입하지만, 실제 코드는 `ChatOpenAI` (OpenAI API)만 사용 — 템플릿 잔여 설정
