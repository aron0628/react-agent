<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# react_agent

## Purpose

ReAct 에이전트의 핵심 구현 패키지. LangGraph의 `StateGraph`를 사용하여 추론-행동 루프를 구성하며, LLM 호출(`call_model`)과 도구 실행(`ToolNode`) 사이를 조건부 에지로 순환하는 그래프를 정의한다.

## Architecture

```
                    ┌─────────────┐
                    │  __start__  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌─────│  call_model │◄────────┐
              │     └──────┬──────┘         │
              │            │                │
              │   tool_calls 있음?          │
              │     Yes         No          │
              │      │           │          │
              │ ┌────▼────┐  ┌──▼───┐      │
              │ │  tools   │  │ END  │      │
              │ └────┬─────┘  └──────┘      │
              │      │                      │
              └──────┴──────────────────────┘
```

### 실행 흐름 상세

1. **`__start__` → `call_model`**: 사용자 입력 수신. `InputState.messages`에 `HumanMessage` 추가
2. **`call_model` 내부 처리**:
   - `Configuration.from_runnable_config(config)`로 설정 로드
   - `ChatOpenAI(temperature=0.1, model=config.model)` 인스턴스 생성
   - `.bind_tools(TOOLS)`로 도구 바인딩
   - 시스템 프롬프트에 `{system_time}` 포맷 변수 삽입 (UTC ISO format)
   - LLM 호출 결과를 `AIMessage`로 cast
3. **`route_model_output` 분기**:
   - `AIMessage.tool_calls`가 비어있으면 → `__end__` (최종 응답)
   - tool_calls가 있으면 → `tools` 노드로 라우팅
4. **`tools` (ToolNode)**: LangGraph 내장 `ToolNode`가 tool_calls를 병렬 실행, 결과를 `ToolMessage`로 반환
5. **`tools` → `call_model`**: 도구 결과를 포함한 메시지로 다시 LLM 호출 (루프)
6. **`is_last_step` 가드**: 재귀 한계(`recursion_limit - 1`)에 도달하면 도구 호출 대신 "Sorry, I could not find an answer..." 메시지 반환

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 패키지 진입점. `graph` 객체를 re-export하여 `from react_agent import graph` 또는 `langgraph.json`에서 참조 가능 |
| `graph.py` | **핵심 파일** (125줄). `StateGraph` 정의, `call_model` 노드 (LLM 호출 + 시스템 프롬프트 포맷팅 + is_last_step 가드), `route_model_output` 조건부 라우터, 그래프 컴파일. `langgraph.json`에서 `agent` 그래프의 진입점 (`graph.py:graph`) |
| `configuration.py` | `Configuration` dataclass (59줄). 3개 설정 필드: `model` (기본 `gpt-4.1`, LangGraph Studio에서 LLM 선택 UI 제공), `system_prompt` (기본 `prompts.SYSTEM_PROMPT`), `max_search_results` (기본 10). 팩토리 메서드 2개: `from_runnable_config()` (노드에서 사용), `from_context()` (도구에서 사용, `langgraph.config.get_config()` 활용) |
| `state.py` | 상태 정의 (61줄). `InputState` — 외부 인터페이스, `messages` 필드만 노출 (`add_messages` 리듀서로 append-only 병합). `State(InputState)` — 내부 확장, `is_last_step: IsLastStep` managed 변수 추가. 확장 예시 주석 포함 (retrieved_documents, extracted_entities 등) |
| `prompts.py` | 시스템 프롬프트 상수 (6줄). `SYSTEM_PROMPT = "You are a helpful AI assistant.\n\nSystem time: {system_time}"`. `{system_time}`은 `graph.py`의 `call_model`에서 UTC ISO format으로 포맷 |
| `tools.py` | 도구 정의 (28줄). `search(query: str)` — Tavily 웹 검색. `Configuration.from_context()`로 `max_search_results` 설정 접근. `TOOLS: List[Callable]` 리스트로 export하여 `graph.py`에서 `ToolNode(TOOLS)` 및 `model.bind_tools(TOOLS)`에 사용 |
| `utils.py` | 유틸리티 함수 (28줄). `get_message_text(msg)` — `BaseMessage`에서 텍스트 추출 (str/dict/list 콘텐츠 처리). `load_chat_model(name)` — `"provider/model"` 문자열로 `init_chat_model()` 호출. **주의: 현재 두 함수 모두 코드베이스에서 미사용** |

## Module Dependency Flow

```
graph.py
  ├── imports: configuration.Configuration
  ├── imports: state.InputState, State
  ├── imports: tools.TOOLS
  └── external: ChatOpenAI, StateGraph, ToolNode, AIMessage

configuration.py
  ├── imports: prompts.SYSTEM_PROMPT
  └── external: RunnableConfig, ensure_config, get_config

tools.py
  ├── imports: configuration.Configuration
  └── external: TavilySearch

state.py
  └── external: AnyMessage, add_messages, IsLastStep

prompts.py
  └── (독립 — 의존성 없음)

utils.py
  └── external: init_chat_model, BaseChatModel, BaseMessage
  └── (현재 미사용)
```

## For AI Agents

### Working In This Directory

- **새 도구 추가**: `tools.py`에 `async def` 함수 정의 → `TOOLS` 리스트에 추가. docstring이 LLM에 도구 설명으로 전달되므로 명확하게 작성. 도구 내에서 설정 접근 시 `Configuration.from_context()` 사용
- **모델 변경**: `configuration.py`의 `model` 필드 기본값 수정 또는 런타임에 `configurable={"model": "provider/model-name"}` 전달
- **프롬프트 수정**: `prompts.py`의 `SYSTEM_PROMPT` 수정 또는 Configuration의 `system_prompt` 필드로 오버라이드
- **상태 확장**: `state.py`의 `State` 클래스에 필드 추가 (주석에 예시: `retrieved_documents`, `extracted_entities`)
- **새 노드 추가**: `graph.py`에 `async def node(state: State, config: RunnableConfig)` 정의 → `builder.add_node()` → `builder.add_edge()` 또는 `builder.add_conditional_edges()`로 연결
- **temperature 변경**: 현재 `graph.py:42`에 `temperature=0.1` 하드코딩. 변경 시 해당 라인 수정 또는 `Configuration`에 필드 추가 권장

### Testing Requirements

- 단위 테스트: `tests/unit_tests/test_configuration.py`에서 Configuration 기본 생성 테스트
- 통합 테스트: `tests/integration_tests/test_graph.py`에서 전체 그래프 `ainvoke()` 실행 테스트
- `mypy --strict` 적용 — 모든 함수에 타입 힌트 필수
- ruff 린트: Google style docstring 필수 (`D` 규칙, `D417` parameter docstring 제외)
- 새 모듈 추가 시 `tests/unit_tests/`에 대응하는 `test_*.py` 작성

### Common Patterns

- **Dataclass + field metadata**: Configuration 필드에 `metadata={"description": ...}` 패턴으로 LangGraph Studio UI에 설명 표시
- **Annotated 타입 + 템플릿 메타데이터**: `Annotated[str, {"__template_metadata__": {"kind": "llm"}}]`로 LangGraph 템플릿 시스템에 모델 선택 UI 힌트 제공
- **cast() 타입 단언**: `model.ainvoke()` 반환값을 `cast(AIMessage, ...)` — mypy strict 호환을 위한 패턴
- **is_last_step 가드**: `state.is_last_step and response.tool_calls` 조건으로 재귀 한계 보호
- **비동기 함수**: 모든 노드 함수와 도구는 `async def`로 정의
- **팩토리 메서드 패턴**: `Configuration`에 `from_runnable_config()` (노드 함수용)과 `from_context()` (도구 함수용, LangGraph 컨텍스트 자동 해석) 두 가지 생성 경로
- **InputState/State 분리**: `InputState`는 외부에 노출되는 좁은 인터페이스 (messages만), `State`는 내부 처리용 확장 (is_last_step 등)

### Code Quality Notes

- `graph.py:6` — `import os` 미사용 (제거 가능)
- `utils.py` — `get_message_text()`와 `load_chat_model()` 모두 코드베이스에서 호출되지 않음. 확장 시 활용하거나 정리 가능
- `graph.py`에서 `ChatOpenAI`를 직접 인스턴스화하므로 OpenAI 호환 모델만 지원. 다중 프로바이더 지원 시 `utils.load_chat_model()` 활용 또는 `langchain.chat_models.init_chat_model` 직접 사용 고려
- `tools.py`의 `search()` 함수에 에러 핸들링 없음 — Tavily API 장애 시 예외가 그대로 전파됨 (ToolNode가 자동으로 `ToolMessage`에 에러 포함하여 LLM에 전달)

## Dependencies

### Internal

```
graph.py → configuration.py → prompts.py
graph.py → state.py
graph.py → tools.py → configuration.py
```

### External

| Package | Import | Used In |
|---------|--------|---------|
| `langchain_openai.ChatOpenAI` | LLM 호출 | graph.py |
| `langchain_tavily.TavilySearch` | 웹 검색 도구 | tools.py |
| `langgraph.graph.StateGraph` | 그래프 빌더 | graph.py |
| `langgraph.prebuilt.ToolNode` | 도구 실행 노드 | graph.py |
| `langgraph.managed.IsLastStep` | 재귀 한계 managed 변수 | state.py |
| `langgraph.config.get_config` | 현재 컨텍스트 설정 조회 | configuration.py |
| `langchain_core.messages.AIMessage` | LLM 응답 타입 | graph.py |
| `langchain_core.messages.AnyMessage` | 메시지 시퀀스 타입 | state.py |
| `langgraph.graph.add_messages` | 메시지 리듀서 (append-only 병합) | state.py |
| `langchain_core.runnables.RunnableConfig` | 실행 설정 전달 | graph.py, configuration.py |

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
