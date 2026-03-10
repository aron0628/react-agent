<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

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

## Key Files

| File | Description |
|------|-------------|
| `__init__.py` | 패키지 진입점. `graph` 객체를 re-export |
| `graph.py` | **핵심 파일**. `StateGraph` 정의, `call_model` 노드 (LLM 호출 + 시스템 프롬프트 포맷팅), `route_model_output` 조건부 라우터, 그래프 컴파일. `langgraph.json`에서 `graph`로 참조됨 |
| `configuration.py` | `Configuration` dataclass — 모델명(`gpt-4.1` 기본), 시스템 프롬프트, 최대 검색 결과 수. `from_runnable_config()`과 `from_context()` 두 가지 팩토리 메서드 제공 |
| `state.py` | `InputState`(메시지 시퀀스)와 `State`(InputState + `is_last_step` managed 변수) dataclass 정의. `add_messages` 리듀서로 메시지 병합 |
| `prompts.py` | 기본 시스템 프롬프트 상수 (`SYSTEM_PROMPT`). `{system_time}` 포맷 변수 포함 |
| `tools.py` | 도구 정의. 현재 Tavily 웹 검색(`search`) 하나만 등록. `TOOLS` 리스트로 export |
| `utils.py` | 유틸리티 함수 — `get_message_text()` (메시지 텍스트 추출), `load_chat_model()` (provider/model 문자열로 모델 로드) |

## For AI Agents

### Working In This Directory

- **새 도구 추가**: `tools.py`에 함수 정의 후 `TOOLS` 리스트에 추가. 도구 함수는 docstring이 LLM에 설명으로 전달되므로 명확하게 작성
- **모델 변경**: `configuration.py`의 `model` 필드 기본값 수정 또는 런타임에 configurable로 전달
- **프롬프트 수정**: `prompts.py`의 `SYSTEM_PROMPT` 수정 또는 Configuration의 `system_prompt` 필드로 오버라이드
- **상태 확장**: `state.py`의 `State` 클래스에 필드 추가 (예: `retrieved_documents`, `extracted_entities`)
- `graph.py`의 `call_model`에서 `ChatOpenAI`를 직접 사용 중 (utils.py의 `load_chat_model`은 미사용)
- `Configuration.from_context()`는 도구 함수 내에서 LangGraph 컨텍스트로부터 설정을 가져올 때 사용

### Testing Requirements

- 단위 테스트: `tests/unit_tests/test_configuration.py`에서 Configuration 테스트
- 통합 테스트: `tests/integration_tests/test_graph.py`에서 전체 그래프 실행 테스트
- `mypy --strict` 적용 — 모든 함수에 타입 힌트 필수
- ruff 린트: Google style docstring 필수 (`D` 규칙, `D417` 제외)

### Common Patterns

- **Dataclass + field metadata**: Configuration 필드에 `metadata={"description": ...}` 패턴으로 LangGraph Studio UI에 설명 표시
- **Annotated 타입**: `Annotated[str, {"__template_metadata__": ...}]`로 LangGraph 템플릿 메타데이터 첨부
- **cast() 패턴**: `model.ainvoke()` 반환값을 `cast(AIMessage, ...)`로 타입 단언
- **is_last_step 가드**: 재귀 한계 도달 시 도구 호출 대신 사과 메시지 반환
- **비동기 함수**: 모든 노드 함수와 도구는 `async def`로 정의

## Dependencies

### Internal

- 모듈 간 의존 관계: `graph.py` → `configuration.py`, `state.py`, `tools.py` → `prompts.py`
- `tools.py` → `configuration.py` (런타임 설정 접근)

### External

- `langchain_openai.ChatOpenAI` — LLM 호출 (graph.py)
- `langchain_tavily.TavilySearch` — 웹 검색 도구 (tools.py)
- `langgraph.graph.StateGraph` — 그래프 빌더 (graph.py)
- `langgraph.prebuilt.ToolNode` — 도구 실행 노드 (graph.py)
- `langgraph.managed.IsLastStep` — 재귀 한계 관리 (state.py)
- `langgraph.config.get_config` — 현재 컨텍스트 설정 조회 (configuration.py)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
