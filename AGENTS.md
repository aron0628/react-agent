<!-- Generated: 2026-03-10 | Updated: 2026-03-17 -->

# react-agent

## Purpose

LangGraph ReAct Agent 템플릿 프로젝트. [ReAct(Reasoning and Action)](https://arxiv.org/abs/2210.03629) 패턴을 구현한 AI 에이전트로, 사용자 질의에 대해 반복적으로 추론(Reasoning)하고 도구를 실행(Action)하여 최종 답변을 생성한다. LangGraph Studio와 통합되어 시각적 디버깅 및 개발이 가능하다.

## Architecture Overview

```
User Query → [call_model] → Tool Calls? ─Yes→ [ToolNode] → [call_model] (반복)
                                         ─No→  __end__ (최종 응답)
```

핵심 흐름:
1. `call_model` 노드가 LLM을 호출하여 추론
2. LLM이 tool_calls를 반환하면 `ToolNode`가 해당 도구 실행
3. 도구 결과를 다시 `call_model`에 전달 (루프)
4. tool_calls가 없으면 최종 응답으로 종료
5. `is_last_step` 가드가 재귀 한계 도달 시 도구 호출 대신 사과 메시지 반환

## Module Dependency Graph

```
graph.py ──────────► configuration.py ──► prompts.py
    │                       ▲
    ├──► state.py           │
    │                       │
    └──► tools.py ──────────┘
         (Configuration.from_context())

utils.py  ← 독립 모듈 (현재 미사용, 확장용 유틸리티)
```

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | 프로젝트 메타데이터, 의존성 (langgraph, langchain-openai, langchain-tavily), ruff 린트 설정. 듀얼 패키지 매핑: `langgraph.templates.react_agent`(LangGraph 템플릿 호환) + `react_agent`(직접 import) |
| `langgraph.json` | LangGraph 배포 설정 — 그래프 진입점을 `src/react_agent/graph.py:graph`로 지정, `.env` 환경변수 로딩 |
| `Makefile` | 테스트(`pytest`), 린트(`ruff`, `mypy`), 포맷팅 자동화. `make lint`는 ruff check + format diff + isort + mypy strict 순차 실행 |
| `.env.example` | 필수 환경변수 템플릿: `OPENAI_API_KEY`, `TAVILY_API_KEY`, LangSmith 설정 |
| `uv.lock` | uv 패키지 매니저 락 파일 — 재현 가능한 빌드 보장 |
| `.codespellignore` | CI 맞춤법 검사 예외 단어 목록 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | 에이전트 소스 코드 (see `src/AGENTS.md`) |
| `tests/` | 단위/통합 테스트 및 VCR 카세트 (see `tests/AGENTS.md`) |
| `.github/` | CI/CD 워크플로우 (see `.github/AGENTS.md`) |
| `static/` | 정적 에셋 — LangGraph Studio UI 스크린샷 (`studio_ui.png`) |
| `.langgraph_api/` | LangGraph Studio 런타임 캐시 (store.pckl, store.vectors.pckl) — git에 포함하지 않음 |
| `react_agent.egg-info/` | Python 패키지 메타데이터 (setuptools 자동 생성) |

## For AI Agents

### Working In This Directory

- Python >= 3.11 필수. 패키지 관리는 `uv` 사용
- `.env` 파일에 API 키 설정 필요: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`
- 모델은 `configuration.py`에서 `model` 필드로 지정 (기본값: `gpt-4.1`)
- LangGraph Studio에서 `langgraph.json`을 열어 그래프를 시각적으로 디버깅 가능
- `python-dotenv`는 의존성에 포함되지만 소스 코드에서 직접 `load_dotenv()`를 호출하지 않음 — LangGraph 런타임이 `langgraph.json`의 `"env": ".env"` 설정을 통해 자동 로딩

### Testing Requirements

- 단위 테스트: `make test` 또는 `python -m pytest tests/unit_tests/`
- 통합 테스트: `python -m pytest tests/integration_tests/` (API 키 필요)
- 린트: `make lint` (ruff check + ruff format + mypy strict)
- 포맷: `make format` (ruff format + isort)
- CI: push/PR 시 단위 테스트 자동 실행 (Python 3.11/3.12), 통합 테스트는 매일 14:37 UTC 스케줄

### Common Patterns

- **Dataclass 기반 설정**: `Configuration` 클래스가 `RunnableConfig`에서 configurable 값 추출. 두 가지 팩토리 메서드: `from_runnable_config()` (노드 함수), `from_context()` (도구 함수)
- **StateGraph 패턴**: `State` dataclass → `StateGraph` 빌더 → 노드 추가 → 조건부 에지 → `compile()`
- **도구 바인딩**: `ChatOpenAI.bind_tools(TOOLS)`로 LLM에 도구 연결
- **Google style docstring**: ruff의 `pydocstyle` 규칙 적용 (`convention = "google"`)
- **비동기 전용**: 모든 노드 함수와 도구는 `async def`로 정의
- **듀얼 패키지 매핑**: `pyproject.toml`에서 `langgraph.templates.react_agent`(LangGraph 생태계 호환)과 `react_agent`(로컬 개발) 두 경로로 동일 소스 매핑

### Code Quality Notes

- `graph.py`에 `import os`가 있으나 사용되지 않음 (불필요한 import)
- `utils.py`의 `load_chat_model()`은 현재 어디에서도 호출되지 않음 — `graph.py`는 `ChatOpenAI`를 직접 인스턴스화. 다중 프로바이더 지원 시 활용 가능
- `graph.py`의 `temperature=0.1`이 하드코딩됨 — `Configuration`에 설정 필드로 분리 가능
- 통합 테스트 CI에서 `ANTHROPIC_API_KEY`를 secrets로 주입하지만, 실제 코드는 `ChatOpenAI` (OpenAI API) 사용 — 템플릿 잔여 설정 가능성

## Dependencies

### External

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >= 0.6.10 | 그래프 기반 에이전트 오케스트레이션 프레임워크. StateGraph, ToolNode, IsLastStep 등 |
| `langchain-openai` | >= 0.3.35 | OpenAI/호환 모델 채팅 인터페이스 (`ChatOpenAI`) |
| `langchain-tavily` | >= 0.2.12 | Tavily 웹 검색 도구 (`TavilySearch`) |
| `langchain` | >= 0.3.27 | LangChain 코어 유틸리티 (`init_chat_model`, `BaseMessage` 등) |
| `python-dotenv` | >= 1.0.1 | `.env` 파일 로딩 (LangGraph 런타임에서 사용) |

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `ruff` >= 0.6.1 | 린트 + 포맷팅 (pycodestyle, pyflakes, isort, pydocstyle, pyupgrade) |
| `mypy` >= 1.11.1 | 정적 타입 검사 (strict 모드) |
| `pytest` >= 8.3.5 | 테스트 프레임워크 |
| `langgraph-cli[inmem]` >= 0.1.71 | LangGraph 로컬 개발 서버 (인메모리 모드) |

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
