<!-- Generated: 2026-03-10 | Updated: 2026-03-10 -->

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

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | 프로젝트 메타데이터, 의존성 (langgraph, langchain-openai, langchain-tavily), ruff 린트 설정 |
| `langgraph.json` | LangGraph 배포 설정 — 그래프 진입점을 `src/react_agent/graph.py:graph`로 지정 |
| `Makefile` | 테스트(`pytest`), 린트(`ruff`, `mypy`), 포맷팅 자동화 명령 |
| `.env.example` | 필수 환경변수 템플릿 (OPENAI_API_KEY, TAVILY_API_KEY, LANGSMITH 설정) |
| `uv.lock` | uv 패키지 매니저 락 파일 (gitignored) |
| `.codespellignore` | CI 맞춤법 검사 예외 단어 목록 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/` | 에이전트 소스 코드 (see `src/AGENTS.md`) |
| `tests/` | 단위/통합 테스트 (see `tests/AGENTS.md`) |
| `.github/` | CI/CD 워크플로우 (see `.github/AGENTS.md`) |
| `static/` | 정적 에셋 — LangGraph Studio UI 스크린샷 |

## For AI Agents

### Working In This Directory

- Python >= 3.11 필수. 패키지 관리는 `uv` 사용
- `.env` 파일에 API 키 설정 필요: `OPENAI_API_KEY`, `TAVILY_API_KEY`, `LANGSMITH_API_KEY`
- 모델은 `configuration.py`에서 `provider/model-name` 형식으로 지정 (기본값: `gpt-4.1`)
- LangGraph Studio에서 직접 열어서 그래프를 시각적으로 디버깅 가능

### Testing Requirements

- 단위 테스트: `make test` 또는 `python -m pytest tests/unit_tests/`
- 통합 테스트: `python -m pytest tests/integration_tests/` (API 키 필요)
- 린트: `make lint` (ruff check + ruff format + mypy strict)
- 포맷: `make format` (ruff format + isort)

### Common Patterns

- **Dataclass 기반 설정**: `Configuration` 클래스가 `RunnableConfig`에서 configurable 값 추출
- **StateGraph 패턴**: `State` dataclass → `StateGraph` 빌더 → 노드 추가 → 조건부 에지 → `compile()`
- **도구 바인딩**: `ChatOpenAI.bind_tools(TOOLS)`로 LLM에 도구 연결
- **Google style docstring**: ruff의 `pydocstyle` 규칙 적용 (`convention = "google"`)

## Dependencies

### External

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | >= 0.6.10 | 그래프 기반 에이전트 오케스트레이션 프레임워크 |
| `langchain-openai` | >= 0.3.35 | OpenAI/호환 모델 채팅 인터페이스 |
| `langchain-tavily` | >= 0.2.12 | Tavily 웹 검색 도구 |
| `langchain` | >= 0.3.27 | LangChain 코어 유틸리티 (`init_chat_model` 등) |
| `python-dotenv` | >= 1.0.1 | `.env` 파일 로딩 |

### Dev Dependencies

| Package | Purpose |
|---------|---------|
| `ruff` | 린트 + 포맷팅 |
| `mypy` | 정적 타입 검사 (strict 모드) |
| `pytest` | 테스트 프레임워크 |
| `langgraph-cli[inmem]` | LangGraph 로컬 개발 서버 |

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
