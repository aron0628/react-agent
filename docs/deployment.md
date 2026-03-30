# 배포 방식: LangGraph Cloud vs 셀프 호스팅

## LangGraph 템플릿의 기본 Dockerfile

LangGraph 공식 템플릿으로 프로젝트를 생성하면 Dockerfile이 `langchain/langgraph-api` 이미지를 베이스로 사용하도록 설정되어 있다. 이 이미지는 **LangGraph Cloud 배포 전용**이며, Go 코어 서버 + Python 워커 구조로 PostgreSQL과 Redis를 필수로 요구한다.

## 두 가지 배포 방식 비교

| | LangGraph Cloud | 셀프 호스팅 (현재 방식) |
|---|---|---|
| 인프라 | LangGraph가 관리 (AWS 등) | 직접 서버 (OCI 등) |
| Docker 이미지 | `langchain/langgraph-api` 필수 | 필요 없음, 자유롭게 구성 |
| Redis | 자동 제공됨 (필수) | 불필요 |
| 비용 | LangGraph에 과금 | 서버 비용만 |
| Dockerfile | 템플릿 기본 제공 | 직접 작성 |

## 실행 방식 비교

| | `langgraph dev` (로컬/셀프호스팅) | `langchain/langgraph-api` (Cloud) |
|---|---|---|
| 구조 | Python만 실행 | Go 코어 서버 + Python 워커 |
| 저장소 | PostgreSQL 직접 연결 | PostgreSQL + Redis 필수 |
| 용도 | 로컬 개발 / 셀프 호스팅 배포 | LangGraph Cloud 배포 |
| 인증 | 선택 | 라이선스 키 기반 |

## 현재 프로젝트 설정

- Dockerfile: `python:3.11-slim` 베이스 + `langgraph dev`로 실행
- DB: OCI 서버의 PostgreSQL 사용 (Redis 없음)
- 로컬 실행: `uv run langgraph dev --no-browser --no-reload`
- Docker 실행: `docker compose up --build` (`/Users/aron/Documents/lab/docker-compose.yml`)

## Docker Compose 구성 (루트: /Users/aron/Documents/lab/)

| 서비스 | 포트 | 설명 |
|--------|:----:|------|
| react-agent | 2024 | LangGraph API 백엔드 |
| document-parser-server | 9997 | 문서 파싱 서버 |
| file-manager-admin | 8000 | 관리자 서버 (document-parser-client 라이브러리 포함) |
| agent-chat-ui | 3000 | Next.js 프론트엔드 |

- DB는 OCI 서버에서 실행 중이므로 PostgreSQL 컨테이너 없음
- `file-manager-admin`은 `document-parser-server` health check 통과 후 기동
- `agent-chat-ui`는 `react-agent` health check 통과 후 기동

## Docker Compose 운영 가이드

### 전체 실행/종료

```bash
cd /Users/aron/Documents/lab

# 전체 빌드 + 실행 (포그라운드, 모든 로그 출력)
docker compose up --build

# 전체 빌드 + 백그라운드 실행
docker compose up -d --build

# 전체 종료
docker compose down
```

### 로그 확인

```bash
# 특정 서비스 로그만 보기 (-f: 실시간 follow)
docker compose logs -f react-agent
docker compose logs -f document-parser-server

# 여러 서비스 조합
docker compose logs -f react-agent agent-chat-ui

# 최근 100줄만 보기
docker compose logs --tail 100 react-agent
```

### 개별 서비스 재시작

코드 수정 후 해당 서비스만 리빌드/재시작할 수 있다. 다른 컨테이너는 그대로 유지된다.

```bash
# 코드 수정 후 리빌드 + 재시작 (Dockerfile 또는 소스 변경 시)
docker compose up -d --build react-agent

# 단순 재시작 (설정 변경만 했을 때)
docker compose restart file-manager-admin

# 특정 서비스 중지/시작
docker compose stop agent-chat-ui
docker compose start agent-chat-ui
```

### 상태 확인

```bash
# 컨테이너 상태 확인
docker compose ps

# 특정 서비스 리소스 사용량
docker compose stats
```

### Docker 관련 주의사항

- `file-manager-admin`의 빌드 컨텍스트는 루트(`/Users/aron/Documents/lab/`)이다. `document-parser-client`를 COPY하기 때문에 반드시 상위 디렉토리에서 빌드해야 한다.
- `agent-chat-ui`는 Docker 환경에서 `AUTH_TRUST_HOST=true`가 필요하다 (NextAuth 호스트 신뢰 설정).
- `PARSER_SERVER_URL`은 Docker 내부 네트워크 주소(`http://document-parser-server:9997`)로 자동 오버라이드된다.
- `LANGGRAPH_API_URL`은 Docker 내부 네트워크 주소(`http://react-agent:2024`)로 자동 오버라이드된다.
