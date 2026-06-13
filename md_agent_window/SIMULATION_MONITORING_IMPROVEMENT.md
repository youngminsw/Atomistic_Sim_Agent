# 시뮬레이션 모니터링 개선 작업 요약

## 📋 문제점

**이전 구현:**
- `run_simulation()` 호출 시 Slurm 작업을 제출한 후 **동기적으로 대기** (최대 5분)
- LLM Agent가 시뮬레이션이 완료될 때까지 **폴링 루프**에서 계속 대기
- 매 턴마다 LLM이 `check_simulation_progress()`를 호출하여 **토큰 사용량 과다**
- 타임아웃 발생 시 재시도로 인한 추가 비용

## ✅ 해결 방법

### 1. **Executor 수정** (`src/executor.py`)
**변경 내용:**
- `_run_slurm()` 메서드를 **비동기(non-blocking)** 방식으로 변경
- Slurm 작업 제출 후 **즉시 반환**
- 백그라운드에서 `monitor_simulation.py` 자동 실행 (선택적)
  - `nohup` + `subprocess.Popen` + `start_new_session=True`로 완전히 분리된 프로세스로 실행
  - 부모 프로세스(Agent)와 독립적으로 동작

**효과:**
- Agent는 시뮬레이션 제출 후 즉시 다음 작업으로 진행 가능
- 모니터링은 Python 코드가 자동으로 처리
- LLM 호출 횟수 대폭 감소

### 2. **Agent 프롬프트 수정** (`src/agent_core.py`)
**변경 내용:**
- Tool 설명 업데이트: `run_simulation`이 non-blocking임을 명시
- 시스템 메시지에 새로운 규칙 추가:
  ```
  SIMULATION MONITORING:
  - run_simulation() with Slurm returns IMMEDIATELY after job submission.
  - A background monitor process will track the simulation automatically.
  - DO NOT call check_simulation_progress() immediately after run_simulation().
  - Only check progress if the user asks, or after sufficient time has passed.
  ```

**효과:**
- LLM이 시뮬레이션 제출 후 즉시 확인하지 않음
- 불필요한 `check_simulation_progress()` 호출 방지

## 📊 예상 개선 효과

### Token 사용량 절감
**이전:**
```
Turn 1: run_simulation() → LLM 대기 중... (폴링 중 타임아웃)
Turn 2: LLM 재시도... (폴링 계속)
Turn 3: LLM 재시도... (폴링 계속)
Turn 4: check_simulation_progress() → "RUNNING"
Turn 5: check_simulation_progress() → "RUNNING"
...
Turn N: check_simulation_progress() → "COMPLETE"
```
**예상 토큰:** ~15,000-20,000 tokens (매 턴마다 full context 전송)

**개선 후:**
```
Turn 1: run_simulation() → "Submitted. Job ID: 12345" (즉시 반환)
        [Background: monitor_simulation.py가 자동으로 추적]
Turn 2: (다른 작업 진행 또는 종료)
...
(사용자가 확인 요청 시에만)
Turn N: check_simulation_progress() → "COMPLETE"
```
**예상 토큰:** ~2,000-3,000 tokens (1-2회 호출)

### 시간 절약
- **이전:** Agent가 5분 이상 대기 (타임아웃 포함)
- **개선 후:** 즉시 반환 (< 1초)

### 안정성 향상
- **이전:** Slurm 작업이 5분 이상 걸리면 타임아웃
- **개선 후:** 제한 없음 (최대 2시간까지 백그라운드 모니터가 추적)

## 🔍 변경된 파일

1. **`src/executor.py`**
   - `_run_slurm()`: 동기 → 비동기 변경
   - 백그라운드 모니터 자동 실행 추가

2. **`src/agent_core.py`**
   - Tool schema: `run_simulation` 설명 업데이트
   - System prompt: SIMULATION MONITORING 규칙 추가

## 📝 사용 방법

### Agent 관점
```python
# 시뮬레이션 제출 (즉시 반환)
result = run_simulation("in.sputtering")
# Output: "Submitted batch job 12345"

# [Background] monitor_simulation.py가 자동으로 작동
# Agent는 다른 작업 수행 또는 종료

# (나중에) 필요 시 수동 확인
status = check_simulation_progress(job_id="12345")
```

### 백그라운드 모니터
```bash
# 자동으로 다음 명령이 실행됨 (detached)
nohup python3 monitor_simulation.py /path/to/workdir \
    --job-id 12345 \
    --interval 30 \
    --max-runtime 7200 &
```

## ⚠️ 주의사항

1. **Windows 호환성:**
   - `nohup`은 Unix 계열 명령어
   - Windows에서는 다른 방식 필요 (`start /B` 또는 Task Scheduler)

2. **모니터 출력:**
   - 현재 `stdout`/`stderr`를 `DEVNULL`로 리다이렉트
   - 로그 확인 필요 시 별도 파일로 저장 필요

3. **종료 조건:**
   - 시뮬레이션 완료 시 모니터는 자동 종료
   - 타임아웃 시 exit code 2로 종료

## 🚀 다음 단계 (선택사항)

1. **알림 시스템 추가:**
   - 시뮬레이션 완료 시 Slack/Email 알림
   - Webhook으로 Agent에게 완료 신호 전송

2. **Windows 지원:**
   - OS 감지 후 플랫폼별 백그라운드 실행 로직 분기

3. **모니터 로그:**
   - 모니터 진행 상황을 별도 파일에 기록

4. **재시작 기능:**
   - 중단된 시뮬레이션 자동 재시작 로직

---

**작성일:** 2026-01-24  
**작성자:** Antigravity AI Assistant  
**버전:** 1.0
