## 자동 모니터링 시스템 사용법

### 1. 백그라운드 모니터

Slurm 작업 제출 후 LLM 없이 자동으로 완료를 감지합니다.

#### 수동 실행:
```bash
python monitor_simulation.py /path/to/work_dir --job-id 2066 --interval 30
```

#### Exit Codes:
- `0`: 정상 완료
- `1`: 오류 발생
- `2`: 타임아웃
- `130`: 사용자 중단 (Ctrl+C)

### 2. Agent 통합 방식

#### 방법 A: `wait_for_completion` 툴 사용 (권장)
```python
# Slurm 제출 후
result = run_simulation("in.sputtering")
job_id = extract_job_id(result)

# Agent가 대기 (LLM 호출 없이)
completion_status = wait_for_completion(job_id=job_id, poll_interval=30)

# 완료 후 Agent 재개
if "COMPLETE" in completion_status:
    # 결과 분석
    pass
```

#### 방법 B: 외부 스크립트 + 콜백
```bash
# 1. Slurm 제출
sbatch run.slurm

# 2. 백그라운드 모니터 시작
python monitor_simulation.py $WORK_DIR --job-id $JOB_ID &

# 3. 완료 시 Agent 재호출
# (monitor_simulation.py가 종료되면 wrapper script가 Agent 재실행)
```

### 3. 현재 구현 상태

- [x] `monitor_simulation.py`: 백그라운드 모니터 스크립트
- [ ] `wait_for_completion` 툴: tools_lib.py에 추가 필요
- [ ] Agent 워크플로우 수정: run_simulation 후 자동 대기

### 4. 권장 워크플로우

```
1. research_crystal() → build_substrate() → ...
2. generate_lammps_input() → generate_slurm_script()
3. run_simulation()  # Slurm 제출
   └─ 즉시 return (Job ID 포함)
4. wait_for_completion(job_id)  # LLM 호출 없이 폴링
   └─ 완료/오류 시 return
5. check_simulation_progress()  # 최종 결과 확인
   또는
   analyze_results()  # LLM으로 분석
```

### 5. 원격 서버 배포

수정된 파일들을 원격 서버에 복사:
```bash
scp src/executor.py user@server:/path/to/md_agent/src/
scp src/tools_lib.py user@server:/path/to/md_agent/src/
scp monitor_simulation.py user@server:/path/to/md_agent/
```
