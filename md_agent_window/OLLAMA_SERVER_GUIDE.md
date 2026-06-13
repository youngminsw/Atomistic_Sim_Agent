# 📢 사내/연구실 전용 LLM 서버 사용 안내

연구실(팀) 구성원 여러분, 
고성능 GPU(RTX 4090) 서버에서 구동되는 **로컬 LLM(Ollama)**을 이제 누구나 자신의 PC에서 원격으로 사용할 수 있습니다.

클라우드 API(OpenAI 등) 비용 걱정 없이, 내부망을 통해 빠르고 안전하게 AI 모델을 활용해 보세요.

---

## 🖥️ 서버 정보 (접속 정보)

| 구분 | ⚡ AI 서버 (RTX 4090) |
| :--- | :--- |
| **IP 주소** | **`10.24.12.81`** |
| **포트** | `11434` |
| **모델명** | **`qwen3-next:latest`** (코딩/추론 최적화) |

> ⚠️ **주의**: 반드시 **내부망(사내 네트워크/VPN)**에 연결된 상태여야 접속이 가능합니다.

---

## 🛠️ 사용 방법 (3가지 케이스)

### 1. Python 코드에서 사용하기 (OpenAI SDK 호환)
기존에 OpenAI API를 쓰던 코드에서 `base_url`과 `api_key` 부분만 아래처럼 바꾸면 즉시 작동합니다.

```python
from openai import OpenAI

# 4090 서버 사용 예시
client = OpenAI(
    base_url="http://10.24.12.81:11434/v1",  # 👈 서버 주소
    api_key="ollama",                        # 키는 아무거나 입력 (필수 아님)
)

response = client.chat.completions.create(
    model="qwen3-next:latest",               # 👈 서버에 설치된 모델 이름
    messages=[{"role": "user", "content": "안녕하세요! 자기소개 부탁해."}],
)

print(response.choices[0].message.content)
```

### 2. LangChain 사용 시
LangChain을 쓰시는 분들은 `ChatOllama` 클래스를 사용하면 더 간단합니다.

```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    base_url="http://10.24.12.81:11434",
    model="qwen3-next:latest"
)

response = llm.invoke("LLM 에이전트란 무엇인가요?")
print(response.content)
```

### 3. 터미널(CLI)에서 바로 쓰기 (내 PC엔 모델 없이!)
내 컴퓨터 용량을 차지하지 않고, 터미널 명령어로 4090의 모델을 불러다 쓸 수 있습니다.
(Ollama 클라이언트가 내 PC에 설치되어 있어야 합니다)

**Mac/Linux:**
```bash
export OLLAMA_HOST=10.24.12.81:11434
ollama run qwen3-next:latest "피보나치 수열 파이썬 코드로 짜줘"
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_HOST="10.24.12.81:11434"
ollama run qwen3-next:latest "피보나치 수열 파이썬 코드로 짜줘"
```

---

## ❓ 자주 묻는 질문 (FAQ)

**Q. 접속이 안 돼요 (Timeout / Connection Refused)**
A. 사내 와이파이나 유선망에 연결되어 있는지 확인해 주세요. (외부망에서는 접속 불가)

**Q. 사용 가능한 모델 목록을 보고 싶어요.**
A. 브라우저 주소창에 `http://10.24.12.81:11434/api/tags`를 입력하면 전체 모델 목록(JSON)이 뜹니다.

**Q. 속도가 느려요.**
A. 여러 사람이 동시에 무거운 작업(긴 코딩, 논문 요약 등)을 시키면 순차적으로 처리하느라 느려질 수 있습니다.
