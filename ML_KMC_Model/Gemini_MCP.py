#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini_MCP.py
- Gemini API(Brain) + FastMCP STDIO(Server tools) 오케스트레이터
- 실행하면 MCP_server.py를 subprocess로 자동 실행하고, STDIO로 연결해서 tool 호출함.

필수:
  export GEMINI_API_KEY="YOUR_KEY"

실행 예:
  python Gemini_MCP.py --workdir . --server MCP_server.py
"""

import os
import json
import asyncio
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from google import genai


SYSTEM = """You are an AI research assistant controlling a multiscale simulation platform
through MCP (Model Context Protocol) tools.

You MUST respond with ONLY valid JSON. No extra text, no markdown, no explanation.

--------------------------------
JSON FORMAT RULES
--------------------------------

If you need to call an MCP tool:
{"tool": "<tool_name>", "args": {...}}

If no tool should be called yet (e.g., you must ask a question first):
{"tool": null, "final": "..."}

Never output anything outside JSON.

--------------------------------
AVAILABLE TOOL PHILOSOPHY
--------------------------------

This platform supports a multiscale workflow:

MD (atomistic) →
ML surrogate (MDN) →
KMC (macro-scale trench evolution)

The tools are designed to be composed automatically.

--------------------------------
TOOL SELECTION RULES (VERY IMPORTANT)
--------------------------------

1) End-to-end requests

If the user asks for:
- "처음부터 다"
- "전체 파이프라인"
- "ML까지"
- "KMC까지 포함"
- "dump부터 KMC까지"

Then:

→ Prefer using `run_pipeline_kmc`

This tool performs:
dump2csv → train_random → infer_best → run_kmc

Use this whenever the user's intent includes BOTH
ML surrogate construction AND KMC simulation.

--------------------------------

2) ML-only pipeline

If the user asks for:
- "surrogate 만들기"
- "ML 학습"
- "pipeline 돌려줘"
- "run_pipeline"

Then:

→ Use `run_pipeline`

This includes:
dump2csv → train_random → infer_best

--------------------------------

3) KMC-only execution

If the user asks for:
- "KMC 돌려줘"
- "trench simulation"
- "etch profile"
- "이전에 학습한 모델로 KMC"

Then:

→ Use `run_kmc`

Assume that a trained surrogate already exists
(random_search_summary.json is present).

--------------------------------
INTERACTIVE SAFETY RULES
--------------------------------

1) workdir handling (CRITICAL)

If the user intent implies running any tool
(dump / train / infer / pipeline / kmc),
but the user has NOT provided a workdir path:

DO NOT call any tool yet.

Ask exactly this question (in Korean):

"workdir(작업 폴더) 경로가 어디야?
예: /home/username/project/04_MCP_server"

Wait for the user's reply.

When calling any tool, ALWAYS include:
{"workdir": "<path>"}

--------------------------------

2) n_trials handling (random search)

If training is involved (train_random or run_pipeline),
and the user did NOT specify the number of trials:

DO NOT call tools yet.

Ask (in Korean):

"랜덤 트라이얼(모델) 몇 개 돌릴까?
예: 5 / 10 / 20 / 50"

--------------------------------

3) overwrite handling

If training may be repeated and rs_runs already exists,
or the user says:
- "다시 돌려"
- "rerun"
- "또 학습"
- "기존 결과 덮어쓰기"

Ask first:

"기존 rs_runs 결과를 덮어쓸까?
덮어쓰려면 overwrite=true,
아니면 run_id(출력 이름)를 새로 줄게."

--------------------------------

4) KMC parameter confirmation (run_kmc)

If the user intent implies running KMC (run_kmc),
but the user did NOT specify any of these:

- energy_ev (입사 에너지, eV)
- angle_deg (입사 각도, degree)

DO NOT call run_kmc yet.

Ask in Korean, using the minimum questions needed:

- If BOTH are missing:
"입사 에너지 E_in(eV)는 몇으로 할까? (예: 50 / 100 / 200)"
"입사 각도 angle(°)는 몇으로 할까? (예: 0 / 5 / 10)"
(Optional) "이온 개수 num_ions는 몇 개로 할까? (예: 2000 / 10000 / 50000)"

- If ONLY energy_ev is missing:
"입사 에너지 E_in(eV)는 몇으로 할까? (예: 50 / 100 / 200)"

- If ONLY angle_deg is missing:
"입사 각도 angle(°)는 몇으로 할까? (예: 0 / 5 / 10)"

If the user says "기본값으로" or "default":
use energy_ev=100.0, angle_deg=0.0, num_ions=10000.

When calling run_kmc, ALWAYS include:
energy_ev, angle_deg, num_ions
(even if defaults are used).

--------------------------------

5) Parameter diversity

If the user asks for:
- "더 다양한 파라미터"
- "하드코딩 말고"
- "넓은 탐색"

Then provide space_json_str with expanded space
ONLY using supported keys:

num_gaussians
hidden_dim
dropout
lr
batch

Do NOT invent new hyperparameter names.

--------------------------------
RESPONSE LANGUAGE
--------------------------------

- JSON keys MUST remain in English.
- Human-facing messages in "final" MUST be written in Korean.

--------------------------------
GOAL
--------------------------------

Your goal is to operate the simulation system correctly,
safely, and with minimal user friction,
while respecting the multiscale physics workflow.
"""



def _must_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"Missing environment variable: {name}")
    return v


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        pass

    l = t.find("{")
    r = t.rfind("}")
    if l != -1 and r != -1 and r > l:
        try:
            return json.loads(t[l : r + 1])
        except Exception:
            return None
    return None


def build_prompt_with_history(
    system_text: str,
    tool_names: List[str],
    chat_history: List[Dict[str, str]],
    current_default_workdir: str,
    max_turns: int = 16,
) -> str:
    """
    Build a single text prompt that includes SYSTEM + tools + remembered context + recent dialogue.
    This is the key fix: without history, the model keeps forgetting that workdir was already provided.
    """
    h = chat_history[-max_turns:]
    lines = [
        system_text,
        "",
        f"Available tools: {tool_names}",
        f"Default workdir (if user does not specify): {current_default_workdir}",
        "",
        "Conversation (most recent last):",
    ]
    for m in h:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            lines.append(f"User> {content}")
        else:
            lines.append(f"Assistant> {content}")
    lines.append("")
    lines.append("Now decide the next action. Respond with ONLY valid JSON.")
    return "\n".join(lines)


async def gemini_decide_tool(
    gclient: genai.Client,
    model: str,
    chat_history: List[Dict[str, str]],
    tool_names: list[str],
    current_default_workdir: str,
    max_retries: int = 2,
) -> Dict[str, Any]:
    prompt0 = build_prompt_with_history(
        system_text=SYSTEM,
        tool_names=tool_names,
        chat_history=chat_history,
        current_default_workdir=current_default_workdir,
        max_turns=16,
    )

    prompt = prompt0
    for _ in range(max_retries + 1):
        resp = gclient.models.generate_content(model=model, contents=prompt)
        raw = (resp.text or "").strip()
        decision = _extract_json(raw)

        if decision is None:
            prompt = prompt0 + "\nERROR: Not valid JSON. Respond again with ONLY valid JSON."
            continue

        if "tool" not in decision:
            prompt = prompt0 + "\nERROR: JSON must include key 'tool'. Respond again with ONLY valid JSON."
            continue

        tool = decision.get("tool", None)

        if tool is None:
            decision.setdefault("final", "")
            return decision

        if not isinstance(tool, str) or tool not in tool_names:
            prompt = prompt0 + f"\nERROR: tool must be one of {tool_names}. Respond again with ONLY valid JSON."
            continue

        args = decision.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            prompt = prompt0 + "\nERROR: 'args' must be an object/dict. Respond again with ONLY valid JSON."
            continue

        decision["args"] = args
        return decision

    return {"tool": None, "final": "모델 응답이 JSON 규칙을 반복해서 위반했어. 입력을 조금 더 구체적으로 다시 말해줘."}


def make_stdio_transport(pyexe: str, server_py: Path, workdir: Path):
    """
    fastmcp 버전별 StdioTransport 생성자 시그니처가 달라서
    여러 패턴을 순서대로 시도한다.
    """
    env = os.environ.copy()

    # 패턴 A) (신버전) args=... 만 받는 경우
    try:
        return StdioTransport(
            args=[pyexe, str(server_py)],
            cwd=str(workdir),
            env=env,
        )
    except TypeError:
        pass

    # 패턴 B) (다른 버전) command + args 를 받는 경우
    try:
        return StdioTransport(
            command=pyexe,
            args=[str(server_py)],
            cwd=str(workdir),
            env=env,
        )
    except TypeError:
        pass

    # 패턴 C) (구버전 스타일) command=list 형태
    try:
        return StdioTransport(
            command=[pyexe, str(server_py)],
            cwd=str(workdir),
            env=env,
        )
    except TypeError as e:
        raise TypeError(
            "Could not construct StdioTransport with any known signature.\n"
            "Your fastmcp version has a different API.\n"
            f"Last error: {e}\n"
            "Tip: run `python -c \"import inspect; from fastmcp.client.transports import StdioTransport; "
            "print(inspect.signature(StdioTransport))\"`"
        )


def _is_path_like(text: str) -> bool:
    t = (text or "").strip()
    return (t.startswith("/") and len(t) > 1) or ("/home/" in t) or ("/mnt/" in t)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="MCP_server.py", help="Path to MCP server python file")
    ap.add_argument("--workdir", default=".", help="Default working directory (incident.dump 등이 있는 폴더)")
    ap.add_argument("--python", default=None, help="Python executable for MCP server (default: current python)")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    args = ap.parse_args()

    # ---- Gemini client ----
    api_key = _must_env("GEMINI_API_KEY")
    gclient = genai.Client(api_key=api_key)

    # ---- Paths ----
    default_workdir = Path(args.workdir).resolve()
    server_py = Path(args.server).resolve()
    if not server_py.is_file():
        raise FileNotFoundError(f"Server file not found: {server_py}")
    if not default_workdir.is_dir():
        raise NotADirectoryError(f"workdir is not a directory: {default_workdir}")

    pyexe = args.python or os.sys.executable

    # ---- MCP client via STDIO ----
    transport = make_stdio_transport(pyexe=pyexe, server_py=server_py, workdir=default_workdir)

    # ---- Conversation memory (THIS FIXES THE LOOP) ----
    chat_history: List[Dict[str, str]] = []

    async with Client(transport) as mcp:
        tools = await mcp.list_tools()
        tool_names = [t.name for t in tools]
        print("✅ Connected to MCP server.")
        print("🧰 MCP tools:", tool_names)
        print("Type 'exit' to quit.\n")

        while True:
            user = input("User> ").strip()
            if user.lower() in ("exit", "quit"):
                print("Bye!")
                break

            # store user turn
            chat_history.append({"role": "user", "content": user})

            # If user provides a path alone (common), store it as context by echoing it back.
            # This helps the model stop re-asking workdir.
            # (We do NOT restart server; tools can use workdir arg freely.)
            if _is_path_like(user):
                # Add a lightweight assistant acknowledgment into history
                chat_history.append(
                    {"role": "assistant", "content": f"확인! workdir 후보로 '{user}'를 받았어. 다음 단계(예: run_pipeline) 요청해줘."}
                )

            decision = await gemini_decide_tool(
                gclient=gclient,
                model=args.model,
                chat_history=chat_history,
                tool_names=tool_names,
                current_default_workdir=str(default_workdir),
                max_retries=2,
            )

            # store assistant decision in history (important for next turn)
            chat_history.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})

            if decision.get("tool") is None:
                print("Assistant>", decision.get("final", ""))
                # also store the human-facing final
                chat_history.append({"role": "assistant", "content": decision.get("final", "")})
                continue

            tool = decision["tool"]
            targs = decision.get("args", {})

            print(f"\n[Calling MCP tool] {tool} args={targs}")
            try:
                result = await mcp.call_tool(tool, targs)
            except Exception as e:
                err_msg = f"Tool call failed: {repr(e)}"
                print("Assistant>", err_msg)
                chat_history.append({"role": "assistant", "content": err_msg})
                continue

            # Store tool result into history so the model can reference it next turn
            chat_history.append({"role": "assistant", "content": f"[Tool result for {tool}] {result}"})

            # Ask the model to produce a user-facing final response, using history
            follow_prompt = build_prompt_with_history(
                system_text=SYSTEM,
                tool_names=tool_names,
                chat_history=chat_history,
                current_default_workdir=str(default_workdir),
                max_turns=20,
            ) + "\nReturn a final user message as JSON: {\"tool\": null, \"final\": \"...\"}"

            resp2 = gclient.models.generate_content(model=args.model, contents=follow_prompt)
            out = _extract_json((resp2.text or "").strip())
            if out and out.get("tool") is None:
                print("Assistant>", out.get("final", ""))
                chat_history.append({"role": "assistant", "content": out.get("final", "")})
            else:
                # fallback
                txt = (resp2.text or "").strip()
                print("Assistant>", txt)
                chat_history.append({"role": "assistant", "content": txt})


if __name__ == "__main__":
    asyncio.run(main())
