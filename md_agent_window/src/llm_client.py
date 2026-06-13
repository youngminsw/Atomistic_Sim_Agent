import json
import os
import time
import requests
import jwt # PyJWT
from src.config import Config


class GLMDirectClient:
    """Direct GLM API client (bypasses opencode CLI)."""
    
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    TIMEOUT = 60
    
    # Rate limit keywords
    RATE_LIMIT_KEYWORDS = ["rate limit", "quota", "429", "too many requests"]
    
    def __init__(self, api_key: str, model: str = "glm-4", timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        
    def _generate_token(self, exp_seconds: int = 3600) -> str:
        """Generate JWT token for GLM API auth."""
        try:
            id, secret = self.api_key.split(".")
        except Exception:
            raise ValueError("Invalid GLM API Key format (expected 'id.secret')")
        
        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }
        
        return jwt.encode(
            payload,
            secret.encode("utf-8"),
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )
    
    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error message indicates rate limiting."""
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in self.RATE_LIMIT_KEYWORDS)
    
    def generate_content(self, prompt: str):
        """Generate content via GLM Direct API."""
        token = self._generate_token()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        print(f"      [GLMDirectClient] Sending request to {self.model}...", flush=True)
        
        try:
            response = requests.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # [Token Tracking]
            try:
                usage = result.get("usage", {})
                if usage:
                    from src.token_tracker import TokenTracker
                    # Use current working directory or find project root
                    work_dir = os.getcwd() 
                    tracker = TokenTracker(work_dir)
                    tracker.log_usage(
                        self.model, 
                        usage.get("prompt_tokens", 0), 
                        usage.get("completion_tokens", 0)
                    )
            except Exception as e:
                print(f"      [GLMDirectClient] Tracking Error: {e}")
            
            # Return wrapper with .text attribute for compatibility
            class ResponseWrapper:
                def __init__(self, text):
                    self.text = text
            
            return ResponseWrapper(content)
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"GLM API timed out after {self.TIMEOUT}s") from None
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if self._is_rate_limit_error(error_msg):
                raise RuntimeError(f"Rate limit detected - switching model: {error_msg}") from None
            raise RuntimeError(f"GLM API error: {error_msg}") from None
            
        except Exception as e:
            raise RuntimeError(f"GLM API failed: {str(e)}") from None

class LLMClient:
    def __init__(self, model_name=None, api_key=None):
        self.api_key = api_key if api_key else Config.API_KEY
        self.model = model_name if model_name else Config.MODEL_NAME
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self._failover_attempts = 0
  # Track total failover switches
        self._current_fallback_index = 0

        
    def _generate_token(self, apikey: str, exp_seconds: int = 3600):
        try:
            id, secret = apikey.split(".")
        except Exception:
            raise ValueError("Invalid API Key format")

        payload = {
            "api_key": id,
            "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
            "timestamp": int(round(time.time() * 1000)),
        }

        return jwt.encode(
            payload,
            secret.encode("utf-8"),
            algorithm="HS256",
            headers={"alg": "HS256", "sign_type": "SIGN"},
        )

    def _try_failover(self):
        """Switch to next available model in FALLBACK_MODELS chain (Looping)."""
        fallback_models = getattr(Config, "FALLBACK_MODELS", [])
        if not fallback_models:
            return False
            
        max_loops = getattr(Config, "MAX_FAILOVER_LOOPS", 5)
        total_limit = len(fallback_models) * max_loops
        
        if self._failover_attempts >= total_limit:
            print("[LLM-Client] [ERROR] Max failover loops exhausted.")
            if self._ask_user_to_continue():
                print("[LLM-Client] Resetting failover counters and continuing...")
                self._failover_attempts = 0
                return self._try_failover()
            return False
            
        # Round-robin selection
        next_model = fallback_models[self._current_fallback_index % len(fallback_models)]
        self._current_fallback_index += 1
        self._failover_attempts += 1
        
        print(f"[LLM-Client] [WARNING] Switching to Fallback: {next_model} (Attempt {self._failover_attempts}/{total_limit})")
        self.model = next_model
        
        # Update API key if switching to GLM (requires different auth)
        if "glm" in next_model.lower() or "zhipu" in next_model.lower():
            self.api_key = getattr(Config, "GLM_API_KEY", Config.API_KEY)
        else:
            self.api_key = None  # Antigravity uses CLI auth
            
        return True

    def _ask_user_to_continue(self):
        """
        Asks the user whether to continue trying failover models or stop.
        Supports both interactive input and basic file-based communication (if AGENT_MODE=1).
        """
        print(f"\n{'='*60}")
        print(f"[LLM CLIENT] Max failover loops ({self._failover_attempts}) exhausted.")
        print(f"Do you want to reset the counter and continue looping through fallback models? (y/n)")
        print(f"{'='*60}\n")
        
        # Check environment for agent mode (file-based)
        agent_mode = os.environ.get("AGENT_MODE", "0") == "1"
        
        if agent_mode:
            # Simple file-based logic (similar to tools_lib but simplified)
            # We assume current working directory is where we should write
            work_dir = os.getcwd()
            q_file = os.path.join(work_dir, "agent_question.txt")
            a_file = os.path.join(work_dir, "agent_answer.txt")
            
            try:
                with open(q_file, "w") as f:
                    f.write("Max failover loops exhausted. Continue? (y/n)")
                
                print(f"[System] Waiting for answer in {a_file}...")
                
                # Wait loop (e.g. 5 mins)
                for _ in range(60): 
                    if os.path.exists(a_file):
                        time.sleep(1) # Wait for write
                        with open(a_file, "r") as f:
                            ans = f.read().strip().lower()
                        # Cleanup
                        if os.path.exists(q_file): os.remove(q_file)
                        if os.path.exists(a_file): os.remove(a_file)
                        return ans.startswith('y')
                    time.sleep(5)
            except Exception as e:
                print(f"[System] File comms failed: {e}")
                
            return False # Default to stop if agent mode fails
        
        else:
            # Interactive
            try:
                choice = input(">>> Continue? (y/n): ").strip().lower()
                return choice.startswith('y')
            except Exception:
                return False

    def generate_response(self, messages, temperature=0.5, tools=None, images=None):
        """
        Sends a request to LLM and returns response.
        Args:
            images (list): List of image paths or PIL images (for Gemini/GPT-4V).
        """

        # --- Antigravity Logic ---
        # --- Antigravity Logic ---
        # Supports "direct/" prefix for fast API access, or standard opencode/ names for CLI
        # Opencode CLI models: google/*, zai-*/*, or anything with "antigravity" or "direct/"
        is_opencode_model = (
            "antigravity" in self.model or 
            "direct/" in self.model or 
            self.model.startswith("google/") or 
            self.model.startswith("zai-")
        )
        if is_opencode_model:
            # Construct prompt_str FIRST so it's available for fallbacks if client init fails
            # Convert messages to prompt (string) for simple generation
            full_prompt = []
            system_instruction = "You are a helpful AI Agent."
            
            for m in messages:
                role = m["role"]
                content = m["content"]
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    full_prompt.append(content)
                elif role == "assistant" or role == "model":
                    full_prompt.append(f"Model: {content}")
            
            # --- STRUCTURED PROMPT FOR TOOL USE ---
            prompt_str = f"""
System: {system_instruction}

HISTORY:
"""
            for m in messages:
                role = m['role'].upper()
                content = m['content']
                if role == "TOOL":
                    prompt_str += f"\nObservation: {content}"
                elif role == "ASSISTANT":
                    prompt_str += f"\nAssistant: {content}"
                elif role == "USER":
                    prompt_str += f"\nUser: {content}"
                elif role == "SYSTEM":
                    prompt_str += f"\nSystem: {content}"
                else:
                    prompt_str += f"\n{role}: {content}"
            
            if tools:
                # Inject Tool Definitions
                tools_desc = json.dumps(tools, indent=2)
                prompt_str += f"""

## AVAILABLE TOOLS
You have access to the following tools. To call a tool, you MUST output a JSON object strictly in this format:
{{
    "tool_calls": [
        {{ "id": "call_unique_id", "type": "function", "function": {{ "name": "tool_name", "arguments": {{ "arg1": "val1" }} }} }}
    ]
}}
IMPORTANT: Output ONLY ONE tool call per turn. Do not parallelize steps. Wait for the result before proceeding.

Tools Schema:
{tools_desc}

INSTRUCTIONS:
1. Review the HISTORY.
2. If you need to perform an action, use a tool by outputting the JSON above.
3. If you have completed the task or need to ask a question, output plain text.
4. DO NOT narrate what you are doing. JUST ACT.
"""

            try:
                from src.antigravity_client import AntigravityClient
                # Inject timeout and API key for direct access
                timeout = getattr(Config, "LLM_TIMEOUT", 60)
                
                # Determine mode: Direct Gemini, Direct GLM, or Opencode CLI
                target_model = self.model
                text_response = ""

                # [NEW] Local Model Retry Logic
                is_local_model = "local-network" in self.model or "localhost" in self.model
                local_max_retries = 5 if is_local_model else 1
                
                last_error = None
                
                for attempt in range(local_max_retries):
                    try:
                        if self.model == "direct/local-network-5090":
                             from src.ollama_brain import OllamaBrain 
                             ip = getattr(Config, "LOCAL_SERVER_5090_IP", "10.24.12.85")
                             model_name = getattr(Config, "LOCAL_SERVER_5090_MODEL", "glm-4.7-flash:latest")
                             if attempt > 0: print(f"   [LLM-Client] Retry {attempt+1}/{local_max_retries}: Local Ollama (5090)...")
                             brain = OllamaBrain(server_ip=ip, model=model_name)
                             text_response = brain.generate_response(prompt_str, system_prompt=system_instruction, timeout=timeout)
                        
                        elif self.model == "direct/local-network-4090":
                             from src.ollama_brain import OllamaBrain 
                             ip = getattr(Config, "LOCAL_SERVER_4090_IP", "10.24.12.81")
                             model_name = getattr(Config, "LOCAL_SERVER_4090_MODEL", "qwen3:32b")
                             if attempt > 0: print(f"   [LLM-Client] Retry {attempt+1}/{local_max_retries}: Local Ollama (4090)...")
                             brain = OllamaBrain(server_ip=ip, model=model_name)
                             text_response = brain.generate_response(prompt_str, system_prompt=system_instruction, timeout=timeout)

                        elif self.model.startswith("direct/glm"):
                            # Direct GLM API Path
                            glm_model = self.model.replace("direct/", "")
                            glm_key = getattr(Config, "GLM_API_KEY", None)
                            if not glm_key:
                                raise ValueError(f"direct/glm model requested but GLM_API_KEY is missing")
                            if attempt == 0: print(f"   [LLM-Client] Sending Request to {self.model} (GLM Direct)...")

                            client = GLMDirectClient(api_key=glm_key, model=glm_model, timeout=timeout)
                            response = client.generate_content(prompt_str)
                            text_response = response.text
                        
                        elif self.model.startswith("direct/"):
                            # Direct Gemini API Path
                            target_model = self.model.replace("direct/", "")
                            direct_key = getattr(Config, "GEMINI_API_KEY", None)
                            if not direct_key:
                                raise ValueError("GEMINI_API_KEY not found in Config or Environment")
                            if attempt == 0: print(f"   [LLM-Client] Sending Request to {self.model} (Gemini Direct)...")
                            client = AntigravityClient(model_name=target_model, timeout=timeout, api_key=direct_key)
                            response = client.generate_content(prompt_str)
                            text_response = response.text
                        
                        else:
                            # Opencode CLI Path
                            if attempt == 0: print(f"   [LLM-Client] Sending Request to {self.model} (Antigravity CLI)...")
                            client = AntigravityClient(model_name=target_model, timeout=timeout)
                            response = client.generate_content(prompt_str)
                            text_response = response.text

                        # Check for error in response
                        if text_response.startswith("Error:"):
                             raise RuntimeError(text_response)
                             
                        # Success! Break the retry loop
                        break
                    
                    except Exception as e:
                        last_error = e
                        if is_local_model and attempt < local_max_retries - 1:
                            print(f"   [LLM-Client] Local Model Error ({e}). Retrying ({attempt+1}/{local_max_retries})...")
                            import time
                            time.sleep(2)
                            continue
                        else:
                            # Re-raise to trigger global fallback
                            raise e

                # Parse for JSON Tool Calls (Shared logic)
                return self._parse_tool_calls(text_response)
                
            except Exception as e:
                print(f"   [LLM-Client] Primary Model Error ({self.model}): {e}")
                # Chain failover - keep trying until we succeed or exhaust all models
                while self._try_failover():
                    try:
                        print(f"   [LLM-Client] [WARNING] Switching to Fallback: {self.model} (Attempt {self._failover_attempts}/{Config.MAX_FAILOVER_LOOPS})")
                        timeout = getattr(Config, "LLM_TIMEOUT", 60)
                        
                        if self.model == "direct/local-network-5090":
                             from src.ollama_brain import OllamaBrain 
                             ip = getattr(Config, "LOCAL_SERVER_5090_IP", "10.24.12.85")
                             m_name = getattr(Config, "LOCAL_SERVER_5090_MODEL", "glm-4.7-flash:latest")
                             print(f"   [LLM-Client] Routing to Local Network Ollama (5090 - {ip})...")
                             brain = OllamaBrain(server_ip=ip, model=m_name)
                             text_response = brain.generate_response(prompt_str, system_prompt=system_instruction, timeout=timeout)
                             
                        elif self.model == "direct/local-network-4090":
                             from src.ollama_brain import OllamaBrain 
                             ip = getattr(Config, "LOCAL_SERVER_4090_IP", "10.24.12.81")
                             m_name = getattr(Config, "LOCAL_SERVER_4090_MODEL", "qwen3:32b")
                             print(f"   [LLM-Client] Routing to Local Network Ollama (4090 - {ip})...")
                             brain = OllamaBrain(server_ip=ip, model=m_name)
                             text_response = brain.generate_response(prompt_str, system_prompt=system_instruction, timeout=timeout)
                        
                        elif self.model.startswith("direct/glm"):
                            glm_model = self.model.replace("direct/", "")
                            glm_key = getattr(Config, "GLM_API_KEY", None)
                            if not glm_key: continue
                            print(f"   [LLM-Client] Sending Request to {self.model} (GLM Direct)...")

                            client = GLMDirectClient(api_key=glm_key, model=glm_model, timeout=timeout)
                            response = client.generate_content(prompt_str)
                            text_response = response.text
                        
                        elif self.model.startswith("direct/"):
                            target_model = self.model.replace("direct/", "")
                            direct_key = getattr(Config, "GEMINI_API_KEY", None)
                            if not direct_key: continue
                            print(f"   [LLM-Client] Sending Request to {self.model} (Gemini Direct)...")
                            client = AntigravityClient(model_name=target_model, timeout=timeout, api_key=direct_key)
                            response = client.generate_content(prompt_str)
                            text_response = response.text
                        
                        else:
                            print(f"   [LLM-Client] Sending Request to {self.model} (Antigravity CLI)...")
                            client = AntigravityClient(model_name=self.model, timeout=timeout)
                            response = client.generate_content(prompt_str)
                            text_response = response.text
                        
                        if text_response.startswith("Error:"):
                             raise RuntimeError(text_response)

                        return self._parse_tool_calls(text_response)

                    except Exception as retry_e:
                        error_str = str(retry_e).lower()
                        rate_limit_keywords = ["rate limit", "quota", "429", "too many requests", "resource exhausted", "exceeded", "billing", "usage limit"]
                        if any(kw in error_str for kw in rate_limit_keywords):
                            print(f"   [LLM-Client] [WARNING] Rate Limit/Quota Error ({self.model}). Switching immediately...")
                        else:
                            print(f"   [LLM-Client] Fallback Error ({self.model}): {retry_e}")
                        continue
                
                # All models exhausted
                return None

        if "gemini" in self.model:
            try:
                return self._generate_gemini_response(messages, temperature, tools, images)
            except Exception as e:

                # If Gemini fails, try failover if we are NOT already on backup?
                # Usually backup is the last resort. But maybe we have circular backup? No.
                # Assuming backup is Gemini, if it fails, we are done.
                # UNLESS we started with Gemini as primary and have GLM as backup.
                # Let's keep it simple: Try failover if configured.
                if self._try_failover():
                    return self.generate_response(messages, temperature, tools)
                raise e
            
        # --- GLM-4 Logic (zhipuai SDK) ---
        # --- GLM-4 Logic (zhipuai SDK) ---
        try:
            # Check for conflict: zhipuai < 2.2 needs pyjwt < 2.9, but mcp needs > 2.10
            # If pyjwt is too new, ZhipuAI SDK might fail at import or runtime.
            # We explicitly check this to trigger fallback to robust legacy requests.
            import jwt
            if jwt.__version__ >= "2.10.0":
                 print("   [LLM-Client] PyJWT version too high for ZhipuAI SDK. Forcing Legacy Mode.")
                 raise ImportError("Force Legacy due to PyJWT conflict")

            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=self.api_key)
            print(f"   [LLM-Client] Sending Request to {self.model} (GLM-4 SDK)...")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=0.9,
                tools=tools,
                tool_choice="auto" if tools else None,
                timeout=getattr(Config, "LLM_TIMEOUT", 300)
            )
            
            msg = response.choices[0].message
            # Convert to dict for compatibility
            return {
                "role": msg.role,
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in (msg.tool_calls or [])
                ] if msg.tool_calls else None
            }

            
        except ImportError:
            print("[LLM-Client] zhipuai package not found. Using legacy requests...")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "401" in err_str or "400" in err_str or "1113" in err_str or "1211" in err_str:
                if self._try_failover():
                    return self.generate_response(messages, temperature, tools)
            print(f"   [LLM-Client] GLM SDK Error: {e}. Falling back to legacy...")

        # --- Legacy GLM-4 Logic ---
        token = self._generate_token(self.api_key)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.9
        }
        if tools:
            data["tools"] = tools
            data["tool_choice"] = "auto"
            
        if "web_search" not in [t.get("type") for t in (data.get("tools") or [])]:
             web_tool = {"type": "web_search", "web_search": {"enable": True, "search_result": True}}
             if "tools" not in data: data["tools"] = []
             data["tools"].append(web_tool)
        
        try:
            print(f"   [LLM-Client] Sending Request to {self.model} (GLM-4)...")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=data,
                timeout=getattr(Config, "LLM_TIMEOUT", 300)
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]
        except Exception as e:
            print(f"   [LLM-Client] GLM Error: {e}")
            return None

    def _generate_gemini_response(self, messages, temperature, tools_schema, images=None):
        """
        Adapter for Gemini API. Supports Images.
        """
        import google.generativeai as genai
        import PIL.Image

        
        # 1. Setup Model
        gemini_key = getattr(Config, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
        if gemini_key:
            genai.configure(api_key=gemini_key)
            
        # 2. Convert Tools (JSON Schema -> Gemini Function Declaration)
        # Note: genai.GenerativeModel can accept 'tools' as a list of functions OR Declarations.
        # However, converting raw JSON schema to Gemini Tools is complex.
        # STRATEGY: For this Sim Agent, we will use a simplified approach.
        # We will pass the raw functions if available, or just rely on text generation if tools are complex.
        # BUT User wants Sim Agent to work. Sim Agent relies on function calling.
        # Gemini Auto-Function calling needs actual python functions or careful schema.
        # WORKAROUND: We will assume 'tools' argument provides descriptions for System Prompt
        # and we force JSON output, then parse it manually as a tool call if strictly needed.
        # OR: We use the Google System Instruction to emulate tool use.
        
        system_instruction = "You are an agent. "
        chat_history = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction += f"\n{content}"
            elif role == "user":
                chat_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                text = content if content else ""
                if not text and "tool_calls" in msg:
                     # Describe tool calls for simple history replay
                     funcs = [tc['function']['name'] for tc in msg['tool_calls']]
                     text = f"I will call the following tools: {', '.join(funcs)}"
                if not text: text = "Thinking..."
                chat_history.append({"role": "model", "parts": [text]})
            elif role == "tool":
                # Gemini doesn't easily support replaying tool outputs in this ad-hoc way without full chat session reconstruction
                # For simplicity in this 'Adapter', we append tool outputs as User messages for context.
                chat_history.append({"role": "user", "parts": [f"Tool Output ({msg.get('tool_call_id', 'unknown')}): {content}"]})

        # Convert tools_schema to text description for System Prompt (Robust fallback)
        if tools_schema:
            tools_desc = json.dumps(tools_schema, indent=2)
            system_instruction += f"""
            
            ## AVAILABLE TOOLS
            You have access to the following tools. To call a tool, you MUST output a JSON object:
            {{
                "tool_calls": [
                    {{ "id": "call_unique_id", "type": "function", "function": {{ "name": "tool_name", "arguments": {{ "arg1": "val1" }} }} }}
                ]
            }}
            
            Tools Schema:
            {tools_desc}
            
            If no tool is needed, just output text.
            IMPORTANT: If you call a tool, do NOT output any other text, just the JSON.
            """

        if "gemini" in self.model:
            if self.api_key:
                genai.configure(api_key=self.api_key)
            else:
                 print("   [LLM-Client] Warning: No API Key provided for Gemini.")
                 
        model = genai.GenerativeModel(self.model, system_instruction=system_instruction)
        
        try:
            total_chars = len(system_instruction) + sum([sum([len(p) if isinstance(p, str) else 0 for p in msg.get('parts', [])]) for msg in chat_history])
            print(f"   [LLM-Client] Sending Request to {self.model} (Gemini)...")
            print(f"   [Debug] Input Size: ~{total_chars} chars (System: {len(system_instruction)}, History: {total_chars - len(system_instruction)})")
            # We treat the last user message as the prompt, and the rest as history
            # Retry Loop for Rate Limits
            max_retries = 5
            wait_time = 15 # Start with 15s for Gemini Flash Free Tier (5 RPM = 12s per req)
            
            for attempt in range(max_retries):
                try:
                    if chat_history and chat_history[-1]["role"] == "user":
                        last_msg = chat_history.pop()
                        # Inject images into the last user message parts if provided
                        if images:
                            for img_path in images:
                                try:
                                    img = PIL.Image.open(img_path)
                                    last_msg["parts"].append(img)
                                    print(f"   [LLM-Client] Attached Image: {img_path}")
                                except Exception as e:
                                    print(f"   [LLM-Client] Failed to load image {img_path}: {e}")
                        
                        chat = model.start_chat(history=chat_history)
                        response = chat.send_message(last_msg["parts"])

                    else:
                        # Single turn (e.g. if system prompt rich)
                        parts = chat_history[-1]["parts"] if chat_history else ["Start"]
                        if images:
                             for img_path in images:
                                try:
                                    img = PIL.Image.open(img_path)
                                    parts.append(img)
                                except Exception as e:
                                    print(f"   [LLM-Client] Failed to load image {img_path}: {e}")
                        response = model.generate_content(parts)

                    break # Success
                except Exception as e:
                    if "429" in str(e) or "ResourceExhausted" in str(e) or "InternalServerError" in str(e) or "500" in str(e):
                        print(f"   [LLM-Client] API Error ({e}). Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        wait_time = min(wait_time * 2, 60) # Exponential Backoff capped at 60s
                        # Put back the popped message if needed for next retry
                        if 'last_msg' in locals():
                             chat_history.append(last_msg)
                    else:
                        raise e
            
            # Handle StopCandidateException implicitly by checking parts
            try:
                text_response = response.text
            except Exception:
                # E.g. StopCandidateException due to safety or malformed function call
                # Try to extract partial parts
                if response.parts:
                    text_response = response.parts[0].text
                else:
                    return {"role": "assistant", "content": "Error: Model stopped unexpectedly (Safety or Malformed Output)."}
            
            # 3. Parse for Pseudo-Tool Calls (JSON mode)
            # We instructed the model to output OpenAI-style JSON for tools.
            try:
                # Attempt to find JSON blob
                import re
                json_match = re.search(r"\{.*\}", text_response, re.DOTALL)
                if json_match:
                    possible_json = json.loads(json_match.group(0))
                    if "tool_calls" in possible_json:
                        return {
                            "content": None,
                            "tool_calls": possible_json["tool_calls"],
                            "role": "assistant"
                        }
            except:
                pass # Not a tool call, just text
                
            return {"role": "assistant", "content": text_response}
            
        except Exception as e:
            print(f"   [LLM-Client] Gemini Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_json_from_text(self, text: str) -> dict | None:
        """
        Extract JSON object from text that may contain non-JSON prefix/suffix.
        Handles cases like log messages mixed with JSON response.

        Example:
            "[auto-update-checker] log message\n{\"key\": \"value\"}"
            -> {"key": "value"}
        """
        # Find the first '{' and match braces to find the complete JSON
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Brace counting to find matching '}'
        brace_count = 0
        end_idx = -1
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break

        if end_idx == -1:
            return None

        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _parse_tool_calls(self, text_response: str):
        """Robustly parse for JSON Tool Calls in LLM response using brace counting."""
        try:
            # Locate start of JSON
            start_marker = "{"
            start_idx = text_response.find(start_marker)
            if start_idx == -1:
                return {"role": "assistant", "content": text_response}

            # Brace counting to find end of JSON object
            brace_count = 0
            json_str = ""
            found_json = False
            
            for i in range(start_idx, len(text_response)):
                char = text_response[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                
                if brace_count == 0:
                    # Potential end of JSON
                    json_candidates = [text_response[start_idx:i+1]]
                    
                    # Try parsing
                    for candidate in json_candidates:
                        try:
                            # Cleanup potential markdown wrapper inside the block
                            clean_cand = candidate.replace("```json", "").replace("```", "")
                            data = json.loads(clean_cand)
                            if "tool_calls" in data:
                                thought = text_response[:start_idx].strip()
                                return {
                                    "content": thought,
                                    "tool_calls": data["tool_calls"],
                                    "role": "assistant"
                                }
                        except json.JSONDecodeError:
                            continue
                    
                    # If we reached 0 but didn't return, maybe we shouldn't stop? 
                    # But usually top-level object closes at 0.
                    # If parsing failed, it might be partial text.
                    # Continue searching if this wasn't it? 
                    # Actually, if brace_count hits 0, that's a closed block.
                    # If it wasn't valid JSON, we might have started at a wrong '{'.
                    # But for now, let's assume the first '{' is the start.
                    pass

        except Exception as e:
            print(f"[LLM-Client] Parse Error: {e}")
            pass
        
        return {"role": "assistant", "content": text_response}

    def generate_json(self, messages, temperature=0.1):
        """
        Requests JSON output from LLM and parses it.
        """
        # [PREVIOUS] if "gemini" in self.model:
        #     # Gemini Native JSON mode - Used hardcoded google.generativeai
        #     import google.generativeai as genai
        #     gemini_key = getattr(Config, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY"))
        #     genai.configure(api_key=gemini_key)
        #     model = genai.GenerativeModel(self.model)
        #     try:
        #         response = model.generate_content(
        #             full_prompt,
        #             generation_config={"response_mime_type": "application/json"}
        #         )
        #         return json.loads(response.text)
        #     except:
        #         return {}

        # NEW: For Antigravity and all other providers, use generate_response with JSON instruction
        # Enforce JSON in system prompt if not present
        if messages[0]["role"] == "system":
            if "JSON" not in messages[0]["content"]:
                messages[0]["content"] += " Return response in pure JSON format."
        
        msg_obj = self.generate_response(messages, temperature)
        if not msg_obj or not msg_obj.get("content"):
            return {}
            
        content = msg_obj["content"]
        
        # Strip markdown codes if any
        clean_content = content.replace("```json", "").replace("```", "").strip()
        
        # 1차 시도: 직접 파싱
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            pass

        # 2차 시도: JSON 블록 추출 후 파싱 (로그가 섞여있는 경우 대응)
        extracted = self._extract_json_from_text(clean_content)
        if extracted:
            print(f"   [LLM-Client] JSON extracted from mixed content.")
            return extracted

        # 최종 실패
        print(f"   [LLM-Client] JSON Parse Failed. Content: {clean_content[:200]}...")
        return {}

    # --- Domain Specific Helpers ---

    def query_physics_knowledge(self, material_formula):
        """
        Retrieves crystal structure data.
        """
        prompt = f"""
        You are an expert material scientist. 
        Provide the crystal structure and lattice constants for: {material_formula}.
        
        Format your response as a JSON object with keys:
        - structure: (e.g., 'diamond', 'fcc', 'hcp', 'wurtzite')
        - a: (float, Angstroms)
        - b: (float, optional)
        - c: (float, optional)
        - alpha, beta, gamma: (optional if non-90)
        
        If precise data is unknown, provide a best scientific estimate.
        """
        
        messages = [
            {"role": "system", "content": "You are a physics API helper. Respond in JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        return self.generate_json(messages)

    def analyze_potential_strategy(self, substrate_elements, ion_elements, available_files=None):
        """
        Decides on Forcefield strategy.
        Args:
            substrate_elements: List of substrate element symbols
            ion_elements: List of ion element symbols  
            available_files: List of available force field files in library
        """
        # Format available files for prompt
        files_info = ""
        if available_files:
            files_info = f"""
        Available Force Field Files in Library:
        {available_files}
        """
        
        prompt = f"""
        Design a LAMMPS forcefield strategy for a sputtering simulation.
        
        System:
        - Substrate Elements: {substrate_elements}
        - Ion Elements: {ion_elements}
        {files_info}
        
        Task:
        1. Select the best potential style for the substrate (e.g., 'tersoff', 'eam', 'sw').
        2. Check if a suitable file exists in the Available Force Field Files list.
        3. If NO suitable file exists locally, you may specify a 'download_url' to a valid potential file (e.g. from NIST or GitHub).
        4. For ion-ion interactions: Use 'zbl' for noble gas ions (Ar, He, Ne, Kr, Xe).
        5. MANDATORY: The interaction between Substrate Elements and Ion Elements MUST be 'zbl'.
        6. MANDATORY: The interaction between Ion Elements MUST be 'zbl'.
        
        Output JSON:
        {{
            "substrate_style": "...",
            "substrate_file": "filename (from list) OR new_filename (if downloading)",
            "download_url": "URL (optional, ONLY if file not in library)",
            "ion_style": "zbl",
            "ion_file": "NULL",
            "interaction_policy": "hybrid/overlay",
            "reasoning": "..."
        }}
        """
        
        messages = [
            {"role": "system", "content": "You are a MD Simulation Architect. Respond in JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        return self.generate_json(messages)
