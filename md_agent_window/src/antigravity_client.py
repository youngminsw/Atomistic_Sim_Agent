import subprocess
import os
import time

class AntigravityClient:
    """
    Wrapper for 'opencode run' CLI.
    Bypasses direct API calls by using the CLI which handles OAuth sessions.
    """
    MAX_RETRIES = 5  # Increased from 2
    TIMEOUT_SECONDS = 60  # 1 minute
    
    # Rate limit keywords - trigger immediate model switch
    RATE_LIMIT_KEYWORDS = ["rate limit", "quota", "429", "too many requests", "resource exhausted"]
    
    def __init__(self, model_name="google/antigravity-gemini-3-flash", timeout=None, api_key=None):
        self.model_name = model_name
        self.timeout = timeout if timeout is not None else self.TIMEOUT_SECONDS
        self.api_key = api_key
        self.genai_model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                # Model name mapping - extract real model name from direct/ prefix
                real_model = model_name.replace("direct/", "") if model_name.startswith("direct/") else model_name
                # Handle various prefixes
                if "/" in real_model:
                    real_model = real_model.split("/")[-1]
                
                self.genai_model = genai.GenerativeModel(real_model)
                print(f"[AntigravityClient] Direct API Mode Enabled ({real_model})")
            except Exception as e:
                print(f"[AntigravityClient] Failed to init Direct API: {e}. Aborting Direct Init.")
                # Important: Do NOT revert to CLI if Direct API was requested (api_key present).
                # CLI likely won't support the 'direct/' model name without mapping.
                if self.api_key:
                    raise RuntimeError(f"Direct API Initialization Failed: {e}. Use 'pip install google-generativeai' if missing.")

    def _is_rate_limit_error(self, error_msg: str) -> bool:
        """Check if error message indicates rate limiting."""
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in self.RATE_LIMIT_KEYWORDS)

    def generate_content(self, prompt):
        """
        Executes generation via Direct API or 'opencode run' CLI.
        Returns an object compatible with google.generativeai response (response.text).
        
        Smart retry logic:
        - Rate limit errors: raise immediately to trigger model switch
        - Timeout/other errors: retry up to MAX_RETRIES times
        """
        # 1. Direct API Path (Fast)
        if self.genai_model:
            try:
                response = self.genai_model.generate_content(prompt)
                
                # [Token Tracking]
                try:
                    meta = response.usage_metadata
                    if meta:
                        from src.token_tracker import TokenTracker
                        work_dir = os.getcwd()
                        tracker = TokenTracker(work_dir)
                        tracker.log_usage(
                            self.model_name,
                            meta.prompt_token_count,
                            meta.candidates_token_count
                        )
                except Exception as e:
                    print(f"      [AntigravityClient] Tracking Warning: {e}")
                    
                return response
            except Exception as e:
                error_str = str(e)
                print(f"      [AntigravityClient] [WARNING] Direct API Error: {e}", flush=True)
                
                # Rate limit -> switch model immediately
                if self._is_rate_limit_error(error_str):
                    raise RuntimeError(f"Rate limit detected - switching model: {error_str}") from None
                
                print(f"      [AntigravityClient] Falling back to CLI...", flush=True)
                # Fallthrough to CLI
        
        # CLI Path (Slow/Robust Fallback)
        from src.config import Config

        # Check if we should use WSL to call opencode
        use_wsl = getattr(Config, "OPENCODE_USE_WSL", False)

        if use_wsl:
            # Build WSL command
            wsl_distro = getattr(Config, "OPENCODE_WSL_DISTRO", "Ubuntu")
            opencode_wsl_path = getattr(Config, "OPENCODE_PATH", "$HOME/.opencode/bin/opencode")
            # WSL command: wsl -d Ubuntu -- bash -c '$HOME/.opencode/bin/opencode run --model ...'
            cmd = ["wsl", "-d", wsl_distro, "--", "bash", "-c",
                   f"{opencode_wsl_path} run --model {self.model_name}"]
        else:
            # Native Windows path
            opencode_cmd = os.path.normpath(getattr(Config, "OPENCODE_PATH", "opencode"))
            cmd = [opencode_cmd, "run", "--model", self.model_name]
        
        # Determine timeout
        timeout = self.timeout if self.timeout else 120
        
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"      [AntigravityClient] Attempt {attempt}/{self.MAX_RETRIES} started (via stdin, timeout={timeout}s)...", flush=True)
                result = subprocess.run(
                    cmd,
                    input=prompt,   # Pass prompt via STDIN to avoid WinError 206 (CLI length limit)
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=timeout,
                    shell=False,    # Using shell=False is cleaner for stdin piping
                    encoding='utf-8',
                    errors='replace'
                )
                
                output = result.stdout.strip()
                print(f"      [AntigravityClient] CLI Response Received ({len(output)} chars).")
                
                # Wrap in a simple object to match existing API usage (response.text)
                class ResponseWrapper:
                    def __init__(self, text):
                        self.text = text
                        
                return ResponseWrapper(output)
                
            except subprocess.TimeoutExpired:
                print(f"      [AntigravityClient] [WARNING] Timeout (attempt {attempt}/{self.MAX_RETRIES}).", flush=True)
                if attempt < self.MAX_RETRIES:
                    print(f"      [AntigravityClient] Retrying in 3s...", flush=True)
                    time.sleep(3)
                else:
                    raise RuntimeError(f"Opencode CLI timed out after {self.MAX_RETRIES} attempts") from None
                    
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr or str(e)
                print(f"      [AntigravityClient] [ERROR] CLI Error: {error_msg}", flush=True)
                
                # Rate limit -> switch model immediately
                if self._is_rate_limit_error(error_msg):
                    raise RuntimeError(f"Rate limit detected - switching model: {error_msg}") from None
                
                # Other errors: check if retryable
                if "endpoints failed" in error_msg.lower():
                    if attempt < self.MAX_RETRIES:
                        print(f"      [AntigravityClient] Retrying in 3s...", flush=True)
                        time.sleep(3)
                        continue
                
                raise RuntimeError(f"Opencode CLI failed: {error_msg}") from None
                
            except FileNotFoundError as e:
                raise RuntimeError(f"Opencode CLI not found at path: '{opencode_cmd}' (exists={os.path.exists(opencode_cmd)}). Original error: {e}")

class AntigravityAuth:
    # Legacy/Unused stub to satisfy any imports
    def get_token(self):
        return "stub-token-managed-by-cli"

