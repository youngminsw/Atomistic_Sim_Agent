import os

class Config:
    # =========================================================================
    # MULTI-AGENT CONFIGURATION (Server - Antigravity via Opencode)
    # =========================================================================
    
    # Server supports OAuth, so use Antigravity (rate limit free)
    # Make sure to run 'opencode auth login' first
    
    # [1] SIMULATION AGENT
    # Note: "direct/" prefix bypasses Opencode CLI and uses API directly
    # SIM_PROVIDER is deprecated - routing is determined by model name prefix
    # "direct/" prefix = Direct API (requires GEMINI_API_KEY), no opencode needed
    # "google/" prefix = Opencode CLI (requires WSL + auth setup)
    SIM_MODEL_NAME = "google/antigravity-gemini-3-flash"  # Opencode CLI (antigravity)
    SIM_API_KEY = None  # Only needed for "direct/" models
    
    # Path to Opencode CLI (via WSL Ubuntu)
    # opencode is installed in WSL Ubuntu at ~/.opencode/bin/opencode
    OPENCODE_USE_WSL = True  # Set to False if opencode is installed natively on Windows
    OPENCODE_WSL_DISTRO = "Ubuntu-20.04"
    OPENCODE_PATH = "$HOME/.opencode/bin/opencode"  # WSL path (used when OPENCODE_USE_WSL=True)
    
    # [NEW] LOCAL NETWORK OLLAMA SERVER 1 (RTX 5090 - High Spec)
    LOCAL_SERVER_5090_IP = "10.24.12.85"
    LOCAL_SERVER_5090_MODEL = "glm-4.7-flash:latest"
    
    # [NEW] LOCAL NETWORK OLLAMA SERVER 2 (RTX 4090 - Standard Spec)
    LOCAL_SERVER_4090_IP = "10.24.12.81"
    LOCAL_SERVER_4090_MODEL = "qwen3:32b"  # 32B fits in 24GB VRAM (Fast)
    
    # GLM API Key (for fallback)
    GLM_API_KEY = os.environ.get("GLM_API_KEY")
    
    # Direct Gemini API Key (Bypass Opencode CLI)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    # [2] INSPECTION AGENT (MCP Server)
    # Note: Inspection Agent runs as a separate MCP server
    # Model is configured in the MCP server's own config, not here
    INSPECTION_MODEL_NAME = "google/antigravity-gemini-3-pro-high"  # Opencode CLI (antigravity)
    INSPECTION_API_KEY = None  # Deprecated - kept for backward compatibility

    # [3] FALLBACK MODELS (in order of preference)
    # When rate limit hit, try next model in chain
    # "direct/" prefix = bypass CLI, use API key directly
    FALLBACK_MODELS = [
        # PRIORITY 1: Local Network (Fast & No CLI dependency)
        "direct/local-network-5090",             # RTX 5090 (GLM 4.7 flash)
        "direct/local-network-4090",
        
        # PRIORITY 2: Direct API (Need Keys)
        "direct/gemini-3-pro-preview",
        "direct/gemini-3-flash-preview",
        "direct/glm-4.7",
        
        # PRIORITY 3: Opencode CLI (Only if installed)
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
        "zai-coding-plan/glm-4.7",
    ]
    


    # =========================================================================
    # SYSTEM SETTINGS
    # =========================================================================
    
    # Timeout for LLM requests (seconds)
    LLM_TIMEOUT = 60  # 5 minutes
    MAX_FAILOVER_LOOPS = 10

    # SYSTEM MAPPINGS (Do Not Edit Below This Line)
    # -------------------------------------------------------------------------
    # LLMClient (Sim Agent) uses these generic keys:
    API_KEY = SIM_API_KEY
    MODEL_NAME = SIM_MODEL_NAME
    # API Keys
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    # Materials Project API Key
    MP_API_KEY = os.environ.get("MP_API_KEY", "xU0NUINqfsF9AvE1EjQd0ltzD3lFqr8A") # User provided key default
    # =========================================================================

    # Paths
    WORK_DIR_PREFIX = "workdir_"
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
    REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "../Reference")
    
    # Execution
    # Execution
    MAX_RETRIES = 3
    # LLM_TIMEOUT and MAX_FAILOVER_LOOPS are defined above
    INSPECTION_SERVER_URL = "http://localhost:8000"
    
    # LAMMPS Configuration
    if os.name == 'nt' or os.environ.get('OS','') == 'Windows_NT':
        LAMMPS_EXECUTABLE = r"C:\Users\swym4\AppData\Local\LAMMPS 64-bit 19Nov2024-MSMPI\bin\lmp.exe"
        MPI_EXECUTABLE = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
    else:
        LAMMPS_EXECUTABLE = "/mnt/c/Users/swym4/AppData/Local/LAMMPS 64-bit 19Nov2024-MSMPI/bin/lmp.exe"
        MPI_EXECUTABLE = "/mnt/c/Program Files/Microsoft MPI/Bin/mpiexec.exe"
    LAMMPS_NP = 8  # Default number of MPI processes
