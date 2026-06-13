"""
Inspection Agent MCP Server - Conversational Interface
Provides a single consult() tool for Sim Agent communication.
"""

import os
import sys
import json
import logging
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any

# Configure logging with prefix for easy identification
logging.basicConfig(
    format='[InspAgent-Server] %(message)s',
    level=logging.INFO
)

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inspection_agent import InspectionAgent

# Initialize FastMCP Server
mcp = FastMCP("Inspection Agent", dependencies=["numpy", "ovito"])


@mcp.tool()
def consult(request: str, info: dict = None, history: str = "") -> Dict[str, Any]:
    """
    Conversational interface to Inspection Agent.
    
    Args:
        request: Natural language request describing what to analyze/validate.
                 Examples:
                 - "Validate the structure file and check stoichiometry for Si"
                 - "Review the LAMMPS input script for errors"
                 - "Inspect simulation results in the work directory"
                 - "Review force field strategy for Si + Ar sputtering"
        info: Context dictionary with relevant paths and parameters.
              Common keys:
              - file_path: Path to a specific file
              - work_dir: Working directory path
              - formula: Chemical formula (e.g., "Si", "SiO2")
              - type_map: Element to type ID mapping
              - strategy: Force field strategy dict
        history: Sim Agent's history summary for context (user goal + recent actions)
    
    Returns:
        dict: Analysis results with structure depending on request type.
              Always includes: success, response
              May include: valid, errors, warnings, recommendations
    """
    print(f"[InspAgent-Server] CallToolRequest: consult(request='{request[:50]}...')")
    # Handle info if passed as string
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except:
            info = {"raw_info": info}
    
    if info is None:
        info = {}
    
    # Initialize agent and process request
    agent = InspectionAgent()
    return agent.process_request(request, info, history)


@mcp.tool()
def read_file(file_path: str, max_lines: int = 0, start_line: int = 0) -> Dict[str, Any]:
    """
    Direct file reading for Inspection Agent.
    Use consult() for analysis - this is for raw file access.
    """
    print(f"[InspAgent-Server] CallToolRequest: read_file(path='{file_path}')")
    if not os.path.exists(file_path):
        return {"success": False, "error": f"File not found: {file_path}"}
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if start_line > 0:
            lines = lines[start_line:]
        if max_lines > 0:
            lines = lines[:max_lines]
        
        return {
            "success": True,
            "path": file_path,
            "content": "".join(lines),
            "total_lines": total_lines,
            "lines_read": len(lines)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    port = int(os.environ.get("INSPECTION_PORT", 8000))
    print(f"[InspAgent-Server] Starting MCP Server on port {port}...")
    
    mcp.settings.host = "0.0.0.0"
    mcp.settings.port = port
    
    mcp.run(transport="sse")
