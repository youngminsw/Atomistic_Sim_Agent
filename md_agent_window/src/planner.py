"""
Task Planner Module
Generates, saves, and manages execution plans for MD simulations.
"""

import json
import os
from datetime import datetime
from src.llm_client import LLMClient


class TaskPlanner:
    """
    Generates structured task plans and tracks execution progress.
    Plans are saved to plan.json in the work directory.
    """
    
    PLAN_FILENAME = "plan.json"
    
    def __init__(self, work_dir: str, llm_client: LLMClient = None):
        self.work_dir = work_dir
        self.client = llm_client
        self.current_plan = None
    
    def create_plan(self, user_goal: str, workflow_guide: str = "") -> dict:
        """
        Generate a structured plan from user goal using LLM.
        
        Args:
            user_goal: The user's goal
            workflow_guide: Optional guide string describing the required workflow
        
        Returns:
            {
                "goal": str,
                "created_at": str,
                "status": "pending" | "approved" | "in_progress" | "completed",
                "steps": [
                    {"id": 1, "description": str, "tools": [str], "status": "pending"}
                ]
            }
        """
        prompt = f"""You are a MD Simulation Planning Expert.
Given a user goal, create a structured execution plan.

USER GOAL: {user_goal}

WORKFLOW GUIDE (Follow this strictly):
{workflow_guide if workflow_guide else "No specific guide provided. Use standard MD workflow."}

Create a step-by-step plan with specific actions. Each step should be:
1. Concrete and actionable
2. Include which tools to use (from: fetch_cif_file, build_structure_from_cif, build_substrate, create_projectile, generate_lammps_input, generate_slurm_script, run_simulation, bash, read_file)
3. Have clear success criteria

Output JSON ONLY:
{{
    "goal": "{user_goal}",
    "steps": [
        {{"id": 1, "description": "...", "tools": ["tool1", "tool2"], "success_criteria": "..."}},
        {{"id": 2, "description": "...", "tools": ["tool1"], "success_criteria": "..."}}
    ]
}}
"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate_json(messages, temperature=0.3)
            if response and "steps" in response:
                plan = {
                    "goal": user_goal,
                    "created_at": datetime.now().isoformat(),
                    "status": "pending",
                    "steps": []
                }
                for step in response["steps"]:
                    plan["steps"].append({
                        "id": step.get("id", len(plan["steps"]) + 1),
                        "description": step.get("description", ""),
                        "tools": step.get("tools", []),
                        "success_criteria": step.get("success_criteria", ""),
                        "status": "pending"
                    })
                self.current_plan = plan
                return plan
        except Exception as e:
            print(f"[Planner] Error creating plan: {e}")
        
        # Fallback: Simple plan
        return self._create_fallback_plan(user_goal)
    
    def _create_fallback_plan(self, user_goal: str) -> dict:
        """Create a basic plan when LLM fails."""
        plan = {
            "goal": user_goal,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
            "steps": [
                {"id": 1, "description": "Research material structure", "tools": ["fetch_cif_file"], "status": "pending"},
                {"id": 2, "description": "Build crystal structure", "tools": ["build_structure_from_cif", "build_substrate"], "status": "pending"},
                {"id": 3, "description": "Prepare force field", "tools": ["bash", "research_potential"], "status": "pending"},
                {"id": 4, "description": "Generate LAMMPS input", "tools": ["generate_lammps_input"], "status": "pending"},
                {"id": 5, "description": "Run simulation", "tools": ["run_simulation"], "status": "pending"}
            ]
        }
        self.current_plan = plan
        return plan
    
    def revise_plan(self, plan: dict, feedback: str) -> dict:
        """
        Revise plan based on Inspection Agent feedback.
        """
        prompt = f"""You are revising an MD simulation plan based on expert feedback.

ORIGINAL PLAN:
{json.dumps(plan, indent=2)}

EXPERT FEEDBACK:
{feedback}

Revise the plan to address the feedback. Keep the same JSON format.
Output the REVISED plan as JSON only.
"""
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.client.generate_json(messages, temperature=0.3)
            if response and "steps" in response:
                revised = {
                    "goal": plan["goal"],
                    "created_at": plan["created_at"],
                    "revised_at": datetime.now().isoformat(),
                    "status": "pending",
                    "revision_reason": feedback,
                    "steps": response["steps"]
                }
                # Ensure all steps have status
                for step in revised["steps"]:
                    if "status" not in step:
                        step["status"] = "pending"
                self.current_plan = revised
                return revised
        except Exception as e:
            print(f"[Planner] Error revising plan: {e}")
        
        return plan  # Return original if revision fails
    
    def save_plan(self, plan: dict = None, filepath: str = None) -> str:
        """
        Save plan to JSON file.
        
        Returns:
            Path to saved file
        """
        if plan is None:
            plan = self.current_plan
        if plan is None:
            raise ValueError("No plan to save")
        
        if filepath is None:
            filepath = os.path.join(self.work_dir, self.PLAN_FILENAME)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)
        
        print(f"[Planner] Plan saved to: {filepath}")
        return filepath
    
    def load_plan(self, filepath: str = None) -> dict:
        """Load plan from JSON file."""
        if filepath is None:
            filepath = os.path.join(self.work_dir, self.PLAN_FILENAME)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            self.current_plan = json.load(f)
        
        return self.current_plan
    
    def update_step_status(self, step_id: int, status: str, result: str = None):
        """
        Update status of a specific step.
        
        Args:
            step_id: Step ID to update
            status: "pending" | "in_progress" | "done" | "failed" | "skipped"
            result: Optional result/error message
        """
        if self.current_plan is None:
            return
        
        for step in self.current_plan["steps"]:
            if step["id"] == step_id:
                step["status"] = status
                step["updated_at"] = datetime.now().isoformat()
                if result:
                    step["result"] = result
                break
        
        # Update overall status
        all_statuses = [s["status"] for s in self.current_plan["steps"]]
        if all(s == "done" for s in all_statuses):
            self.current_plan["status"] = "completed"
        elif any(s == "in_progress" for s in all_statuses):
            self.current_plan["status"] = "in_progress"
        elif any(s == "failed" for s in all_statuses):
            self.current_plan["status"] = "failed"
        
        # Auto-save after update
        self.save_plan()
    
    def get_current_step(self) -> dict:
        """Get the first pending step."""
        if self.current_plan is None:
            return None
        
        for step in self.current_plan["steps"]:
            if step["status"] == "pending":
                return step
        
        return None  # All steps done
    
    def get_plan_summary(self) -> str:
        """Get human-readable plan summary."""
        if self.current_plan is None:
            return "No plan loaded."
        
        lines = [f"Goal: {self.current_plan['goal']}", "Steps:"]
        for step in self.current_plan["steps"]:
            status_icon = {
                "pending": "⬜",
                "in_progress": "🔄",
                "done": "✅",
                "failed": "❌",
                "skipped": "⏭️"
            }.get(step["status"], "?")
            lines.append(f"  {status_icon} {step['id']}. {step['description']}")
        
        return "\n".join(lines)
