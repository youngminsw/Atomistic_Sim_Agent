from src.llm_client import LLMClient

class PhysicsResearcher:
    def __init__(self):
        self.brain = LLMClient()

    def get_crystal_params(self, formula):
        """
        Asks the AI Agent to research the material structure.
        """
        print(f"[Research] Asking AI Agent about: {formula}")
        return self.brain.query_physics_knowledge(formula)

    def recommend_potential(self, substrate_elements, ion_elements, available_files=None):
        """
        Asks the AI Agent to design the simulation physics.
        Args:
            substrate_elements: List of substrate element symbols
            ion_elements: List of ion element symbols
            available_files: List of available force field files in library
        """
        print(f"[Research] Asking AI Agent to design Force Field strategy...")
        if available_files:
            print(f"[Research] Available files provided: {len(available_files)} files")
        return self.brain.analyze_potential_strategy(substrate_elements, ion_elements, available_files)
