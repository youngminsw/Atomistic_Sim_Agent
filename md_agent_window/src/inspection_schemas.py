from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class BaseRequest(BaseModel):
    work_dir: str = Field(..., description="Absolute path to the working directory where files are located.")

class ValidateStructureRequest(BaseRequest):
    data_file: str = Field(..., description="Name of the LAMMPS data file to validate (e.g., 'structure.data').")
    formula: str = Field(..., description="Chemical formula to check stoichiometry against (e.g., 'SiO2').")

class ValidateLammpsInputRequest(BaseRequest):
    input_file: str = Field(..., description="Name of the LAMMPS input script to validate (e.g., 'in.sputtering').")
    type_map: Dict[str, int] = Field(..., description="Mapping of element names to LAMMPS type IDs (e.g., {'Si': 1, 'O': 2}).")
    data_file: Optional[str] = Field(None, description="Optional name of the data file referenced by the input script.")

class ValidateParamsRequest(BaseRequest):
    tool_name: str = Field(..., description="Name of the tool being validated (e.g., 'create_projectile').")
    params: Dict[str, Any] = Field(..., description="Dictionary of parameters to validate.")
    log_file_path: Optional[str] = Field("", description="Optional path to a log file for context.")

class ReviewPlanRequest(BaseRequest):
    input_script_name: str = Field(..., description="Name of the input script being generated.")
    template_content: str = Field(..., description="Content of the template being used.")
    parameters: Dict[str, Any] = Field(..., description="Parameters intended to populate the template.")

class InspectSimulationRequest(BaseRequest):
    log_filename: str = Field(..., description="Name of the LAMMPS log file (e.g., 'log.lammps').")
    dump_filename: str = Field(..., description="Name of the dump/trajectory file (e.g., 'dump.lammpstrj').")
    context_description: str = Field(..., description="Context of what is being checked (e.g., 'Equilibration phase').")
