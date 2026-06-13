from Orchestrator import Orchestrator

orch = Orchestrator()
req = {
    "stages": ["MD"],
    "params": {
        "ion": "Ar",
        "sub": "Si",
        "energy": 100.0,
        "events": 3
    }
}
print("Starting Programmatic MD Test...")
orch.execute_pipeline(req)
print("Test Script Finished.")
