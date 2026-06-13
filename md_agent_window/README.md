# MD Agent Docker Environment

Docker-based environment for running the Multi-Agent Molecular Dynamics system.

## Quick Start

### 1. Build the Image
```bash
docker build -t md_agent .
```

### 2. Run the Simulation
```bash
# Run with bind mount (entire folder mounted - all files visible)
docker-compose up

# Or manual docker run
docker run -v $(pwd):/app md_agent
```

### 3. Custom Simulation Parameters
```bash
docker run -v $(pwd):/app md_agent \
    python autonomous_agent.py --ion Ar --sub Si --energy 100 --events 50
```

## Output Structure
All files in `md_agent/` are bind-mounted. LAMMPS outputs go to `results/`:
```
md_agent/
├── src/                    # Source code (visible)
├── templates/              # LAMMPS templates (visible)
├── results/
│   └── run_CF4_Ru_10evts/  # Simulation outputs
│       ├── reflected.dump
│       ├── sputtered.dump
│       ├── traj.dump
│       ├── log.lammps
│       └── in.sputtering
```

## Environment Variables
- `OPENCODE_API_KEY` - For Antigravity/opencode authentication

## Troubleshooting
- **OpenGL errors**: Container uses Xvfb for headless rendering
- **Permission issues**: Run `chmod -R 777 results/` if needed
