from __future__ import annotations

import json

from sim_agent.schemas._parse import JsonMap

from .input_deck_types import (
    DeckContract,
    DeckSchedule,
    DeckSurface,
    IncidentSpec,
    LAMMPSInputDeckError,
)


def render_lammps_script(
    contract: DeckContract,
    schedule: DeckSchedule,
    surface: DeckSurface,
    events: tuple[IncidentSpec, ...],
    manifest: JsonMap,
) -> str:
    lines = _header_lines(contract, schedule, surface, manifest)
    for event in events:
        if event.ion_species != schedule.ion_species:
            raise LAMMPSInputDeckError("mixed_ion_schedule_not_supported")
        lines.extend(_event_block(event))
    lines.append("write_data surface_snapshot_after.data")
    return "\n".join(lines) + "\n"


def _header_lines(
    contract: DeckContract,
    schedule: DeckSchedule,
    surface: DeckSurface,
    manifest: JsonMap,
) -> list[str]:
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    lines = [
        "# sim_agent atomistic campaign input",
        f"# run_id {contract.run_id}",
        f"# schedule_id {schedule.schedule_id}",
        f"# surface_state_id {surface.surface_state_id}",
        f"# material_phase {surface.material_id}:{surface.phase}",
        "clear",
        f"units {contract.unit_style}",
        "dimension 3",
        "atom_style charge",
        "boundary p p f",
        "read_data surface_snapshot_before.data",
        "mass 1 28.0855",
        "mass 2 39.948",
        "pair_style hybrid/overlay tersoff zbl 0.5 2.0",
        "pair_coeff * * tersoff Si.tersoff Si NULL",
        "pair_coeff 1 2 zbl 14 18",
        "pair_coeff 2 2 zbl 18 18",
        "neighbor 2.5 bin",
        "neigh_modify every 1 delay 0 check yes",
        "variable ev_to_j equal 1.602176634e-19",
        "variable projectile_mass_kg equal 39.948*1.66053906660e-27",
        "variable collision_steps equal 1000",
        "variable zhi_box equal zhi",
        "variable z_in_lo equal v_zhi_box-5.0",
        "region incident_region block INF INF INF INF ${z_in_lo} ${zhi_box} units box",
        "region escape_region block INF INF INF INF ${zhi_box} INF units box",
        "group substrate type 1",
        "group addatom type 2",
        "compute cVel all property/atom vx vy vz",
        "compute cKE all ke/atom",
        "dump dAll all custom 100 traj.dump id type x y z "
        "c_cVel[1] c_cVel[2] c_cVel[3] c_cKE",
        "dump_modify dAll append yes sort id",
        f"print '{manifest_json}' file run_manifest.json",
    ]
    lines.extend(f"# required_output {filename}" for filename in contract.required_outputs)
    return lines


def _event_block(event: IncidentSpec) -> tuple[str, ...]:
    suffix = event.lammps_suffix
    seed = 900_000 + int(suffix.rsplit("_", maxsplit=1)[-1])
    return (
        "",
        f"# incident {event.event_id}",
        "delete_atoms group addatom",
        "group addatom clear",
        "group addatom type 2",
        "variable E_in delete",
        "variable polar_deg delete",
        "variable azimuth_deg delete",
        f"variable E_in equal {_format_float(event.energy_eV)}",
        f"variable polar_deg equal {_format_float(event.polar_deg)}",
        f"variable azimuth_deg equal {_format_float(event.azimuth_deg)}",
        "variable polar_rad equal v_polar_deg*PI/180.0",
        "variable azimuth_rad equal v_azimuth_deg*PI/180.0",
        "variable speed_mps equal sqrt(2.0*v_E_in*v_ev_to_j/v_projectile_mass_kg)",
        "variable speed_angs_ps equal v_speed_mps*0.01",
        "variable vx_in equal -v_speed_angs_ps*sin(v_polar_rad)*cos(v_azimuth_rad)",
        "variable vy_in equal -v_speed_angs_ps*sin(v_polar_rad)*sin(v_azimuth_rad)",
        "variable vz_in equal -v_speed_angs_ps*cos(v_polar_rad)",
        f"fix dep_{suffix} addatom deposit 1 2 1 {seed} region incident_region "
        "near 2.0 vx ${vx_in} ${vx_in} vy ${vy_in} ${vy_in} "
        "vz ${vz_in} ${vz_in} units box",
        "run 1",
        f"unfix dep_{suffix}",
        "dump dInc addatom custom 1 incident.dump id type x y z vx vy vz v_E_in",
        "dump_modify dInc append yes",
        "run 0",
        "undump dInc",
        "fix fIon addatom nve",
        "fix fSub substrate nve",
        "run ${collision_steps}",
        "unfix fIon",
        "unfix fSub",
        "group reflected intersect addatom escape_region",
        "group sputtered intersect substrate escape_region",
        "dump dRefl reflected custom 1 reflected.dump id type x y z vx vy vz c_cKE",
        "dump dSput sputtered custom 1 sputtered.dump id type x y z vx vy vz c_cKE",
        "dump dImpl addatom custom 1 implanted.dump id type x y z vx vy vz c_cKE",
        "dump_modify dRefl append yes",
        "dump_modify dSput append yes",
        "dump_modify dImpl append yes",
        "run 0",
        "undump dRefl",
        "undump dSput",
        "undump dImpl",
        "group reflected clear",
        "group sputtered clear",
    )


def _format_float(value: float) -> str:
    return f"{value:.6f}"
