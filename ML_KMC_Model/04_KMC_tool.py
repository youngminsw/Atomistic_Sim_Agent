# KMC_tool.py
# -----------------------------------------------------------------------------
# KMC trench histogram + heatmap tool (original "형 파일" 로직 호환 버전)
#
# - MCP_server.py expects: run_kmc_with_mdn(...)
# - This implementation follows the received logic in physics_engine.py/total_model.py:
#   * Geometry binning along trench walls
#   * Ray-segment intersection / multiple bounces
#   * Deposit energy on hit bin (bottom absorbs; others use ML surrogate if available)
#   * Heatmap visualization uses outward normals (no "empty vertical wall" bug)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Prefer config.OUTPUT_DIR if present (same convention as received files)
try:
    from config import OUTPUT_DIR  # type: ignore
except Exception:
    OUTPUT_DIR = "outputs"

# Try to load the surrogate model exactly like the received files (total_model.py).
# If not available, we fall back to "dummy physics" (deposit all remaining energy and stop).
try:
    from total_model import convert_2d_to_3d, surrogate_predict, convert_3d_to_2d  # type: ignore
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False
    convert_2d_to_3d = None  # type: ignore
    surrogate_predict = None  # type: ignore
    convert_3d_to_2d = None  # type: ignore


class KMCHistogramEngine:
    """
    KMC histogram engine (2D trench) matching the collaborator's original logic.
    """

    def __init__(self, segment_bin_size: float = 2.0):
        # Trench polyline points (centered at X=0)
        self.points = np.array(
            [(-68, 700), (-60, 700), (-60, 0), (60, 0), (60, 700), (68, 700)],
            dtype=float,
        )

        # Bin resolution along each segment (nm units in this coordinate system)
        self.SEGMENT_BIN_SIZE = float(segment_bin_size)

        self.segments, self.bin_centers, self.bin_energies = self._setup_bins()

    def _setup_bins(self):
        segments = []
        bin_centers = []
        bin_energies = []

        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            seg_vec = p2 - p1
            seg_len = np.linalg.norm(seg_vec)
            if seg_len == 0:
                continue

            unit_vec = seg_vec / seg_len
            # use ceil so bins cover the full edge without gaps
            num_bins = int(np.ceil(seg_len / self.SEGMENT_BIN_SIZE))
            if num_bins < 1:
                num_bins = 1
            actual_bin_size = seg_len / num_bins

            for b in range(num_bins):
                start_pt = p1 + unit_vec * (b * actual_bin_size)
                end_pt = p1 + unit_vec * ((b + 1) * actual_bin_size)
                center = (start_pt + end_pt) / 2.0

                segments.append((start_pt, end_pt))
                bin_centers.append(center)
                bin_energies.append(0.0)

        return segments, np.array(bin_centers), bin_energies

    def _reset_energies(self):
        self.bin_energies = [0.0] * len(self.bin_energies)

    @staticmethod
    def _compute_segment_normal(p1, p2, incident_dir):
        edge = p2 - p1
        # 2D normal (rotate 90°)
        rawN = np.array([-edge[1], edge[0]], dtype=float)
        # ensure facing the ion (opposite to incident direction)
        if np.dot(rawN, incident_dir) > 0:
            rawN = -rawN
        return -rawN / np.linalg.norm(rawN)

    @staticmethod
    def _ray_segment_intersection(origin, direction, p1, p2):
        seg = p2 - p1
        d = seg[1] * direction[0] - seg[0] * direction[1]
        if abs(d) < 1e-8:
            return None
        diff = origin - p1
        ua = (seg[0] * diff[1] - seg[1] * diff[0]) / d
        ub = (direction[0] * diff[1] - direction[1] * diff[0]) / d
        if ua >= 0 and 0 <= ub <= 1:
            return origin + ua * direction
        return None

    def _find_closest_bin_index(self, collision_pt):
        dists = np.linalg.norm(self.bin_centers - collision_pt, axis=1)
        return int(np.argmin(dists))

    def run_simulation(self, energy_ev: float, angle_deg: float, num_ions: int):
        """
        Run KMC: accumulate deposited energy into nearest bins.
        - Bottom segment absorbs and terminates.
        - If ML surrogate unavailable, deposit remaining energy once and terminate.
        """
        self._reset_energies()

        min_x, max_x = self.points[:, 0].min(), self.points[:, 0].max()
        max_y = self.points[:, 1].max()

        # Index of the vertical wall-to-bottom segment in the polyline (matches received file)
        BOTTOM_SEG_IDX = 2

        for _ in range(int(num_ions)):
            origin = np.array([np.random.uniform(min_x + 5, max_x - 5), max_y + 3])

            sign = 1 if np.random.rand() > 0.5 else -1
            curr_angle = np.random.normal(0, 0.5) + (float(angle_deg) * sign)
            angle_rad = np.radians(curr_angle)

            direction = np.array([np.sin(angle_rad), -np.cos(angle_rad)])
            direction /= np.linalg.norm(direction)

            pos = origin.copy()
            dir_vec = direction.copy()
            E_curr = float(energy_ev)
            bounces = 0

            while bounces < 10:
                nearest_inter, nearest_idx, min_dist = None, -1, np.inf

                # find nearest intersected segment
                for i in range(len(self.points) - 1):
                    p1, p2 = self.points[i], self.points[i + 1]
                    inter = self._ray_segment_intersection(pos, dir_vec, p1, p2)
                    if inter is not None:
                        dist = np.linalg.norm(inter - pos)
                        if 1e-8 < dist < min_dist:
                            min_dist, nearest_inter, nearest_idx = dist, inter, i

                if nearest_inter is None:
                    break

                hit_bin_idx = self._find_closest_bin_index(nearest_inter)

                # bottom absorbs (deposit full remaining)
                if nearest_idx == BOTTOM_SEG_IDX:
                    self.bin_energies[hit_bin_idx] += E_curr
                    break

                # if no surrogate: deposit and stop
                if not MODEL_LOADED:
                    self.bin_energies[hit_bin_idx] += E_curr
                    break

                # surrogate bounce
                p1, p2 = self.points[nearest_idx], self.points[nearest_idx + 1]
                normal = self._compute_segment_normal(p1, p2, -dir_vec)
                cos_i = np.dot(-dir_vec, normal)

                theta_i_rad = np.arccos(np.clip(cos_i, -1, 1))
                theta_i_deg = np.degrees(theta_i_rad)

                try:
                    vx, vy, vz = convert_2d_to_3d(E_curr, theta_i_deg)
                    pred = surrogate_predict(vx, vy, vz, E_curr)
                    if pred is None:
                        self.bin_energies[hit_bin_idx] += E_curr
                        break

                    vx_p, vy_p, vz_p, E_out = pred

                    dep = E_curr - float(E_out)
                    if dep < 0:
                        dep = 0.0
                    self.bin_energies[hit_bin_idx] += dep

                    if float(E_out) <= 5:
                        break

                    proj = convert_3d_to_2d(vx_p, vy_p, vz_p)
                    if proj is None:
                        break
                    _, theta_o = proj

                    tangent = (p2 - p1) / np.linalg.norm(p2 - p1)
                    y_axis = -normal
                    ix = np.dot(dir_vec, tangent)
                    sign_ix = np.sign(ix) if ix != 0 else 1

                    local_x = sign_ix * np.sin(theta_o)
                    local_y = -1 * np.cos(theta_o)

                    refl_vec = local_x * tangent + local_y * y_axis
                    refl_vec /= np.linalg.norm(refl_vec)

                    pos = nearest_inter
                    dir_vec = refl_vec
                    E_curr = float(E_out)
                    bounces += 1

                except Exception:
                    # fail-safe: stop current ion
                    break

        return self.bin_energies

    def plot_histogram_heatmap(
        self,
        run_id: str,
        energy: float,
        angle: float,
        out_png: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Create & save heatmap image. Returns (filename_or_path, max_energy).
        If out_png is provided, it's treated as a path (absolute or relative to OUTPUT_DIR).
        """
        ep_data = np.array(self.bin_energies, dtype=float)
        max_E = float(ep_data.max()) if ep_data.size else 0.0
        if max_E <= 0:
            max_E = 1.0

        # scaling used only for visualization height
        visual_max_length = 60.0
        scale_factor = visual_max_length / max_E

        fig, ax = plt.subplots(figsize=(9, 10))
        #plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.weight"] = "bold"

        # trench outline
        ax.plot(self.points[:, 0], self.points[:, 1], color="black", linewidth=2, zorder=10)

        patches = []
        colors = []
        cmap = plt.cm.get_cmap("hot")

        for i in range(len(self.segments)):
            E = float(self.bin_energies[i])
            p1, p2 = self.segments[i]
            p1, p2 = np.array(p1), np.array(p2)
            seg_center = (p1 + p2) / 2.0

            # visualization normal (original heuristic)
            if seg_center[0] < -50:
                normal = np.array([-1.0, 0.0])   # left wall outward
            elif seg_center[0] > 50:
                normal = np.array([1.0, 0.0])    # right wall outward
            elif seg_center[1] < 10:
                normal = np.array([0.0, -1.0])   # bottom outward
            else:
                normal = np.array([0.0, 1.0])    # top outward

            h = E * scale_factor
            v1, v2 = p1, p2
            v3, v4 = p2 + normal * h, p1 + normal * h

            poly = Polygon([v1, v2, v3, v4], closed=True)
            patches.append(poly)
            colors.append(E)

        pc = PatchCollection(patches, cmap=cmap, alpha=0.9)
        pc.set_array(np.array(colors, dtype=float))
        pc.set_clim(0, max_E)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.3)
        ax.add_collection(pc)

        ax.set_aspect("equal")
        ax.set_xlim(-120, 120)
        ax.set_ylim(-80, 750)
        ax.set_xticks([-100, -50, 0, 50, 100])
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlabel("X Position (nm)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Y Position (nm)", fontsize=16, fontweight="bold")

        cbar = plt.colorbar(pc, ax=ax, pad=0.02)
        cbar.set_label("Deposited Energy (a.u.)", rotation=270, labelpad=20, fontsize=14, fontweight="bold")
        plt.title(f"KMC Heatmap | run={run_id} | E_in={energy:.1f} eV | angle={angle:.1f}°", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # output path
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if out_png is None or str(out_png).strip() == "":
            filename = f"{run_id}_E{energy:.1f}_A{angle:.1f}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            out_ref = save_path
        else:
            # if user passes a filename, store under OUTPUT_DIR; if path, respect it
            out_png = str(out_png)
            if os.path.isabs(out_png):
                save_path = out_png
            else:
                save_path = os.path.join(OUTPUT_DIR, out_png)
            out_ref = save_path

        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        return out_ref, float(max_E)


def run_kmc_with_mdn(
    workdir=".",
    energy_ev: float = 100.0,
    angle_deg: float = 0.0,
    num_ions: int = 2000,
    run_id: str = "kmc_run",
    out_png: Optional[str] = None,
    seed: Optional[int] = None,
    auto_best: bool = True,  # kept for MCP_server compatibility (unused here)
    device: str = "auto",    # kept for compatibility (model device is handled in total_model.py)
    mode: str = "sample",    # kept for compatibility
) -> Dict[str, Any]:
    """
    MCP_server.py 호환 엔트리포인트.
    - 내부 로직은 '형 파일'(physics_engine.py + total_model.py) 방식 그대로 동작.
    - auto_best/device/mode는 기존 MCP 인터페이스 유지용으로만 받고, 여기서는 무시합니다.

    Returns JSON-serializable dict:
      {
        "run_id": ...,
        "energy_ev": ...,
        "angle_deg": ...,
        "num_ions": ...,
        "model_loaded": bool,
        "heatmap_png": "path/to/png",
        "max_energy": ...,
        "bin_energies": [...],
      }
    """
    # workdir is accepted for interface consistency; output dir is controlled by config.OUTPUT_DIR
    _ = workdir

    if seed is not None:
        np.random.seed(int(seed))

    engine = KMCHistogramEngine(segment_bin_size=2.0)
    engine.run_simulation(float(energy_ev), float(angle_deg), int(num_ions))
    out_path, max_e = engine.plot_histogram_heatmap(run_id=str(run_id), energy=float(energy_ev), angle=float(angle_deg), out_png=out_png)

    return {
        "run_id": str(run_id),
        "energy_ev": float(energy_ev),
        "angle_deg": float(angle_deg),
        "num_ions": int(num_ions),
        "model_loaded": bool(MODEL_LOADED),
        "heatmap_png": out_path,
        "max_energy": float(max_e),
        "bin_energies": [float(x) for x in engine.bin_energies],
    }

if __name__ == "__main__":
    print("Running KMC Tool Test...")
    res = run_kmc_with_mdn(energy_ev=100.0, angle_deg=0.0, num_ions=100, run_id="test_run")
    print("Result:", res)
