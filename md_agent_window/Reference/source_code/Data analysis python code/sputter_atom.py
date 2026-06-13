#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

def parse_sputtered_dump(sput_path: str) -> pd.DataFrame:
    """
    Parse a LAMMPS dump with blocks like:
      ITEM: TIMESTEP
      <timestep>
      ITEM: NUMBER OF ATOMS
      <N>
      ITEM: BOX BOUNDS ...
      <3 lines>
      ITEM: ATOMS id type x y z v_vxout v_vyout v_vzout v_ke
      <N lines of atoms>

    Returns DataFrame with columns:
      ['x','y','z','v_vxout','v_vyout','v_vzout','v_ke']
    """
    cols = ['x','y','z','v_vxout','v_vyout','v_vzout','v_ke']
    records = []

    with open(sput_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i, L = 0, len(lines)
    while i < L:
        if lines[i].startswith("ITEM: TIMESTEP"):
            # timestep value (unused)
            i += 2
            # number of atoms
            n_atoms = 0
            if i < L and lines[i].startswith("ITEM: NUMBER OF ATOMS"):
                i += 1
                if i < L:
                    try:
                        n_atoms = int(lines[i].strip())
                    except Exception:
                        n_atoms = 0
                i += 1
            # box bounds header + 3 lines
            if i < L and lines[i].startswith("ITEM: BOX BOUNDS"):
                i += 4
            # atoms header
            if i < L and lines[i].startswith("ITEM: ATOMS"):
                i += 1
                for _ in range(max(n_atoms, 0)):
                    if i >= L:
                        break
                    parts = lines[i].strip().split()
                    # expecting: id type x y z v_vxout v_vyout v_vzout v_ke
                    if len(parts) >= 9:
                        try:
                            rec = [float(parts[2]), float(parts[3]), float(parts[4]),
                                   float(parts[5]), float(parts[6]), float(parts[7]), float(parts[8])]
                            records.append(rec)
                        except Exception:
                            pass
                    i += 1
            else:
                # malformed block; continue searching
                i += 1
        else:
            i += 1

    return pd.DataFrame(records, columns=cols)

def main():
    ap = argparse.ArgumentParser(description="Parse sputtered.dump and save selected columns to CSV.")
    ap.add_argument("--path", default="sputtered.dump", help="Input dump path (default: sputtered.dump)")
    ap.add_argument("--out",  default="mdn_sputtered.csv", help="Output CSV path (default: mdn_sputtered.csv)")
    args = ap.parse_args()

    df = parse_sputtered_dump(args.path)
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
