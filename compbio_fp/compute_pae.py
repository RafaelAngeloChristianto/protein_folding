#!/usr/bin/env python3
"""
Compute empirical PAE matrix from an ensemble of predicted structures.

Requirements:
  pip install biopython numpy matplotlib

Usage:
  - Put ensemble pdbs in a folder (same residue numbering).
  - Adjust ENSEMBLE_FILES list or create code to glob.
  - Run: python compute_pae.py
Outputs:
  - pae.npy   : numpy array (L x L)
  - pae.png   : heatmap plot
"""

import numpy as np
import math
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
import glob
import os

# -------------- User settings --------------
# Pattern or list of PDB files for the ensemble:
ENSEMBLE_GLOB = "ensemble/*.pdb"   # change to your path
# Use C-alpha atoms and a local fragment size for alignment (must be >= 3 to define rotation)
FRAG_RADIUS = 1  # number of residues on each side, so radius=1 -> residues j-1, j, j+1
# Aggregation method: 'mean' or 'median'
AGG_METHOD = 'mean'
# Reference model index (0 means first file found)
REF_INDEX = 0
# -------------------------------------------

def load_ca_coords(pdb_file):
    """Return dict {resid_index: np.array([x,y,z])} using residue sequence index (starting 1)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(os.path.basename(pdb_file), pdb_file)
    # take first model if multiple
    model = next(structure.get_models())
    coords = {}
    # Use a simple residue counter (assumes consistent numbering across ensemble)
    for chain in model:
        for res in chain:
            # ignore hetero/water residues; Biopython uses id[0] == ' '
            if res.id[0] != ' ':
                continue
            resnum = res.id[1]  # residue number
            if 'CA' in res:
                coords[resnum] = res['CA'].get_coord().copy()
    return coords

def kabsch(P, Q):
    """Return rotation R and translation t that minimizes ||R P + t - Q||.
    P and Q are (N,3) arrays. Assumes N >= 3 and centered.
    """
    # covariance
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = np.linalg.det(np.dot(V, Wt))
    D = np.eye(3)
    D[2,2] = d
    R = np.dot(np.dot(V, D), Wt)
    return R

def superpose_coords(mobile_coords, fixed_coords, res_list):
    """Compute rotation+translation to align mobile->fixed using residues in res_list (list of resnums).
    Returns transformed mobile_coords dict (a new dict).
    """
    # Build arrays P (mobile) and Q (fixed)
    P_list = []
    Q_list = []
    for r in res_list:
        if r not in mobile_coords or r not in fixed_coords:
            return None  # cannot align if any anchor missing
        P_list.append(mobile_coords[r])
        Q_list.append(fixed_coords[r])
    P = np.array(P_list)
    Q = np.array(Q_list)
    # center
    P_cent = P.mean(axis=0)
    Q_cent = Q.mean(axis=0)
    Pc = P - P_cent
    Qc = Q - Q_cent
    # need at least 3 non-collinear points; we assume frag >=3
    R = kabsch(Pc, Qc)
    t = Q_cent - R.dot(P_cent)
    # apply to all mobile coords
    transformed = {}
    for r, coord in mobile_coords.items():
        transformed[r] = (R.dot(coord) + t)
    return transformed

def compute_pae(ensemble_files, frag_radius=1, agg='mean'):
    # load ensemble coords
    ensemble = []
    for f in ensemble_files:
        ensemble.append(load_ca_coords(f))
    L_resnums = sorted(list(ensemble[0].keys()))
    L = len(L_resnums)
    # map index -> residue number for heatmap ordering
    resnum_to_index = {resnum: idx for idx, resnum in enumerate(L_resnums)}
    # choose reference coords
    ref_coords = ensemble[REF_INDEX]
    # prepare storage: errors[k][i_idx][j_idx]
    # We'll accumulate errors over samples (exclude the reference sample from averaging optionally)
    samples = [e for idx,e in enumerate(ensemble) if idx != REF_INDEX]  # exclude ref from samples
    if len(samples) == 0:
        raise ValueError("Need at least 2 structures (reference + at least one sample).")
    n_samples = len(samples)
    # initialize accumulation arrays
    pae_accum = np.zeros((L, L, n_samples))  # (i, j, sample_k)
    sample_k = 0
    for sample_coords in samples:
        # for each anchor residue j
        for j_idx, j_resnum in enumerate(L_resnums):
            # define fragment for alignment: residues j-r .. j+r
            frag = [r for r in range(j_resnum - frag_radius, j_resnum + frag_radius + 1)]
            # attempt superposition of sample -> reference using frag
            transformed = superpose_coords(sample_coords, ref_coords, frag)
            if transformed is None:
                # if frag missing (e.g., start/end), try expanding fragment or fall back to translation only
                # here we skip and set NaNs for this j
                for i_idx in range(L):
                    pae_accum[i_idx, j_idx, sample_k] = np.nan
                continue
            # now for every residue i compute euclidean distance between transformed sample and ref
            for i_idx, i_resnum in enumerate(L_resnums):
                if i_resnum not in transformed or i_resnum not in ref_coords:
                    pae_accum[i_idx, j_idx, sample_k] = np.nan
                else:
                    d = np.linalg.norm(transformed[i_resnum] - ref_coords[i_resnum])
                    pae_accum[i_idx, j_idx, sample_k] = d
        sample_k += 1

    # aggregate across samples
    if agg == 'mean':
        pae = np.nanmean(pae_accum, axis=2)
    elif agg == 'median':
        pae = np.nanmedian(pae_accum, axis=2)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    return pae, L_resnums

def plot_pae(pae, resnums, out_png="pae.png", vmax=None):
    plt.figure(figsize=(7,7))
    im = plt.imshow(pae, origin='lower', interpolation='nearest', cmap='Greens', vmin=0, vmax=vmax)
    plt.xlabel("Scored residue (index)")
    plt.ylabel("Aligned residue (index)")
    plt.title("Empirical PAE")
    # colorbar with typical range 0-30 Å
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Expected position error (Å)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

if __name__ == "__main__":
    files = sorted(glob.glob(ENSEMBLE_GLOB))
    if len(files) < 2:
        raise SystemExit("Need at least two PDB files in ensemble folder. Adjust ENSEMBLE_GLOB.")
    print(f"Found {len(files)} ensemble files.")
    pae_mat, resnums = compute_pae(files, frag_radius=FRAG_RADIUS, agg=AGG_METHOD)
    np.save("pae.npy", pae_mat)
    print("Saved pae.npy")
    plot_pae(pae_mat, resnums, out_png="pae.png", vmax=30)
    print("Saved pae.png")
