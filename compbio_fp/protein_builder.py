import numpy as np

############################################################
# GEOMETRY HELPERS
############################################################

def place_atom(a, b, c, bond_length, angle_deg, dihedral_deg):
    """
    Place an atom in 3D using internal coordinates:
    a, b, c = known atoms
    returns new atom d

    angle is at c (b-c-d), dihedral is (a-b-c-d)
    """
    angle = np.deg2rad(angle_deg)
    dihedral = np.deg2rad(dihedral_deg)

    bc = b - c
    bc /= np.linalg.norm(bc)

    # normal vector
    n = np.cross(a - b, bc)
    n /= (np.linalg.norm(n) + 1e-10)

    m = np.cross(n, bc)

    d = (c +
         bond_length * np.cos(angle) * bc +
         bond_length * np.sin(angle) * (np.cos(dihedral) * m + np.sin(dihedral) * n))

    return d


############################################################
# BACKBONE FROM CA TRACE
############################################################

def build_backbone_from_CA(ca_coords):
    """
    Simple backbone reconstruction from CA trace.
    Returns arrays for N, CA, C, O atoms.
    """

    # Standard peptide geometry
    d_N_CA = 1.46
    d_CA_C = 1.52
    d_C_O = 1.24

    angle_N_CA_C = 110.4
    angle_CA_C_N = 116.2

    N, CA, C, O = [], [], [], []

    # Initialize first residue with a reasonable guess
    n0 = ca_coords[0] + np.array([-1.2, 0.7, 0.0])
    c0 = ca_coords[0] + np.array([1.2, 0.2, 0.0])

    N.append(n0)
    CA.append(ca_coords[0])
    C.append(c0)
    O.append(c0 + np.array([0.0, 0.0, 1.2]))

    for i in range(1, len(ca_coords)):
        ca_prev = CA[-1]
        c_prev = C[-1]
        n_prev = N[-1]

        CA_i = ca_coords[i]

        # Build N_i
        N_i = place_atom(c_prev, ca_prev, n_prev,
                         d_N_CA, angle_CA_C_N, 180)

        # Build C_i
        C_i = place_atom(N_i, CA_i, c_prev,
                         d_CA_C, angle_N_CA_C, 180)

        O_i = C_i + np.array([0.0, 0.0, 1.24])  # crude but acceptable

        N.append(N_i)
        CA.append(CA_i)
        C.append(C_i)
        O.append(O_i)

    return np.array(N), np.array(CA), np.array(C), np.array(O)


############################################################
# SIDECHAIN GEOMETRY (CB + pseudo sidechain)
############################################################

def place_CB(N, CA, C):
    """Compute Cβ from backbone geometry (tetrahedral)."""
    v1 = N - CA
    v2 = C - CA
    n = np.cross(v1, v2)
    n /= (np.linalg.norm(n) + 1e-10)
    cb = CA + (0.7 * v1 + 0.7 * v2 + 0.7 * n)
    return cb


def place_rotamer(CA, CB, chi_deg):
    """Place first sidechain pseudo-atom based on χ1 angle."""
    v = CB - CA
    v /= np.linalg.norm(v)

    # perpendicular vector
    perp = np.array([v[1], -v[0], 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0, 1, 0])
    perp /= np.linalg.norm(perp)

    chi = np.deg2rad(chi_deg)
    sc = CB + 1.5 * (np.cos(chi) * perp +
                     np.sin(chi) * np.cross(v, perp))
    return sc


############################################################
# MINIMAL ROTAMER LIBRARY
############################################################

ROT_LIB = {
    "ARG": [60, -60, 180],
    "LYS": [60, -60, 180],
    "GLU": [60, -60, 180],
    "ASP": [60, -60, 180],
    "LEU": [60, -60, 180],
    "VAL": [60, -60, 180],
    "ILE": [60, -60, 180],
    "PHE": [60, -60, 180],
    "TYR": [60, -60, 180],
    "TRP": [60, -60, 180],
    "MET": [60, -60, 180],
}


############################################################
# SCORING FUNCTION
############################################################

def steric_score(points):
    """
    Sum of steric clashes.
    Penalizes d < 2.0 Å.
    Lower score = better.
    """
    total = 0.0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 2.0:
                total += np.exp(-(d - 2.0)**2)
    return total


############################################################
# ROTAMER PACKING
############################################################

def pack_sidechains(seq, N, CA, C):
    CB = []
    SC = []  # pseudo sidechain atoms

    for i, aa in enumerate(seq):
        cb = place_CB(N[i], CA[i], C[i])
        CB.append(cb)

        if aa in ROT_LIB:
            best_chi = None
            best_sc = None
            best_score = 1e9

            for chi in ROT_LIB[aa]:
                sc = place_rotamer(CA[i], cb, chi)
                score = steric_score(SC + [sc])
                if score < best_score:
                    best_score = score
                    best_chi = chi
                    best_sc = sc

            SC.append(best_sc)
        else:
            # For ALA/GLY, no sidechain beyond CB
            SC.append(cb)

    return np.array(CB), np.array(SC)


############################################################
# PDB OUTPUT
############################################################

def write_pdb(filename, N, CA, C, O, CB, SC):
    with open(filename, "w") as f:
        atom_id = 1
        for i in range(len(CA)):
            atoms = [
                ("N", N[i]),
                ("CA", CA[i]),
                ("C", C[i]),
                ("O", O[i]),
                ("CB", CB[i]),
                ("SC", SC[i])
            ]

            for name, coord in atoms:
                f.write(
                    f"ATOM  {atom_id:5d} {name:<4} RES A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n"
                )
                atom_id += 1


############################################################
# DEMO: 10-residue alpha helix (poly-Ala)
############################################################

def demo():
    # use 80-residue poly-Ala sequence
    seq = ["ALA"] * 80

    # Generate an ideal helical Cα trace matching the sequence length
    N_RES = len(seq)
    ca_coords = []
    for i in range(N_RES):
        angle = i * 100.0 * np.pi / 180.0
        x = np.cos(angle) * 5.0
        y = np.sin(angle) * 5.0
        z = i * 1.5
        ca_coords.append([x, y, z])

    ca_coords = np.array(ca_coords)

    # Build backbone
    N, CA, C, O = build_backbone_from_CA(ca_coords)

    # Pack sidechains
    CB, SC = pack_sidechains(seq, N, CA, C)

    # Write structure
    write_pdb("demo_structure.pdb", N, CA, C, O, CB, SC)
    print("Wrote demo_structure.pdb")


if __name__ == "__main__":
    demo()
