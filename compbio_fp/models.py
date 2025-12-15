import numpy as np

AMINO_PROPS = {
    'A': {'hydro': 1.8,  'charge': 0},
    'C': {'hydro': 2.5,  'charge': 0},
    'D': {'hydro': -3.5, 'charge': -1},
    'E': {'hydro': -3.5, 'charge': -1},
    'F': {'hydro': 2.8,  'charge': 0},
    'G': {'hydro': -0.4, 'charge': 0},
    'H': {'hydro': -3.2, 'charge': 0},
    'I': {'hydro': 4.5,  'charge': 0},
    'K': {'hydro': -3.9, 'charge': +1},
    'L': {'hydro': 3.8,  'charge': 0},
    'M': {'hydro': 1.9,  'charge': 0},
    'N': {'hydro': -3.5, 'charge': 0},
    'P': {'hydro': -1.6, 'charge': 0},
    'Q': {'hydro': -3.5, 'charge': 0},
    'R': {'hydro': -4.5, 'charge': +1},
    'S': {'hydro': -0.8, 'charge': 0},
    'T': {'hydro': -0.7, 'charge': 0},
    'V': {'hydro': 4.2,  'charge': 0},
    'W': {'hydro': -0.9, 'charge': 0},
    'Y': {'hydro': -1.3, 'charge': 0},
}


def distance(a, b):
    return np.linalg.norm(a - b)


def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cos_theta)


class Protein:
    def __init__(self, sequence, ca_distance=3.8, include_sidechains=False):
        if not sequence:
            raise ValueError("Sequence must be non-empty")
        self.sequence = sequence
        self.N = len(sequence)
        # normalize hydrophobicity to 0..1 for use as contact propensity
        raw_props = [AMINO_PROPS.get(c, {'hydro': 0.0, 'charge': 0}) for c in sequence]
        hydro_values = [p['hydro'] for p in raw_props]
        hmin = min(hydro_values)
        hmax = max(hydro_values)
        if hmax - hmin == 0:
            norm = [0.0 for _ in hydro_values]
        else:
            norm = [(h - hmin) / (hmax - hmin) for h in hydro_values]
        self.props = []
        for p, hn in zip(raw_props, norm):
            self.props.append({'hydro': hn, 'charge': p.get('charge', 0)})
        self.ca_distance = ca_distance
        self.coords = self._init_coords()
        self.include_sidechains = include_sidechains
        if include_sidechains:
            # simple model: one sidechain bead per residue offset from CA along +y
            self.sidechain_coords = self._init_sidechains()
        else:
            self.sidechain_coords = None

    def _init_coords(self):
        coords = np.zeros((self.N, 3))
        for i in range(self.N):
            coords[i, 0] = i * self.ca_distance
        return coords

    def _init_sidechains(self, offset=1.8):
        # Place sidechain bead at a small offset from CA in +y direction as a simple placeholder
        sc = np.zeros((self.N, 3))
        for i in range(self.N):
            sc[i, 0] = self.coords[i, 0]
            sc[i, 1] = self.coords[i, 1] + offset
            sc[i, 2] = self.coords[i, 2]
        return sc

    def rmsd_to(self, other_coords):
        L = min(len(self.coords), len(other_coords))
        diff = self.coords[:L] - other_coords[:L]
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def initialize_extended_chain(sequence_length, ca_distance=3.8):
    """Initialize protein coordinates in an extended beta-strand conformation.
    
    This provides a better starting structure than random coordinates,
    placing all CA atoms along a straight line with proper spacing.
    
    Args:
        sequence_length: Number of residues
        ca_distance: Distance between consecutive CA atoms (default 3.8 Å)
    
    Returns:
        coords: Array of shape (sequence_length, 3) with extended chain coordinates
    """
    coords = np.zeros((sequence_length, 3))
    for i in range(sequence_length):
        coords[i] = [i * ca_distance, 0, 0]  # Extended along x-axis
    return coords


def initialize_helical_chain(sequence_length, ca_distance=3.8):
    """Initialize protein coordinates in an alpha-helix conformation.
    
    Places CA atoms in an ideal alpha-helix geometry:
    - 3.6 residues per turn
    - 5.4 Å pitch (rise per turn)
    - 2.3 Å radius
    
    Args:
        sequence_length: Number of residues
        ca_distance: Average CA-CA distance (adjusted for helix, ~3.8 Å)
    
    Returns:
        coords: Array of shape (sequence_length, 3) with helical coordinates
    """
    coords = np.zeros((sequence_length, 3))
    
    # Alpha helix parameters
    residues_per_turn = 3.6
    pitch = 5.4  # Angstroms per turn (rise along helix axis)
    radius = 2.3  # Angstroms
    
    for i in range(sequence_length):
        # Angle around helix axis
        theta = (i / residues_per_turn) * 2 * np.pi
        
        # Position along helix axis
        z = (i / residues_per_turn) * pitch
        
        # Coordinates in cylindrical system, then convert to Cartesian
        coords[i] = [
            radius * np.cos(theta),  # x
            radius * np.sin(theta),  # y
            z                         # z (helix axis)
        ]
    
    return coords


__all__ = ['Protein', 'AMINO_PROPS', 'distance', 'angle_between', 
           'initialize_extended_chain', 'initialize_helical_chain']
