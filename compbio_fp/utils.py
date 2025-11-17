from .models import AMINO_PROPS


def validate_sequence(seq: str):
    if seq is None:
        return None
    s = seq.strip().upper()
    if len(s) == 0:
        return None
    for c in s:
        if c not in AMINO_PROPS:
            return None
    return s


__all__ = ['validate_sequence']

def max_bond_deviation(coords, r0=3.8):
    """Return maximum absolute deviation from r0 for consecutive CA-CA distances.

    coords: array-like (N,3)
    r0: target CA-CA distance
    """
    import numpy as _np
    coords = _np.asarray(coords)
    if coords.shape[0] < 2:
        return 0.0
    diffs = coords[1:] - coords[:-1]
    dists = _np.linalg.norm(diffs, axis=1)
    dev = _np.abs(dists - r0)
    return float(_np.max(dev))

__all__ = ['validate_sequence', 'max_bond_deviation']


def open_with_default_viewer(path):
    """Open `path` with the OS default viewer. Returns True on success, False otherwise."""
    import sys, os, subprocess, logging
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
        return True
    except OSError as e:
        logging.warning("Could not open %s: %s", path, e)
    except Exception as e:
        logging.exception("Unexpected error opening %s: %s", path, e)
    return False


def energy_series(history):
    """Return a dict of energy component series extracted from history.

    history: iterable of (coords, energy_dict)
    Returns dict with keys for each energy component and numpy arrays as values.
    """
    import numpy as _np
    energy_dicts = [h[1] for h in history]
    keys = set()
    for d in energy_dicts:
        keys.update(d.keys())
    out = {}
    for k in sorted(keys):
        out[k] = _np.array([d.get(k, 0.0) for d in energy_dicts])
    return out


def rotation_matrix(axis, angle):
    """Return a 3x3 rotation matrix rotating about `axis` (3-vector) by `angle` radians.

    Axis need not be unit length; function will normalize it.
    """
    import numpy as _np, math
    axis = _np.asarray(axis, dtype=float)
    norm = _np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("rotation axis must be non-zero")
    ux, uy, uz = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    return _np.array([
        [c + ux*ux*(1-c),     ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s,  c + uy*uy*(1-c),    uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s,  uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)]
    ])
