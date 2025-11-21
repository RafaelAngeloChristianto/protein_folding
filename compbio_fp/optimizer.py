import numpy as np
import math
from .utils import rotation_matrix


class SimulatedAnnealer:
    def __init__(self, protein, energy_fn, temp_K=300.0, k_B=0.0019872041, cooling=0.995, max_steps=2000, step_size=0.4, max_energy_jump=1e6, debug=False, energy_term_threshold=1e5, gui_exit=True):
        self.protein = protein
        self.energy_fn = energy_fn
        # store temperature in Kelvin and convert to energy units via k_B (kcal/mol·K)
        self.temp_K = float(temp_K)
        self.k_B = float(k_B)
        self.T_energy = self.k_B * self.temp_K
        self.cooling = cooling
        self.max_steps = max_steps
        self.step_size = step_size
        self.history = []
        # maximum allowed energy jump between consecutive accepted evaluations (treat larger jumps as invalid)
        self.max_energy_jump = float(max_energy_jump)
        # debug: print per-term energies when suspicious
        self.debug = bool(debug)
        # threshold for any single energy term magnitude to be considered suspicious
        self.energy_term_threshold = float(energy_term_threshold)
        # If True, detect GUI window closes and exit the run loop
        self.gui_exit = bool(gui_exit)

    def propose_move(self):
        coords_old = self.protein.coords.copy()
        coords_new = coords_old.copy()
        N = self.protein.N
        if N <= 2:
            return coords_new, coords_old

        # Mostly use torsion-angle (bond-axis) moves which preserve covalent bond lengths
        r = np.random.rand()
        if r < 0.7:
            # pick a bond index b between 0 and N-2
            b = np.random.randint(0, N - 1)
            # rotation axis is the vector from coords[b] to coords[b+1]
            p1 = coords_old[b]
            p2 = coords_old[b+1]
            axis = p2 - p1
            axis_len = np.linalg.norm(axis)
            if axis_len == 0:
                return coords_new, coords_old
            axis = axis / axis_len
            # sample a torsion angle (radians)
            angle = np.random.normal(scale=self.step_size)
            R = rotation_matrix(axis, angle)
            # rotate downstream segment (b+1..N-1) around axis passing through p1->p2
            idx = np.arange(b+1, N)
            if idx.size > 0:
                rel = coords_old[idx] - p1
                # project rel onto plane orthogonal to axis and rotate
                rotated = (R @ rel.T).T + p1
                coords_new[idx] = rotated
            return coords_new, coords_old
        elif r < 0.9:
            # occasional larger pivot move (random axis through a pivot)
            pivot = np.random.randint(0, N - 1)
            angle = np.random.normal(scale=2.0*self.step_size)
            axis = np.random.normal(size=3)
            norm = np.linalg.norm(axis)
            if norm == 0:
                return coords_new, coords_old
            axis = axis / norm
            R = rotation_matrix(axis, angle)
            pivot_point = coords_old[pivot]
            idx = np.arange(pivot + 1, N)
            if idx.size > 0:
                rel = coords_old[idx] - pivot_point
                rotated = (R @ rel.T).T + pivot_point
                coords_new[idx] = rotated
            return coords_new, coords_old
        else:
            # fallback: small local displacement similar to original implementation
            i = np.random.randint(1, N - 1)
            displacement = np.random.normal(scale=self.step_size, size=3)
            coords_new[i] += displacement
            if i - 1 >= 0:
                coords_new[i-1] += 0.2 * displacement
            if i + 1 < N:
                coords_new[i+1] += 0.2 * displacement
            return coords_new, coords_old

    def run(self):
        coords = self.protein.coords.copy()
        energy_dict = self.energy_fn.total_energy()
        e = energy_dict['total']

        # initial sanity check: print per-term energies if unphysical
        def _check_and_report(ed, label="energy"):
            if not isinstance(ed, dict):
                if self.debug:
                    print(f"[DEBUG] {label} not a dict: {ed}")
                return False
            bad = False
            for k, v in ed.items():
                try:
                    val = float(v)
                except Exception:
                    val = float('nan')
                if not np.isfinite(val) or abs(val) > self.energy_term_threshold:
                    bad = True
                    if self.debug:
                        print(f"[DEBUG] Unphysical term: {k} = {v} (label={label})")
            if bad and self.debug:
                # print full per-term breakdown for debugging
                print(f"[DEBUG] Full {label} breakdown:")
                for k, v in ed.items():
                    print(f"  {k}: {v}")
            return not bad

        # report initial if suspicious
        _check_and_report(energy_dict, "initial_energy")

        # attempt to detect matplotlib figures so we can exit when the user closes the visualizer
        prev_fig_count = None
        if self.gui_exit:
            try:
                import matplotlib.pyplot as plt
                prev_fig_count = len(plt.get_fignums())
            except Exception:
                # matplotlib not available or import failed; ignore GUI detection
                prev_fig_count = None

        best_coords = coords.copy()
        best_e = e
        self.history.append((coords.copy(), energy_dict))
        from .utils import max_bond_deviation
        BOND_TOL = 1e-6
        for step in range(self.max_steps):
            # If GUI-exit detection is enabled, check for figure count decreasing
            if prev_fig_count is not None:
                try:
                    import matplotlib.pyplot as plt
                    curr_count = len(plt.get_fignums())
                    if curr_count < prev_fig_count:
                        if self.debug:
                            print(f"[DEBUG] Detected figure close (prev={prev_fig_count}, curr={curr_count}) - exiting run loop")
                        # close remaining figures and exit cleanly
                        try:
                            plt.close('all')
                        except Exception:
                            pass
                        break
                    # update previous count in case new figures were created
                    prev_fig_count = curr_count
                except Exception:
                    # ignore matplotlib errors and continue
                    prev_fig_count = None

            coords_new, coords_old = self.propose_move()
            # Quick geometric/bond constraint pre-check to avoid expensive energy
            # evaluations for moves that will be rejected because they violate
            # covalent CA-CA bond lengths beyond tolerance.
            max_dev = max_bond_deviation(coords_new, r0=self.protein.ca_distance)
            if max_dev > BOND_TOL:
                # Reject early: restore previous coords, record same energy and
                # continue (still perform cooling/update step-size as usual).
                self.protein.coords = coords.copy()
                self.history.append((coords.copy(), energy_dict))
                # cool in Kelvin and update energy-scale temperature
                self.temp_K *= self.cooling
                self.T_energy = self.k_B * self.temp_K
                if step % 500 == 0 and step > 0:
                    self.step_size *= 0.9
                if self.temp_K < 1e-6:
                    break
                continue

            # Acceptable geometry; evaluate energy.
            self.protein.coords = coords_new
            energy_dict_new = self.energy_fn.total_energy()
            new_e = energy_dict_new['total']

            # Check per-term energies and report if suspicious; treat as rejection if unphysical
            physical = _check_and_report(energy_dict_new, label=f"step_{step}_proposed")
            if not physical:
                if self.debug:
                    print(f"[DEBUG] Rejecting move at step {step} due to unphysical per-term energy.")
                # Treat as rejection: restore previous coords and record previous energy (no spike)
                self.protein.coords = coords.copy()
                self.history.append((coords.copy(), energy_dict))
                # cool/update and continue
                self.temp_K *= self.cooling
                self.T_energy = self.k_B * self.temp_K
                if step % 500 == 0 and step > 0:
                    self.step_size *= 0.9
                if self.temp_K < 1e-6:
                    break
                continue

            # Reject moves that produce non-finite energies or absurdly large energy jumps
            if (not np.isfinite(new_e)) or (not np.isfinite(e)) or (abs(new_e - e) > self.max_energy_jump):
                if self.debug:
                    print(f"[DEBUG] Rejecting move at step {step} due to new_e={new_e}, old_e={e}, dE={new_e-e}")
                # Treat as rejection: restore previous coords and record previous energy (no spike)
                self.protein.coords = coords.copy()
                self.history.append((coords.copy(), energy_dict))
                # cool/update and continue
                self.temp_K *= self.cooling
                self.T_energy = self.k_B * self.temp_K
                if step % 500 == 0 and step > 0:
                    self.step_size *= 0.9
                if self.temp_K < 1e-6:
                    break
                continue

            dE = new_e - e
            accept = False
            if dE < 0:
                accept = True
            else:
                # safe divide (T_energy guarded in denominator)
                if np.random.rand() < math.exp(-dE / max(self.T_energy, 1e-12)):
                    accept = True

            if accept:
                # check bond deviations (torsion moves should preserve bonds)
                max_dev = max_bond_deviation(self.protein.coords, r0=self.protein.ca_distance)
                if max_dev > BOND_TOL:
                    # reject move if bond constraints violated beyond tolerance
                    self.protein.coords = coords.copy()
                    accept = False
                else:
                    coords = coords_new.copy()
                    e = new_e
                    energy_dict = energy_dict_new
                    if e < best_e:
                        best_e = e
                        best_coords = coords.copy()

            if not accept:
                # ensure coords and energy_dict reflect the preserved state, and avoid recording spike values
                self.protein.coords = coords.copy()
                self.history.append((coords.copy(), energy_dict))
            else:
                # record the accepted energy (no spike)
                self.history.append((coords.copy(), energy_dict_new))

            # cool in Kelvin and update energy-scale temperature
            self.temp_K *= self.cooling
            self.T_energy = self.k_B * self.temp_K
            if step % 500 == 0 and step > 0:
                self.step_size *= 0.9
            if self.temp_K < 1e-6:
                break
        self.protein.coords = best_coords.copy()
        return best_coords, best_e, self.history


__all__ = ['SimulatedAnnealer']
