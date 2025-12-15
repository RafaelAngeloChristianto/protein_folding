import numpy as np
import math
from .utils import rotation_matrix
from multiprocessing import Pool, cpu_count
import os


# Module-level worker function for parallel replica evaluation (must be picklable)
def _evaluate_replica_worker(args):
    """Worker function to propose and evaluate a move for one replica.
    
    Args:
        args: tuple of (replica_dict, step_size, ca_distance, BOND_TOL, k_B)
    Returns:
        updated replica_dict with new coords and energy_dict
    """
    replica_dict, step_size, ca_distance, BOND_TOL, k_B = args
    from .utils import max_bond_deviation, rotation_matrix
    
    # Extract current state
    coords_old = replica_dict['coords'].copy()
    coords_new = coords_old.copy()
    N = coords_old.shape[0]
    
    # Propose move (local copy of propose logic)
    if N > 2:
        r = np.random.rand()
        if r < 0.7:
            b = np.random.randint(0, N - 1)
            p1 = coords_old[b]
            p2 = coords_old[b+1]
            axis = p2 - p1
            axis_len = np.linalg.norm(axis)
            if axis_len > 0:
                axis = axis / axis_len
                angle = np.random.normal(scale=step_size)
                R = rotation_matrix(axis, angle)
                idx = np.arange(b+1, N)
                if idx.size > 0:
                    rel = coords_old[idx] - p1
                    rotated = (R @ rel.T).T + p1
                    coords_new[idx] = rotated
        elif r < 0.9:
            pivot = np.random.randint(0, N - 1)
            angle = np.random.normal(scale=2.0*step_size)
            axis = np.random.normal(size=3)
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
                R = rotation_matrix(axis, angle)
                pivot_point = coords_old[pivot]
                idx = np.arange(pivot + 1, N)
                if idx.size > 0:
                    rel = coords_old[idx] - pivot_point
                    rotated = (R @ rel.T).T + pivot_point
                    coords_new[idx] = rotated
        else:
            i = np.random.randint(1, N - 1)
            displacement = np.random.normal(scale=step_size, size=3)
            coords_new[i] += displacement
            if i - 1 >= 0:
                coords_new[i-1] += 0.2 * displacement
            if i + 1 < N:
                coords_new[i+1] += 0.2 * displacement
    
    # Bond check
    max_dev = max_bond_deviation(coords_new, r0=ca_distance)
    if max_dev > BOND_TOL:
        # Reject - return unchanged
        return replica_dict
    
    # Evaluate energy
    from .models import Protein
    prot = Protein.__new__(Protein)
    prot.__dict__.update(replica_dict['protein_dict'])
    prot.coords = coords_new
    prot.ca_distance = ca_distance
    ef = replica_dict['energy_fn_class'](prot)
    energy_dict_new = ef.total_energy()
    new_e = energy_dict_new['total']
    old_e = replica_dict['energy_dict']['total']
    
    # Acceptance
    if not np.isfinite(new_e) or not np.isfinite(old_e):
        return replica_dict
    
    if new_e < old_e:
        accept = True
    else:
        T_energy = k_B * replica_dict['temp_K']
        if np.random.rand() < math.exp(-(new_e - old_e) / max(T_energy, 1e-12)):
            accept = True
        else:
            accept = False
    
    if accept:
        replica_dict['coords'] = coords_new.copy()
        replica_dict['energy_dict'] = energy_dict_new
    
    return replica_dict


class ReplicaExchange:
    """Enhanced Parallel Tempering / Replica Exchange Monte Carlo for maximum accuracy.

    This runs `num_replicas` Monte-Carlo replicas at different fixed temperatures
    and attempts swaps between neighboring replicas every `exchange_interval`
    steps. Enhanced for 80%+ accuracy target.
    """
    def __init__(self, protein, energy_fn, num_replicas=6, exchange_interval=8, base_temp_K=400.0,
                 temp_scale=1.4, k_B=0.0019872041, max_steps=25000, step_size=0.3, debug=False, n_workers=None):
        import copy
        from .utils import max_bond_deviation

        self.copy = copy
        self.max_bond_deviation = max_bond_deviation
        self.energy_fn_class = energy_fn.__class__
        # create independent replicas (deep copies of the protein and independent energy_fns)
        self.replicas = []
        self.num_replicas = int(max(2, num_replicas))
        self.exchange_interval = int(max(1, exchange_interval))
        self.k_B = float(k_B)
        self.max_steps = int(max_steps)
        self.step_size = float(step_size)
        self.debug = bool(debug)
        self.n_workers = n_workers if n_workers is not None else min(cpu_count() or 1, self.num_replicas)

        # temperature ladder (geometric)
        self.temps = [base_temp_K * (temp_scale ** i) for i in range(self.num_replicas)]
        for i in range(self.num_replicas):
            prot_copy = self.copy.deepcopy(protein)
            ef = self.energy_fn_class(prot_copy)
            # Store protein state dict for parallel workers
            prot_dict = {k: v for k, v in prot_copy.__dict__.items() if k != 'coords'}
            self.replicas.append({
                'protein': prot_copy,
                'energy_fn': ef,
                'energy_fn_class': self.energy_fn_class,
                'protein_dict': prot_dict,
                'coords': prot_copy.coords.copy(),
                'temp_K': float(self.temps[i]),
                'T_energy': float(self.k_B * self.temps[i]),
                'step_size': float(self.step_size),
                'energy_dict': ef.total_energy()
            })

    def _propose_move_for_coords(self, coords, step_size):
        # Local copy of torsion/pivot/random displacement logic
        coords_old = coords.copy()
        coords_new = coords_old.copy()
        N = coords_old.shape[0]
        if N <= 2:
            return coords_new

        r = np.random.rand()
        if r < 0.7:
            b = np.random.randint(0, N - 1)
            p1 = coords_old[b]
            p2 = coords_old[b+1]
            axis = p2 - p1
            axis_len = np.linalg.norm(axis)
            if axis_len == 0:
                return coords_new
            axis = axis / axis_len
            angle = np.random.normal(scale=step_size)
            R = rotation_matrix(axis, angle)
            idx = np.arange(b+1, N)
            if idx.size > 0:
                rel = coords_old[idx] - p1
                rotated = (R @ rel.T).T + p1
                coords_new[idx] = rotated
            return coords_new
        elif r < 0.9:
            pivot = np.random.randint(0, N - 1)
            angle = np.random.normal(scale=2.0*step_size)
            axis = np.random.normal(size=3)
            norm = np.linalg.norm(axis)
            if norm == 0:
                return coords_new
            axis = axis / norm
            R = rotation_matrix(axis, angle)
            pivot_point = coords_old[pivot]
            idx = np.arange(pivot + 1, N)
            if idx.size > 0:
                rel = coords_old[idx] - pivot_point
                rotated = (R @ rel.T).T + pivot_point
                coords_new[idx] = rotated
            return coords_new
        else:
            i = np.random.randint(1, N - 1)
            displacement = np.random.normal(scale=step_size, size=3)
            coords_new[i] += displacement
            if i - 1 >= 0:
                coords_new[i-1] += 0.2 * displacement
            if i + 1 < N:
                coords_new[i+1] += 0.2 * displacement
            return coords_new

    def run(self):
        # history will mirror the coldest replica (index 0) over time
        history = []
        best_coords = self.replicas[0]['coords'].copy()
        best_e = self.replicas[0]['energy_dict']['total']

        BOND_TOL = 1e-6
        
        # Setup multiprocessing pool for parallel replica evaluation
        pool = None
        use_parallel = self.n_workers > 1 and self.num_replicas > 1
        if use_parallel:
            try:
                pool = Pool(processes=self.n_workers)
            except Exception:
                use_parallel = False
        
        try:
            for step in range(self.max_steps):
                # perform one MC step for each replica - in parallel if possible
                if use_parallel:
                    # Prepare tasks for parallel execution
                    tasks = [(rep.copy(), rep['step_size'], rep['protein'].ca_distance, BOND_TOL, self.k_B) 
                             for rep in self.replicas]
                    # Evaluate all replicas in parallel
                    updated_replicas = pool.map(_evaluate_replica_worker, tasks)
                    # Update replicas with results
                    for i, updated_rep in enumerate(updated_replicas):
                        self.replicas[i]['coords'] = updated_rep['coords'].copy()
                        self.replicas[i]['energy_dict'] = updated_rep['energy_dict']
                        self.replicas[i]['protein'].coords = self.replicas[i]['coords']
                        
                        # track best overall
                        cur_e = self.replicas[i]['energy_dict']['total']
                        if cur_e < best_e:
                            best_e = cur_e
                            best_coords = self.replicas[i]['coords'].copy()
                else:
                    # Sequential evaluation (original logic)
                    for i, rep in enumerate(self.replicas):
                        coords_old = rep['coords'].copy()
                        coords_new = self._propose_move_for_coords(coords_old, rep['step_size'])
                        # bond check
                        max_dev = self.max_bond_deviation(coords_new, r0=rep['protein'].ca_distance)
                        if max_dev > BOND_TOL:
                            # reject
                            rep['coords'] = coords_old.copy()
                            # keep energy_dict unchanged
                            continue

                        # evaluate
                        rep['protein'].coords = coords_new
                        energy_dict_new = rep['energy_fn'].total_energy()
                        new_e = energy_dict_new['total']
                        old_e = rep['energy_dict']['total']

                        # acceptance
                        if not np.isfinite(new_e) or not np.isfinite(old_e):
                            rep['protein'].coords = coords_old.copy()
                            continue

                        if new_e < old_e:
                            accept = True
                        else:
                            if np.random.rand() < math.exp(-(new_e - old_e) / max(rep['T_energy'], 1e-12)):
                                accept = True
                            else:
                                accept = False

                        if accept:
                            rep['coords'] = coords_new.copy()
                            rep['energy_dict'] = energy_dict_new
                        else:
                            rep['coords'] = coords_old.copy()

                        # track best overall
                        cur_e = rep['energy_dict']['total']
                        if cur_e < best_e:
                            best_e = cur_e
                            best_coords = rep['coords'].copy()

                # attempt exchanges between neighboring replicas every exchange_interval steps
                if (step + 1) % self.exchange_interval == 0:
                    for i in range(self.num_replicas - 1):
                        rep_i = self.replicas[i]
                        rep_j = self.replicas[i+1]
                        Ei = rep_i['energy_dict']['total']
                        Ej = rep_j['energy_dict']['total']
                        beta_i = 1.0 / (self.k_B * rep_i['temp_K'])
                        beta_j = 1.0 / (self.k_B * rep_j['temp_K'])
                        # acceptance for swap: exp((beta_i - beta_j) * (Ej - Ei))
                        try:
                            prob = math.exp((beta_i - beta_j) * (Ej - Ei))
                        except OverflowError:
                            prob = float('inf') if (beta_i - beta_j) * (Ej - Ei) > 0 else 0.0
                        if np.random.rand() < min(1.0, prob):
                            # swap coordinates and energy_dict
                            tmp_coords = rep_i['coords'].copy()
                            tmp_ed = rep_i['energy_dict']
                            rep_i['coords'] = rep_j['coords'].copy()
                            rep_i['protein'].coords = rep_i['coords']
                            rep_i['energy_dict'] = rep_j['energy_dict']
                            rep_j['coords'] = tmp_coords.copy()
                            rep_j['protein'].coords = rep_j['coords']
                            rep_j['energy_dict'] = tmp_ed

                # append coldest replica state to history for GUI plotting
                cold = self.replicas[0]
                history.append((cold['coords'].copy(), cold['energy_dict']))
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # ensure protein ends at best found
        return best_coords, best_e, history


class SimulatedAnnealer:
    def __init__(self, protein, energy_fn, temp_K=2000.0, k_B=0.0019872041, cooling=0.9985, max_steps=25000, step_size=0.08, max_energy_jump=1e6, debug=False, energy_term_threshold=1e5, n_workers=None, adaptive_cooling=True):
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
        # parallel evaluation: use all available cores by default
        self.n_workers = n_workers if n_workers is not None else (cpu_count() or 1)
        self.adaptive_cooling = adaptive_cooling
        self.energy_plateau_count = 0
        self.last_best_energy = float('inf')

    def propose_move(self):
        coords_old = self.protein.coords.copy()
        coords_new = coords_old.copy()
        N = self.protein.N
        if N <= 2:
            return coords_new, coords_old

        # Enhanced move set for maximum accuracy
        r = np.random.rand()
        if r < 0.6:  # More frequent torsion moves
            # Enhanced torsion-angle moves with better sampling
            b = np.random.randint(0, N - 1)
            p1 = coords_old[b]
            p2 = coords_old[b+1]
            axis = p2 - p1
            axis_len = np.linalg.norm(axis)
            if axis_len == 0:
                return coords_new, coords_old
            axis = axis / axis_len
            
            # Enhanced angle sampling with temperature dependence
            temp_factor = max(0.1, self.temp_K / 2000.0)  # Scale with temperature
            angle = np.random.normal(scale=self.step_size * temp_factor)
            
            R = rotation_matrix(axis, angle)
            idx = np.arange(b+1, N)
            if idx.size > 0:
                rel = coords_old[idx] - p1
                rotated = (R @ rel.T).T + p1
                coords_new[idx] = rotated
            return coords_new, coords_old
        elif r < 0.8:  # More pivot moves
            # Enhanced pivot moves with better axis selection
            pivot = np.random.randint(1, N - 1)  # Avoid endpoints
            
            # Bias toward middle of chain for better global moves
            if N > 10:
                center = N // 2
                pivot = np.random.choice([center + np.random.randint(-N//4, N//4), 
                                        np.random.randint(1, N-1)])
                pivot = max(1, min(N-2, pivot))
            
            temp_factor = max(0.2, self.temp_K / 2000.0)
            angle = np.random.normal(scale=1.5*self.step_size * temp_factor)
            
            # Better axis selection (sometimes use local geometry)
            if np.random.rand() < 0.3 and pivot > 0 and pivot < N-1:
                # Use local geometry for axis
                v1 = coords_old[pivot] - coords_old[pivot-1]
                v2 = coords_old[pivot+1] - coords_old[pivot]
                axis = np.cross(v1, v2)
                if np.linalg.norm(axis) > 1e-6:
                    axis = axis / np.linalg.norm(axis)
                else:
                    axis = np.random.normal(size=3)
                    axis = axis / np.linalg.norm(axis)
            else:
                axis = np.random.normal(size=3)
                axis = axis / np.linalg.norm(axis)
            
            R = rotation_matrix(axis, angle)
            pivot_point = coords_old[pivot]
            idx = np.arange(pivot + 1, N)
            if idx.size > 0:
                rel = coords_old[idx] - pivot_point
                rotated = (R @ rel.T).T + pivot_point
                coords_new[idx] = rotated
            return coords_new, coords_old
        elif r < 0.95:  # Local moves
            # Enhanced local displacement with cooperative motion
            i = np.random.randint(1, N - 1)
            temp_factor = max(0.1, self.temp_K / 2000.0)
            displacement = np.random.normal(scale=self.step_size * temp_factor, size=3)
            
            # Cooperative motion with neighbors
            coords_new[i] += displacement
            if i - 1 >= 0:
                coords_new[i-1] += 0.3 * displacement  # Stronger coupling
            if i + 1 < N:
                coords_new[i+1] += 0.3 * displacement  # Stronger coupling
            
            # Sometimes include next neighbors
            if np.random.rand() < 0.3:
                if i - 2 >= 0:
                    coords_new[i-2] += 0.1 * displacement
                if i + 2 < N:
                    coords_new[i+2] += 0.1 * displacement
            
            return coords_new, coords_old
        else:
            # New: segment flip moves for better sampling
            if N > 6:
                # Flip a small segment
                start = np.random.randint(1, N - 4)
                length = min(np.random.randint(2, 5), N - start - 1)
                end = start + length
                
                # Flip the segment
                segment = coords_old[start:end+1].copy()
                coords_new[start:end+1] = segment[::-1]
            
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

        best_coords = coords.copy()
        best_e = e
        self.history.append((coords.copy(), energy_dict))
        from .utils import max_bond_deviation
        BOND_TOL = 1e-6
        for step in range(self.max_steps):
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

            # Fast adaptive cooling (less frequent checks)
            if self.adaptive_cooling and step % 200 == 0:  # Less frequent for speed
                if abs(best_e - self.last_best_energy) < 0.01:
                    self.energy_plateau_count += 1
                    if self.energy_plateau_count > 3:
                        self.cooling *= 0.98
                        self.step_size *= 1.05
                        self.energy_plateau_count = 0
                else:
                    self.energy_plateau_count = 0
                    self.last_best_energy = best_e
            
            # Enhanced cooling schedule
            self.temp_K *= self.cooling
            self.T_energy = self.k_B * self.temp_K
            
            # Fast step size reduction
            if step % 1000 == 0 and step > 0:
                self.step_size *= 0.95
            
            # Lower temperature threshold for longer optimization
            if self.temp_K < 1e-8:
                break
        self.protein.coords = best_coords.copy()
        return best_coords, best_e, self.history


class MultiScaleOptimizer:
    """Multi-scale optimization using coarse-to-fine approach for better accuracy"""
    
    def __init__(self, protein, energy_fn_class=None, debug=False):
        from .energy import EnergyFunction
        self.protein = protein
        self.energy_fn_class = energy_fn_class if energy_fn_class is not None else EnergyFunction
        self.debug = debug
        
    def run(self, max_steps=25000):
        """Fast 3-stage multi-scale optimization"""
        
        # Stage 1: Coarse exploration
        coarse_energy = self.energy_fn_class(self.protein)
        original_total = coarse_energy.total_energy
        def coarse_total_energy():
            result = original_total()
            result['bond'] *= 0.3
            result['angle'] *= 0.3
            result['hydro'] *= 3.0
            result['compact'] *= 3.0
            result['ss'] *= 0.8
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        coarse_energy.total_energy = coarse_total_energy
        
        coarse_optimizer = SimulatedAnnealer(
            self.protein, coarse_energy,
            temp_K=3000.0, cooling=0.990, max_steps=max_steps//3,
            step_size=0.15, adaptive_cooling=True
        )
        
        coords1, energy1, history1 = coarse_optimizer.run()
        self.protein.coords = coords1
        
        # Stage 2: Medium refinement
        medium_energy = self.energy_fn_class(self.protein)
        def medium_total_energy():
            result = original_total()
            result['hydro'] *= 2.0
            result['compact'] *= 2.0
            result['ss'] *= 1.5
            if 'hbond' in result: result['hbond'] *= 1.5
            if 'torsion' in result: result['torsion'] *= 1.2
            if 'loop' in result: result['loop'] *= 1.2
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        medium_energy.total_energy = medium_total_energy
        
        medium_optimizer = SimulatedAnnealer(
            self.protein, medium_energy,
            temp_K=1500.0, cooling=0.994, max_steps=max_steps//3,
            step_size=0.10, adaptive_cooling=True
        )
        
        coords2, energy2, history2 = medium_optimizer.run()
        self.protein.coords = coords2
        
        # Stage 3: Fine optimization
        fine_energy = self.energy_fn_class(self.protein)
        
        fine_optimizer = SimulatedAnnealer(
            self.protein, fine_energy,
            temp_K=800.0, cooling=0.996, max_steps=max_steps//3,
            step_size=0.06, adaptive_cooling=True
        )
        
        coords3, energy3, history3 = fine_optimizer.run()
        
        combined_history = history1 + history2 + history3
        return coords3, energy3, combined_history


class AdaptiveOptimizer:
    """Adaptive optimizer that adjusts strategy based on sequence properties"""
    
    def __init__(self, protein, energy_fn_class=None, debug=False):
        from .energy import EnergyFunction
        self.protein = protein
        self.energy_fn_class = energy_fn_class if energy_fn_class is not None else EnergyFunction
        self.debug = debug
        
    def analyze_sequence(self):
        """Analyze sequence to determine optimal folding strategy"""
        sequence = self.protein.sequence
        
        # Calculate hydrophobic content
        hydrophobic_residues = set('AILMFWYV')
        hydrophobic_fraction = sum(1 for aa in sequence if aa in hydrophobic_residues) / len(sequence)
        
        # Calculate charged content
        charged_residues = set('DEKR')
        charged_fraction = sum(1 for aa in sequence if aa in charged_residues) / len(sequence)
        
        # Determine strategy
        if hydrophobic_fraction > 0.4:
            strategy = 'hydrophobic_collapse'
        elif charged_fraction > 0.3:
            strategy = 'electrostatic_guided'
        else:
            strategy = 'balanced'
            
        return {
            'strategy': strategy,
            'hydrophobic_fraction': hydrophobic_fraction,
            'charged_fraction': charged_fraction,
            'length': len(sequence)
        }
    
    def run(self, max_steps=25000):
        """Run adaptive optimization based on sequence analysis"""
        
        analysis = self.analyze_sequence()
        strategy = analysis['strategy']
        
        if self.debug:
            print(f"Using strategy: {strategy}")
            print(f"Hydrophobic fraction: {analysis['hydrophobic_fraction']:.2f}")
            print(f"Charged fraction: {analysis['charged_fraction']:.2f}")
        
        if strategy == 'hydrophobic_collapse':
            return self._hydrophobic_strategy(max_steps)
        elif strategy == 'electrostatic_guided':
            return self._electrostatic_strategy(max_steps)
        else:
            return self._balanced_strategy(max_steps)
    
    def _hydrophobic_strategy(self, max_steps):
        """Ultra-enhanced hydrophobic strategy for maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Ultra-enhanced energy function for hydrophobic proteins
        original_total = energy_fn.total_energy
        def hydrophobic_total_energy():
            result = original_total()
            result['hydro'] *= 4.5  # Much stronger hydrophobic driving force
            result['compact'] *= 3.5  # Much stronger compactness
            result['lj'] *= 2.2  # Enhanced LJ interactions
            result['ss'] *= 2.0  # Much stronger secondary structure
            if 'hbond' in result: result['hbond'] *= 1.8  # Enhanced hydrogen bonding
            if 'rama' in result: result['rama'] *= 2.5  # Strong Ramachandran bias
            if 'torsion' in result: result['torsion'] *= 1.5  # Enhanced torsional constraints
            if 'loop' in result: result['loop'] *= 1.3  # Better loop closure
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = hydrophobic_total_energy
        
        # Fast 2-stage optimization
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2500.0, cooling=0.994, max_steps=max_steps//2,
            step_size=0.10, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1000.0, cooling=0.997, max_steps=max_steps//2,
            step_size=0.06, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history
    
    def _electrostatic_strategy(self, max_steps):
        """Ultra-enhanced strategy for charged proteins with maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Ultra-enhanced energy function for charged proteins
        original_total = energy_fn.total_energy
        def electrostatic_total_energy():
            result = original_total()
            result['elec'] *= 3.5  # Much stronger electrostatics
            if 'hbond' in result: result['hbond'] *= 2.5  # Enhanced hydrogen bonding for charged residues
            result['compact'] *= 0.6  # Reduced compactness to allow charge separation
            result['ss'] *= 1.8  # Enhanced secondary structure
            if 'rama' in result: result['rama'] *= 2.0  # Strong backbone constraints
            if 'torsion' in result: result['torsion'] *= 1.3  # Enhanced torsional bias
            if 'loop' in result: result['loop'] *= 1.5  # Better loop formation for charge accommodation
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = electrostatic_total_energy
        
        # Multi-stage optimization for charged proteins
        # Stage 1: Extended conformation search
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2800.0, cooling=0.990, max_steps=max_steps//2,
            step_size=0.14, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        # Stage 2: Fine electrostatic optimization
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1000.0, cooling=0.995, max_steps=max_steps//2,
            step_size=0.08, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history
    
    def _balanced_strategy(self, max_steps):
        """Ultra-enhanced balanced strategy for mixed sequences with maximum accuracy"""
        energy_fn = self.energy_fn_class(self.protein)
        
        # Enhanced balanced energy function
        original_total = energy_fn.total_energy
        def balanced_total_energy():
            result = original_total()
            # Balanced enhancement of all terms for maximum accuracy
            result['hydro'] *= 2.8
            result['elec'] *= 2.2
            result['compact'] *= 2.0
            result['ss'] *= 2.5
            if 'hbond' in result: result['hbond'] *= 2.0
            if 'rama' in result: result['rama'] *= 2.8
            if 'torsion' in result: result['torsion'] *= 1.8
            if 'loop' in result: result['loop'] *= 1.6
            result['lj'] *= 1.5
            result['total'] = sum(v for k, v in result.items() if k != 'total')
            return result
        
        energy_fn.total_energy = balanced_total_energy
        
        # Fast 2-stage balanced optimization
        optimizer1 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=2800.0, cooling=0.992, max_steps=max_steps//2,
            step_size=0.12, adaptive_cooling=True
        )
        coords1, energy1, history1 = optimizer1.run()
        self.protein.coords = coords1
        
        optimizer2 = SimulatedAnnealer(
            self.protein, energy_fn,
            temp_K=1200.0, cooling=0.996, max_steps=max_steps//2,
            step_size=0.06, adaptive_cooling=True
        )
        coords2, energy2, history2 = optimizer2.run()
        
        combined_history = history1 + history2
        return coords2, energy2, combined_history


__all__ = ['SimulatedAnnealer', 'ReplicaExchange', 'MultiScaleOptimizer', 'AdaptiveOptimizer']
