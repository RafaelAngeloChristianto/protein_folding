#!/usr/bin/env python3
"""Comprehensive GUI for protein folding demo with embedded visualization"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from compbio_fp.models import Protein
from compbio_fp.energy import EnergyFunction
from compbio_fp.optimizer import SimulatedAnnealer, ReplicaExchange
from compbio_fp.protein_builder import build_backbone_from_CA, pack_sidechains, write_pdb

class ProteinFoldingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Protein Folding Simulation")
        self.root.geometry("1200x800")
        
        # Variables
        self.sequence_var = tk.StringVar(value="ACDEFGHIKLMNPQR")
        self.steps_var = tk.StringVar(value="1500")
        self.temp_var = tk.StringVar(value="300")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()
        
        self.history = []
        self.current_frame = 0
        self.animation_running = False
        self.start_temp = None
        # store results when running multiple optimizers
        self.optimizer_results = []
        # references to colorbar/legend so we can remove them between redraws
        self.hydro_colorbar = None
        self.hydro_legend = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))
        
        # Sequence
        ttk.Label(control_frame, text="Sequence:").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.sequence_var, width=20).pack(pady=(0,10))
        
        # Steps
        ttk.Label(control_frame, text="Steps:").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.steps_var, width=10).pack(pady=(0,10))
        
        # Temperature
        ttk.Label(control_frame, text="Temperature (K):").pack(anchor=tk.W)
        ttk.Entry(control_frame, textvariable=self.temp_var, width=10).pack(pady=(0,10))
        
        # Buttons
        self.run_btn = ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation)
        self.run_btn.pack(pady=10)
        
        self.animate_btn = ttk.Button(control_frame, text="Animate", command=self.toggle_animation, state='disabled')
        self.animate_btn.pack(pady=5)
        # View in PyMOL button
        self.pymol_btn = ttk.Button(control_frame, text="View in PyMOL", command=self.view_in_pymol, state='disabled')
        self.pymol_btn.pack(pady=5)
        # Run all optimizers button
        self.run_all_btn = ttk.Button(control_frame, text="Run All Optimizers", command=self.run_all_optimizers)
        self.run_all_btn.pack(pady=5)
        # Hydrophobicity display mode
        ttk.Label(control_frame, text="Hydrophobicity view:").pack(anchor=tk.W, pady=(10,0))
        self.hydro_mode_var = tk.StringVar(value='Continuous')
        ttk.Combobox(control_frame, textvariable=self.hydro_mode_var, values=['Continuous'], state='readonly', width=12).pack(pady=(0,5))

        # Optimizer selection
        ttk.Label(control_frame, text="Optimizer:").pack(anchor=tk.W, pady=(10,0))
        self.optimizer_var = tk.StringVar(value='Simulated Annealing')
        self.optimizer_combo = ttk.Combobox(control_frame, textvariable=self.optimizer_var,
                            values=['Simulated Annealing', 'Parallel Tempering'], state='readonly', width=18)
        self.optimizer_combo.pack(pady=(0,5))
        self.optimizer_combo.bind('<<ComboboxSelected>>', lambda e: self._on_optimizer_change())

        # Parallel Tempering params (hidden unless PT selected)
        self.replicas_var = tk.IntVar(value=4)
        self.exchange_interval_var = tk.IntVar(value=10)
        self.pt_frame = ttk.Frame(control_frame)
        ttk.Label(self.pt_frame, text="Replicas:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.pt_frame, textvariable=self.replicas_var, width=6).grid(row=0, column=1, padx=(4,0))
        ttk.Label(self.pt_frame, text="Exchange int:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(self.pt_frame, textvariable=self.exchange_interval_var, width=6).grid(row=1, column=1, padx=(4,0))
        # not packed by default

        
        # Progress
        ttk.Label(control_frame, text="Progress:").pack(anchor=tk.W, pady=(20,0))
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=150).pack(pady=10)
        
        # Right panel - visualization + results (tabs)
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Notebook with three tabs: Structure, Graphs and Results
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.structure_tab = ttk.Frame(self.notebook)
        self.graphs_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.structure_tab, text="Structure")
        self.notebook.add(self.graphs_tab, text="Graphs")
        self.notebook.add(self.results_tab, text="Results")

        # Create matplotlib figure for the 3D final structure (placed in Structure tab)
        self.fig_structure = Figure(figsize=(6, 6))
        self.ax3d = self.fig_structure.add_subplot(111, projection='3d')
        self.canvas_structure = FigureCanvasTkAgg(self.fig_structure, self.structure_tab)
        self.canvas_structure.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure for the three graphs (placed in Graphs tab)
        # Use a 2x2 grid and place plots at: (1,1)=energy, (1,2)=rmsd, (2,1)=temp
        # leave (2,2) empty
        self.fig_graphs = Figure(figsize=(8, 8))
        self.ax_energy = self.fig_graphs.add_subplot(2, 2, 1)
        self.ax_rmsd = self.fig_graphs.add_subplot(2, 2, 2)
        self.ax_temp = self.fig_graphs.add_subplot(2, 2, 3)
        # leave (2,2) empty in this figure; the comparison matrix will be in its own tab
        self.canvas_graphs = FigureCanvasTkAgg(self.fig_graphs, self.graphs_tab)
        self.canvas_graphs.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # Matrixes tab: dedicated space for pairwise comparison matrices
        self.matrixes_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.matrixes_tab, text="Matrixes")
        self.fig_matrix = Figure(figsize=(5, 5))
        self.ax_matrix = self.fig_matrix.add_subplot(111)
        self.canvas_matrix = FigureCanvasTkAgg(self.fig_matrix, self.matrixes_tab)
        self.canvas_matrix.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Additional small structures figure to compare optimizer 3D results
        self.fig_structs = Figure(figsize=(8, 3))
        self.ax_struct1 = self.fig_structs.add_subplot(1, 2, 1, projection='3d')
        self.ax_struct2 = self.fig_structs.add_subplot(1, 2, 2, projection='3d')
        # Place the comparison 3D canvases in the Structure tab so
        # "Run All Optimizers" shows the two optimizer results there.
        self.canvas_structs = FigureCanvasTkAgg(self.fig_structs, self.structure_tab)
        # initially hide the comparison canvas; show it only after Run All
        self.canvas_structs.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5,0))
        try:
            self.canvas_structs.get_tk_widget().pack_forget()
        except Exception:
            pass

        # Attempt to make each axes square-shaped where supported
        for _ax in (self.ax_energy, self.ax_rmsd, self.ax_temp):
            try:
                _ax.set_box_aspect(1)
            except Exception:
                # Older matplotlib may not support set_box_aspect; ignore
                pass

        # Results tab: scrolled text for numeric output
        self.results_text = scrolledtext.ScrolledText(self.results_tab, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig_structure.tight_layout()
        self.fig_graphs.tight_layout()
        # Ensure optimizer UI initial state
        self._on_optimizer_change()

    def _on_optimizer_change(self):
        """Show/hide optimizer-specific parameter controls."""
        try:
            if self.optimizer_var.get() == 'Parallel Tempering':
                # show PT params
                self.pt_frame.pack(pady=(0,5))
            else:
                # hide PT params
                self.pt_frame.pack_forget()
        except Exception:
            pass
        
    def run_simulation(self):
        sequence = self.sequence_var.get().strip().upper()
        try:
            steps = int(self.steps_var.get())
            temp = float(self.temp_var.get())
        except ValueError:
            self.status_var.set("Error: Invalid numeric values")
            return
        # record the starting temperature chosen by the user for verification
        self.start_temp = temp
            
        self.run_btn.config(state='disabled')
        self.animate_btn.config(state='disabled')
        self.status_var.set("Running simulation...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self._run_simulation, args=(sequence, steps, temp))
        thread.daemon = True
        thread.start()

    def run_all_optimizers(self):
        """Run all available optimizers on the same sequence and collect comparisons."""
        sequence = self.sequence_var.get().strip().upper()
        try:
            steps = int(self.steps_var.get())
            temp = float(self.temp_var.get())
        except ValueError:
            self.status_var.set("Error: Invalid numeric values")
            return
        # record starting temp
        self.start_temp = temp

        # disable controls
        self.run_btn.config(state='disabled')
        self.run_all_btn.config(state='disabled')
        self.animate_btn.config(state='disabled')
        self.status_var.set("Running all optimizers...")
        self.progress_var.set(0)

        thread = threading.Thread(target=self._run_all_thread, args=(sequence, steps, temp))
        thread.daemon = True
        thread.start()

    def _run_all_thread(self, sequence, steps, temp):
        try:
            results = []
            # define optimizer configurations to run
            optimizers = [
                ('Simulated Annealing', 'SA'),
                ('Parallel Tempering', 'PT')
            ]

            total = len(optimizers)
            for idx, (name, key) in enumerate(optimizers):
                self.root.after(0, lambda n=name: self.status_var.set(f"Running: {n}"))
                start = time.time()
                protein = Protein(sequence)
                # capture hydrophobicity values for this protein so we can color comparison plots
                try:
                    hydro_vals = [p['hydro'] for p in protein.props]
                except Exception:
                    hydro_vals = None
                energy_fn = EnergyFunction(protein)
                if key == 'PT':
                    nrep = int(self.replicas_var.get()) if hasattr(self, 'replicas_var') else 4
                    exch = int(self.exchange_interval_var.get()) if hasattr(self, 'exchange_interval_var') else 10
                    optimizer = ReplicaExchange(protein, energy_fn, num_replicas=nrep, exchange_interval=exch, base_temp_K=temp, max_steps=steps)
                else:
                    optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp, max_steps=steps)

                best_coords, best_e, history = optimizer.run()
                elapsed = time.time() - start
                results.append({
                    'name': name,
                    'time': elapsed,
                    'best_energy': best_e,
                    'best_coords': best_coords,
                    'history': history,
                    'hydro': hydro_vals
                })
                # update progress
                pct = int(((idx + 1) / total) * 100)
                self.root.after(0, lambda p=pct: self.progress_var.set(p))

            # store optimizer_results and pick best overall by energy
            self.optimizer_results = results
            winner = min(results, key=lambda r: r['best_energy'] if r['best_energy'] is not None else float('inf'))
            self.best_coords = winner['best_coords']
            self.best_energy = winner['best_energy']
            # use winner history for plotting
            self.history = winner['history']

            self.root.after(0, self._all_complete)
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))
            self.root.after(0, lambda: self.run_btn.config(state='normal'))
            self.root.after(0, lambda: self.run_all_btn.config(state='normal'))

    def _all_complete(self):
        self.status_var.set("All optimizers completed")
        self.progress_var.set(100)
        self.run_btn.config(state='normal')
        self.run_all_btn.config(state='normal')
        self.animate_btn.config(state='normal')
        self.pymol_btn.config(state='normal')
        # plot winner results
        # hide the main single-structure view and show the two comparison views
        try:
            self.show_only_comparisons()
        except Exception:
            pass
        self.plot_results()
        # if multiple optimizers ran, also draw their best structures side-by-side
        try:
            if getattr(self, 'optimizer_results', None):
                self.plot_structs_comparison(self.optimizer_results)
                # also draw pairwise RMSD matrix
                try:
                    self.plot_structs_matrix(self.optimizer_results)
                except Exception:
                    pass
            else:
                # clear comparison axes
                self.ax_struct1.clear(); self.ax_struct2.clear(); self.canvas_structs.draw()
        except Exception:
            pass
        try:
            self.update_results_text()
        except Exception:
            pass
        
    def _run_simulation(self, sequence, steps, temp):
        try:
            protein = Protein(sequence)
            # store hydrophobicity values (0..1) for coloring
            try:
                self.protein_hydro = [p['hydro'] for p in protein.props]
            except Exception:
                self.protein_hydro = None
            energy_fn = EnergyFunction(protein)
            # debug: print the temperature being used to construct the optimizer
            print(f"[DEBUG] Starting temperature (K) passed to optimizer: {temp}")
            # construct selected optimizer
            opt_choice = self.optimizer_var.get() if hasattr(self, 'optimizer_var') else 'Simulated Annealing'
            if opt_choice == 'Parallel Tempering':
                nrep = int(self.replicas_var.get()) if hasattr(self, 'replicas_var') else 4
                exch = int(self.exchange_interval_var.get()) if hasattr(self, 'exchange_interval_var') else 10
                optimizer = ReplicaExchange(protein, energy_fn, num_replicas=nrep, exchange_interval=exch, base_temp_K=temp, max_steps=steps)
            else:
                optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp, max_steps=steps)
            # capture initial energy-scale temperature (kcal/mol units) for display
            try:
                self.start_T_energy = optimizer.T_energy
            except Exception:
                self.start_T_energy = None
            # capture cooling rate so we can mirror the temperature decay in the plot
            try:
                self.start_cooling = optimizer.cooling
            except Exception:
                self.start_cooling = 0.995
            
            best_coords, best_e, self.history = optimizer.run()
            # store best results for later display
            self.best_coords = best_coords
            self.best_energy = best_e
            self.root.after(0, self._simulation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.run_btn.config(state='normal'))
    
    def _simulation_complete(self):
        self.status_var.set("Simulation completed!")
        self.progress_var.set(100)
        self.run_btn.config(state='normal')
        self.animate_btn.config(state='normal')
        self.pymol_btn.config(state='normal')
        # when running a single simulation, hide comparison canvases and show the main structure
        try:
            self.show_only_main_structure()
        except Exception:
            pass
        self.plot_results()
        # populate the Results tab with numeric outputs
        try:
            self.update_results_text()
        except Exception:
            pass
        # Display contact map for the single optimizer
        try:
            opt_name = self.optimizer_var.get() if hasattr(self, 'optimizer_var') else 'Optimizer'
            single_result = [{
                'name': opt_name,
                'best_coords': self.best_coords,
                'hydro': getattr(self, 'protein_hydro', None)
            }]
            self.plot_structs_matrix(single_result)
        except Exception as e:
            print(f"Error plotting single optimizer contact map: {e}")
            pass
        
    def plot_results(self):
        if not self.history:
            return
        # Plot final structure in its own tab
        self.plot_structure(self.history[-1][0], self.history[-1][1].get('total') if isinstance(self.history[-1][1], dict) else None)

        # Plot graphs (energy, RMSD, temperature) in the Graphs tab
        self.plot_graphs()
        # If optimizer comparison results exist, update small 3D comparison axes
        try:
            if getattr(self, 'optimizer_results', None):
                self.plot_structs_comparison(self.optimizer_results)
        except Exception:
            pass

    def plot_structs_comparison(self, optimizer_results):
        """Draw best-structure 3D views side-by-side for each optimizer in `optimizer_results`.

        Expects `optimizer_results` to be a list of dicts with keys 'name' and 'best_coords'.
        """
        try:
            # Prepare axes
            self.ax_struct1.clear()
            self.ax_struct2.clear()

            # Draw first two optimizers if available
            if len(optimizer_results) >= 1:
                r0 = optimizer_results[0]
                coords0 = np.asarray(r0.get('best_coords'))
                hydro0 = r0.get('hydro')
                if coords0.size:
                    self.ax_struct1.plot(coords0[:,0], coords0[:,1], coords0[:,2], color='gray', linewidth=1)
                    # color by hydrophobicity when available
                    if hydro0 is None:
                        self.ax_struct1.scatter(coords0[:,0], coords0[:,1], coords0[:,2], c='C0', s=30)
                    else:
                        try:
                            import matplotlib.cm as cm
                            import matplotlib.colors as mcolors
                            mode = self.hydro_mode_var.get() if hasattr(self, 'hydro_mode_var') else 'Continuous'
                            if mode == 'Continuous':
                                cmap = cm.get_cmap('coolwarm')
                                norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                                self.ax_struct1.scatter(coords0[:,0], coords0[:,1], coords0[:,2], c=hydro0, cmap=cmap, norm=norm, s=30)
                            else:
                                thresh = float(getattr(self, 'hydro_thresh_var', 0.5)) if not hasattr(self, 'hydro_thresh_var') else float(self.hydro_thresh_var.get())
                                cols = ['orange' if h > thresh else 'cyan' for h in hydro0]
                                self.ax_struct1.scatter(coords0[:,0], coords0[:,1], coords0[:,2], c=cols, s=30)
                        except Exception:
                            self.ax_struct1.scatter(coords0[:,0], coords0[:,1], coords0[:,2], c='C0', s=30)
                    self.ax_struct1.set_title(r0.get('name'))
            if len(optimizer_results) >= 2:
                r1 = optimizer_results[1]
                coords1 = np.asarray(r1.get('best_coords'))
                hydro1 = r1.get('hydro')
                if coords1.size:
                    self.ax_struct2.plot(coords1[:,0], coords1[:,1], coords1[:,2], color='gray', linewidth=1)
                    # color by hydrophobicity when available
                    if hydro1 is None:
                        self.ax_struct2.scatter(coords1[:,0], coords1[:,1], coords1[:,2], c='C1', s=30)
                    else:
                        try:
                            import matplotlib.cm as cm
                            import matplotlib.colors as mcolors
                            mode = self.hydro_mode_var.get() if hasattr(self, 'hydro_mode_var') else 'Continuous'
                            if mode == 'Continuous':
                                cmap = cm.get_cmap('coolwarm')
                                norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                                self.ax_struct2.scatter(coords1[:,0], coords1[:,1], coords1[:,2], c=hydro1, cmap=cmap, norm=norm, s=30)
                            else:
                                thresh = float(getattr(self, 'hydro_thresh_var', 0.5)) if not hasattr(self, 'hydro_thresh_var') else float(self.hydro_thresh_var.get())
                                cols = ['orange' if h > thresh else 'cyan' for h in hydro1]
                                self.ax_struct2.scatter(coords1[:,0], coords1[:,1], coords1[:,2], c=cols, s=30)
                        except Exception:
                            self.ax_struct2.scatter(coords1[:,0], coords1[:,1], coords1[:,2], c='C1', s=30)
                    self.ax_struct2.set_title(r1.get('name'))

            # If fewer than 2 results, leave the other empty
            self.canvas_structs.draw()
        except Exception:
            pass

    # Helpers to control which canvases are visible in the Structure tab
    def show_only_main_structure(self):
        """Show the large main structure canvas and hide the comparison canvases."""
        try:
            # ensure comparison canvas is hidden
            self.canvas_structs.get_tk_widget().pack_forget()
        except Exception:
            pass
        try:
            # make sure main structure canvas is visible
            self.canvas_structure.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception:
            pass

    def show_only_comparisons(self):
        """Hide the main structure canvas and show the two comparison canvases."""
        try:
            self.canvas_structure.get_tk_widget().pack_forget()
        except Exception:
            pass
        try:
            self.canvas_structs.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5,0))
        except Exception:
            pass

    def update_results_text(self):
        """Populate the Results tab with numeric summaries."""
        if not self.history:
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert(tk.END, "No results available.\n")
            return

        # initial energy and breakdown
        initial_energy = self.history[0][1].get('total') if isinstance(self.history[0][1], dict) else None
        energy_breakdown = self.history[0][1] if isinstance(self.history[0][1], dict) else {}

        # best energy found
        best_energy = getattr(self, 'best_energy', None)
        if best_energy is None:
            energies = [h[1].get('total') for h in self.history if isinstance(h[1], dict)]
            best_energy = min(energies) if energies else None

        # RMSD to initial structure (use best_coords if available, otherwise final)
        try:
            initial_coords = self.history[0][0]
            target_coords = getattr(self, 'best_coords', self.history[-1][0])
            diff = target_coords - initial_coords
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        except Exception:
            rmsd = None

        # Helper metrics
        def radius_of_gyration(coords):
            c = np.asarray(coords)
            centroid = c.mean(axis=0)
            rg = np.sqrt(np.mean(np.sum((c - centroid) ** 2, axis=1)))
            return float(rg)

        def rmsd(a, b):
            a = np.asarray(a)
            b = np.asarray(b)
            try:
                return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
            except Exception:
                return None

        def acceptance_rate_from_history(hist):
            if not hist or len(hist) < 2:
                return None
            changes = 0
            total = 0
            prev = np.asarray(hist[0][0])
            for frame in hist[1:]:
                cur = np.asarray(frame[0])
                total += 1
                if not np.allclose(prev, cur):
                    changes += 1
                prev = cur
            return changes / total if total > 0 else None

        def contact_fraction(coords, cutoff=8.0):
            c = np.asarray(coords)
            N = c.shape[0]
            if N < 2:
                return 0.0
            # compute pairwise distances upper triangle
            dists = np.sqrt(((c[None, :, :] - c[:, None, :]) ** 2).sum(axis=2))
            iu = np.triu_indices(N, k=1)
            pairs = dists[iu]
            contacts = np.sum(pairs < cutoff)
            total_pairs = len(pairs)
            return float(contacts) / total_pairs if total_pairs > 0 else 0.0

        # Build text lines
        lines = []
        # If multiple optimizers were run, include a comparison summary table
        if getattr(self, 'optimizer_results', None):
            lines.append("Optimizer comparisons:")
            for res in self.optimizer_results:
                name = res.get('name')
                t = res.get('time')
                be = res.get('best_energy')
                coords = res.get('best_coords')
                hist = res.get('history', [])
                N = len(coords) if coords is not None else (len(hist[0][0]) if hist else 0)
                energy_per_res = (be / N) if (be is not None and N > 0) else None
                rg = radius_of_gyration(coords) if coords is not None else None
                rmsd_val = None
                try:
                    if hist:
                        rmsd_val = rmsd(hist[0][0], coords)
                except Exception:
                    rmsd_val = None
                acc = acceptance_rate_from_history(hist)
                evals_sec = (len(hist) / t) if (t and t > 0) else None
                contact_frac = contact_fraction(coords) if coords is not None else None
                lines.append(f"- {name}: time={t:.2f}s, best_energy={be}, energy_per_res={energy_per_res}, Radius_of_Gyration={rg}, RMSD_init={rmsd_val}, acceptance_rate={acc}, evals/s={evals_sec}, contact_frac={contact_frac}")
            # indicate winner
            try:
                winner = min(self.optimizer_results, key=lambda r: r.get('best_energy') if r.get('best_energy') is not None else float('inf'))
                lines.append(f"Winner: {winner.get('name')} (energy={winner.get('best_energy')})\n")
            except Exception:
                pass
        # include starting temperature used
        if getattr(self, 'start_temp', None) is not None:
            lines.append(f"Starting temperature (K): {self.start_temp}\n")
        if getattr(self, 'start_T_energy', None) is not None:
            lines.append(f"Starting temperature (energy units, k_BT): {self.start_T_energy}\n")

        if initial_energy is not None:
            lines.append(f"Initial energy: {initial_energy}\n")
        else:
            lines.append("Initial energy: N/A\n")

        lines.append(f"Energy breakdown: {energy_breakdown}\n")

        if best_energy is not None:
            lines.append(f"Best energy found: {best_energy}\n")
        else:
            lines.append("Best energy found: N/A\n")

        if rmsd is not None:
            lines.append(f"RMSD to initial structure: {rmsd}\n")
        else:
            lines.append("RMSD to initial structure: N/A\n")

        # Insert into scrolled text
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        
    def toggle_animation(self):
        if not self.animation_running:
            self.animation_running = True
            self.animate_btn.config(text="Stop")
            self.animate_structure()
        else:
            self.animation_running = False
            self.animate_btn.config(text="Animate")
            
    def animate_structure(self):
        if not self.animation_running or not self.history:
            return
        coords = self.history[self.current_frame][0]
        energy = self.history[self.current_frame][1].get('total') if isinstance(self.history[self.current_frame][1], dict) else None
        self.ax3d.clear()
        # draw backbone line in neutral color
        self.ax3d.plot(coords[:,0], coords[:,1], coords[:,2], color='gray', linewidth=1)

        # color CA points by hydrophobicity if available
        hydro = getattr(self, 'protein_hydro', None)
        mode = self.hydro_mode_var.get() if hasattr(self, 'hydro_mode_var') else 'Continuous'
        if hydro is None:
            self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c='red', s=50)
        else:
            if mode == 'Continuous':
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                cmap = cm.get_cmap('coolwarm')
                norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                colors = [cmap(norm(h)) for h in hydro]
                self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=colors, s=50)
            else:
                thresh = float(self.hydro_thresh_var.get()) if hasattr(self, 'hydro_thresh_var') else 0.5
                cols = ['orange' if h > thresh else 'cyan' for h in hydro]
                self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=cols, s=50)
        title = f'Step {self.current_frame}'
        if energy is not None:
            title += f', Energy: {energy:.2f}'
        self.ax3d.set_title(title)

        self.canvas_structure.draw()

        self.current_frame = (self.current_frame + 1) % len(self.history)
        if self.animation_running:
            self.root.after(50, self.animate_structure)

    def plot_structure(self, coords, energy=None):
        """Draw the 3D final structure into the Structure tab."""
        try:
            self.ax3d.clear()
            # draw backbone line in neutral color
            self.ax3d.plot(coords[:,0], coords[:,1], coords[:,2], color='gray', linewidth=1)

            # color CA points by hydrophobicity if available
            hydro = getattr(self, 'protein_hydro', None)
            mode = self.hydro_mode_var.get() if hasattr(self, 'hydro_mode_var') else 'Continuous'
            if hydro is None:
                self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c='red', s=50)
            else:
                if mode == 'Continuous':
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors

                    cmap = cm.get_cmap('coolwarm')
                    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                    sc = self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=hydro, cmap=cmap, norm=norm, s=60)
                    # Don't use make_axes_locatable with a 3D axes (it can create
                    # axes that interfere with mplot3d rendering). Instead create
                    # a small standalone axes in figure coordinates and reuse it.
                    try:
                        pos = self.ax3d.get_position()
                        fig = self.fig_structure
                        cax_width = 0.025
                        pad = 0.01
                        cax_x = pos.x1 + pad
                        cax_y = pos.y0
                        cax_h = pos.height
                        cax_rect = [cax_x, cax_y, cax_width, cax_h]

                        # create or reposition reusable colorbar axes
                        if getattr(self, 'hydro_cax', None) is None or self.hydro_cax not in fig.axes:
                            self.hydro_cax = fig.add_axes(cax_rect)
                        else:
                            try:
                                self.hydro_cax.set_position(cax_rect)
                                self.hydro_cax.cla()
                            except Exception:
                                pass

                        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                        mappable.set_array(hydro)
                        cb = fig.colorbar(mappable, cax=self.hydro_cax)
                        cb.set_label('Hydrophobicity (0..1)')
                        self.hydro_colorbar = cb
                    except Exception:
                        pass
                else:
                    thresh = float(self.hydro_thresh_var.get()) if hasattr(self, 'hydro_thresh_var') else 0.5
                    cols = ['orange' if h > thresh else 'cyan' for h in hydro]
                    self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=cols, s=60)
                    # legend: remove previous then add
                    from matplotlib.patches import Patch
                    legend_patches = [Patch(color='orange', label='Hydrophobic'), Patch(color='cyan', label='Hydrophilic')]
                    try:
                        if getattr(self, 'hydro_legend', None) is not None:
                            try:
                                self.hydro_legend.remove()
                            except Exception:
                                pass
                            self.hydro_legend = None
                        self.hydro_legend = self.ax3d.legend(handles=legend_patches)
                    except Exception:
                        pass

            title = 'Final Structure'
            if energy is not None:
                title += f' (Energy: {energy:.2f})'
            self.ax3d.set_title(title)
            self.canvas_structure.draw()
        except Exception:
            pass

    def plot_graphs(self):
        """Plot the energy, RMSD, and temperature graphs into the Graphs tab."""
        try:
            # Clear graph axes
            self.ax_energy.clear()
            self.ax_rmsd.clear()
            self.ax_temp.clear()

            # Energy evolution (plot vs step index)
            steps = np.arange(len(self.history))
            energies = [h[1]['total'] for h in self.history if isinstance(h[1], dict) and 'total' in h[1]]
            if energies:
                self.ax_energy.plot(steps, energies)
            self.ax_energy.set_title('Energy vs Step')
            self.ax_energy.set_ylabel('Energy')
            self.ax_energy.set_xlim(0, max(1, len(self.history)-1))

            # RMSD from initial
            initial_coords = self.history[0][0]
            rmsds = []
            for coords, _ in self.history:
                diff = coords - initial_coords
                rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
                rmsds.append(rmsd)
            self.ax_rmsd.plot(steps, rmsds)
            self.ax_rmsd.set_title('RMSD from Initial')
            self.ax_rmsd.set_ylabel('RMSD (Å)')
            self.ax_rmsd.set_xlim(0, max(1, len(self.history)-1))

            # Temperature (mock)
            # compute temperature decay using the starting temperature entered by the user
            start_temp = getattr(self, 'start_temp', None)
            if start_temp is None:
                start_temp = 300.0
            cooling = getattr(self, 'start_cooling', 0.995)
            temps = [start_temp * (cooling ** i) for i in range(len(self.history))]
            # plot temperature vs step index and set limits so axis isn't wrongly capped
            self.ax_temp.plot(steps, temps)
            self.ax_temp.set_title('Temperature')
            self.ax_temp.set_xlabel('Step')
            self.ax_temp.set_ylabel('Temperature (K)')
            # set sensible limits with a small margin
            if temps:
                tmin = min(temps)
                tmax = max(temps)
                margin = (tmax - tmin) * 0.05 if tmax != tmin else tmax * 0.05
                self.ax_temp.set_ylim(tmin - margin, tmax + margin)
            self.ax_temp.set_xlim(0, max(1, len(self.history)-1))

            self.fig_graphs.tight_layout()
            self.canvas_graphs.draw()
        except Exception:
            pass

    def _kabsch_rmsd(self, A, B):
        """Compute RMSD between two coordinate sets A and B after optimal superposition (Kabsch).

        A and B should be arrays shape (N,3). Returns scalar RMSD.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        if A.size == 0 or B.size == 0:
            return None
        # ensure same length
        n = min(A.shape[0], B.shape[0])
        A = A[:n]
        B = B[:n]

        # center
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # covariance
        H = AA.T.dot(BB)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T.dot(U.T)
        # correct reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T.dot(U.T)

        A_rot = AA.dot(R)
        diff = A_rot - BB
        rmsd = np.sqrt((diff * diff).sum() / n)
        return float(rmsd)

    def plot_structs_matrix(self, optimizer_results):
        """Compute contact/distance maps for each optimizer's best structure and display side-by-side.

        Shows pairwise CA-CA distances for each optimizer as separate contact maps.
        """
        try:
            self.ax_matrix.clear()
            
            coords_list = [np.asarray(r.get('best_coords')) for r in optimizer_results]
            names = [r.get('name', f'opt{i}') for i, r in enumerate(optimizer_results)]
            m = len(coords_list)
            
            if m == 0:
                self.canvas_matrix.draw()
                return
            
            # Create subplots for each optimizer's contact map
            self.ax_matrix.clear()
            self.fig_matrix.clear()
            
            # Create grid of subplots (1 row, m columns for up to m optimizers)
            n_cols = min(m, 3)  # max 3 columns to keep readable
            n_rows = (m + n_cols - 1) // n_cols
            
            axes = []
            for idx in range(m):
                ax = self.fig_matrix.add_subplot(n_rows, n_cols, idx + 1)
                axes.append(ax)
            
            # Compute and plot contact map for each optimizer
            for idx, (coords, name) in enumerate(zip(coords_list, names)):
                if coords.size == 0:
                    continue
                    
                N = coords.shape[0]
                # Compute pairwise distance matrix (residue i vs residue j)
                dist_mat = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        dist_mat[i, j] = np.linalg.norm(coords[i] - coords[j])
                
                # Plot contact map (distance matrix)
                ax = axes[idx]
                im = ax.imshow(dist_mat, cmap='viridis_r', interpolation='nearest', vmin=0, vmax=30)
                ax.set_title(f'{name}\n(Residue Distances)', fontsize=10)
                ax.set_xlabel('Residue Index', fontsize=8)
                ax.set_ylabel('Residue Index', fontsize=8)
                ax.tick_params(labelsize=7)
                
                # Add colorbar for each subplot
                try:
                    cbar = self.fig_matrix.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Distance (Å)', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                except Exception:
                    pass
            
            self.fig_matrix.tight_layout()
            self.canvas_matrix.draw()
        except Exception as e:
            print(f"Error plotting contact maps: {e}")
            pass

    def _sequence_one_to_three(self, seq_one):
        """Convert a single-letter sequence string to a list of three-letter codes.

        Unknown residues are mapped to 'ALA' as a safe fallback.
        """
        mapping = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        seq_one = seq_one.strip().upper()
        return [mapping.get(ch, 'ALA') for ch in seq_one]

    def view_in_pymol(self):
        """Write the current best/final structure to a PDB and open it in an external PyMOL process.

        Uses `write_pdb` from `protein_builder.py` to write a reasonable PDB built from the CA trace.
        """
        # choose coordinates: prefer best_coords, else last history frame
        if getattr(self, 'best_coords', None) is not None:
            ca_coords = self.best_coords
        elif self.history:
            ca_coords = self.history[-1][0]
        else:
            self.status_var.set("No structure available to view")
            return

        try:
            # Build backbone and sidechains
            N_arr, CA_arr, C_arr, O_arr = build_backbone_from_CA(np.asarray(ca_coords))
            seq_three = self._sequence_one_to_three(self.sequence_var.get())
            CB_arr, SC_arr = pack_sidechains(seq_three, N_arr, CA_arr, C_arr)

            # write PDB to repo root with a timestamp
            out_name = f"sim_result_{int(threading.get_ident())}.pdb"
            out_path = os.path.join(os.getcwd(), out_name)
            write_pdb(out_path, N_arr, CA_arr, C_arr, O_arr, CB_arr, SC_arr)

            # launch external PyMOL
            try:
                pymol_cmd = os.environ.get('PYMOL_PATH', 'pymol')
                subprocess.Popen([pymol_cmd, out_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.status_var.set(f"Opened structure in PyMOL: {out_name}")
            except FileNotFoundError:
                self.status_var.set("PyMOL not found. Install PyMOL, add to PATH, or set PYMOL_PATH environment variable to the full executable path.")
        except Exception as e:
            self.status_var.set(f"Error preparing structure for PyMOL: {e}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ProteinFoldingGUI()
    app.run()