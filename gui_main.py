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
from compbio_fp.optimizer import SimulatedAnnealer, ReplicaExchange, MultiScaleOptimizer, AdaptiveOptimizer
from compbio_fp.protein_builder import build_backbone_from_CA, pack_sidechains, write_pdb
from compbio_fp.fasta_db import load_database_sequences
from compbio_fp.alphafold_compare import compare_with_alphafold

class ProteinFoldingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Protein Folding Simulation")
        self.root.geometry("1200x800")
        
        # Variables
        self.sequence_var = tk.StringVar(value="ACDEFGHIKLMNPQR")
        self.steps_var = tk.StringVar(value="10000")
        self.temp_var = tk.StringVar(value="2000")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar()
        self.db_sequence_var = tk.StringVar()
        
        # Load database sequences
        self.db_sequences = load_database_sequences()
        
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
        
        # Database sequence selection
        ttk.Label(control_frame, text="Database Sequences:").pack(anchor=tk.W)
        db_options = ["Custom"] + [f"{header[:50]}..." if len(header) > 50 else header for header in self.db_sequences.keys()]
        self.db_combo = ttk.Combobox(control_frame, textvariable=self.db_sequence_var, values=db_options, state='readonly', width=25)
        self.db_combo.set("Custom")
        self.db_combo.pack(pady=(0,5))
        self.db_combo.bind('<<ComboboxSelected>>', self.on_db_sequence_selected)
        
        # Manual sequence entry
        ttk.Label(control_frame, text="Sequence:").pack(anchor=tk.W)
        self.sequence_entry = ttk.Entry(control_frame, textvariable=self.sequence_var, width=20)
        self.sequence_entry.pack(pady=(0,10))
        
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
        self.optimizer_var = tk.StringVar(value='Adaptive')
        self.optimizer_combo = ttk.Combobox(control_frame, textvariable=self.optimizer_var,
                            values=['Adaptive', 'Multi-Scale', 'Simulated Annealing', 'Parallel Tempering'], state='readonly', width=18)
        self.optimizer_combo.pack(pady=(0,5))
        self.optimizer_combo.bind('<<ComboboxSelected>>', lambda e: self._on_optimizer_change())

        # Parallel Tempering params (hidden unless PT selected)
        self.replicas_var = tk.IntVar(value=6)
        self.exchange_interval_var = tk.IntVar(value=8)
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

        # Additional small structures figure to compare optimizer 3D results (2x2 grid for 4 optimizers)
        self.fig_structs = Figure(figsize=(10, 8))
        self.ax_struct1 = self.fig_structs.add_subplot(2, 2, 1, projection='3d')
        self.ax_struct2 = self.fig_structs.add_subplot(2, 2, 2, projection='3d')
        self.ax_struct3 = self.fig_structs.add_subplot(2, 2, 3, projection='3d')
        self.ax_struct4 = self.fig_structs.add_subplot(2, 2, 4, projection='3d')
        # Place the comparison 3D canvases in the Structure tab so
        # "Run All Optimizers" shows all four optimizer results there.
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

        # Results tab: scrolled text for numeric output with improved aesthetics
        self.results_text = scrolledtext.ScrolledText(
            self.results_tab, 
            wrap=tk.WORD, 
            height=20,
            font=('Consolas', 10),  # Monospace font for better alignment
            bg='#f5f5f5',  # Light gray background
            fg='#2c3e50',  # Dark text
            padx=10,
            pady=10,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure text tags for styling
        self.results_text.tag_config('header', font=('Consolas', 12, 'bold'), foreground='#2c3e50')
        self.results_text.tag_config('section', font=('Consolas', 10, 'bold'), foreground='#3498db')
        self.results_text.tag_config('winner', font=('Consolas', 10, 'bold'), foreground='#27ae60')
        self.results_text.tag_config('metric', foreground='#34495e')
        self.results_text.tag_config('value', foreground='#e74c3c')
        
        # AlphaFold comparison button
        af_frame = ttk.Frame(self.results_tab)
        af_frame.pack(fill=tk.X, padx=5, pady=5)
        self.alphafold_btn = ttk.Button(af_frame, text="Compare with AlphaFold", command=self.compare_alphafold)
        self.alphafold_btn.pack(side=tk.LEFT)

        self.fig_structure.tight_layout()
        self.fig_graphs.tight_layout()
        # Ensure optimizer UI initial state
        self._on_optimizer_change()
    
    def on_db_sequence_selected(self, event=None):
        """Handle database sequence selection"""
        selected = self.db_sequence_var.get()
        if selected == "Custom":
            self.sequence_entry.config(state='normal')
            return
        
        # Find the full header that matches the truncated display
        for header, sequence in self.db_sequences.items():
            display_header = f"{header[:50]}..." if len(header) > 50 else header
            if display_header == selected:
                self.sequence_var.set(sequence)
                self.sequence_entry.config(state='readonly')
                break

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
                ('Adaptive', 'ADAPTIVE'),
                ('Multi-Scale', 'MULTISCALE'),
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
                    best_coords, best_e, history = optimizer.run()
                elif key == 'MULTISCALE':
                    optimizer = MultiScaleOptimizer(protein, EnergyFunction, debug=False)
                    best_coords, best_e, history = optimizer.run(steps)
                elif key == 'ADAPTIVE':
                    optimizer = AdaptiveOptimizer(protein, EnergyFunction, debug=False)
                    best_coords, best_e, history = optimizer.run(steps)
                else:  # SA
                    optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp, max_steps=steps, adaptive_cooling=True)
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
                self.ax_struct1.clear(); self.ax_struct2.clear(); self.ax_struct3.clear(); self.ax_struct4.clear(); self.canvas_structs.draw()
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
            opt_choice = self.optimizer_var.get() if hasattr(self, 'optimizer_var') else 'Adaptive'
            if opt_choice == 'Parallel Tempering':
                nrep = int(self.replicas_var.get()) if hasattr(self, 'replicas_var') else 4
                exch = int(self.exchange_interval_var.get()) if hasattr(self, 'exchange_interval_var') else 10
                optimizer = ReplicaExchange(protein, energy_fn, num_replicas=nrep, exchange_interval=exch, base_temp_K=temp, max_steps=steps)
            elif opt_choice == 'Multi-Scale':
                optimizer = MultiScaleOptimizer(protein, EnergyFunction, debug=True)
            elif opt_choice == 'Adaptive':
                optimizer = AdaptiveOptimizer(protein, EnergyFunction, debug=True)
            else:  # Simulated Annealing
                optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp, max_steps=steps, adaptive_cooling=True)
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
            
            if opt_choice in ['Multi-Scale', 'Adaptive']:
                best_coords, best_e, self.history = optimizer.run(steps)
            else:
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
            # Auto-run AlphaFold comparison
            self.compare_alphafold()
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
        """Draw best-structure 3D views for all optimizers in `optimizer_results` in a 2x2 grid.

        Expects `optimizer_results` to be a list of dicts with keys 'name' and 'best_coords'.
        """
        try:
            # Prepare all axes
            self.ax_struct1.clear()
            self.ax_struct2.clear()
            self.ax_struct3.clear()
            self.ax_struct4.clear()
            
            axes = [self.ax_struct1, self.ax_struct2, self.ax_struct3, self.ax_struct4]

            # Draw all available optimizers (up to 4)
            for idx, result in enumerate(optimizer_results[:4]):
                if idx >= 4:
                    break
                    
                ax = axes[idx]
                coords = np.asarray(result.get('best_coords'))
                hydro = result.get('hydro')
                
                if coords.size == 0:
                    continue
                    
                # Draw backbone
                ax.plot(coords[:,0], coords[:,1], coords[:,2], color='gray', linewidth=1)
                
                # Color by hydrophobicity if available
                if hydro is None:
                    ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=f'C{idx}', s=30)
                else:
                    try:
                        import matplotlib.cm as cm
                        import matplotlib.colors as mcolors
                        mode = self.hydro_mode_var.get() if hasattr(self, 'hydro_mode_var') else 'Continuous'
                        if mode == 'Continuous':
                            cmap = cm.get_cmap('coolwarm')
                            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=hydro, cmap=cmap, norm=norm, s=30)
                        else:
                            thresh = float(getattr(self, 'hydro_thresh_var', 0.5)) if hasattr(self, 'hydro_thresh_var') else 0.5
                            cols = ['orange' if h > thresh else 'cyan' for h in hydro]
                            ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=cols, s=30)
                    except Exception:
                        ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=f'C{idx}', s=30)
                
                ax.set_title(result.get('name'), fontsize=10)
                ax.set_xlabel('X', fontsize=8)
                ax.set_ylabel('Y', fontsize=8)
                ax.set_zlabel('Z', fontsize=8)
                ax.tick_params(labelsize=7)
            
            self.fig_structs.tight_layout()
            self.canvas_structs.draw()
        except Exception as e:
            print(f"Error in plot_structs_comparison: {e}")
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
        """Populate the Results tab with numeric summaries using styled text."""
        self.results_text.delete('1.0', tk.END)
        
        if not self.history:
            self.results_text.insert(tk.END, "No results available.\n")
            return

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
            dists = np.sqrt(((c[None, :, :] - c[:, None, :]) ** 2).sum(axis=2))
            iu = np.triu_indices(N, k=1)
            pairs = dists[iu]
            contacts = np.sum(pairs < cutoff)
            total_pairs = len(pairs)
            return float(contacts) / total_pairs if total_pairs > 0 else 0.0

        # Header
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, " PROTEIN FOLDING SIMULATION RESULTS\n", 'header')
        self.results_text.insert(tk.END, " " + "="*68 + "\n\n")
        
        # If multiple optimizers were run, show comparison
        if getattr(self, 'optimizer_results', None):
            self.results_text.insert(tk.END, " OPTIMIZER COMPARISON\n", 'section')
            self.results_text.insert(tk.END, " " + "-"*68 + "\n\n")
            
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
                    pass
                acc = acceptance_rate_from_history(hist)
                evals_sec = (len(hist) / t) if (t and t > 0) else None
                contact_frac = contact_fraction(coords) if coords is not None else None
                
                self.results_text.insert(tk.END, f"  {name}\n", 'metric')
                self.results_text.insert(tk.END, f"    Time:              {t:.2f} s\n")
                self.results_text.insert(tk.END, f"    Best Energy:       {be:.2f}\n" if be is not None else "    Best Energy:       N/A\n")
                self.results_text.insert(tk.END, f"    Energy/Residue:    {energy_per_res:.2f}\n" if energy_per_res is not None else "    Energy/Residue:    N/A\n")
                self.results_text.insert(tk.END, f"    Radius of Gyr.:    {rg:.2f} A\n" if rg is not None else "    Radius of Gyr.:    N/A\n")
                self.results_text.insert(tk.END, f"    RMSD (initial):    {rmsd_val:.2f} A\n" if rmsd_val is not None else "    RMSD (initial):    N/A\n")
                self.results_text.insert(tk.END, f"    Acceptance Rate:   {acc:.1%}\n" if acc is not None else "    Acceptance Rate:   N/A\n")
                self.results_text.insert(tk.END, f"    Evaluations/sec:   {evals_sec:.0f}\n" if evals_sec is not None else "    Evaluations/sec:   N/A\n")
                self.results_text.insert(tk.END, f"    Contact Fraction:  {contact_frac:.2%}\n\n" if contact_frac is not None else "    Contact Fraction:  N/A\n\n")
            
            # Show winner
            try:
                winner = min(self.optimizer_results, key=lambda r: r.get('best_energy') if r.get('best_energy') is not None else float('inf'))
                self.results_text.insert(tk.END, f"  WINNER: {winner.get('name')} (Energy: {winner.get('best_energy'):.2f})\n\n", 'winner')
            except Exception:
                pass
        
        # Simulation parameters
        self.results_text.insert(tk.END, " SIMULATION PARAMETERS\n", 'section')
        self.results_text.insert(tk.END, " " + "-"*68 + "\n\n")
        
        if getattr(self, 'start_temp', None) is not None:
            self.results_text.insert(tk.END, f"  Starting Temperature (K):        {self.start_temp}\n")
        if getattr(self, 'start_T_energy', None) is not None:
            self.results_text.insert(tk.END, f"  Temperature (k_BT energy units): {self.start_T_energy:.4f}\n")
        self.results_text.insert(tk.END, "\n")

        # Energy metrics
        initial_energy = self.history[0][1].get('total') if isinstance(self.history[0][1], dict) else None
        energy_breakdown = self.history[0][1] if isinstance(self.history[0][1], dict) else {}
        best_energy = getattr(self, 'best_energy', None)
        if best_energy is None:
            energies = [h[1].get('total') for h in self.history if isinstance(h[1], dict)]
            best_energy = min(energies) if energies else None

        self.results_text.insert(tk.END, " ENERGY METRICS\n", 'section')
        self.results_text.insert(tk.END, " " + "-"*68 + "\n\n")
        
        if initial_energy is not None:
            self.results_text.insert(tk.END, f"  Initial Energy:    {initial_energy:.2f} kcal/mol\n")
        else:
            self.results_text.insert(tk.END, "  Initial Energy:    N/A\n")

        if best_energy is not None:
            self.results_text.insert(tk.END, f"  Best Energy:       ", 'metric')
            self.results_text.insert(tk.END, f"{best_energy:.2f} kcal/mol\n", 'value')
            if initial_energy is not None:
                improvement = initial_energy - best_energy
                self.results_text.insert(tk.END, f"  Energy Reduction:  {improvement:.2f} kcal/mol\n")
        else:
            self.results_text.insert(tk.END, "  Best Energy:       N/A\n")
        
        self.results_text.insert(tk.END, "\n  Energy Components:\n")
        if energy_breakdown:
            # Filter out inactive/misleading terms
            excluded_terms = {'hydro', 'ss', 'hbond'}
            for key, value in energy_breakdown.items():
                if key != 'total' and key not in excluded_terms:
                    self.results_text.insert(tk.END, f"    {key:12s}: {value:8.2f}\n")
        self.results_text.insert(tk.END, "\n")

        # Structural metrics
        self.results_text.insert(tk.END, " STRUCTURAL METRICS\n", 'section')
        self.results_text.insert(tk.END, " " + "-"*68 + "\n\n")
        
        try:
            initial_coords = self.history[0][0]
            target_coords = getattr(self, 'best_coords', self.history[-1][0])
            diff = target_coords - initial_coords
            rmsd_val = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            self.results_text.insert(tk.END, f"  RMSD from Initial:  {rmsd_val:.2f} A\n")
        except Exception:
            self.results_text.insert(tk.END, "  RMSD from Initial:  N/A\n")
        
        self.results_text.insert(tk.END, "\n" + " " + "="*68 + "\n")
        
    def compare_alphafold(self):
        """Compare current structure with AlphaFold"""
        if not hasattr(self, 'best_coords') or self.best_coords is None:
            self.results_text.insert(tk.END, "\nNo structure available for comparison\n")
            return
            
        sequence = self.sequence_var.get().strip().upper()
        
        # Create temporary protein object for comparison
        from compbio_fp.models import Protein
        protein = Protein(sequence)
        protein.coords = self.best_coords
        
        # Compare with AlphaFold
        af_result = compare_with_alphafold(protein)
        
        if af_result:
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, "─" * 70 + "\n")
            self.results_text.insert(tk.END, "  ALPHAFOLD VALIDATION\n")
            self.results_text.insert(tk.END, "─" * 70 + "\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, f"  Reference Structure\n")
            self.results_text.insert(tk.END, f"  ├─ UniProt ID:          {af_result['uniprot_id']}\n")
            self.results_text.insert(tk.END, f"  ├─ AlphaFold Length:    {af_result.get('alphafold_length', 'N/A')} residues\n")
            self.results_text.insert(tk.END, f"  └─ Simulated Length:    {af_result.get('simulated_length', 'N/A')} residues\n")
            self.results_text.insert(tk.END, "\n")
            
            self.results_text.insert(tk.END, f"  Accuracy Metrics\n")
            self.results_text.insert(tk.END, f"  ├─ RMSD:                {af_result['rmsd']:.2f} Å\n")
            self.results_text.insert(tk.END, f"  ├─ GDT-TS Score:        {af_result['gdt_ts']:.1f}%\n")
            
            # Display new enhanced metrics if available
            if 'tm_score' in af_result:
                self.results_text.insert(tk.END, f"  ├─ TM-Score:            {af_result['tm_score']:.3f}\n")
            if 'local_accuracy' in af_result:
                self.results_text.insert(tk.END, f"  ├─ Local Accuracy:      {af_result['local_accuracy']:.1%}\n")
            
            self.results_text.insert(tk.END, f"  └─ Coverage:            {af_result['coverage']:.1%}\n")
            self.results_text.insert(tk.END, "\n")
            
            # Enhanced accuracy assessment using multiple metrics
            rmsd = af_result['rmsd']
            gdt_ts = af_result['gdt_ts']
            tm_score = af_result.get('tm_score', 0)
            
            if rmsd < 3.0 and gdt_ts > 80 and tm_score > 0.7:
                accuracy = "★★★ Excellent (Near-native)"
                symbol = "✓"
            elif rmsd < 5.0 and gdt_ts > 60 and tm_score > 0.5:
                accuracy = "★★ Very Good"
                symbol = "✓"
            elif rmsd < 8.0 and gdt_ts > 40 and tm_score > 0.3:
                accuracy = "★ Good"
                symbol = "○"
            elif rmsd < 12.0 and gdt_ts > 25:
                accuracy = "Fair"
                symbol = "○"
            else:
                accuracy = "Poor"
                symbol = "✗"
            
            self.results_text.insert(tk.END, f"  {symbol} Quality Assessment: {accuracy}\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, "=" * 70 + "\n")
        else:
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, "─" * 70 + "\n")
            self.results_text.insert(tk.END, "  ALPHAFOLD VALIDATION\n")
            self.results_text.insert(tk.END, "─" * 70 + "\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, "  ✗ No matching AlphaFold structure found\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.insert(tk.END, "=" * 70 + "\n")
        
        self.results_text.see(tk.END)
        
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

        Shows pairwise CA-CA distances for each optimizer as separate contact maps in a 2x2 grid.
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
            
            # Create 2x2 grid of subplots for up to 4 optimizers
            n_cols = 2
            n_rows = 2
            
            axes = []
            for idx in range(min(m, 4)):  # max 4 optimizers in 2x2 grid
                ax = self.fig_matrix.add_subplot(n_rows, n_cols, idx + 1)
                axes.append(ax)
            
            # Compute and plot contact map for each optimizer (up to 4 in 2x2 grid)
            for idx, (coords, name) in enumerate(zip(coords_list[:4], names[:4])):
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
                ax.set_title(f'{name}\n(Residue Distances)', fontsize=9)
                ax.set_xlabel('Residue Index', fontsize=7)
                ax.set_ylabel('Residue Index', fontsize=7)
                ax.tick_params(labelsize=6)
                
                # Add colorbar for each subplot
                try:
                    cbar = self.fig_matrix.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Distance (Å)', fontsize=7)
                    cbar.ax.tick_params(labelsize=6)
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