#!/usr/bin/env python3
"""Comprehensive GUI for protein folding demo with embedded visualization"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from compbio_fp.models import Protein
from compbio_fp.energy import EnergyFunction
from compbio_fp.optimizer import SimulatedAnnealer
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
        # Hydrophobicity display mode
        ttk.Label(control_frame, text="Hydrophobicity view:").pack(anchor=tk.W, pady=(10,0))
        self.hydro_mode_var = tk.StringVar(value='Continuous')
        ttk.Combobox(control_frame, textvariable=self.hydro_mode_var, values=['Continuous', 'Binary'], state='readonly', width=12).pack(pady=(0,5))
        ttk.Label(control_frame, text="Binary threshold (0-1):").pack(anchor=tk.W)
        self.hydro_thresh_var = tk.DoubleVar(value=0.5)
        ttk.Entry(control_frame, textvariable=self.hydro_thresh_var, width=6).pack(pady=(0,10))
        
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
        # (2,2) intentionally left empty
        self.canvas_graphs = FigureCanvasTkAgg(self.fig_graphs, self.graphs_tab)
        self.canvas_graphs.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        self.plot_results()
        # populate the Results tab with numeric outputs
        try:
            self.update_results_text()
        except Exception:
            pass
        
    def plot_results(self):
        if not self.history:
            return
        # Plot final structure in its own tab
        self.plot_structure(self.history[-1][0], self.history[-1][1].get('total') if isinstance(self.history[-1][1], dict) else None)

        # Plot graphs (energy, RMSD, temperature) in the Graphs tab
        self.plot_graphs()

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

        # Build text
        lines = []
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
                    # add colorbar to the structure figure
                    try:
                        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
                        mappable.set_array(hydro)
                        cb = self.fig_structure.colorbar(mappable, ax=self.ax3d, shrink=0.6, pad=0.05)
                        cb.set_label('Hydrophobicity (0..1)')
                    except Exception:
                        pass
                else:
                    thresh = float(self.hydro_thresh_var.get()) if hasattr(self, 'hydro_thresh_var') else 0.5
                    cols = ['orange' if h > thresh else 'cyan' for h in hydro]
                    self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c=cols, s=60)
                    # legend
                    from matplotlib.patches import Patch
                    legend_patches = [Patch(color='orange', label='Hydrophobic'), Patch(color='cyan', label='Hydrophilic')]
                    try:
                        self.ax3d.legend(handles=legend_patches)
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