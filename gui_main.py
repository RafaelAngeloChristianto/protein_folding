#!/usr/bin/env python3
"""Comprehensive GUI for protein folding demo with embedded visualization"""

import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from compbio_fp.models import Protein
from compbio_fp.energy import EnergyFunction
from compbio_fp.optimizer import SimulatedAnnealer

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
        
        # Progress
        ttk.Label(control_frame, text="Progress:").pack(anchor=tk.W, pady=(20,0))
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status
        ttk.Label(control_frame, textvariable=self.status_var, wraplength=150).pack(pady=10)
        
        # Right panel - visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization")
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8))
        self.ax3d = self.fig.add_subplot(221, projection='3d')
        self.ax_energy = self.fig.add_subplot(222)
        self.ax_rmsd = self.fig.add_subplot(223)
        self.ax_temp = self.fig.add_subplot(224)
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.fig.tight_layout()
        
    def run_simulation(self):
        sequence = self.sequence_var.get().strip().upper()
        try:
            steps = int(self.steps_var.get())
            temp = float(self.temp_var.get())
        except ValueError:
            self.status_var.set("Error: Invalid numeric values")
            return
            
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
            energy_fn = EnergyFunction(protein)
            optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp, max_steps=steps)
            
            best_coords, best_e, self.history = optimizer.run()
            self.root.after(0, self._simulation_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            self.root.after(0, lambda: self.run_btn.config(state='normal'))
    
    def _simulation_complete(self):
        self.status_var.set("Simulation completed!")
        self.progress_var.set(100)
        self.run_btn.config(state='normal')
        self.animate_btn.config(state='normal')
        self.plot_results()
        
    def plot_results(self):
        if not self.history:
            return
            
        # Clear axes
        self.ax3d.clear()
        self.ax_energy.clear()
        self.ax_rmsd.clear()
        self.ax_temp.clear()
        
        # Plot final structure
        final_coords = self.history[-1][0]
        self.ax3d.plot(final_coords[:,0], final_coords[:,1], final_coords[:,2], 'b-', linewidth=2)
        self.ax3d.scatter(final_coords[:,0], final_coords[:,1], final_coords[:,2], c='red', s=50)
        self.ax3d.set_title('Final Structure')
        
        # Plot energy evolution
        energies = [h[1]['total'] for h in self.history]
        self.ax_energy.plot(energies)
        self.ax_energy.set_title('Energy vs Step')
        self.ax_energy.set_ylabel('Energy')
        
        # Plot RMSD from initial
        initial_coords = self.history[0][0]
        rmsds = []
        for coords, _ in self.history:
            diff = coords - initial_coords
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsds.append(rmsd)
        self.ax_rmsd.plot(rmsds)
        self.ax_rmsd.set_title('RMSD from Initial')
        self.ax_rmsd.set_ylabel('RMSD (Å)')
        
        # Plot temperature (mock)
        temps = [300 * (0.995 ** i) for i in range(len(self.history))]
        self.ax_temp.plot(temps)
        self.ax_temp.set_title('Temperature')
        self.ax_temp.set_ylabel('Temperature (K)')
        
        self.canvas.draw()
        
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
        energy = self.history[self.current_frame][1]['total']
        
        self.ax3d.clear()
        self.ax3d.plot(coords[:,0], coords[:,1], coords[:,2], 'b-', linewidth=2)
        self.ax3d.scatter(coords[:,0], coords[:,1], coords[:,2], c='red', s=50)
        self.ax3d.set_title(f'Step {self.current_frame}, Energy: {energy:.2f}')
        
        self.canvas.draw()
        
        self.current_frame = (self.current_frame + 1) % len(self.history)
        if self.animation_running:
            self.root.after(50, self.animate_structure)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ProteinFoldingGUI()
    app.run()