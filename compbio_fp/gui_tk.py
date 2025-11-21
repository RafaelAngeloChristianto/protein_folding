import threading
import tempfile
import logging
import tkinter as tk
from tkinter import ttk, messagebox

from .models import Protein
from .energy import EnergyFunction
from .optimizer import SimulatedAnnealer
from .viz import animate_history
from .utils import open_with_default_viewer, validate_sequence

logging.basicConfig(level=logging.INFO)

def run_simulation(sequence, steps, step_size, temp_K, include_sidechains, status_var, start_btn):
    try:
        status_var.set("Initializing...")
        prot = Protein(sequence, include_sidechains=include_sidechains)
        ef = EnergyFunction(prot)
        annealer = SimulatedAnnealer(prot, ef, temp_K=temp_K, max_steps=steps, step_size=step_size)
        status_var.set("Running annealing...")
        best_coords, best_e, history = annealer.run()
        status_var.set(f"Done. Best energy: {best_e:.3f}. Saving animation...")
        # save animation to temp file
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        tf.close()
        try:
            animate_history(history, sequence, save_as=tf.name, show=False)
            status_var.set(f"Saved animation: {tf.name}")
            open_with_default_viewer(tf.name)
        except Exception as e:
            logging.exception("Failed to save/show animation: %s", e)
            status_var.set("Finished but failed to save animation.")
    except Exception as e:
        logging.exception("Simulation failed: %s", e)
        status_var.set("Simulation failed. See logs.")
    finally:
        start_btn.config(state=tk.NORMAL)

def on_start(sequence_var, steps_var, step_size_var, temp_var, side_var, status_var, start_btn):
    seq = sequence_var.get().strip().upper()
    if not validate_sequence(seq):
        messagebox.showerror("Invalid sequence", "Sequence contains invalid amino-acid codes.")
        return
    try:
        steps = int(steps_var.get())
    except ValueError:
        messagebox.showerror("Invalid steps", "Steps must be an integer.")
        return
    try:
        step_size = float(step_size_var.get())
        temp_K = float(temp_var.get())
    except ValueError:
        messagebox.showerror("Invalid numeric value", "Step size and temperature must be numbers.")
        return

    start_btn.config(state=tk.DISABLED)
    status_var.set("Queued...")
    thread = threading.Thread(
        target=run_simulation,
        args=(seq, steps, step_size, temp_K, bool(side_var.get()), status_var, start_btn),
        daemon=True
    )
    thread.start()

def main():
    root = tk.Tk()
    root.title("Simple Protein Folding Demo")

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")

    sequence_var = tk.StringVar(value="ACDEFGHIKLMNPQR")
    steps_var = tk.StringVar(value="1500")
    step_size_var = tk.StringVar(value="0.4")
    temp_var = tk.StringVar(value="300.0")
    side_var = tk.IntVar(value=0)
    status_var = tk.StringVar(value="Idle")

    ttk.Label(frm, text="Sequence:").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=sequence_var, width=30).grid(row=0, column=1, columnspan=3, sticky="we")

    ttk.Label(frm, text="Steps:").grid(row=1, column=0, sticky="w")
    ttk.Entry(frm, textvariable=steps_var, width=10).grid(row=1, column=1, sticky="w")

    ttk.Label(frm, text="Step size:").grid(row=1, column=2, sticky="w")
    ttk.Entry(frm, textvariable=step_size_var, width=10).grid(row=1, column=3, sticky="w")

    ttk.Label(frm, text="Temp (K):").grid(row=2, column=0, sticky="w")
    ttk.Entry(frm, textvariable=temp_var, width=10).grid(row=2, column=1, sticky="w")

    ttk.Checkbutton(frm, text="Include sidechains", variable=side_var).grid(row=2, column=2, columnspan=2, sticky="w")

    start_btn = ttk.Button(frm, text="Start", command=lambda: on_start(sequence_var, steps_var, step_size_var, temp_var, side_var, status_var, start_btn))
    start_btn.grid(row=3, column=0, columnspan=4, pady=(10,0))

    ttk.Label(frm, textvariable=status_var, foreground="blue").grid(row=4, column=0, columnspan=4, sticky="w", pady=(10,0))

    root.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()