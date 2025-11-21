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

import numpy as np
logging.basicConfig(level=logging.INFO)

def run_simulation(sequence, steps, step_size, temp_K, include_sidechains, status_var, start_btn, ui_callback=None):
    try:
        status_var.set("Initializing...")
        prot = Protein(sequence, include_sidechains=include_sidechains)
        initial_coords = prot.coords.copy()
        ef = EnergyFunction(prot)
        # enable debug/gui_exit as desired; reuse defaults
        annealer = SimulatedAnnealer(prot, ef, temp_K=temp_K, max_steps=steps, step_size=step_size)
        status_var.set("Running annealing...")
        best_coords, best_e, history = annealer.run()
        status_var.set(f"Done. Best energy: {best_e:.3f}. Preparing results...")
        # Compute energy series and find best index for per-term breakdown
        energies = [ed.get('total', float('nan')) if isinstance(ed, dict) else float('nan') for _, ed in history]
        # locate best index (closest to best_e)
        best_idx = None
        if len(energies) > 0:
            diffs = [abs(val - best_e) if np.isfinite(val) else float('inf') for val in energies]
            best_idx = int(np.argmin(diffs))
        # prepare summary dict
        initial_energy_dict = history[0][1] if len(history) > 0 else {}
        best_energy_dict = history[best_idx][1] if (best_idx is not None and len(history) > best_idx) else {}
        # RMSD between initial and best coords (CA trace assumed)
        def compute_rmsd(a, b):
            try:
                a = np.asarray(a)
                b = np.asarray(b)
                if a.shape != b.shape:
                    # try to align by truncation
                    n = min(a.shape[0], b.shape[0])
                    a = a[:n]
                    b = b[:n]
                return float(np.sqrt(np.mean(np.sum((a - b)**2, axis=1))))
            except Exception:
                return float('nan')
        rmsd = compute_rmsd(initial_coords, best_coords)
        results = {
            "energies": energies,
            "initial_energy": initial_energy_dict,
            "best_energy": best_energy_dict,
            "best_e": best_e,
            "rmsd": rmsd,
            "history": history
        }
        # call UI callback (must schedule UI update on main thread)
        if ui_callback is not None:
            try:
                ui_callback(results)
            except Exception:
                # if callback fails, log but continue
                logging.exception("ui_callback failed")
        # optional: save animation as before (kept out of main GUI thread)
        try:
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
            tf.close()
            animate_history(history, sequence, save_as=tf.name, show=False)
            open_with_default_viewer(tf.name)
        except Exception as e:
            logging.exception("Failed to save/show animation: %s", e)
            # don't override UI callback results
    except Exception as e:
        logging.exception("Simulation failed: %s", e)
        status_var.set("Simulation failed. See logs.")
    finally:
        start_btn.config(state=tk.NORMAL)

def on_start(sequence_var, steps_var, step_size_var, temp_var, side_var, status_var, start_btn, ui_callback, root):
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
        args=(seq, steps, step_size, temp_K, bool(side_var.get()), status_var, start_btn, ui_callback),
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

    # Result area (embedded plot + text)
    result_frame = ttk.Frame(root, padding=6, relief="groove")
    result_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6,10))
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)

    # left: matplotlib canvas (energy plot), right: text summary
    canvas_container = ttk.Frame(result_frame)
    canvas_container.grid(row=0, column=0, sticky="nsew")
    text_container = ttk.Frame(result_frame)
    text_container.grid(row=0, column=1, sticky="nsew", padx=(10,0))

    # placeholders for plot and text widgets
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = Figure(figsize=(5,3))
        ax = fig.add_subplot(111)
        ax.set_title("Total energy vs step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Total energy")
        canvas = FigureCanvasTkAgg(fig, master=canvas_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
    except Exception:
        fig = None
        ax = None
        canvas = None
        canvas_widget = ttk.Label(canvas_container, text="matplotlib unavailable", foreground="red")
        canvas_widget.pack(fill="both", expand=True)

    # text summary area
    summary_text = tk.Text(text_container, width=40, height=12, wrap="word")
    summary_text.pack(fill="both", expand=True)
    summary_text.insert("end", "Results will appear here...\n")
    summary_text.config(state="disabled")

    # UI callback to update plot and text; scheduled via root.after for thread-safety
    def schedule_ui_update(results):
        def _update():
            try:
                # update summary text
                summary_text.config(state="normal")
                summary_text.delete("1.0", "end")
                summary_text.insert("end", f"Initial energy: {results.get('initial_energy') and results['initial_energy'].get('total')}\n")
                summary_text.insert("end", f"Best energy found: {results.get('best_e')}\n")
                summary_text.insert("end", f"RMSD to initial structure: {results.get('rmsd')}\n\n")
                summary_text.insert("end", "Initial energy breakdown:\n")
                ie = results.get('initial_energy') or {}
                for k,v in ie.items():
                    summary_text.insert("end", f"  {k}: {v}\n")
                summary_text.insert("end", "\nBest energy breakdown:\n")
                be = results.get('best_energy') or {}
                for k,v in be.items():
                    summary_text.insert("end", f"  {k}: {v}\n")
                summary_text.config(state="disabled")
                # update plot
                if fig is not None and ax is not None and canvas is not None:
                    energies = results.get('energies', [])
                    ax.clear()
                    ax.plot(energies, '-', lw=1)
                    ax.set_title("Total energy vs step")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Total energy")
                    # mark best point
                    best_e = results.get('best_e', None)
                    if best_e is not None and len(energies) > 0:
                        # find index of best
                        idx = int(np.argmin([abs(e - best_e) if np.isfinite(e) else np.nan for e in energies]))
                        ax.plot(idx, energies[idx], 'ro', label="best")
                        ax.legend()
                    canvas.draw_idle()
            except Exception:
                logging.exception("Failed updating UI with results")
        root.after(0, _update)

    # bind start button with ui_callback
    start_btn = ttk.Button(frm, text="Start", command=lambda: on_start(sequence_var, steps_var, step_size_var, temp_var, side_var, status_var, start_btn, schedule_ui_update, root))
    start_btn.grid(row=3, column=0, columnspan=4, pady=(10,0))

    ttk.Label(frm, textvariable=status_var, foreground="blue").grid(row=4, column=0, columnspan=4, sticky="w", pady=(10,0))

    root.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()