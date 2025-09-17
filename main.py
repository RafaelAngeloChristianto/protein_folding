# A protein is simplified into a string of letters, only H (hydrophobic) or P (polar).
# Instead of real 3D continuous folding, we place the chain on a 2D grid (like graph paper).
# The program tries to fold the chain (move it around on the grid) to find a structure with the lowest possible energy.

import random
import math
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# HP Model Utilities (3D)
# -----------------------------
directions = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1)
]

def add_coords(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def init_chain(seq):
    return [(i, 0, 0) for i in range(len(seq))]

def energy(seq, coords):
    e = 0
    occupied = {c: i for i, c in enumerate(coords)}
    for i, aa in enumerate(seq):
        if aa == "H":
            for d in directions:
                neigh = add_coords(coords[i], d)
                if neigh in occupied:
                    j = occupied[neigh]
                    if abs(i - j) > 1 and seq[j] == "H":
                        e -= 1
    return e // 2

# Exact 90° rotations
def rotate_point(point, pivot, axis, angle):
    px, py, pz = pivot
    x, y, z = point
    dx, dy, dz = x - px, y - py, z - pz
    angle = angle % 360

    if axis == 'x':
        if angle == 90:  dx, dy, dz = dx, -dz, dy
        elif angle == 180: dx, dy, dz = dx, -dy, -dz
        elif angle == 270: dx, dy, dz = dx, dz, -dy
    elif axis == 'y':
        if angle == 90:  dx, dy, dz = dz, dy, -dx
        elif angle == 180: dx, dy, dz = -dx, dy, -dz
        elif angle == 270: dx, dy, dz = -dz, dy, dx
    elif axis == 'z':
        if angle == 90:  dx, dy, dz = -dy, dx, dz
        elif angle == 180: dx, dy, dz = -dx, -dy, dz
        elif angle == 270: dx, dy, dz = dy, -dx, dz

    return (px + dx, py + dy, pz + dz)

def neighbor(coords):
    n = len(coords)
    if n < 3:
        return coords[:]
    pivot = random.randint(1, n - 2)
    before = coords[:pivot]
    after = coords[pivot:]
    axis = random.choice(['x', 'y', 'z'])
    angle = random.choice([90, 180, 270])
    new_after = [rotate_point(p, coords[pivot], axis, angle) for p in after]
    new_coords = before + new_after
    if len(set(new_coords)) < len(new_coords):
        return coords[:]
    return new_coords

# -----------------------------
# Simulated Annealing
# -----------------------------
def simulated_annealing_with_history(seq, T0=5.0, cooling=0.995, max_steps=500):
    coords = init_chain(seq)
    e = energy(seq, coords)
    best_coords, best_e = coords[:], e
    history = [(coords[:], e)]
    T = T0
    for step in range(max_steps):
        new_coords = neighbor(coords)
        new_e = energy(seq, new_coords)
        dE = new_e - e
        if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-12)):
            coords, e = new_coords, new_e
            if e < best_e:
                best_coords, best_e = coords[:], e
        history.append((coords[:], e))
        T *= cooling
    return best_coords, best_e, history

# -----------------------------
# Tkinter + Matplotlib GUI
# -----------------------------
class FoldingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Protein Folding Simulation (HP Model 3D)")

        # Input UI
        self.label = tk.Label(root, text="Enter HP Sequence:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, width=40)
        self.entry.pack(pady=5)
        self.entry.insert(0, "HHPHHPPHPH")  # default

        self.run_button = tk.Button(root, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(pady=5)

        # Figure
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.ani = None

    def set_axes_equal(self):
        x_limits, y_limits, z_limits = self.ax.get_xlim3d(), self.ax.get_ylim3d(), self.ax.get_zlim3d()
        x_range, y_range, z_range = abs(x_limits[1]-x_limits[0]), abs(y_limits[1]-y_limits[0]), abs(z_limits[1]-z_limits[0])
        x_middle, y_middle, z_middle = sum(x_limits)/2, sum(y_limits)/2, sum(z_limits)/2
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        self.ax.set_xlim3d(x_middle - plot_radius, x_middle + plot_radius)
        self.ax.set_ylim3d(y_middle - plot_radius, y_middle + plot_radius)
        self.ax.set_zlim3d(z_middle - plot_radius, z_middle + plot_radius)

    def run_simulation(self):
        seq = self.entry.get().strip().upper()
        if not seq:
            messagebox.showwarning("Warning", "Enter a sequence!")
            return
        if any(ch not in "HP" for ch in seq):
            messagebox.showerror("Error", "Sequence must only contain H and P!")
            return

        best_coords, best_e, history = simulated_annealing_with_history(seq)

        print("Best Energy:", best_e)

        self.ax.clear()
        self.ax.set_title("3D Protein Folding Simulation")

        line, = self.ax.plot([], [], [], 'k-', lw=2)
        h_scatter = self.ax.scatter([], [], [], c='red', s=60, label='H')
        p_scatter = self.ax.scatter([], [], [], c='blue', s=60, label='P')
        text = self.ax.text2D(0.02, 0.95, '', transform=self.ax.transAxes)
        self.ax.legend()

        all_coords = [c for (c, _) in history]
        xs = [x for coords in all_coords for (x, _, _) in coords]
        ys = [y for coords in all_coords for (_, y, _) in coords]
        zs = [z for coords in all_coords for (_, _, z) in coords]
        margin = 2
        self.ax.set_xlim(min(xs)-margin, max(xs)+margin)
        self.ax.set_ylim(min(ys)-margin, max(ys)+margin)
        self.ax.set_zlim(min(zs)-margin, max(zs)+margin)
        self.set_axes_equal()

        def update(frame):
            coords, e = history[frame]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            zs = [c[2] for c in coords]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            hx = [coords[i][0] for i in range(len(seq)) if seq[i] == "H"]
            hy = [coords[i][1] for i in range(len(seq)) if seq[i] == "H"]
            hz = [coords[i][2] for i in range(len(seq)) if seq[i] == "H"]
            px = [coords[i][0] for i in range(len(seq)) if seq[i] == "P"]
            py = [coords[i][1] for i in range(len(seq)) if seq[i] == "P"]
            pz = [coords[i][2] for i in range(len(seq)) if seq[i] == "P"]

            h_scatter._offsets3d = (hx, hy, hz)
            p_scatter._offsets3d = (px, py, pz)
            text.set_text(f"Step: {frame}, Energy: {e}")
            return line, h_scatter, p_scatter, text

        if self.ani:
            self.ani.event_source.stop()
            self.ani = None


        self.ani = FuncAnimation(self.fig, update, frames=len(history), interval=80, blit=False, repeat=False)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FoldingApp(root)
    root.protocol("WM_DELETE_WINDOW", root.quit)  # force quit loop
    root.mainloop()


