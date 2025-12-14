import argparse
import sys
import os
import tempfile
import logging
from .models import Protein, AMINO_PROPS
from .energy import EnergyFunction
from .optimizer import SimulatedAnnealer
from .viz import animate_history
from .utils import validate_sequence, open_with_default_viewer, energy_series
from .alphafold_compare import compare_with_alphafold, get_sequence_from_uniprot_id


def demo_run(sequence="ACDEFGHIKLMNPQR", steps=1500, show_plot=True, save_animation=None, step_size=None, temp_K=300.0, include_sidechains=False, uniprot_id=None):
    if steps is None:
        # interactive prompt for steps
        try:
            s_in = input("Enter number of annealing steps (default 1200): ").strip()
        except EOFError:
            s_in = ""
        if s_in == "":
            steps = 1200
        else:
            try:
                steps = int(s_in)
                if steps <= 0:
                    print("Steps must be positive. Using default=1200")
                    steps = 1200
            except (ValueError, TypeError) as e:
                logging.warning("Invalid number entered for steps: %s. Using default=1200", e)
                steps = 1200

    protein = Protein(sequence, include_sidechains=include_sidechains)
    energy_fn = EnergyFunction(protein)
    if step_size is None:
        step_size = 0.4
    optimizer = SimulatedAnnealer(protein, energy_fn, temp_K=temp_K, cooling=0.9985, max_steps=steps, step_size=step_size)

    init_e = energy_fn.total_energy()
    print("Initial energy:", init_e['total'])
    print("Energy breakdown:", init_e)
    best_coords, best_e, history = optimizer.run()
    print("Best energy found:", best_e)
    print("RMSD to initial structure:", protein.rmsd_to(protein._init_coords()))
    
    # AlphaFold comparison
    af_result = compare_with_alphafold(protein)
    if af_result:
        print(f"\n=== AlphaFold Comparison (UniProt: {af_result['uniprot_id']}) ===")
        print(f"RMSD to AlphaFold: {af_result['rmsd']:.2f} Å")
        print(f"GDT-TS Score: {af_result['gdt_ts']:.1f}%")
        print(f"Structure Coverage: {af_result['coverage']:.1f}%")
        print(f"Lengths - Simulated: {af_result['simulated_length']}, AlphaFold: {af_result['alphafold_length']}")
    else:
        print("\n=== AlphaFold Comparison ===")
        print("No matching AlphaFold structure found in database")
    # Plot detailed energies (optional)
    try:
        import matplotlib.pyplot as plt
        import numpy as _np
        series = energy_series(history)
        steps_idx = range(len(series.get('total', [])))
        # extract named series with safe defaults (convert to lists for plotting)
        bond = list(series.get('bond', []))
        angle = list(series.get('angle', []))
        lj = list(series.get('lj', []))
        hydro = list(series.get('hydro', []))
        elec = list(series.get('elec', []))
        excl = list(series.get('excl', []))
        total = list(series.get('total', []))

        plt.figure(figsize=(10, 5))
        plt.plot(steps_idx, bond, label='Bond', lw=1)
        plt.plot(steps_idx, angle, label='Angle', lw=1)
        plt.plot(steps_idx, lj, label='LJ', lw=1)
        plt.plot(steps_idx, hydro, label='Hydrophobic', lw=1)
        plt.plot(steps_idx, elec, label='Electrostatic', lw=1)
        plt.plot(steps_idx, excl, label='Excluded', lw=1)
        plt.plot(steps_idx, total, label='Total', color='black', lw=2)

        # Detect extreme outliers in total energy that would squash other traces.
        total_arr = _np.array(total)
        if total_arr.size > 0:
            max_total = float(_np.nanmax(total_arr))
            p99 = float(_np.nanpercentile(total_arr, 99))
            median = float(_np.nanmedian(total_arr))
            # define an adaptive threshold: if the max is much larger than the 99th pct or median,
            # we'll clip the y-axis to the 99th percentile for visibility and annotate the plot.
            clip_needed = (max_total > max(p99 * 1.5, median + 1e4))
            if clip_needed:
                y_max = p99 * 1.2 if p99 > 0 else median + 1.0
                plt.ylim(bottom=min(_np.nanmin(total_arr), -1.0), top=y_max)
                plt.gca().axhline(y=y_max, color='gray', linestyle='--', lw=1)
                plt.gca().text(0.99, 0.95, 'Note: extreme values clipped for visibility',
                               horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
                               fontsize='small', color='gray')

        plt.title("Energy components vs. step")
        plt.xlabel("Step")
        plt.ylabel("Energy (arbitrary units)")
        plt.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if show_plot:
            plt.show()
        else:
            if save_animation:
                plt.savefig(save_animation + "_energy.png")
    except (ImportError, OSError, RuntimeError) as e:
        logging.warning("Could not produce energy plot: %s", e)
    except Exception as e:
        logging.exception("Unexpected error while plotting energies: %s", e)

    # If the user didn't request a save path but wants to see the animation,
    # create a temporary file to save the animation (GIF) and open it with the
    # OS default viewer after the energy plot window is closed. This gives a
    # quick way to view the folding trajectory even in headless environments.
    try:
        if save_animation:
            out_path = save_animation
        else:
            if show_plot:
                # create a temporary gif file path
                tf = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
                tf.close()
                out_path = tf.name
            else:
                out_path = None

        animate_history(history, sequence, save_as=out_path, show=show_plot)

        # If we wrote a file, attempt to open it with the OS default viewer
        if out_path is not None and os.path.exists(out_path):
            try:
                open_with_default_viewer(out_path)
            except Exception as e:
                logging.exception("Unexpected error when attempting to open animation %s", out_path)
    except (ImportError, OSError, RuntimeError) as e:
        # If animation saving/showing fails due to system errors or missing dependencies,
        # log a warning and try a best-effort call to animate_history without opening.
        logging.warning("Animation display/save failed: %s", e)
        try:
            animate_history(history, sequence, save_as=save_animation, show=show_plot)
        except Exception as e2:
            logging.exception("Fallback animate_history call failed: %s", e2)
    except Exception as e:
        logging.exception("Unexpected error in animation/display logic: %s", e)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simple protein folding demo (simulated annealing)")
    parser.add_argument("--sequence", "-s", help="Amino-acid sequence (one-letter codes). Example: ACDEFGHIKLMNPQR")
    parser.add_argument("--steps", "-n", type=int, default=None, help="Number of annealing steps (positive integer). If omitted you'll be prompted.")
    parser.add_argument("--no-plot", dest="show_plot", action="store_false", help="Do not display plots/animation (useful for headless runs)")
    parser.add_argument("--save-animation", dest="save_animation", help="Path prefix to save animation and energy plot (optional)")
    parser.add_argument("--step-size", dest="step_size", type=float, default=None, help="Torsion move step size (radians). If not given, demo default is used.")
    parser.add_argument("--temp", dest="temp", type=float, default=300.0, help="Simulation temperature in Kelvin (default 300 K)")
    parser.add_argument("--sidechains", dest="sidechains", action="store_true", help="Include simple side-chain beads (one extra bead per residue)")
    parser.add_argument("--uniprot", dest="uniprot_id", help="UniProt ID for AlphaFold comparison (auto-detected if not provided)")
    args = parser.parse_args(argv)

    seq = None
    if args.sequence:
        seq = validate_sequence(args.sequence)
        if seq is None:
            print("Invalid sequence provided. Allowed one-letter codes:", ''.join(sorted(AMINO_PROPS.keys())))
            return 2

    if seq is None:
        try:
            inp = input(f"Enter amino-acid sequence (default ACDEFGHIKLMNPQR): ").strip()
        except EOFError:
            inp = ""
        if inp == "":
            seq = "ACDEFGHIKLMNPQR"
        else:
            seq = validate_sequence(inp)
            if seq is None:
                print("Invalid sequence entered. Exiting.")
                return 3

    # allow overriding default step_size
    extra = {}
    if args.step_size is not None:
        extra['step_size'] = args.step_size
    extra['temp_K'] = args.temp
    extra['include_sidechains'] = args.sidechains
    extra['uniprot_id'] = args.uniprot_id
    demo_run(sequence=seq, steps=args.steps, show_plot=args.show_plot, save_animation=args.save_animation, **extra)


if __name__ == '__main__':
    sys.exit(main())


__all__ = ['demo_run', 'main']
