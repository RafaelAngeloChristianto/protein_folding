import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_history(history, sequence, save_as=None, show=True):
    coords_list = [h[0] for h in history]
    # use energy_series helper to extract components
    from .utils import energy_series
    series = energy_series(history)
    energies = list(series.get('total', []))
    Nframes = len(coords_list)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    line, = ax.plot([], [], [], lw=2)
    scatter = ax.scatter([], [], [], s=40)
    title = ax.set_title("")

    all_coords = __import__('numpy').concatenate(coords_list, axis=0)
    xmin, ymin, zmin = __import__('numpy').min(all_coords, axis=0) - 2.0
    xmax, ymax, zmax = __import__('numpy').max(all_coords, axis=0) + 2.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    def update(frame):
        c = coords_list[frame]
        x, y, z = c[:,0], c[:,1], c[:,2]
        line.set_data(x, y)
        line.set_3d_properties(z)
        scatter._offsets3d = (x, y, z)
        title.set_text(f"Step {frame}/{Nframes-1}  Energy={energies[frame]:.2f}")
        return line, scatter, title

    ani = None

    # Determine whether the current backend is interactive (can open windows)
    backend = matplotlib.get_backend().lower()
    interactive_backends = ('tkagg', 'qt5agg', 'qtagg', 'wxagg', 'gtk3agg')
    is_interactive = backend in interactive_backends

    needs_animation = bool(save_as) or bool(show)
    # For interactive display prefer a manual loop using plt.pause to avoid
    # FuncAnimation event loop issues; reserve FuncAnimation for file saving.
    if needs_animation and not (show and is_interactive):
        ani = FuncAnimation(fig, update, frames=list(range(Nframes)) + [Nframes-1]*400, interval=30, blit=False)
        # The extra frames at the end allow a pause on the final frame to view it.

    def _save_with_fallback(animation_obj, base_name, final_coords=None):
        """Try saving animation with multiple writers in order and fall back to a PNG snapshot.

        Tries: ffmpeg (mp4) -> Pillow (gif) -> ImageMagick (gif).
        Any writer-specific exceptions are kept local and printed only as a summary.
        If all writers fail and final_coords is provided, a PNG snapshot of final_coords
        will be written as base_name + '_final.png'. Returns True if something was saved.
        """
        if animation_obj is None:
            return False

        errors = []

        # 1) ffmpeg -> mp4
        try:
            Writer = matplotlib.animation.FFMpegWriter
            writer = Writer(fps=30)
            out_path = base_name if base_name.lower().endswith('.mp4') else base_name + '.mp4'
            animation_obj.save(out_path, writer=writer, dpi=150)
            logging.info("Saved animation to %s", out_path)
            try:
                animation_obj._stop()
            except AttributeError:
                # some animation objects may not expose _stop
                pass
            return True
        except Exception as e_ffmpeg:
            errors.append(('ffmpeg', e_ffmpeg))

        # 2) Pillow -> gif
        try:
            from matplotlib.animation import PillowWriter
            out_path = base_name if base_name.lower().endswith('.gif') else base_name + '.gif'
            animation_obj.save(out_path, writer=PillowWriter(fps=20), dpi=150)
            logging.info("Saved animation to %s", out_path)
            try:
                animation_obj._stop()
            except AttributeError:
                pass
            return True
        except Exception as e_pillow:
            errors.append(('pillow', e_pillow))

        try:
            out_path = base_name if base_name.lower().endswith('.gif') else base_name + '.gif'
            animation_obj.save(out_path, writer='imagemagick', fps=30)
            logging.info("Saved animation to %s", out_path)
            try:
                animation_obj._stop()
            except AttributeError:
                pass
            return True
        except Exception as e_im:
            errors.append(('imagemagick', e_im))

        logging.warning("Failed to save animation with available writers.")
        for name, ex in errors:
            logging.warning("%s error: %s", name, ex)

        if final_coords is not None:
            try:
                fig_final = plt.figure(figsize=(6,6))
                axf = fig_final.add_subplot(111, projection='3d')
                axf.plot(final_coords[:,0], final_coords[:,1], final_coords[:,2], lw=2)
                axf.scatter(final_coords[:,0], final_coords[:,1], final_coords[:,2], s=40)
                out_png = base_name + '_final.png'
                fig_final.savefig(out_png)
                plt.close(fig_final)
                logging.info("Saved final structure snapshot to %s", out_png)
                return True
            except Exception as e_snapshot:
                logging.exception("Failed to save final snapshot: %s", e_snapshot)

        return False

    # If user explicitly requested a save path, try to save via the consolidated helper.
    if save_as:
        _save_with_fallback(ani, save_as, final_coords=coords_list[-1] if coords_list else None)
    if show:
                # If interactive backend is available, open a 3D window and play frames via plt.pause
                if is_interactive:
                    try:
                        # Drive the frames manually and then block until the user closes the window.
                        for frame in range(Nframes):
                            update(frame)
                            plt.draw()
                            plt.pause(0.03)
                        # Block until the window is closed by the user (prevents a second non-blocking show)
                        plt.show()
                    except Exception as e:
                        logging.warning("Interactive display failed: %s", e)
                        # fall back to saving if possible using consolidated helper
                        default_name = save_as if save_as else f"animation_{sequence[:6]}"
                        _save_with_fallback(ani, default_name, final_coords=coords_list[-1] if coords_list else None)
                    finally:
                        plt.close('all')
                else:
                    # Non-interactive backend: if we have an animation object, try to auto-save
                    if ani is not None:
                        default_name = (save_as if save_as else f"animation_{sequence[:6]}")
                        _save_with_fallback(ani, default_name, final_coords=coords_list[-1] if coords_list else None)
                # leave saving to helper; nothing else to do here
    else:
        # when not showing, just close figures to free resources
        plt.close('all')


__all__ = ['animate_history']
