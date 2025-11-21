"""Module execution entrypoint.

Allows running the package with ``python -m compbio_fp``.
Launches GUI if no arguments, otherwise delegates to CLI.
"""

import sys
from .cli import main


def _run():
	if len(sys.argv) == 1:  # No arguments - launch GUI
		import os
		gui_path = os.path.join(os.path.dirname(__file__), '..', 'gui_main.py')
		if os.path.exists(gui_path):
			exec(open(gui_path).read())
		else:
			from .gui_tk import main as gui_main
			return gui_main()
	else:  # Has arguments - use CLI
		return main()


if __name__ == "__main__":
	raise SystemExit(_run())
