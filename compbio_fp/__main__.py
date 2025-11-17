"""Module execution entrypoint.

Allows running the package with ``python -m compbio_fp``.
Delegates to ``compbio_fp.cli.main`` so there is a single canonical
implementation of argument parsing & demo execution.
"""

from .cli import main


def _run():  # tiny indirection for potential future hooks
	return main()


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(_run())
