"""Execute quickstart.ipynb in place, with outputs saved, from the quickstart folder.

Figures go to $QUICKSTART_OUT (default figures/), the ESPEI export to $QUICKSTART_EXPORT (default export/, git-ignored).

  python scripts/run_notebook.py [--no-save]     # --no-save: execute without rewriting the notebook (tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

here = Path(__file__).resolve().parents[1]


def main() -> None:
    """Execute the notebook, drop volatile execution metadata and save it unless --no-save is given."""
    nb = nbformat.read(str(here / "quickstart.ipynb"), as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(here)}}).execute()
    for c in nb.cells:  # drop volatile metadata so the saved notebook diff stays small
        c.get("metadata", {}).pop("execution", None)
    if "--no-save" not in sys.argv:
        nbformat.write(nb, str(here / "quickstart.ipynb"))
    print("executed quickstart.ipynb")


if __name__ == "__main__":
    main()
