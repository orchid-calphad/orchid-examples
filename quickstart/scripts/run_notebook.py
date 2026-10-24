"""Execute quickstart.ipynb in place (outputs saved) from the quickstart folder. Figures go to $QUICKSTART_OUT
(default figures/), the ESPEI export to $QUICKSTART_EXPORT (default export/, git-ignored).
  python scripts/run_notebook.py [--no-save]     # --no-save: execute without rewriting the notebook (tests)"""
import os, sys
import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbformat.read(os.path.join(HERE, "quickstart.ipynb"), as_version=4)
NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": HERE}}).execute()
for c in nb.cells:      # drop volatile metadata so the saved notebook diff stays small
    c.get("metadata", {}).pop("execution", None)
if "--no-save" not in sys.argv:
    nbformat.write(nb, os.path.join(HERE, "quickstart.ipynb"))
print("executed quickstart.ipynb")
