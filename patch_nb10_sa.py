"""Patch notebook 10: Fix SA_Score import path."""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

NB = Path(r"e:\PhD_Projecttttttttttttt\PhD_QSAR_Leishmania\notebooks\10_virtual_screening.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

sa_old = "from rdkit.Chem.SA_Score import sascorer\n"
sa_new = [
    "from rdkit.Chem import RDConfig\n",
    "import os\n",
    'sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))\n',
    "import sascorer\n",
]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    if sa_old in cell["source"]:
        idx = cell["source"].index(sa_old)
        cell["source"] = cell["source"][:idx] + sa_new + cell["source"][idx + 1 :]
        print(f"  Fixed SA_Score import at cell {i}")
        break

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! SA_Score import fixed.")
