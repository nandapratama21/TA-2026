import ast
from pathlib import Path

import nbformat


project_root = Path(__file__).resolve().parents[1]
notebook_path = project_root / "notebooks" / "18_single_image_inference_fft_clip.ipynb"
nb = nbformat.read(notebook_path, as_version=4)

code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
for idx, cell in enumerate(code_cells, start=1):
    ast.parse(cell.source, filename=f"{notebook_path.name}:code_cell_{idx}")

print("Notebook valid")
print("Cells:", len(nb.cells))
print("Code cells:", len(code_cells))
print("Path:", notebook_path)
