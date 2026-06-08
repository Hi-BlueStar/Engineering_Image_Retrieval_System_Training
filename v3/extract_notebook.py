import json
from pathlib import Path

notebook_path = Path("/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training/v3/simsiam_training.ipynb")
output_path = Path("/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training/v3/simsiam_training_extracted.py")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(output_path, "w", encoding="utf-8") as f:
    for idx, cell in enumerate(nb["cells"]):
        f.write(f"\n# ==========================================\n# CELL {idx} ({cell['cell_type'].upper()})\n# ==========================================\n")
        if cell["cell_type"] == "code":
            f.write("".join(cell["source"]))
            f.write("\n")
        else:
            # Markdown cells are commented out
            for line in cell["source"]:
                f.write(f"# {line}")
            f.write("\n")

print("Extracted successfully!")
