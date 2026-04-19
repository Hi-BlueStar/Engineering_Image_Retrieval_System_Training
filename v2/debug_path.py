from pathlib import Path
from collections import defaultdict
import os

src = Path("data/preprocessed_images")
structure_map = defaultdict(list)

print(f"Current CWD: {os.getcwd()}")
print(f"Scanning {src.resolve()}...")
count = 0
for original_file in src.rglob("*_original.png"):
    count += 1
    instance_dir = original_file.parent
    class_name = instance_dir.parent.name
    if instance_dir not in structure_map[class_name]:
        structure_map[class_name].append(instance_dir)

print(f"Found {count} original PNGs.")
print(f"Found {len(structure_map)} classes.")
for i, (cls, insts) in enumerate(structure_map.items()):
    if i >= 5: break
    print(f"Class: {cls}, Instances: {len(insts)}")
    if insts:
        print(f"  First instance: {insts[0]}")
        pngs = list(insts[0].glob('*.png'))
        print(f"  PNGs in first instance: {len(pngs)}")
        print(f"  Example PNGs: {[p.name for p in pngs[:3]]}")
