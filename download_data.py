"""
Downloads the PlantVillage dataset from Kaggle (via kagglehub) and prepares it
as data/PlantVillage/<class_name>/*.jpg, ready for train.py.

Usage:
    python download_data.py                 # all 38 classes
    python download_data.py --crops tomato potato pepper apple grape corn
"""
import argparse, os, shutil
import kagglehub

p = argparse.ArgumentParser()
p.add_argument("--crops", nargs="*", default=None,
               help="Keep only classes starting with these crop names (case-insensitive)")
p.add_argument("--out_dir", default="data/PlantVillage")
args = p.parse_args()

print("Downloading PlantVillage from Kaggle (≈1 GB, first run only)...")
root = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
print("Downloaded to:", root)

# Find the 'color' image folder (the dataset also ships grayscale + segmented versions)
color_dir = None
for dirpath, dirnames, _ in os.walk(root):
    if os.path.basename(dirpath).lower() == "color":
        color_dir = dirpath
        break
if color_dir is None:
    raise SystemExit("Could not find the 'color' folder inside the download: " + root)

crops = [c.lower() for c in args.crops] if args.crops else None
os.makedirs(args.out_dir, exist_ok=True)
kept = 0
for cls in sorted(os.listdir(color_dir)):
    src = os.path.join(color_dir, cls)
    if not os.path.isdir(src):
        continue
    if crops and not any(cls.lower().startswith(c) for c in crops):
        continue
    dst = os.path.join(args.out_dir, cls)
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
    kept += 1
    print(f"  {cls}: {len(os.listdir(dst))} images")

print(f"\nDone. {kept} classes ready in '{args.out_dir}'.")
print(f"Next: python train.py --data_dir {args.out_dir}")
