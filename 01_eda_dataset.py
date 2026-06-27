"""
=========================================================
   01 - EDA (Exploratory Data Analysis) - DDD Dataset
=========================================================
Had script kaydi ga3 les info 3la dataset 9bel ma ndarbo:
    - 3dad dyal soura f kola class
    - Shape dyal soura (taille, channels)
    - Statistiques (mean, std, min, max pixels)
    - Samples visuels
    - Class balance (wach dataset balanced wla la)

Installation:
    pip install opencv-python matplotlib numpy pandas seaborn pillow
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# ---------- CONFIG ----------
DATASET_DIR = "ddd_dataset"   # bdel hada 3la fin 3andek dataset
CLASSES = ["Drowsy", "Non Drowsy"]
OUTPUT_DIR = "eda_report"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1) COUNT IMAGES PER CLASS ----------
print("=" * 60)
print("1) COMPTAGE DYAL SOURA F KOLA CLASS")
print("=" * 60)

class_counts = {}
class_filepaths = {}

for cls in CLASSES:
    cls_path = os.path.join(DATASET_DIR, cls)
    if not os.path.exists(cls_path):
        print(f"⚠️ Path machi mojoud: {cls_path}")
        continue
    files = [f for f in os.listdir(cls_path)
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    class_counts[cls] = len(files)
    class_filepaths[cls] = [os.path.join(cls_path, f) for f in files]
    print(f"  {cls}: {len(files)} soura")

total = sum(class_counts.values())
print(f"\nTotal: {total} soura")

# ---------- 2) CLASS BALANCE PLOT ----------
plt.figure(figsize=(6, 4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title("Class Distribution")
plt.ylabel("Number of images")
for i, v in enumerate(class_counts.values()):
    plt.text(i, v + total * 0.01, str(v), ha="center")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.close()
print(f"\n[OK] class_distribution.png sauvegardi f {OUTPUT_DIR}/")

# Imbalance ratio
if len(class_counts) == 2:
    vals = list(class_counts.values())
    ratio = max(vals) / min(vals)
    print(f"\nImbalance ratio: {ratio:.2f}x")
    if ratio > 1.5:
        print("⚠️ Dataset machi balanced bzaf - khasna class_weight f training")
    else:
        print("✅ Dataset balanced mzyan")

# ---------- 3) IMAGE SHAPE / RESOLUTION ANALYSIS ----------
print("\n" + "=" * 60)
print("2) SHAPE DYAL SOURA (sample dyal 200 soura)")
print("=" * 60)

shapes = []
sample_files = []
for cls in CLASSES:
    files = class_filepaths.get(cls, [])[:100]  # 100 mn kol class
    sample_files.extend([(f, cls) for f in files])

for filepath, cls in sample_files:
    img = cv2.imread(filepath)
    if img is not None:
        shapes.append(img.shape)

shapes_df = pd.DataFrame(shapes, columns=["height", "width", "channels"])
print(shapes_df.describe())

unique_shapes = shapes_df.drop_duplicates()
print(f"\nNombre dyal shapes uniques: {len(unique_shapes)}")
if len(unique_shapes) == 1:
    print(f"✅ Ga3 soura nfs shape: {unique_shapes.values[0]}")
else:
    print("⚠️ Soura 3andhom shapes mokhtalifin - khasna resize f preprocessing")
    print(unique_shapes.head(10))

# ---------- 4) PIXEL STATISTICS ----------
print("\n" + "=" * 60)
print("3) PIXEL STATISTICS")
print("=" * 60)

pixel_means = []
pixel_stds = []
for filepath, cls in sample_files[:100]:
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        pixel_means.append(img.mean())
        pixel_stds.append(img.std())

print(f"Mean pixel value (moyenne): {np.mean(pixel_means):.2f} / 255")
print(f"Std pixel value: {np.mean(pixel_stds):.2f}")
print(f"Min: {np.min(pixel_means):.2f}  |  Max: {np.max(pixel_means):.2f}")

# ---------- 5) SAMPLE IMAGES GRID ----------
print("\n" + "=" * 60)
print("4) SAMPLE IMAGES VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for row, cls in enumerate(CLASSES):
    files = class_filepaths.get(cls, [])[:5]
    for col, filepath in enumerate(files):
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[row, col].imshow(img)
        axes[row, col].set_title(cls, fontsize=10)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
plt.close()
print(f"[OK] sample_images.png sauvegardi f {OUTPUT_DIR}/")

# ---------- 6) FULL SUMMARY REPORT (TXT FILE) ----------
report_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("EDA REPORT - Driver Drowsiness Dataset\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total images: {total}\n\n")
    f.write("Class distribution:\n")
    for cls, count in class_counts.items():
        pct = (count / total) * 100 if total else 0
        f.write(f"  - {cls}: {count} ({pct:.1f}%)\n")
    f.write(f"\nImage shape (mode): {unique_shapes.values[0] if len(unique_shapes) > 0 else 'N/A'}\n")
    f.write(f"Number of unique shapes detected: {len(unique_shapes)}\n")
    f.write(f"\nMean pixel value: {np.mean(pixel_means):.2f}\n")
    f.write(f"Std pixel value: {np.mean(pixel_stds):.2f}\n")

print(f"\n[OK] Rapport kamel sauvegardi f: {report_path}")
print("\n" + "=" * 60)
print("EDA KMLAT! Chof folder 'eda_report/' fih ga3 les graphes w rapport.")
print("=" * 60)
