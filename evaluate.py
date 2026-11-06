"""
Evaluate an already-trained model (no retraining) and write the report files.
Also (re)creates outputs/class_names.json, which app.py needs.

Usage:  python evaluate.py --data_dir data/PlantVillage
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/PlantVillage")
p.add_argument("--img_size", type=int, default=128)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--out_dir", default="outputs")
args = p.parse_args()

SIZE = (args.img_size, args.img_size)
# Same split settings as train.py -> identical validation set
val_ds = tf.keras.utils.image_dataset_from_directory(
    args.data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=SIZE, batch_size=args.batch_size, label_mode="int", shuffle=False)
class_names = val_ds.class_names
num_classes = len(class_names)
labels = list(range(num_classes))

with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
    json.dump({"class_names": class_names, "img_size": args.img_size}, f, indent=2)
print("Saved class_names.json")

model = tf.keras.models.load_model(os.path.join(args.out_dir, "plant_disease_model.keras"))
loss, acc = model.evaluate(val_ds)
print(f"Validation accuracy: {acc*100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_pred = np.argmax(model.predict(val_ds), axis=1)
report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                               digits=3, zero_division=0)
print(report)
open(os.path.join(args.out_dir, "classification_report.txt"), "w").write(report)

cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(max(8, num_classes * 0.45),) * 2)
sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
print("Saved report + confusion matrix to", args.out_dir)
