"""
Plant Disease Detection - training script (CNN, TensorFlow/Keras)

Usage:
    python train.py --data_dir /path/to/PlantVillage --epochs 15
    python train.py --data_dir /path/to/PlantVillage --model mobilenet   # optional transfer learning

Expected dataset layout (one folder per class):
    PlantVillage/
        Tomato_Early_blight/ img1.jpg ...
        Potato___healthy/    img1.jpg ...
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data/PlantVillage")
p.add_argument("--img_size", type=int, default=128)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--model", choices=["cnn", "mobilenet"], default="cnn")
p.add_argument("--out_dir", default="outputs")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
SIZE = (args.img_size, args.img_size)
tf.keras.utils.set_random_seed(42)

# ---------------- 1. Load data (80% train / 20% validation+test) ----------------
common = dict(validation_split=0.2, seed=42, image_size=SIZE,
              batch_size=args.batch_size, label_mode="int")
train_ds = tf.keras.utils.image_dataset_from_directory(
    args.data_dir, subset="training", shuffle=True, **common)
val_ds = tf.keras.utils.image_dataset_from_directory(
    args.data_dir, subset="validation", shuffle=False, **common)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Found {num_classes} classes")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# ---------------- 2. Build model ----------------
# Preprocessing (rescaling) lives INSIDE the model, so the app can pass raw 0-255 pixels.
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="augmentation")

def build_cnn():
    inp = layers.Input(shape=SIZE + (3,))
    x = augment(inp)
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)  # conv + ReLU
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)                                         # max pooling
    x = layers.Flatten()(x)                                                  # flatten
    x = layers.Dense(256, activation="relu")(x)                              # dense
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)                 # softmax
    return models.Model(inp, out, name="plant_cnn")

def build_mobilenet():
    base = tf.keras.applications.MobileNetV2(
        input_shape=SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False
    inp = layers.Input(shape=SIZE + (3,))
    x = augment(inp)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out, name="plant_mobilenet")

model = build_cnn() if args.model == "cnn" else build_mobilenet()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------- 3. Train ----------------
model_path = os.path.join(args.out_dir, "plant_disease_model.keras")
cbs = [
    callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
    callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
]
history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)

# ---------------- 4. Evaluate ----------------
loss, acc = model.evaluate(val_ds)
print(f"\nValidation accuracy: {acc*100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in val_ds])
y_pred = np.argmax(model.predict(val_ds), axis=1)
report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print(report)
with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train"); ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train"); ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "training_curves.png"), dpi=150); plt.close()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(8, num_classes * 0.45),) * 2)
sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=90); plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150); plt.close()

# Metadata for the app
with open(os.path.join(args.out_dir, "class_names.json"), "w") as f:
    json.dump({"class_names": class_names, "img_size": args.img_size}, f, indent=2)

print(f"\nSaved model + artifacts to '{args.out_dir}/'")
