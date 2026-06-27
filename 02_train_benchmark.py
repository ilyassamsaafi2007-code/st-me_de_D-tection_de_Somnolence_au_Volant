"""
=========================================================
   02 - Model Benchmarking - Drowsiness Detection
=========================================================
Had script kaydreb 3 modeles mokhtalfin 3la nfs dataset, w kayqarn
binathom bach n3rfo chkon ahsen:

    1) Simple CNN   (bsita, fast, light)
    2) Deep CNN     (3miqa shwiya, akthar params)
    3) MobileNetV2  (Transfer Learning - pretrained 3la ImageNet)

F lakher, kaykhrej:
    - model_comparison.csv      (jadwal kamel dyal results)
    - model_comparison.png      (graph dyal accuracy/time)
    - model_summaries.txt       (summary() dyal kola model + optimizer info)
    - best_model.keras          (ahsen model automatiquement)

Installation:
    pip install tensorflow scikit-learn pandas matplotlib seaborn
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# ---------- CONFIG ----------
DATASET_DIR = "ddd_dataset"
IMG_SIZE = (64, 64)          # sghira bzaf bach matatqalch l PC
BATCH_SIZE = 16               # batch sghir = ahaf 3la RAM
EPOCHS = 5                    # kafi bach n3orfo chkon ahsen model
OUTPUT_DIR = "benchmark_results"

# LIGHT MODE: ghir subset mn dataset f kola epoch (bach ysra3 w mayakhodch
# RAM/CPU bzaf). Hadi tjarib sari3a bach n3orfo ahsen model, mnba3d
# nrj3o ndarbo l ahsen wahd b dataset kamla ila bghiti accuracy ahsen.
STEPS_PER_EPOCH = 60           # ~960 soura f kola epoch bdal 33000+
VALIDATION_STEPS = 15          # ~240 soura l validation

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- DATA GENERATORS ----------
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="training", color_mode="rgb",
)
val_gen = datagen.flow_from_directory(
    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", subset="validation", color_mode="rgb", shuffle=False,
)

print("Classes:", train_gen.class_indices)
CLASS_INDICES = train_gen.class_indices

# ---------- MODEL DEFINITIONS ----------
def build_simple_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="Simple_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_deep_cnn():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),
        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="Deep_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_mobilenet():
    base = applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False  # Transfer learning: ma kandarboch les poids l9dam

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ], name="MobileNetV2_Transfer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


MODELS_TO_TEST = {
    "Simple_CNN": build_simple_cnn,
    "Deep_CNN": build_deep_cnn,
    "MobileNetV2_Transfer": build_mobilenet,
}

# ---------- TRAINING LOOP + BENCHMARKING ----------
results = []
summaries_text = []

for name, builder in MODELS_TO_TEST.items():
    print("\n" + "=" * 70)
    print(f"TRAINING: {name}")
    print("=" * 70)

    model = builder()

    # Capture model.summary() f text
    summary_lines = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    summaries_text.append(f"\n{'='*70}\nMODEL: {name}\n{'='*70}\n")
    summaries_text.append("\n".join(summary_lines))

    # Optimizer info
    opt_config = model.optimizer.get_config()
    summaries_text.append(f"\n\nOptimizer config:\n{json.dumps(opt_config, indent=2, default=str)}\n")

    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    start_time = time.time()
    history = model.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks, verbose=1,
    )
    train_time = time.time() - start_time

    # Evaluation
    val_loss, val_acc = model.evaluate(val_gen, steps=VALIDATION_STEPS, verbose=0)

    # Predictions for detailed report
    val_gen.reset()
    preds = model.predict(val_gen, steps=VALIDATION_STEPS, verbose=0)
    pred_labels = (preds > 0.5).astype(int).flatten()
    true_labels = val_gen.classes[:len(pred_labels)]

    report = classification_report(true_labels, pred_labels, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)

    # Save confusion matrix plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_INDICES.keys(), yticklabels=CLASS_INDICES.keys())
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))
    plt.close()

    # Save model
    model.save(os.path.join(OUTPUT_DIR, f"{name}.keras"))

    results.append({
        "model_name": name,
        "total_params": total_params,
        "trainable_params": int(trainable_params),
        "train_time_sec": round(train_time, 2),
        "epochs_ran": len(history.history["loss"]),
        "val_accuracy": round(val_acc, 4),
        "val_loss": round(val_loss, 4),
        "precision": round(report["weighted avg"]["precision"], 4),
        "recall": round(report["weighted avg"]["recall"], 4),
        "f1_score": round(report["weighted avg"]["f1-score"], 4),
    })

    # Plot training curves per model
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title(f"{name} - Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title(f"{name} - Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"curves_{name}.png"))
    plt.close()

# ---------- SAVE COMPARISON TABLE ----------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("val_accuracy", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

print("\n" + "=" * 70)
print("RESULTAT FINAL - COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))

# ---------- SAVE SUMMARIES ----------
with open(os.path.join(OUTPUT_DIR, "model_summaries.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summaries_text))

# ---------- COMPARISON PLOT ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.barplot(data=results_df, x="model_name", y="val_accuracy", ax=axes[0])
axes[0].set_title("Validation Accuracy par Model")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=20)

sns.barplot(data=results_df, x="model_name", y="train_time_sec", ax=axes[1])
axes[1].set_title("Training Time (secondes)")
axes[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
plt.close()

# ---------- SAVE BEST MODEL SEPARATELY ----------
best_model_name = results_df.iloc[0]["model_name"]
best_model_path = os.path.join(OUTPUT_DIR, f"{best_model_name}.keras")
final_best_path = os.path.join(OUTPUT_DIR, "best_model.keras")

import shutil
shutil.copy(best_model_path, final_best_path)

# Save class indices mapping (mohim bzaf l deployment)
with open(os.path.join(OUTPUT_DIR, "class_indices.json"), "w") as f:
    json.dump(CLASS_INDICES, f)

print(f"\n✅ AHSEN MODEL: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['val_accuracy']*100:.2f}%")
print(f"   Sauvegardi f: {final_best_path}")
print(f"\nGa3 les fichiers f folder: {OUTPUT_DIR}/")
print("   - model_comparison.csv")
print("   - model_comparison.png")
print("   - model_summaries.txt")
print("   - best_model.keras")
print("   - class_indices.json")
print("   - confusion_matrix_*.png (pour kol model)")
print("   - curves_*.png (training curves pour kol model)")