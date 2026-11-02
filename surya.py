
import os
import json
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

def load_dataset(dataset_path):

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int"
    )

    class_names = train_ds.class_names

    if len(class_names) < 2:
        raise ValueError("At least two classes are required.")

    print("\nDetected disease classes:")
    for name in class_names:
        print("-", name)

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    return train_ds, val_ds, class_names


def build_model(num_classes):

    # Data augmentation
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1)
    ])

    # Pretrained EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)

    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(dataset_path):

    train_ds, val_ds, class_names = load_dataset(
        dataset_path
    )

    model = build_model(len(class_names))

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            "organ_disease_model.keras",
            monitor="val_loss",
            save_best_only=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]

    print("\nStarting model training...\n")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Evaluate model
    loss, accuracy = model.evaluate(val_ds)

    print("\nTraining completed!")
    print("Validation Loss:", round(loss, 4))
    print("Validation Accuracy:", round(accuracy * 100, 2), "%")

    # Save class names
    with open("class_names.json", "w") as file:
        json.dump(class_names, file, indent=4)

    print("\nModel saved successfully!")
    print("File: organ_disease_model.keras")
    print("Classes saved in: class_names.json")

    return history


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Human Organ Disease Recognition"
    )

    parser.add_argument(
        "--data",
        default="dataset",
        help="Path to dataset directory"
    )

    args = parser.parse_args()

    train_model(args.data)
