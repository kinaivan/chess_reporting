"""
Train the ChessVisionAI model.

Run generate_dataset.py first to create the training data.

Usage:
    python train_model.py
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import img_width, img_height, num_classes

# Paths
train_data_dir = "dataset/training"
test_data_dir = "dataset/test"
wip_models_dir = "models_wip"
models_dir = "models"
model_name = "chess_classifier_10k"
current_best = os.path.join(wip_models_dir, f"{model_name}_best.keras")
final = os.path.join(models_dir, f"{model_name}.keras")

# Parameters
batch_size = 32
epochs = 10


def main():
    print("=" * 50)
    print("ChessVisionAI Model Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs(wip_models_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Check if dataset exists
    if not os.path.exists(train_data_dir):
        print(f"Error: Training data not found at {train_data_dir}")
        print("Please run generate_dataset.py first.")
        return

    # Data normalization
    print("\nSetting up data generators...")
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Train and validation generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
    )

    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode="categorical",
    )

    # Calculate steps
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(validation_generator) // batch_size)

    print(f"\nTraining samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")

    # Create model
    print("\nBuilding model (MobileNetV2 + custom layers)...")
    base_model = MobileNetV2(
        weights="imagenet", 
        include_top=False, 
        input_shape=(img_width, img_height, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers for initial training
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        current_best, monitor="val_loss", save_best_only=True
    )

    # Initial training
    print("\n" + "=" * 50)
    print("Phase 1: Initial Training (frozen base layers)")
    print("=" * 50)
    
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Fine-tuning
    print("\n" + "=" * 50)
    print("Phase 2: Fine-tuning (unfreezing last 30 layers)")
    print("=" * 50)

    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Save final model
    print(f"\nSaving final model to {final}...")
    model.save(final)

    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(validation_generator, steps=validation_steps)
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Final Test Loss: {loss:.4f}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {final}")
    print("=" * 50)


if __name__ == "__main__":
    main()
