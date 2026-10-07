"""FreshVision - End-to-End Model Training Pipeline.

This script trains a Transfer Learning model using MobileNetV2 for classifying
food freshness (Fresh vs. Spoiled).

Usage:
    python train.py
    python train.py --epochs 10 --batch-size 32
    python train.py --create-sample-data --epochs 2
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

from src.data_preprocessing import (
    CLASS_NAMES,
    IMAGE_SIZE,
    check_dataset_status,
    load_datasets,
)
from src.evaluation import evaluate_model_performance, plot_training_history
from src.model import build_freshvision_model, enable_fine_tuning


def generate_sample_dataset(dataset_dir: Path) -> None:
    """Generates synthetic sample images for initial testing and pipeline validation.

    This allows a student to verify the full code pipeline immediately before
    downloading a multi-gigabyte real-world dataset.
    """
    print("[*] Generating synthetic sample food images for testing...")
    splits_counts = {
        "train": 16,
        "validation": 6,
        "test": 6,
    }

    color_schemes = {
        "fresh": [(34, 197, 94), (22, 163, 74), (239, 68, 68), (245, 158, 11)],  # Vibrant green, red, yellow
        "spoiled": [(78, 59, 45), (107, 79, 58), (55, 65, 81), (80, 70, 50)],   # Dark brown, grey, decayed tones
    }

    for split, count_per_class in splits_counts.items():
        for category in CLASS_NAMES:
            cat_dir = dataset_dir / split / category
            cat_dir.mkdir(parents=True, exist_ok=True)
            colors = color_schemes[category]

            for i in range(count_per_class):
                img_path = cat_dir / f"sample_{category}_{i + 1}.jpg"
                if not img_path.exists():
                    img = Image.new("RGB", (224, 224), color=colors[i % len(colors)])
                    draw = ImageDraw.Draw(img)
                    # Add circular shapes to mimic fruit / food
                    draw.ellipse([40, 40, 184, 184], fill=colors[(i + 1) % len(colors)], outline=(20, 20, 20))
                    # Add spots
                    if category == "spoiled":
                        draw.ellipse([80, 80, 120, 120], fill=(30, 20, 10))
                        draw.ellipse([130, 110, 160, 140], fill=(20, 20, 15))
                    img.save(str(img_path), "JPEG", quality=90)

    print("[+] Sample dataset generated in dataset/ directory.")


def train_freshvision(
    dataset_dir: str = "dataset",
    model_save_path: str = "models/freshvision_model.keras",
    results_dir: str = "results",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    create_sample_data: bool = False,
):
    """Executes the complete data loading, model compilation, training, and evaluation pipeline."""
    data_path = Path(dataset_dir)
    save_path = Path(model_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    res_dir = Path(results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)

    if create_sample_data:
        generate_sample_dataset(data_path)

    # 1. Verify Dataset
    counts = check_dataset_status(data_path)
    train_total = sum(counts["train"].values())
    val_total = sum(counts["validation"].values())
    test_total = sum(counts["test"].values())

    print("\n" + "=" * 55)
    print("         FreshVision - Dataset Inspection")
    print("=" * 55)
    print(f"Training Images   : {train_total} (Fresh: {counts['train']['fresh']}, Spoiled: {counts['train']['spoiled']})")
    print(f"Validation Images : {val_total} (Fresh: {counts['validation']['fresh']}, Spoiled: {counts['validation']['spoiled']})")
    print(f"Testing Images    : {test_total} (Fresh: {counts['test']['fresh']}, Spoiled: {counts['test']['spoiled']})")
    print("=" * 55 + "\n")

    if train_total == 0:
        print("[!] ERROR: No images found in the dataset training directory!")
        print("\nHow to fix:")
        print("1. Place your food images into:")
        print("   dataset/train/fresh/        <- Put fresh fruit/vegetable photos here")
        print("   dataset/train/spoiled/      <- Put rotten/decayed food photos here")
        print("   dataset/validation/fresh/")
        print("   dataset/validation/spoiled/")
        print("   dataset/test/fresh/")
        print("   dataset/test/spoiled/")
        print("\n2. Or, run with '--create-sample-data' for an instant trial:")
        print("   python train.py --create-sample-data --epochs 2")
        print("\nSee dataset/README.md for download instructions for free public datasets.\n")
        sys.exit(1)

    # 2. Load Datasets
    print("[*] Loading and batching images...")
    train_ds, val_ds, test_ds = load_datasets(data_path, target_size=IMAGE_SIZE, batch_size=batch_size)

    # 3. Build Model
    print("\n[*] Initializing Transfer Learning Model (MobileNetV2)...")
    model = build_freshvision_model(
        num_classes=len(CLASS_NAMES),
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        learning_rate=learning_rate,
    )
    model.summary()

    # 4. Training Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor="val_loss" if val_ds else "loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_ds else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if val_ds else "loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # 5. Train Phase 1 (Transfer Learning with Frozen Base)
    print(f"\n[*] Starting Phase 1 Training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Optional Phase 2: Fine-Tuning
    if fine_tune and val_ds:
        print(f"\n[*] Starting Phase 2: Fine-Tuning for {fine_tune_epochs} additional epochs...")
        model = enable_fine_tuning(model, unfreeze_from_layer=100, fine_tune_lr=1e-5)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=history.epoch[-1] + 1,
            callbacks=callbacks,
            verbose=1,
        )
        # Merge histories for plotting
        for k in history.history:
            history.history[k].extend(fine_history.history.get(k, []))

    # 7. Save Final Model
    model.save(str(save_path))
    print(f"\n[+] Trained model saved to: {save_path}")

    # 8. Plot and Save Training Curves
    acc_loss_path = res_dir / "accuracy_loss.png"
    plot_training_history(history, output_path=acc_loss_path)

    # 9. Evaluate on Test Dataset
    if test_ds and test_total > 0:
        evaluate_model_performance(
            model=model,
            test_dataset=test_ds,
            class_names=CLASS_NAMES,
            output_dir=res_dir,
        )
    else:
        print("[!] Note: Test dataset was empty or not provided. Skipping test evaluation.")
        print("    Add images to dataset/test/ to evaluate generalization accuracy.")

    print("\n" + "=" * 55)
    print("       [+] FreshVision Model Training Finished! [+]")
    print("=" * 55)
    print(f"Model saved to   : {save_path}")
    print(f"Results saved to : {res_dir}")
    print("Next step: Run the web app using:")
    print("    streamlit run app.py\n")


def main():
    parser = argparse.ArgumentParser(description="Train FreshVision Food Freshness Classifier.")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset directory.")
    parser.add_argument("--model-path", type=str, default="models/freshvision_model.keras", help="Model save path.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save evaluation plots.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15).")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Initial Adam learning rate.")
    parser.add_argument("--fine-tune", action="store_true", help="Enable fine-tuning after initial training.")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Epochs for fine-tuning.")
    parser.add_argument("--create-sample-data", action="store_true", help="Generate sample dataset for testing.")
    args = parser.parse_args()

    train_freshvision(
        dataset_dir=args.dataset_dir,
        model_save_path=args.model_path,
        results_dir=args.results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        create_sample_data=args.create_sample_data,
    )


if __name__ == "__main__":
    main()
