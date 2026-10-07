"""FreshVision - Standalone Image Freshness Prediction Script.

Usage:
    python predict.py --image path/to/food.jpg
    python predict.py --image path/to/food.jpg --model models/freshvision_model.keras
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

from src.data_preprocessing import CLASS_NAMES, preprocess_image_for_prediction

DISCLAIMER_TEXT = (
    "Disclaimer: FreshVision is an educational computer-vision project. "
    "Predictions may be incorrect and should not be used as the only "
    "method for determining whether food is safe to eat."
)


def predict_food_freshness(
    image_path: str,
    model_path: str = "models/freshvision_model.keras",
    verbose: bool = True,
):
    """Predicts whether a food image is Fresh or Spoiled.

    Args:
        image_path: Filepath to the input image.
        model_path: Filepath to the trained Keras model (.keras).
        verbose: Whether to print formatted output to the console.

    Returns:
        A dictionary containing predicted_class, confidence, probabilities, and note.
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}\n"
            "Please check the path and make sure the image exists."
        )

    mdl_file = Path(model_path)
    if not mdl_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Tip: You need to train the model first by running:\n"
            "    python train.py\n"
            "or place a trained freshvision_model.keras in the models/ folder."
        )

    if verbose:
        print(f"[*] Loading model from: {model_path} ...")
    model = tf.keras.models.load_model(str(mdl_file))

    if verbose:
        print(f"[*] Preprocessing image: {image_path} ...")
    preprocessed_img = preprocess_image_for_prediction(img_file)

    # Run inference
    predictions = model.predict(preprocessed_img, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx].capitalize()
    confidence = float(predictions[predicted_idx]) * 100.0

    # Explanation message
    if predicted_class.lower() == "fresh":
        message = (
            "The food appears visually fresh with healthy surface color, "
            "normal texture, and no obvious signs of decay or mold."
        )
    else:
        message = (
            "The food shows visual patterns characteristic of spoilage, "
            "such as discoloration, dark spots, fungal growth, or surface softening."
        )

    result = {
        "image_path": str(img_file),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i].capitalize(): float(predictions[i]) * 100.0
            for i in range(len(CLASS_NAMES))
        },
        "message": message,
        "disclaimer": DISCLAIMER_TEXT,
    }

    if verbose:
        print("\n" + "=" * 55)
        print("          FreshVision - Food Freshness Result")
        print("=" * 55)
        print(f"Image File       : {img_file.name}")
        print(f"Prediction       : {result['predicted_class']}")
        print(f"Confidence       : {result['confidence']:.2f}%")
        print("Probabilities    : " + " | ".join(
            f"{k}: {v:.2f}%" for k, v in result["probabilities"].items()
        ))
        print(f"\nAnalysis Summary : {result['message']}")
        print("-" * 55)
        print(f"[Notice] {DISCLAIMER_TEXT}")
        print("=" * 55 + "\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="FreshVision: Classify food freshness from an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the food image file (JPG, JPEG, PNG).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/freshvision_model.keras",
        help="Path to trained Keras model file (default: models/freshvision_model.keras).",
    )
    args = parser.parse_args()

    try:
        predict_food_freshness(args.image, args.model)
    except Exception as exc:
        print(f"\n[Error] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
