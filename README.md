# FreshVision — Food Freshness Classification

[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras 3](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision and Deep Learning application that classifies whether food items (fruits, vegetables, and perishables) are **Fresh** or **Spoiled** using Transfer Learning with **MobileNetV2**.

---

## 1. Project Title
**FreshVision — AI-Powered Food Freshness Classification**

---

## 2. Project Description
**FreshVision** is an educational Deep Learning application developed as a first-semester B.Tech Computer Science project. It uses state-of-the-art computer vision to inspect food photographs, evaluate visual indicators of quality (color consistency, surface texture, dark spots, microbial decay), and deliver an instant freshness assessment with an accompanying confidence score through an interactive web interface.

---

## 3. Problem Statement
Every year, over 1.3 billion tons of food is wasted globally, while consuming spoiled food leads to preventable foodborne illnesses. Traditional food inspection relies entirely on manual observation, which is subjective and error-prone. FreshVision addresses this challenge by providing an accessible, automated, and objective computer vision tool for preliminary freshness verification.

---

## 4. Objective
- Build an end-to-end Deep Learning classification pipeline that runs efficiently on personal laptops without requiring high-end GPUs.
- Leverage **Transfer Learning** using pre-trained **MobileNetV2** weights from ImageNet to achieve accurate results with modest training data.
- Evaluate the model on unbiased test datasets using genuine metrics (Precision, Recall, F1-Score, and Confusion Matrix).
- Deploy an intuitive, beginner-friendly **Streamlit web application** for image upload and real-time inference.

---

## 5. Features
- 📤 **Image Upload:** Supports common image formats (`.jpg`, `.jpeg`, `.png`).
- ⚡ **Instant Inference:** Delivers rapid predictions using a lightweight MobileNetV2 architecture.
- 🎯 **Confidence Scores:** Computes genuine softmax probability distribution and confidence percentages.
- 📊 **Visual Probability Breakdown:** Displays side-by-side progress bars for both Fresh and Spoiled probabilities.
- 📝 **Diagnostic Summary:** Explains the visual cues observed by the model.
- 🛡️ **Safety Disclaimer:** Prominently highlights educational use constraints.
- 🧪 **Offline Sample Generator:** Includes a built-in flag (`--create-sample-data`) allowing immediate testing without downloading large datasets.

---

## 6. Technologies Used
- **Programming Language:** Python 3.13
- **Deep Learning Framework:** TensorFlow 2.21 / Keras 3
- **Computer Vision:** OpenCV (`opencv-python`), Pillow (`PIL`)
- **Data Science & Metrics:** NumPy, Pandas, Scikit-learn
- **Data Visualization:** Matplotlib
- **Web Interface:** Streamlit
- **Version Control:** Git & GitHub

---

## 7. How the Model Works
Instead of training a convolutional neural network (CNN) from scratch—which requires tens of thousands of labeled images and days of compute—FreshVision uses **Transfer Learning**:

1. **Base Feature Extractor:** MobileNetV2 was pre-trained by Google on the ImageNet benchmark (1.4 million images across 1,000 classes).
2. **Feature Preservation:** The base convolutional layers are **frozen** so that generic features (edges, contours, color transitions, textures) are retained.
3. **Data Augmentation:** Random horizontal flips, rotations ($\pm 10\%$), and zooms ($\pm 10\%$) prevent overfitting during training.
4. **Classification Head:** A custom head containing a `GlobalAveragePooling2D` layer, `Dropout(0.3)` regularization, a `Dense(128, ReLU)` feature layer, and a final `Dense(2, Softmax)` output layer learns specifically to distinguish **Fresh** from **Spoiled** food items.

---

## 8. Dataset Structure

Organize your dataset inside the `dataset/` directory with the following structure:

```text
dataset/
├── train/
│   ├── fresh/       # Training images of fresh food
│   └── spoiled/     # Training images of spoiled food
│
├── validation/
│   ├── fresh/       # Validation images for tuning and early stopping
│   └── spoiled/
│
└── test/
    ├── fresh/       # Held-out test images evaluated only after training
    └── spoiled/
```

### Where to Get Datasets
You can download public food freshness datasets from Kaggle:
- **Kaggle Fruits Fresh and Rotten:** [sriramr/fruits-fresh-and-rotten-for-classification](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)
- **Kaggle Fruit Freshness Dataset:** [raghavr/fruit-freshness-dataset](https://www.kaggle.com/datasets/raghavr/fruit-freshness-dataset)

> 💡 *See [dataset/README.md](file:///dataset/README.md) for full download instructions and dataset organization steps.*

---

## 9. Installation & System Requirements
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** Python 3.10 to 3.13
- **RAM:** Minimum 4 GB (8 GB recommended)

Clone or download the project to your local computer:

```powershell
git clone https://github.com/YOUR_USERNAME/FreshVision.git
cd FreshVision
```

---

## 10. Virtual Environment Setup

Isolating dependencies inside a virtual environment prevents version conflicts with other projects.

### Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1
```

*(If you use the Windows Python launcher, you can also run `py -3.13 -m venv .venv`)*

### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 11. Dependency Installation

With your virtual environment activated, install all required packages:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 12. How to Train the Model

### Option A: Quick Test Run (Using Synthetic Sample Data)
If you haven't downloaded a full dataset yet, test the entire training and evaluation pipeline instantly:

```powershell
python train.py --create-sample-data --epochs 2
```

### Option B: Full Training (With Your Dataset)
Once images are placed in `dataset/train/`, `dataset/validation/`, and `dataset/test/`:

```powershell
python train.py --epochs 15 --batch-size 32
```

### Optional Fine-Tuning:
To unfreeze the top layers of MobileNetV2 and fine-tune with a low learning rate:

```powershell
python train.py --epochs 15 --fine-tune --fine-tune-epochs 5
```

The script will automatically:
1. Load and batch images with prefetching.
2. Train using `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`.
3. Save the trained model to `models/freshvision_model.keras`.
4. Plot and save training history to `results/accuracy_loss.png`.
5. Compute precision, recall, F1-score, and save the confusion matrix to `results/confusion_matrix.png`.

---

## 13. How to Run the Streamlit Application

Start the web application locally:

```powershell
streamlit run app.py
```

Your web browser will automatically open at:
```text
http://localhost:8501
```

1. Upload any food image (`.jpg`, `.jpeg`, `.png`).
2. View the uploaded image preview.
3. Review the predicted class, confidence percentage, and observation summary.

---

## 14. Example Predictions

### Example 1: Fresh Apple
```text
==================================================
          FreshVision — Food Freshness Result
==================================================
Image File       : fresh_apple_01.jpg
Prediction       : Fresh
Confidence       : 94.20%
Probabilities    : Fresh: 94.20% | Spoiled: 5.80%
Analysis Summary : The food appears visually fresh with healthy surface color,
                   normal texture, and no obvious signs of decay or mold.
```

### Example 2: Spoiled Banana
```text
==================================================
          FreshVision — Food Freshness Result
==================================================
Image File       : rotten_banana_14.jpg
Prediction       : Spoiled
Confidence       : 91.70%
Probabilities    : Fresh: 8.30% | Spoiled: 91.70%
Analysis Summary : The food shows visual patterns characteristic of spoilage,
                   such as discoloration, dark spots, fungal growth, or surface softening.
```

---

## 15. Model Architecture

```text
Input Layer (224x224x3)
        │
        ▼
Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
        │
        ▼
MobileNetV2 Normalization (Scaling to [-1, 1])
        │
        ▼
Pre-trained MobileNetV2 (Frozen ImageNet Weights, 1280 feature channels)
        │
        ▼
GlobalAveragePooling2D (Reduces 7x7x1280 to 1280 feature vector)
        │
        ▼
BatchNormalization
        │
        ▼
Dropout (30% dropout rate)
        │
        ▼
Dense Layer (128 units, ReLU activation)
        │
        ▼
Dropout (20% dropout rate)
        │
        ▼
Dense Output Layer (2 units, Softmax activation)
```

---

## 16. Evaluation Metrics

During training, the test dataset is evaluated using Scikit-Learn without artificial manipulation:

- **Accuracy:** Overall correct predictions over total predictions.
- **Precision:** True Positives / (True Positives + False Positives).
- **Recall:** True Positives / (True Positives + False Negatives).
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Detail of correct vs. misclassified instances saved to `results/confusion_matrix.png`.

---

## 17. Project Folder Structure

```text
FreshVision/
│
├── app.py                     # Interactive Streamlit web interface
├── train.py                   # Complete model training pipeline
├── predict.py                 # Standalone CLI prediction script
├── requirements.txt           # Project Python dependencies
├── README.md                  # Complete project documentation
├── PROJECT_REPORT.md          # College academic project report
├── .gitignore                 # Excludes cache, venv, and large files
│
├── dataset/
│   ├── README.md              # Dataset download & organization guide
│   ├── train/                 # Training split
│   │   ├── fresh/
│   │   └── spoiled/
│   ├── validation/            # Validation split
│   │   ├── fresh/
│   │   └── spoiled/
│   └── test/                  # Test split
│       ├── fresh/
│       └── spoiled/
│
├── models/
│   ├── .gitkeep
│   └── freshvision_model.keras # Saved trained Keras model
│
├── notebooks/
│   └── FreshVision_Training.ipynb # Jupyter Notebook for training & exploration
│
├── src/
│   ├── __init__.py            # Package initializer
│   ├── data_preprocessing.py  # Image loading, preprocessing & scaling
│   ├── model.py               # MobileNetV2 transfer learning model
│   └── evaluation.py          # Metric reports, loss curves & confusion matrix
│
├── results/
│   ├── accuracy_loss.png      # Training/Validation accuracy and loss curves
│   └── confusion_matrix.png   # Scikit-learn confusion matrix plot
│
└── screenshots/
    └── README.md              # Guide for saving UI preview screenshots
```

---

## 18. Limitations
- **External Surface Only:** Predictions evaluate only visible exterior features. Internal mold or bacterial pathogens without visible symptoms cannot be detected.
- **Lighting Conditions:** Harsh shadows, severe glare, or blurry camera focus can reduce prediction accuracy.
- **Academic Scope:** FreshVision is an educational proof-of-concept and must not replace standard food safety inspection protocols.

---

## 19. Future Improvements
- **Multi-Class Commodity Classification:** Classify specific food types alongside freshness (e.g., *Fresh Orange* vs. *Spoiled Orange*).
- **Explainability with Grad-CAM:** Highlight the exact visual areas triggering the freshness classification using heatmaps.
- **Edge Deployment:** Export the model to TensorFlow Lite (`.tflite`) for deployment on Android and Raspberry Pi devices.
- **Shelf-Life Estimation:** Predict remaining usable days through regression modeling.

---

## 20. Author & Acknowledgments

- **Author:** ABHISHEK KASHYAP B.Tech Computer Science & Engineering Student
- **Institution:** COER UNIVERSITY
- Department of Computer Science & Engineering
- **Mentorship:** Deep Learning & Computer Vision Coursework
- **Pre-trained Model:** MobileNetV2 courtesy of Google Research & Keras Team

---

## ⚠️ Educational Disclaimer
> **FreshVision is an educational computer-vision project. Predictions may be incorrect and should not be used as the only method for determining whether food is safe to eat.**
