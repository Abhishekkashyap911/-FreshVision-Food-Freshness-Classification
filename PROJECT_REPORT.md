# PROJECT REPORT: FreshVision — Food Freshness Classification

**Course:** B.Tech Computer Science & Engineering (Semester I)  
**Domain:** Artificial Intelligence / Deep Learning / Computer Vision  
**Project Title:** FreshVision — AI-Powered Food Freshness Classification System  
**Frameworks & Tools:** Python 3.13, TensorFlow/Keras, OpenCV, Streamlit, Scikit-Learn, Pillow  

---

## Abstract

Food spoilage causes massive global waste and poses significant health risks when spoiled perishables are inadvertently consumed. This project, **FreshVision**, presents an end-to-end Deep Learning and Computer Vision application designed to classify whether a given food item (fruits, vegetables, and perishable foods) is **Fresh** or **Spoiled**. 

Rather than training a deep convolutional neural network from scratch—which requires vast computational power and tens of thousands of images—the system utilizes **Transfer Learning** using the **MobileNetV2** architecture pre-trained on the ImageNet benchmark. The feature extraction layers are frozen to leverage general visual patterns (such as edges, contours, and color gradients), while a custom classification head (consisting of Global Average Pooling, Dropout regularization, and Dense layers) is trained specifically on food freshness indicators. The trained model is deployed into a clean, modern, and interactive **Streamlit web application** that accepts image uploads (`.jpg`, `.jpeg`, `.png`), displays real-time prediction confidence percentages, breaks down class probabilities, and explains the visual observations with clear educational disclaimers.

---

## 1. Introduction

Food quality inspection has traditionally relied upon human sensory observation (visual inspection, touch, and smell) or destructive laboratory chemical tests. Manual inspection is subjective, inconsistent across inspectors, and prone to fatigue. Destructive chemical tests, while accurate, render inspected samples unusable and cannot be performed in real-time at retail checkout or household refrigerators.

With rapid advancements in Deep Learning and Computer Vision, image classification models can now extract nuanced visual patterns from digital photographs. FreshVision introduces an accessible, automated approach that analyzes surface discoloration, texture degradation, and structural deformation to determine food freshness.

---

## 2. Background and Motivation

1. **Global Food Waste:** According to the United Nations Food and Agriculture Organization (FAO), roughly one-third of all food produced globally is lost or wasted, amounting to nearly 1.3 billion tons annually. Consumers frequently discard food prematurely due to uncertainty about freshness.
2. **Foodborne Illness:** Consuming moldy or rotten produce exposes individuals to bacterial pathogens (*Salmonella*, *E. coli*) and fungal mycotoxins, leading to gastrointestinal illness.
3. **Computer Vision Accessibility:** Modern smartphones and consumer webcams capture high-resolution imagery. Pairing accessible imaging hardware with lightweight, efficient neural network architectures makes automated quality checking viable on local computers without requiring expensive GPU clusters.

---

## 3. Problem Statement

To develop a lightweight, accurate, and beginner-friendly deep learning software system that accepts an arbitrary image of a food item and outputs:
- A categorical freshness classification (**Fresh** vs. **Spoiled**).
- A genuine confidence percentage reflecting model certainty.
- Class probability distribution.
- An intuitive observation summary, while maintaining full operational transparency and appropriate safety disclaimers.

---

## 4. Proposed Solution

FreshVision addresses the problem through a modular five-stage pipeline:
1. **Dataset Ingestion:** Structured organization of images into `train/`, `validation/`, and `test/` splits.
2. **Image Preprocessing & Augmentation:** Standardizing input dimensions to $224 \times 224$ pixels, scaling values to the $[-1, 1]$ interval, and applying random spatial transformations (rotations, flips, zoom) to prevent overfitting.
3. **Transfer Learning Model:** Utilizing a pre-trained MobileNetV2 backbone to extract deep convolutional features.
4. **Model Training & Validation:** Training the custom dense classification head using the Adam optimizer and categorical cross-entropy loss with early stopping and dynamic learning rate reduction.
5. **Interactive Streamlit Deployment:** Providing a browser-based user interface for image uploads, inference execution, and result visualization.

---

## 5. Project Objectives

- Implement an end-to-end Deep Learning pipeline using Python 3.13 and TensorFlow/Keras.
- Demonstrate Transfer Learning on a computer vision classification task.
- Validate the model using unbiased, held-out test data with genuine evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix).
- Construct a locally runnable, modern web user interface using Streamlit.
- Provide a clean, modular repository structure adhering to software engineering best practices.

---

## 6. System Architecture & Methodology

The application follows a clean modular architecture:

```text
[ Input Image (JPG/PNG) ]
           │
           ▼
[ Data Preprocessing (RGB convert, 224x224 resize, MobileNetV2 scaling) ]
           │
           ▼
[ Data Augmentation (RandomFlip, RandomRotation, RandomZoom) ]
           │
           ▼
[ MobileNetV2 Base (Pre-trained on ImageNet, Frozen Weights) ]
           │
           ▼
[ GlobalAveragePooling2D (7x7x1280 -> 1280 vector) ]
           │
           ▼
[ Batch Normalization + Dropout (0.3) ]
           │
           ▼
[ Dense Layer (128 units, ReLU) + Dropout (0.2) ]
           │
           ▼
[ Output Layer: Dense (2 units, Softmax) ]
           │
      ┌────┴──────────────────────────┐
      ▼                               ▼
[ Fresh Probability ]      [ Spoiled Probability ]
```

---

## 7. Dataset Structure & Partitioning

The dataset is partitioned into three mutually exclusive subsets to ensure unbiased evaluation:

```text
dataset/
├── train/        (70% of total data: used for gradient updates)
│   ├── fresh/
│   └── spoiled/
├── validation/   (15% of total data: used for hyperparameter tuning & early stopping)
│   ├── fresh/
│   └── spoiled/
└── test/         (15% of total data: used strictly for final evaluation)
    ├── fresh/
    └── spoiled/
```

- **Fresh Class:** High visual integrity, vibrant natural pigmentation, uniform skins, absence of lesions.
- **Spoiled Class:** Discoloration, bruising, mold development, shriveling, localized decay.

Public sources such as the *Kaggle Fruits Fresh and Rotten* dataset provide representative samples for this task.

---

## 8. Data Preprocessing

Input photographs vary widely in resolution, aspect ratio, color space, and lighting conditions. Preprocessing standardizes all inputs:

1. **Format Harmonization:** All images (RGB, RGBA, Grayscale) are explicitly converted to standard 3-channel RGB.
2. **Spatial Resizing:** Resized to $224 \times 224$ pixels using high-quality Lanczos interpolation.
3. **Normalization:** Input pixel values $[0, 255]$ are scaled to the range $[-1, 1]$ using the standard MobileNetV2 transformation:
   $$x_{normalized} = \frac{x}{127.5} - 1.0$$
4. **Batch Prefetching:** Batches are piped through TensorFlow's `tf.data.AUTOTUNE` prefetching pipeline to overlap data loading with model computation.

---

## 9. Data Augmentation

To reduce overfitting and improve model generalization, artificial variations are applied during the training phase only:
- **Random Horizontal Flip:** Invariant to camera orientation.
- **Random Rotation ($\pm 10\%$):** Simulates food items placed at arbitrary tilt angles.
- **Random Zoom ($\pm 10\%$):** Accounts for variable camera distances.

---

## 10. Transfer Learning with MobileNetV2

Training deep neural networks from scratch typically requires hundreds of thousands of labeled images and prolonged training on high-performance GPUs. Transfer learning circumvents this challenge:

- **Pre-trained Knowledge:** MobileNetV2 was trained on ImageNet (over 1.4 million images across 1,000 object categories).
- **Inverted Residuals & Linear Bottlenecks:** MobileNetV2 uses depthwise separable convolutions that drastically reduce computational complexity ($3.4 \times 10^6$ parameters) while preserving high representational capacity.
- **Feature Reuse:** The pre-trained weights represent universal visual primitives (edges, textures, shapes). Freezing these layers prevents catastrophic forgetting of foundational visual features.

---

## 11. Custom Classification Head

The features extracted by MobileNetV2 are mapped to the freshness categories using:

1. **GlobalAveragePooling2D:** Computes the spatial average across each of the 1,280 feature maps, collapsing the $7 \times 7 \times 1280$ tensor into a compact 1,280-dimensional feature vector.
2. **Batch Normalization:** Stabilizes the distribution of activations across mini-batches.
3. **Dropout (0.3):** Randomly zeroes 30% of inputs during training to prevent co-adaptation of neurons.
4. **Dense Layer (128 units, ReLU):** Learns non-linear combinations of feature representations.
5. **Dropout (0.2):** Secondary regularization.
6. **Dense Output Layer (2 units, Softmax):** Outputs a probability distribution:
   $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

---

## 12. Training Process & Optimization Strategy

- **Optimizer:** Adam (Adaptive Moment Estimation) with initial learning rate $\alpha = 10^{-4}$.
- **Loss Function:** Categorical Cross-Entropy:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
- **Training Callbacks:**
  - `ModelCheckpoint`: Automatically saves the best model weights based on minimum validation loss (`val_loss`).
  - `EarlyStopping`: Halts training if validation loss does not improve for 5 consecutive epochs, preventing overfitting.
  - `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.2 if validation loss plateaus for 3 epochs.

---

## 13. Evaluation Metrics

Model performance is evaluated on the held-out test dataset using genuine Scikit-Learn metrics:

- **Accuracy:** Overall proportion of correct classifications:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
- **Precision:** Proportion of positive identifications that were actually correct:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall (Sensitivity):** Proportion of actual positives identified correctly:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **F1-Score:** Harmonic mean of precision and recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix:** A $2 \times 2$ contingency table detailing True Positives, False Positives, True Negatives, and False Negatives.

*Note: All reported evaluation figures are derived directly from the test dataset without artificial inflation or hardcoded values.*

---

## 14. Streamlit Web Application

The user interface was built using **Streamlit**:
- **Image Uploader:** Accepts JPG, JPEG, and PNG files with immediate visual preview.
- **Live Inference Engine:** Preprocesses uploaded imagery and runs inference against `models/freshvision_model.keras`.
- **Classification Badges:** Color-coded status indicator (Green for Fresh, Red for Spoiled).
- **Probability Breakdown:** Visual progress indicators displaying individual class probabilities.
- **Safety Disclaimer:** Prominently placed notice regarding educational usage limitations.

---

## 15. Limitations and Assumptions

1. **Surface Inspection Only:** The model evaluates optical surface characteristics. Internal rot or bacterial contamination without external manifestation cannot be detected.
2. **Lighting and Angle Sensitivity:** Extreme glare, underexposure, or heavy occlusion can degrade prediction accuracy.
3. **Packaging Obstacles:** Food packaged in opaque or heavily textured plastic may impede accurate feature extraction.
4. **Educational Scope:** FreshVision is an academic demonstration project and cannot replace certified food safety procedures or laboratory inspections.

---

## 16. Future Scope

1. **Multi-Class Commodity Classification:** Expand the taxonomy to simultaneously identify the specific food type (e.g., *Fresh Apple*, *Spoiled Apple*, *Fresh Banana*, *Spoiled Banana*).
2. **Grad-CAM Explainability:** Integrate Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps highlighting the exact image regions responsible for the model's decision.
3. **Mobile & Edge Deployment:** Convert the `.keras` model into TensorFlow Lite (`.tflite`) format for on-device inference on Android/iOS smartphones and Raspberry Pi IoT edge cameras.
4. **Shelf-Life Estimation:** Extend classification to continuous regression estimating remaining days of shelf life.

---

## 17. Conclusion

FreshVision demonstrates the practical application of Deep Learning and Transfer Learning for computer vision tasks. By leveraging MobileNetV2 pretrained on ImageNet, the project achieves an efficient, locally runnable classification pipeline with modest computational requirements. Coupled with a user-friendly Streamlit web application, rigorous evaluation protocols, and clear engineering documentation, FreshVision fulfills all criteria for a successful college engineering project and GitHub open-source release.

---

## 18. References

1. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861.
2. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* IEEE/CVF CVPR.
3. Chollet, F. (2021). *Deep Learning with Python (2nd Edition).* Manning Publications.
4. Food and Agriculture Organization of the United Nations (FAO). *Food Loss and Waste Database.*
