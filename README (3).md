# Pneumonia Detection from Chest X-Rays using Deep Learning

## Business Context

Pneumonia is one of the leading causes of morbidity and mortality worldwide, particularly affecting children under five years and elderly populations. According to the WHO, pneumonia accounts for a significant percentage of deaths caused by infectious diseases. Early detection and timely treatment are critical to improving patient outcomes, yet current diagnostic methods present real challenges.

The most common diagnostic method combines clinical evaluation with chest X-ray imaging. However, accurate X-ray interpretation requires skilled radiologists, whose availability is limited in many regions — especially rural or resource-constrained healthcare settings. Even where radiologists are available, fatigue, high patient load, and human error can affect diagnostic accuracy and consistency, leading to delayed treatment, misdiagnosis, or unnecessary antibiotic use.

This project explores how machine learning and deep learning can support automated chest X-ray analysis — acting as a decision-support tool for healthcare professionals, reducing diagnostic workload, improving consistency, and enabling timely intervention in areas with limited medical expertise.

## Objective

Build an intelligent, automated system capable of detecting pneumonia (lung opacity) from chest X-ray images. The system aims to:

1. Accurately classify chest X-rays into pneumonia-positive (Lung Opacity) and pneumonia-negative (Normal) cases.
2. Assist healthcare professionals with a reliable second opinion that reduces diagnostic errors and variability.
3. Improve efficiency by delivering faster diagnoses and reducing radiologist workload.
4. Enhance accessibility via a scalable solution deployable in hospitals, clinics, or resource-limited settings.
5. Support global health efforts through earlier detection, lower pneumonia-related mortality, and more appropriate antibiotic use.

## Dataset

- **Source:** RSNA Pneumonia Detection Challenge-style dataset (`stage_2_train_images.zip`, `stage_2_test_images.zip`, `stage_2_train_labels.csv`, and a detailed class-info file), stored as DICOM (`.dcm`) images.
- **Classes (original):** `Lung Opacity`, `Normal`, `No Lung Opacity / Not Normal` (a third class — abnormal but not pneumonia).
- **Label handling:** The `"No Lung Opacity / Not Normal"` class was merged into `"Normal"`, since the business goal is a binary pneumonia-positive vs. pneumonia-negative decision, and that abnormal-but-non-pneumonia class was the hardest for the model to separate correctly.
- **Target:** Binary `Target` column (0 = Normal, 1 = Lung Opacity/Pneumonia).
- **Class imbalance:** The dataset is imbalanced toward the Normal class; this is addressed with combined over/under-sampling (`RandomOverSampler` + `RandomUnderSampler` from `imbalanced-learn`) on the training split only.

## Methodology

### 1. Data Preparation
- Extract DICOM images from the provided archives.
- Merge `class_info` and `labels` CSVs on `patientId`; drop duplicate patient records.
- Resize images to `256 × 256` and normalize pixel values to `[0, 1]` (a fixed `/255.0` operation applied before the train/val/test split — safe, since it isn't learned from training statistics).
- Images are already single-channel grayscale (DICOM), so no RGB→grayscale conversion is needed.
- Split into train / validation / test sets.
- Balance the training (and validation) sets via oversampling the minority class and undersampling the majority class.

### 2. Exploratory Data Analysis
- Visualize sample X-rays per class.
- Analyze class distribution/imbalance before and after remapping the third class.

### 3. Model Building
Four model configurations were trained and compared, all using TensorFlow/Keras with `tf.data` pipelines:

| Model | Description |
|---|---|
| **Base CNN (from scratch)** | 3-block Conv2D → BatchNorm → ReLU → MaxPool → Dropout architecture with GlobalAveragePooling2D head. |
| **Base CNN + VGG16 (transfer learning)** | VGG16 backbone with a custom classification head. |
| **Base CNN + Frozen EfficientNetB0** | EfficientNetB0 backbone (frozen), custom Dense(128) classification head. |
| **EfficientNetB0 + Extra Layers / Two-Phase Fine-Tuning** | Phase 1: train the classifier head with the backbone frozen. Phase 2: unfreeze and fine-tune the top 40 backbone layers at a low learning rate. |

All models are evaluated on accuracy, precision, recall, F1-score, and AUC, with confusion matrices and training/validation curves plotted for each.

### 4. Threshold Optimization
The classification threshold for the best model is tuned using a precision-recall curve to maximize F1-score (rather than using the default 0.5 cutoff).

## Results Summary

| Model | Val Accuracy | F1 | Val AUC | Verdict |
|---|---|---|---|---|
| Base CNN (from scratch) | 0.623 | 0.716 | — | Retired — too shallow for 256×256 X-rays; high recall is misleading (over-predicts pneumonia) |
| Base CNN + VGG16 | — | 0.503 | — | Discarded — unstable/degenerate training, barely above random |
| **Base CNN + Frozen EfficientNetB0** | **0.779** | **0.788** | 0.852 | **Best overall** — only model with stable, monotonic improvement |
| EfficientNetB0 + Extra Dense Layers | — | — (lower precision) | — | Third place — marginal recall gain, no real improvement |
| Fine-tuned EfficientNetB0 (top 40 layers) | — | 0.779 | **0.856** | Highest AUC but lower F1 than the frozen model; gains plateau quickly |

**Key takeaway:** The simplest frozen EfficientNetB0 configuration (single Dense(128) head) is the best all-round model for production — highest F1 and accuracy, with stable training. Dataset scale, not architecture or fine-tuning strategy, is the primary bottleneck limiting further gains.

Why VGG16 underperformed: VGG16's ImageNet preprocessing (RGB→BGR conversion and subtracting ImageNet channel means) is incompatible with the dark, low-contrast pixel distribution of grayscale-converted chest X-rays, pushing most pixel values negative and breaking the backbone's frozen filters.

## Actionable Insights & Recommendations

1. **Extended classification heads improve recall at the cost of precision** — the right tradeoff depends on the clinical cost function (false negatives vs. false positives).
2. **Fine-tuning improves AUC but not F1** at this dataset scale; a larger labeled dataset is likely needed to unlock the full benefit of fine-tuning.
3. **The frozen EfficientNetB0 + Dense(128) head is recommended for production** — best F1, best accuracy, stable training behavior.
4. **Dataset scale is the primary bottleneck**, not model architecture or training strategy — the fine-tuning plateau and modest absolute metrics both point to a need for more labeled training data.
5. **The two-phase fine-tuning protocol (freeze → selective unfreeze) is methodologically sound** and should be retained as the fine-tuning approach if a larger dataset becomes available.

## Model Deployment

The best-performing model (frozen EfficientNetB0) is serialized and deployed as an interactive **Streamlit** web application:

- Users upload a chest X-ray (`.dcm`, `.png`, or `.jpg`/`.jpeg`).
- The app preprocesses the image (grayscale → resize → RGB → EfficientNet normalization) exactly as in training.
- It outputs a **Normal / Lung Opacity** prediction with a confidence score and probability bars.
- An adjustable classification threshold (auto-loaded from the F1-optimal threshold saved during training) lets users tune sensitivity.
- An optional **Grad-CAM heatmap** overlay highlights the image regions driving the prediction, for interpretability.
- A visible disclaimer marks the tool as **research/educational use only** — not a certified medical device, not for clinical diagnosis.

### Live Demo
🤗 Hosted on Hugging Face Spaces: **https://huggingface.co/spaces/Thaseen75/Pneumonia_Detection_Final**

## Tech Stack

- **Deep Learning:** TensorFlow / Keras, `tf-keras`
- **Transfer Learning Backbones:** VGG16, EfficientNetB0
- **Medical Imaging:** `pydicom`
- **Image Processing:** OpenCV, Pillow, scikit-image
- **Data Handling:** pandas, NumPy
- **Imbalance Handling:** `imbalanced-learn` (RandomOverSampler, RandomUnderSampler)
- **Visualization:** Matplotlib, Seaborn
- **Experiment Tracking:** MLflow
- **Deployment:** Streamlit, Docker
- **Hosting:** Hugging Face Hub (Spaces) via `huggingface_hub`

## Repository Structure

```
Pneumonia_Detection/
├── deployment/
│   ├── app.py              # Streamlit inference app (Grad-CAM, threshold slider)
│   ├── Dockerfile           # Containerized deployment definition
│   └── requirements.txt     # App dependencies
├── hosting/
│   └── hosting.py           # Script to push the app + model to Hugging Face Spaces
├── efficientnet_pneumonia_v1_0.keras   # Serialized best model
└── optimal_threshold.npy    # F1-optimal classification threshold
```

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   ```

2. **Install dependencies**
   ```bash
   pip install -r Pneumonia_Detection/deployment/requirements.txt
   ```

3. **Provide the dataset** (for retraining)
   The notebook expects:
   - `stage_2_train_images.zip`, `stage_2_test_images.zip` — DICOM image archives
   - `stage_2_train_labels.csv` — bounding-box/target labels
   - a detailed class-info CSV — three-class labels (merged with the above on `patientId`)

4. **Run the notebook**
   Open `Mohamed_Rifakath_Pneumonia_Detection_Notebook.ipynb` in Jupyter / Google Colab and run the cells sequentially to reproduce preprocessing, training, evaluation, and model serialization.

5. **Run the Streamlit app locally**
   ```bash
   cd Pneumonia_Detection/deployment
   streamlit run app.py
   ```
   Ensure `efficientnet_pneumonia_v1_0.keras` and `optimal_threshold.npy` are present in the same directory (or a `deployment_files/` subfolder).

6. **Deploy to Hugging Face Spaces** (optional)
   ```bash
   export HF_Pneumonia_Detection=<your_huggingface_token>
   python Pneumonia_Detection/hosting/hosting.py
   ```

## Disclaimer

This project is a prototype/academic capstone demonstrating deep learning for medical image classification. **It is not a certified medical device and must not be used for clinical diagnosis or patient care decisions.** Any real-world deployment would require regulatory review, clinical validation, and a substantially larger, more diverse labeled dataset.
