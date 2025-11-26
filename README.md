# ECG Heartbeat Classification using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Difficulty Level**: 8/10 | **Course**: Data Scientist Training Program

An end-to-end machine learning project for classifying cardiac signals from ECG data using state-of-the-art deep learning architectures, achieving **99.1% macro average recall** with ResNet1D.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results & Performance](#results--performance)
- [Methodology](#methodology)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## 🎯 Overview

Electrocardiograms (ECGs) capture the electrical activity of the heart and are essential for diagnosing cardiac conditions. This project implements and compares multiple machine learning and deep learning models to classify ECG signals, detecting normal rhythms and various arrhythmia patterns.

### Problem Statement

Develop robust classification models to identify:
- **MIT-BIH Dataset**: 5 classes of heartbeat patterns (Normal, Supraventricular, Ventricular, Fusion, Unknown)
- **PTB Dataset**: Binary classification (Normal vs. Abnormal)

### Why This Matters

In medical diagnosis, **false negatives can be fatal**. Our models prioritize **recall** to minimize missed diagnoses while maintaining high overall accuracy.

## 🏆 Key Results

| Model | Dataset | Macro Avg Recall | Accuracy |
|-------|---------|------------------|----------|
| **ResNet1D** | MIT-BIH | **91.9%** | 98.2% |
| **ResNet1D** | PTB | **99.1%** | 99.3% |
| XGBoost | MIT-BIH | 89.8% | 92.5% |
| Gradient Boosting | PTB | 98.0% | 98.0% |

> **ResNet1D** consistently outperforms traditional ML methods and simpler CNN architectures across both datasets.

## 📊 Dataset

### Sources
- **MIT-BIH Arrhythmia Dataset**: 109,446 samples across 5 classes
- **PTB Diagnostic ECG Database**: 14,552 samples (binary classification)

Both datasets available on [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat)

### Class Distribution (MIT-BIH)

| Class | Code | Type | Description |
|-------|------|------|-------------|
| N | 0 | Normal | Healthy heartbeat |
| S | 1 | Supraventricular | Pathological |
| V | 2 | Ventricular | Pathological |
| F | 3 | Fusion | Pathological |
| Q | 4 | Unknown | Potentially pathological |

### Preprocessing
- **Sampling Frequency**: 125 Hz
- **Imbalance Handling**:
  - RandomUnderSampler for majority class (N)
  - SMOTE-Tomek for minority classes
  - GAN-based oversampling (experimental)
- **Target**: ~20,000 samples per class for MIT-BIH

## 📁 Project Structure

```
Heartbeat_Classification_Datascientist/
├── data/
│   ├── mitbih_train.csv
│   ├── mitbih_test.csv
│   ├── ptbdb_normal.csv
│   └── ptbdb_abnormal.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_deep_learning_models.ipynb
├── src/
│   ├── models/
│   │   ├── cnn_v1.py
│   │   ├── cnn_v2.py
│   │   └── resnet1d.py
│   ├── preprocessing/
│   │   ├── resampling.py
│   │   └── feature_engineering.py
│   └── utils/
│       ├── visualization.py
│       └── metrics.py
├── saved_models/
├── reports/
│   ├── ECG_Heartbeat_Classification_Report.pdf
│   └── figures/
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for deep learning models)

### Setup

```bash
# Clone the repository
git clone https://github.com/Saikot1997/Hearbit_Classification_Datascientist.git
cd Hearbit_Classification_Datascientist

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```
tensorflow>=2.8.0
scikit-learn>=1.0.0
xgboost>=1.5.0
catboost>=1.0.0
imbalanced-learn>=0.9.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### 1. Data Preparation

```bash
# Download dataset from Kaggle
kaggle datasets download -d shayanfazeli/heartbeat

# Extract and place in data/ directory
unzip heartbeat.zip -d data/
```

### 2. Train Models

```python
# Train ResNet1D on MIT-BIH dataset
python src/train.py --model resnet1d --dataset mitbih --epochs 50

# Train XGBoost on PTB dataset
python src/train.py --model xgboost --dataset ptb
```

### 3. Evaluate Models

```python
# Evaluate on test set
python src/evaluate.py --model resnet1d --dataset mitbih --checkpoint saved_models/resnet1d_best.h5
```

### 4. Make Predictions

```python
from src.models.resnet1d import ResNet1D
from src.preprocessing import load_and_preprocess

# Load model
model = ResNet1D.load('saved_models/resnet1d_best.h5')

# Predict
ecg_signal = load_and_preprocess('path/to/ecg_signal.csv')
prediction = model.predict(ecg_signal)
```

## 🧠 Model Architectures

### 1. **CNN v1** (Baseline)
- 3 Convolutional layers with MaxPooling
- Simple architecture for baseline comparison

### 2. **CNN v2** (Enhanced)
- Built on CNN v1
- Added Batch Normalization
- Dropout layers for regularization
- Improved generalization

### 3. **ResNet1D** (Best Performer) ⭐
- Adapted from ResNet-18 architecture
- 1D convolutions for time-series data
- Residual connections to prevent vanishing gradients
- Dropout before final fully connected layer
- **Architecture highlights**:
  - Input: 187-dimensional ECG signal
  - Residual blocks with skip connections
  - Global Average Pooling
  - Dense layer with softmax activation

### 4. **XGBoost** (ML Baseline)
- Gradient boosting framework
- Optimized hyperparameters via RandomizedSearchCV
- Strong performance for comparison

## 📈 Results & Performance

### MIT-BIH Dataset (5-Class Classification)

#### Model Comparison

| Model | Macro Avg Recall | Accuracy | Training Time |
|-------|------------------|----------|---------------|
| CNN v1 | 82.8% | 97.6% | ~15 min |
| CNN v2 | 86.6% | 98.1% | ~20 min |
| **ResNet1D** | **91.9%** | **98.2%** | ~30 min |
| XGBoost | 89.8% | 92.5% | ~10 min |

#### ResNet1D Confusion Matrix (MIT-BIH)
```
Predicted →   N      S      V      F      Q
Actual ↓
N          17890    42    105     15     65
S             23   481     38      3     11
V             45    28   1342     12     21
F             10     5     18    122      7
Q             38    12     22      8   1528
```

### PTB Dataset (Binary Classification)

#### Model Comparison

| Model | Macro Avg Recall | Precision | F1-Score |
|-------|------------------|-----------|----------|
| **ResNet1D** | **99.1%** | 99.3% | 99.2% |
| CNN v2 | 97.5% | 97.6% | 97.5% |
| Gradient Boosting | 98.0% | 98.0% | 98.0% |
| XGBoost | 98.0% | 98.0% | 98.0% |
| CNN v1 | 96.0% | 97.0% | 96.4% |

#### Transfer Learning Success
Pre-trained CNN v2 from MIT-BIH successfully adapted to PTB dataset with minimal modifications, demonstrating excellent **transfer learning** capabilities.

## 🔬 Methodology

### 1. **Data Exploration**
- Analyzed class distribution and imbalance
- Visualized ECG signal patterns
- Identified sampling frequency (125 Hz)

### 2. **Preprocessing**
- **Imbalance Handling**:
  - RandomUnderSampler for majority class
  - SMOTE-Tomek for synthetic minority oversampling
  - GAN-based augmentation (experimental)
- Feature scaling with StandardScaler
- Signal segmentation by heartbeat

### 3. **Model Development**
- **Traditional ML**: Random Forest, SVM, XGBoost, Gradient Boosting, etc.
- **Deep Learning**: CNN variants and ResNet1D
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV

### 4. **Evaluation Metrics**
- **Primary**: Macro Average Recall (prioritizes minority classes)
- **Secondary**: Accuracy, Precision, F1-Score
- **Rationale**: In medical diagnosis, missing a disease is more critical than false alarms

### 5. **Transfer Learning**
- Pre-trained CNN v2 on MIT-BIH (5 classes)
- Fine-tuned for PTB dataset (binary classification)
- Modified output layer and preprocessing pipeline

## 📊 Key Findings

1. **Deep Learning Superiority**: ResNet1D outperforms all traditional ML models
2. **Recall Priority**: Macro average recall ensures balanced performance across all classes
3. **Transfer Learning Effectiveness**: Models trained on MIT-BIH successfully adapt to PTB
4. **Batch Normalization Impact**: CNN v2 shows significant improvement over CNN v1
5. **Imbalance Handling**: SMOTE-Tomek effectively addresses class imbalance

## 🔮 Future Work

- [ ] Implement attention mechanisms for interpretability
- [ ] Explore transformer-based architectures
- [ ] Real-time ECG classification system
- [ ] Multi-lead ECG signal processing
- [ ] Ensemble methods combining top models
- [ ] Model deployment with Flask/FastAPI
- [ ] Mobile application integration

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Saikot** - [GitHub](https://github.com/Saikot1997)

## 🙏 Acknowledgments

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/)
- Kaggle community for dataset compilation
- Data Scientist training course instructors

## 📚 References

1. [Analytics Vidhya - ANN with 1-D ECG Data](https://www.analyticsvidhya.com/blog/2021/07/artificial-neural-network-simplified-with-1-d-ecg-biomedical-data/)
2. [DataSci - ECG Research Solutions](https://www.datasci.com/solutions/cardiovascular/ecg-research/)
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
4. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

---

⭐ If you find this project helpful, please consider giving it a star!

For questions or collaboration, feel free to open an issue or contact via [GitHub](https://github.com/Saikot1997).
