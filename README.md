# ECG Arrhythmia Classification

**End-to-end machine learning project for cardiac arrhythmia detection from ECG signals**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

This project implements and compares multiple machine learning and deep learning architectures for classifying cardiac arrhythmias from ECG signals. Using the MIT-BIH and PTB datasets, the models achieve **99.1% macro recall** on binary classification—critical for minimizing false negatives in clinical diagnosis.

### Key Results

| Model | Dataset | Macro Recall | Accuracy |
|-------|---------|--------------|----------|
| **ResNet1D** | **PTB** | **99.1%** | **99.3%** |
| ResNet1D | MIT-BIH | 91.9% | 98.2% |
| CNN V2 | MIT-BIH | 86.6% | 98.1% |
| XGBoost | MIT-BIH | 89.8% | 92.5% |

## 📊 Datasets

### MIT-BIH Arrhythmia Database
- **109,446 samples** across 5 heartbeat classes
- **Classes**: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), Unknown (Q)
- **Challenge**: Severe class imbalance (Normal class >72% of samples)

### PTB Diagnostic ECG Database
- **14,552 samples** for binary classification
- **Classes**: Normal vs. Abnormal heartbeats
- **More balanced distribution**: ~40/60 split

**Dataset Source**: [Kaggle - Heartbeat Dataset](https://www.kaggle.com/shayanfazeli/heartbeat)

## 📁 Repository Structure

```
Hearbit_Classification_Datascientist/
│
├── data/                           # Raw and processed datasets
│   ├── mitbih_train.csv           # MIT-BIH training data
│   ├── mitbih_test.csv            # MIT-BIH test data
│   ├── ptbdb_normal.csv           # PTB normal heartbeats
│   └── ptbdb_abnormal.csv         # PTB abnormal heartbeats
│
├── code_model/                     # MIT-BIH model experiments
│   ├── CNN.ipynb                  # CNN implementation for MIT-BIH
│   ├── XGBoost.ipynb              # XGBoost implementation for MIT-BIH
│   ├── cnn_model.pkl              # Saved CNN model
│   └── xgb_model.json             # Saved XGBoost model
│
├── ML_Models.ipynb                 # Traditional ML baselines (RF, SVM, etc.)
├── CNN_V1.ipynb                    # Simple CNN architecture
├── CNN_V2.ipynb                    # Improved CNN with BatchNorm
├── Restnet1D.ipynb                 # ResNet1D architecture (best model)
│
├── cnn_model_v1.pkl               # Saved CNN V1 model
├── cnn_model_v2.pkl               # Saved CNN V2 model  
├── restnet1d_model.pkl            # Saved ResNet1D model
│
├── ECG Heartbeat Categorization Model report (1).pdf  # Full technical report
├── ECG_Classification (1).pptx                        # Presentation slides
│
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Installation

```bash
# Clone repository
git clone https://github.com/Saikot1997/Hearbit_Classification_Datascientist.git
cd Hearbit_Classification_Datascientist

# Install dependencies
pip install numpy pandas scikit-learn tensorflow keras xgboost
pip install matplotlib seaborn imbalanced-learn jupyter
```

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/shayanfazeli/heartbeat)
2. Extract files to `data/` directory:
   - `mitbih_train.csv`
   - `mitbih_test.csv`
   - `ptbdb_normal.csv`
   - `ptbdb_abnormal.csv`

### Running the Notebooks

**Start with traditional ML baselines:**
```bash
jupyter notebook ML_Models.ipynb
```

**Progress through deep learning architectures:**
```bash
jupyter notebook CNN_V1.ipynb      # Baseline CNN
jupyter notebook CNN_V2.ipynb      # Improved CNN
jupyter notebook Restnet1D.ipynb   # Best performing model
```

### Using Pre-trained Models

```python
import pickle
import numpy as np

# Load trained model
with open('restnet1d_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare ECG signal (187 features)
ecg_signal = np.array([...])  # Your ECG data
ecg_signal = ecg_signal.reshape(1, -1)

# Predict
prediction = model.predict(ecg_signal)
predicted_class = np.argmax(prediction)
```

## 🧠 Model Architectures

### 1. Traditional ML Models (`ML_Models.ipynb`)

Baseline classifiers for comparison:
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost** (best traditional ML: 89.8% recall)
- **Gradient Boosting**
- **Logistic Regression**

### 2. CNN V1 (`CNN_V1.ipynb`)

Simple convolutional baseline:
- 3 convolutional layers with MaxPooling
- Dense layers for classification
- **Result**: 82.8% macro recall

### 3. CNN V2 (`CNN_V2.ipynb`)

Enhanced architecture with regularization:
- Added **Batch Normalization** after conv layers
- **Dropout** layers (0.3-0.5) to prevent overfitting
- Improved activation functions
- **Result**: 86.6% macro recall (MIT-BIH)

### 4. ResNet1D (`Restnet1D.ipynb`) ⭐ Best Model

Adapted ResNet architecture for 1D time-series:
- **Residual blocks** with skip connections
- Prevents vanishing gradients in deep networks
- Global Average Pooling before final classification
- **Results**:
  - MIT-BIH: 91.9% macro recall, 98.2% accuracy
  - PTB: **99.1% macro recall, 99.3% accuracy**

**Why ResNet1D performs best:**
- Skip connections enable deeper networks
- Better gradient flow during training
- Captures both short and long-term temporal patterns in ECG signals

## 📈 Detailed Results

### Confusion Matrix (MIT-BIH - ResNet1D)

```
              Predicted
           N      S      V      F      Q
Actual N   17890  42     105    15     65
       S   23     481    38     3      11
       V   45     28     1342   12     21
       F   10     5      18     122    7
       Q   38     12     22     8      1528
```

**Key Insight**: Model maintains high recall across minority classes (S, V, F, Q), which is critical for detecting rare but dangerous arrhythmias.

### Per-Class Performance (MIT-BIH)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| N (Normal) | 99.4% | 98.9% | 99.1% | 18117 |
| S (Supraventricular) | 84.4% | 86.5% | 85.4% | 556 |
| V (Ventricular) | 87.8% | 93.1% | 90.4% | 1448 |
| F (Fusion) | 75.9% | 75.3% | 75.6% | 162 |
| Q (Unknown) | 93.6% | 95.5% | 94.5% | 1608 |

### PTB Dataset (Binary Classification)

All models achieve >96% performance on this more balanced dataset:

| Model | Macro Recall | Precision | Accuracy |
|-------|-------------|-----------|----------|
| ResNet1D | 99.1% | 99.3% | 99.3% |
| CNN V2 | 97.5% | 97.6% | 97.6% |
| XGBoost | 98.0% | 98.0% | 98.0% |

## 🔧 Methodology

### Data Preprocessing

**Class Imbalance Handling** (MIT-BIH):
- **RandomUnderSampler**: Reduce majority class (Normal) samples
- **SMOTE-Tomek**: Generate synthetic samples for minority classes
- **Target**: ~20,000 samples per class for balanced training

**Feature Engineering**:
- StandardScaler normalization
- Signal segmentation by heartbeat
- Feature extraction from time-series

### Training Strategy

- **Loss Function**: Weighted CrossEntropyLoss (inverse class frequency)
- **Optimizer**: Adam with learning rate scheduling
- **Batch Size**: 64-128
- **Epochs**: 30-50 with early stopping
- **Validation Split**: 20% of training data

### Evaluation Metrics

**Primary**: **Macro Average Recall**
- Treats all classes equally regardless of frequency
- Critical for detecting rare arrhythmias
- Prioritizes minimizing false negatives

**Why recall over accuracy?**  
In medical diagnosis, missing a disease (false negative) is more dangerous than a false alarm. A model with 99% accuracy but low recall for dangerous arrhythmias is clinically useless.

**Secondary Metrics**:
- Accuracy (overall correctness)
- Precision (false positive rate)
- F1-Score (harmonic mean)
- Confusion matrix analysis

## 📄 Documentation

- **[Technical Report](ECG%20Heartbeat%20Categorization%20Model%20report%20(1).pdf)**: Detailed methodology, experiments, and results
- **[Presentation](ECG_Classification%20(1).pptx)**: High-level overview and key findings

## 🔬 Key Findings

1. **Deep learning significantly outperforms traditional ML** for raw ECG signal classification
2. **Residual connections are crucial** - ResNet1D outperforms CNNs by 5-6% in recall
3. **Class imbalance requires sophisticated handling** - SMOTE-Tomek works well for time-series data
4. **Batch normalization improves generalization** - CNN V2 shows 4% improvement over V1
5. **Transfer learning is viable** - Models trained on MIT-BIH can adapt to PTB dataset
6. **Binary classification is easier** - All models achieve >96% on PTB vs 82-92% on MIT-BIH

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras
- **ML Libraries**: Scikit-learn, XGBoost
- **Imbalance Handling**: imbalanced-learn (SMOTE, Tomek links)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook

## 🔮 Future Enhancements

- [ ] Implement attention mechanisms for model interpretability
- [ ] Explore Transformer architectures (ECG-Former, Time Series Transformer)
- [ ] Multi-lead ECG processing (12-lead ECGs)
- [ ] Real-time classification system with streaming data
- [ ] Model deployment with FastAPI + Docker
- [ ] MLflow integration for experiment tracking
- [ ] Hyperparameter optimization with Optuna
- [ ] Grad-CAM visualization for clinical explainability

## 📚 Learning Resources

- **MIT-BIH Database**: [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- **PTB Database**: [PhysioNet](https://physionet.org/content/ptbdb/1.0.0/)
- **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **SMOTE Paper**: [Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **PhysioNet** for providing the MIT-BIH and PTB databases
- **Kaggle community** for dataset compilation and preprocessing
- **DataScientest training program** for project guidance

## 📧 Contact

**Saikot Das Joy**
- 📧 Email: saikotavi@gmail.com
- 💼 LinkedIn: [saikot-das-joy](https://linkedin.com/in/saikot-das-joy)
- 🐙 GitHub: [@Saikot1997](https://github.com/Saikot1997)

---

⭐ **If you find this project helpful, please consider starring the repository!**
