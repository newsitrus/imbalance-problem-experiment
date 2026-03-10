# Implementation Plan: Tourism Sentiment Analysis Experiment Reproduction

## Paper Reference
**Title:** Advancing tourism sentiment analysis: a comparative evaluation of traditional machine learning, deep learning, and transformer models on imbalanced datasets
**Authors:** Sawitree Srianan, Aziz Nanthaamornphong, Chayanon Phucharoen
**Published:** Information Technology & Tourism (2025)

---

## 1. Project Overview

### 1.1 Objective
Reproduce the comparative evaluation of 8 sentiment classification models on tourism reviews:
- **Traditional ML:** Naive Bayes (NB), Logistic Regression (LR), Support Vector Machine (SVM)
- **Deep Learning:** CNN, LSTM, GRU
- **Transformer:** RoBERTa (pretrained), RoBERTa (fine-tuned)

### 1.2 Key Research Questions
1. How do models compare in classification performance, fairness, and reliability?
2. How effective are oversampling techniques (SMOTE, ADASYN, RandomOverSampler)?
3. What are the performance vs. efficiency trade-offs?

### 1.3 Expected Results (Target Metrics)
| Model | Accuracy | F1-Score | Cohen's Kappa | Balanced Accuracy |
|-------|----------|----------|---------------|-------------------|
| NB (baseline) | 82.35% | - | 0.0054 | 50.17% |
| LR | 90.25% | 94.28% | 0.6150 | 76.73% |
| SVM | - | - | 0.1041 | 53.31% |
| LSTM | 91.06% | - | 0.6846 | 83.09% |
| GRU | 90.82% | - | 0.6781 | 81%+ |
| RoBERTa (pretrained) | 86.07% | 91.85% | 0.4424 | 68.89% |
| RoBERTa (fine-tuned) | 92.31% | 95.34% | 0.7321 | 85.92% |

---

## 2. Project Structure

```
sentourism-experiment/
├── data/
│   ├── raw/                    # Original scraped data
│   ├── processed/              # Cleaned and preprocessed data
│   └── splits/                 # Train/test splits
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scraper.py          # TripAdvisor scraping (if needed)
│   │   ├── preprocessing.py    # Data cleaning pipeline
│   │   └── dataset.py          # PyTorch Dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional_ml.py   # NB, LR, SVM implementations
│   │   ├── deep_learning.py    # CNN, LSTM, GRU implementations
│   │   └── transformer.py      # RoBERTa implementations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── oversampling.py     # SMOTE, ADASYN, RandomOverSampler
│   │   └── visualization.py    # Plotting functions
│   └── train/
│       ├── __init__.py
│       ├── train_ml.py         # Training script for ML models
│       ├── train_dl.py         # Training script for DL models
│       └── train_transformer.py # Training script for RoBERTa
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_results_analysis.ipynb
├── configs/
│   └── config.yaml             # Hyperparameters and settings
├── outputs/
│   ├── models/                 # Saved model checkpoints
│   ├── results/                # Evaluation results
│   └── figures/                # Generated plots
├── requirements.txt
├── setup.py
└── main.py                     # Main execution script
```

---

## 3. Implementation Phases

### Phase 1: Environment Setup and Data Acquisition

#### 3.1.1 Dependencies Installation
```bash
# Create requirements.txt with:
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
torch>=2.0.0
torchtext>=0.15.0
transformers>=4.30.0
nltk>=3.8.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
jupyter>=1.0.0
```

#### 3.1.2 Data Acquisition Strategy
**Option A: Web Scraping (Original Method)**
- Scrape TripAdvisor reviews for Bali and Phuket tourist attractions
- Target: 505,980 English-language reviews (2010-2023)
- Required fields: review_text, rating (1-5), year

**Option B: Alternative Dataset**
- Use publicly available TripAdvisor datasets
- Consider: Kaggle TripAdvisor datasets or similar tourism review datasets

#### 3.1.3 Dataset Specifications
- **Total Reviews:** 505,980
- **Language:** English only
- **Locations:** Bali (Indonesia), Phuket (Thailand)
- **Time Period:** 2010-2023
- **Class Distribution:**
  - Positive (ratings 4-5): 415,478 (82.11%)
  - Negative (ratings 1-3): 89,989 (17.79%)

---

### Phase 2: Data Preprocessing Pipeline

#### 3.2.1 Data Cleaning Steps
```python
def clean_data(df):
    """
    1. Remove duplicate rows
    2. Remove rows with missing text or rating
    3. Convert rating to numeric
    4. Convert year to datetime
    """
    df = df.drop_duplicates()
    df = df.dropna(subset=['review_text', 'rating'])
    df['rating'] = pd.to_numeric(df['rating'])
    return df
```

#### 3.2.2 Text Preprocessing Steps
```python
def preprocess_text(text):
    """
    1. Lowercase text
    2. Remove punctuation
    3. Remove extra spaces
    4. Remove stopwords
    5. Lemmatize words
    """
    # Implementation using NLTK
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)
```

#### 3.2.3 Sentiment Labeling
```python
def assign_sentiment(rating):
    """
    Ratings 1-3 -> Negative (0)
    Ratings 4-5 -> Positive (1)
    """
    return 1 if rating >= 4 else 0
```

#### 3.2.4 Train-Test Split
- **Split Ratio:** 70% training, 30% testing
- **Random State:** 42 (for reproducibility)
- **Stratified:** Yes (maintain class distribution)

---

### Phase 3: Traditional ML Models Implementation

#### 3.3.1 Feature Extraction (TF-IDF)
```python
# TF-IDF Parameters
tfidf_params = {
    'ngram_range': (1, 2),
    'max_features': 5000
}
```

#### 3.3.2 Model Implementations

**Naive Bayes:**
```python
from sklearn.naive_bayes import MultinomialNB
nb_model = MultinomialNB()
```

**Logistic Regression:**
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
```

**Support Vector Machine:**
```python
from sklearn.svm import SVC
svm_model = SVC(kernel='linear', random_state=42)
```

---

### Phase 4: Deep Learning Models Implementation

#### 3.4.1 Common Hyperparameters
```python
dl_params = {
    'learning_rate': 1e-3,
    'batch_size': 64,
    'num_epochs': 5,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'embedding_dim': 128,
    'hidden_dim': 128  # For LSTM/GRU
}
```

#### 3.4.2 CNN Model Architecture
```python
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)
```

#### 3.4.3 LSTM Model Architecture
```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))
```

#### 3.4.4 GRU Model Architecture
```python
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))
```

---

### Phase 5: Transformer Model Implementation (RoBERTa)

#### 3.5.1 RoBERTa Parameters
```python
roberta_params = {
    'base_model': 'cardiffnlp/twitter-roberta-base-sentiment',
    'max_length': 512,
    'batch_size': 64,
    'num_epochs': 5,
    'learning_rate': 2e-5,
    'dropout': 0.1,
    'optimizer': 'AdamW'
}
```

#### 3.5.2 Pretrained RoBERTa (Zero-shot)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_pretrained(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    # Map 3-class output to binary
    return outputs.logits.argmax(dim=1)
```

#### 3.5.3 Fine-tuned RoBERTa
```python
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

model = RobertaForSequenceClassification.from_pretrained(
    'cardiffnlp/twitter-roberta-base-sentiment',
    num_labels=2,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)

training_args = TrainingArguments(
    output_dir='./outputs/roberta',
    num_train_epochs=5,
    per_device_train_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)
```

---

### Phase 6: Oversampling Techniques Implementation

#### 3.6.1 SMOTE
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 3.6.2 ADASYN
```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(n_neighbors=5, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
```

#### 3.6.3 RandomOverSampler
```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
```

---

### Phase 7: Evaluation Metrics Implementation

#### 3.7.1 Core Metrics
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
```

#### 3.7.2 Training Time Tracking
```python
import time

def train_with_timing(train_func, *args, **kwargs):
    start_time = time.time()
    model = train_func(*args, **kwargs)
    training_time = time.time() - start_time
    return model, training_time
```

---

### Phase 8: Experiment Execution Plan

#### 3.8.1 Experiment Matrix

| Experiment ID | Model | Oversampling | Config |
|---------------|-------|--------------|--------|
| EXP-001 | NB | None | Baseline |
| EXP-002 | NB | SMOTE | - |
| EXP-003 | NB | ADASYN | - |
| EXP-004 | NB | RandomOverSampler | - |
| EXP-005 | LR | None | Baseline |
| EXP-006 | LR | SMOTE | - |
| EXP-007 | LR | ADASYN | - |
| EXP-008 | LR | RandomOverSampler | - |
| EXP-009 | SVM | None | Baseline |
| EXP-010 | SVM | SMOTE | - |
| EXP-011 | SVM | ADASYN | - |
| EXP-012 | SVM | RandomOverSampler | - |
| EXP-013 | CNN | None | Baseline |
| EXP-014 | CNN | SMOTE | - |
| EXP-015 | CNN | ADASYN | - |
| EXP-016 | CNN | RandomOverSampler | - |
| EXP-017 | LSTM | None | Baseline |
| EXP-018 | LSTM | SMOTE | - |
| EXP-019 | LSTM | ADASYN | - |
| EXP-020 | LSTM | RandomOverSampler | - |
| EXP-021 | GRU | None | Baseline |
| EXP-022 | GRU | SMOTE | - |
| EXP-023 | GRU | ADASYN | - |
| EXP-024 | GRU | RandomOverSampler | - |
| EXP-025 | RoBERTa | Pretrained | Zero-shot |
| EXP-026 | RoBERTa | Fine-tuned | Task-specific |

#### 3.8.2 Execution Order
1. Run all Traditional ML experiments (EXP-001 to EXP-012)
2. Run all Deep Learning experiments (EXP-013 to EXP-024)
3. Run RoBERTa experiments (EXP-025 to EXP-026)
4. Compile results and generate visualizations

---

## 4. Configuration File

```yaml
# configs/config.yaml

# Data Configuration
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  train_test_split: 0.7
  random_state: 42

# Preprocessing
preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true
  lemmatize: true

# TF-IDF Configuration
tfidf:
  ngram_range: [1, 2]
  max_features: 5000

# Deep Learning Configuration
deep_learning:
  learning_rate: 0.001
  batch_size: 64
  num_epochs: 5
  embedding_dim: 128
  hidden_dim: 128
  optimizer: "Adam"
  loss: "CrossEntropyLoss"

# RoBERTa Configuration
roberta:
  model_name: "cardiffnlp/twitter-roberta-base-sentiment"
  max_length: 512
  batch_size: 64
  num_epochs: 5
  learning_rate: 0.00002
  dropout: 0.1
  optimizer: "AdamW"

# Oversampling Configuration
oversampling:
  smote:
    k_neighbors: 5
    random_state: 42
  adasyn:
    n_neighbors: 5
    random_state: 42
  random_oversampler:
    random_state: 42

# Output Configuration
output:
  models_dir: "outputs/models/"
  results_dir: "outputs/results/"
  figures_dir: "outputs/figures/"
```

---

## 5. Implementation Timeline

### Week 1: Setup and Data Preparation
- [ ] Set up project structure and environment
- [ ] Acquire/prepare dataset (scraping or alternative source)
- [ ] Implement data cleaning pipeline
- [ ] Implement text preprocessing pipeline
- [ ] Create train/test splits
- [ ] Verify data statistics match paper (82.11% positive, 17.89% negative)

### Week 2: Traditional ML Models
- [ ] Implement TF-IDF vectorization
- [ ] Implement Naive Bayes classifier
- [ ] Implement Logistic Regression classifier
- [ ] Implement SVM classifier
- [ ] Run baseline experiments (without oversampling)
- [ ] Run experiments with SMOTE, ADASYN, RandomOverSampler

### Week 3: Deep Learning Models
- [ ] Build vocabulary and prepare data loaders
- [ ] Implement CNN architecture
- [ ] Implement LSTM architecture
- [ ] Implement GRU architecture
- [ ] Train models with and without oversampling
- [ ] Implement early stopping and checkpointing

### Week 4: Transformer Models
- [ ] Set up HuggingFace Transformers environment
- [ ] Implement pretrained RoBERTa inference
- [ ] Implement fine-tuning pipeline
- [ ] Run experiments and record training times

### Week 5: Analysis and Documentation
- [ ] Compile all results
- [ ] Generate confusion matrices
- [ ] Create performance comparison visualizations
- [ ] Analyze oversampling effects
- [ ] Document findings and compare with original paper

---

## 6. Expected Computational Resources

### Hardware Requirements
- **Minimum:** 16GB RAM, GPU with 8GB VRAM (for DL/Transformer models)
- **Recommended:** 32GB RAM, GPU with 16GB+ VRAM

### Estimated Training Times (Based on Paper)
| Model | Baseline | With SMOTE |
|-------|----------|------------|
| NB | ~0.38s | ~0.66s |
| LR | ~30.71s | ~71.81s |
| SVM | ~1.28s | ~2.99s |
| CNN | ~3,936s (~1.1h) | ~6,764s (~1.9h) |
| LSTM | ~6,079s (~1.7h) | ~11,130s (~3.1h) |
| GRU | ~5,891s (~1.6h) | ~10,383s (~2.9h) |
| RoBERTa (pretrained) | ~1,025s (~17min) | N/A |
| RoBERTa (fine-tuned) | ~110,830s (~30.8h) | N/A |

---

## 7. Validation Checklist

### Data Validation
- [ ] Total reviews: ~505,980
- [ ] Class distribution: ~82% positive, ~18% negative
- [ ] Train/test split: 70/30

### Model Validation
- [ ] NB baseline kappa near 0 (confirming majority-class bias)
- [ ] LSTM/GRU kappa > 0.67
- [ ] RoBERTa fine-tuned kappa > 0.73
- [ ] SMOTE improves all model fairness metrics

### Results Validation
- [ ] Training times within expected ranges
- [ ] Confusion matrix patterns match paper
- [ ] SMOTE most effective for traditional ML
- [ ] Fine-tuning critical for RoBERTa performance

---

## 8. Notes and Considerations

### 8.1 Data Availability
The original paper notes data availability limitations. If the exact dataset is unavailable:
- Use publicly available TripAdvisor datasets
- Scrape similar tourism reviews from accessible sources
- Document any dataset differences

### 8.2 Reproducibility Tips
- Set all random seeds to 42
- Use fixed library versions
- Document GPU/hardware specifications
- Save all model checkpoints

### 8.3 Known Limitations (from Paper)
- Document-level sentiment may miss mixed sentiments
- Rating-based labeling may misalign with textual sentiment
- Rating 3 may reflect neutral sentiment (classified as negative)
- High computational cost for transformer fine-tuning

---

## 9. Quick Start Commands

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet

# 3. Prepare data
python src/data/preprocessing.py

# 4. Run experiments
python main.py --model all --oversampling all

# 5. Generate results
python src/utils/visualization.py
```

---

## 10. Contact and References

### Original Paper
- DOI: https://doi.org/10.1007/s40558-025-00336-0
- Journal: Information Technology & Tourism (2025)

### Key Libraries Documentation
- scikit-learn: https://scikit-learn.org/
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- imbalanced-learn: https://imbalanced-learn.org/
