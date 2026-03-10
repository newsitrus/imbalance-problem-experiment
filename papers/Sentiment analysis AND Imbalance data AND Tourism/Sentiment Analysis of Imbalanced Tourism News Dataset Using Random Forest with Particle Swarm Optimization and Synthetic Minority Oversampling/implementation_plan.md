# Implementation Plan: Sentiment Analysis of Imbalanced Tourism News Dataset Using Random Forest with PSO and SMOTE

## Overview

This plan replicates the full pipeline from the paper:
**Text Preprocessing → TF Feature Extraction → Binary PSO Feature Selection → SMOTE Oversampling → Random Forest Classification**

Two experiments are conducted:
1. **Experiment 1:** PSO + SMOTE + Random Forest
2. **Experiment 2:** PSO + Random Forest (no SMOTE)

Results are compared across 10 PSO population sizes (10, 20, ..., 100) measuring accuracy and computation time.

---

## Prerequisites

### Python Version
- Python 3.9+

### Required Libraries
```
scikit-learn        # Random Forest, train/test split, metrics
imbalanced-learn    # SMOTE implementation
pyswarms            # Particle Swarm Optimization (or custom binary PSO)
numpy
pandas
Sastrawi            # Indonesian language stemmer (PySastrawi)
nltk                # Tokenization, stopword support
matplotlib          # Visualization (accuracy/runtime charts)
```

### Installation
```bash
pip install scikit-learn imbalanced-learn pyswarms numpy pandas matplotlib nltk PySastrawi
```

---

## Dataset

### Important Note
The original dataset (200 Madura tourism news articles) is **private and not publicly available**. Two options:

#### Option A: Recreate a Similar Dataset (Recommended for faithful replication)
- Scrape ~200 Indonesian tourism news articles from online outlets (e.g., kompas.com travel section, detik.com travel)
- Focus on Madura Island / East Java tourism topics
- Manually label each article as **positive** or **negative** sentiment
- Target distribution: ~65% positive, ~35% negative (to replicate the imbalance)
- Store as CSV: columns `text`, `label` (1=positive, 0=negative)

#### Option B: Substitute with a Public Indonesian Sentiment Dataset
- Use an existing Indonesian sentiment dataset (e.g., from Kaggle or Indonesian NLP repositories)
- Subsample to ~200 instances with a 65/35 class imbalance
- Note: results will differ from the paper but the pipeline remains identical

### Expected Dataset Format
```
dataset/
  tourism_news.csv    # columns: id, text, label
```

---

## Phase 1: Text Preprocessing

### Objective
Clean raw Indonesian news text to extract meaningful tokens.

### Steps

#### 1.1 Case Folding
- Convert all text to lowercase
```python
text = text.lower()
```

#### 1.2 Filtering
- Remove URLs, HTML tags, special characters, numbers, punctuation
- Keep only alphabetic Indonesian words
```python
import re
text = re.sub(r'[^a-z\s]', '', text)
```

#### 1.3 Tokenizing
- Split text into individual word tokens
```python
tokens = text.split()
# or use nltk.word_tokenize
```

#### 1.4 Stop-word Removal
- Remove Indonesian stop words using a custom list or NLTK's Indonesian stopword list
- Supplement with Sastrawi or a community-maintained Indonesian stopword list
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('indonesian'))
tokens = [t for t in tokens if t not in stop_words]
```

#### 1.5 Stemming
- Apply Indonesian stemming using PySastrawi
```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.createStemmer()
tokens = [stemmer.stem(t) for t in tokens]
```

### Output
- List of cleaned token lists, one per document
- Store preprocessed text back into the dataframe

---

## Phase 2: Term Frequency (TF) Feature Extraction

### Objective
Convert preprocessed tokens into numerical TF feature vectors.

### Steps

#### 2.1 Build Vocabulary
- Collect all unique terms across all preprocessed documents

#### 2.2 Compute TF Vectors
- For each document, count the frequency of each term
- The paper mentions multiple TF variants (binary, raw tf, 1+log(tf), etc.) but uses raw TF
```python
from sklearn.feature_extraction.text import CountVectorizer

# Join tokens back to strings for CountVectorizer
corpus = [' '.join(tokens) for tokens in preprocessed_docs]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
```

### Output
- Feature matrix `X` of shape (n_documents, n_features)
- Label array `y` of shape (n_documents,)

---

## Phase 3: Binary PSO Feature Selection

### Objective
Use Binary Particle Swarm Optimization to select the most relevant features, reducing dimensionality and improving classification efficiency.

### Parameters (from paper)
| Parameter | Value |
|-----------|-------|
| C1 (cognitive) | 2 |
| C2 (social) | 2 |
| Population sizes | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| Inertia weight (w) | Initialize randomly or use standard (e.g., 0.7) |
| Iterations | Not specified in paper; use 50–100 |

### Steps

#### 3.1 Define Fitness Function
The fitness function from the paper (Equation 2):
```
fitness(X_i) = k1 * score + k2 * (num_of_features_x)^(-1)
```
- `score`: classification accuracy using selected features (use Random Forest with a quick evaluation, e.g., cross-validation on training set)
- `num_of_features_x`: number of selected features
- `k1`, `k2`: weighting constants (typically k1=0.9, k2=0.1 or similar — not specified in paper, use common defaults)

#### 3.2 Implement Binary PSO
Each particle is a binary vector of length = n_features. A value of 1 means the feature is selected; 0 means not selected.

**Velocity update** (Equation 5):
```
v_ij^(t+1) = w * v_ij^t + c1 * r1 * (Pbest_ij - x_ij^t) + c2 * r2 * (Gbest_j - x_ij^t)
```

**Sigmoid transfer function** (Equation 1):
```
S(v) = 1 / (1 + exp(-v))
if rand < S(v): x = 1
else: x = 0
```

#### 3.3 Implementation Options

**Option A: Using `pyswarms` library**
```python
from pyswarms.discrete import BinaryPSO

optimizer = BinaryPSO(
    n_particles=population_size,
    dimensions=n_features,
    options={'c1': 2, 'c2': 2, 'w': 0.7, 'k': 30, 'p': 2}
)
cost, pos = optimizer.optimize(fitness_function, iters=100)
selected_features = pos == 1
```

**Option B: Custom implementation** (for more control over the exact formulas from the paper)
- Implement the PSO loop manually following Equations 1–5

#### 3.4 Run PSO for Each Population Size
- For each population in [10, 20, 30, ..., 100]:
  - Run Binary PSO
  - Record selected features
  - Record computation time

### Output
- Boolean mask of selected features for each population size
- Reduced feature matrix `X_selected`

---

## Phase 4: SMOTE Oversampling (Experiment 1 Only)

### Objective
Balance the training set by generating synthetic samples for the minority (negative) class.

### Steps

#### 4.1 Split Data
- Split into training and testing sets
- Paper does not specify the exact split ratio; use 80/20 or 70/30
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 4.2 Apply SMOTE on Training Data Only
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### Output
- Balanced training set (equal positive and negative samples)
- Test set remains unchanged

---

## Phase 5: Random Forest Classification

### Objective
Train and evaluate Random Forest on both experimental configurations.

### Parameters (from paper)
| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 10 |

### Steps

#### 5.1 Train Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)  # or X_train_balanced for Experiment 1
```

#### 5.2 Evaluate
```python
from sklearn.metrics import accuracy_score
import time

start = time.time()
y_pred = rf.predict(X_test)
end = time.time()

accuracy = accuracy_score(y_test, y_pred)
runtime = end - start
```

#### 5.3 Record Results
- Store accuracy and runtime for each population size and experiment

---

## Phase 6: Experiment Loop

### Experiment 1: PSO + SMOTE + Random Forest
```
For each population_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    1. Run Binary PSO with population_size → get selected features
    2. Split data into train/test
    3. Apply SMOTE on training data
    4. Train Random Forest on balanced training data
    5. Predict on test data
    6. Record accuracy and total runtime
```

### Experiment 2: PSO + Random Forest (no SMOTE)
```
For each population_size in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    1. Run Binary PSO with population_size → get selected features
    2. Split data into train/test
    3. Train Random Forest on (imbalanced) training data
    4. Predict on test data
    5. Record accuracy and total runtime
```

### Expected Results (from paper)
| Experiment | Avg Accuracy | Peak Accuracy |
|-----------|-------------|--------------|
| PSO + SMOTE + RF | 83.75% | 90% |
| PSO + RF | 91% | 95% |

---

## Phase 7: Visualization & Analysis

### Charts to Reproduce

#### 7.1 Accuracy Comparison Chart (Fig. 6 in paper)
- Line chart: x-axis = population size (10–100), y-axis = accuracy
- Two lines: PSO+SMOTE+RF vs PSO+RF

#### 7.2 Runtime Comparison Chart (Fig. 7 in paper)
- Bar chart: x-axis = population size, y-axis = computation time (seconds)
- Two series: PSO+SMOTE+RF vs PSO+RF

#### 7.3 Class Distribution Pie Chart (Fig. 2 in paper)
- Show 65% positive / 35% negative split

#### 7.4 Individual Experiment Charts
- Accuracy per population for each experiment (Fig. 1, Fig. 5)
- Runtime per population for each experiment (Fig. 3, Fig. 4)

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(populations, acc_exp1, marker='o', label='PSO+SMOTE+RF')
plt.plot(populations, acc_exp2, marker='s', label='PSO+RF')
plt.xlabel('Population Size')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracy Values')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_comparison.png')
plt.show()
```

---

## Project File Structure

```
project/
├── data/
│   └── tourism_news.csv              # Dataset (to be created/obtained)
├── src/
│   ├── preprocessing.py              # Phase 1: text preprocessing
│   ├── feature_extraction.py         # Phase 2: TF computation
│   ├── binary_pso.py                 # Phase 3: Binary PSO feature selection
│   ├── smote_balancing.py            # Phase 4: SMOTE oversampling
│   ├── random_forest_classifier.py   # Phase 5: Random Forest training/evaluation
│   └── experiment_runner.py          # Phase 6: main experiment loop
├── results/
│   ├── experiment1_results.csv       # PSO+SMOTE+RF results
│   ├── experiment2_results.csv       # PSO+RF results
│   └── figures/                      # Generated charts
├── requirements.txt
└── README.md
```

---

## Estimated Timeline

| Phase | Task | Time Estimate |
|-------|------|---------------|
| 0 | Dataset collection/preparation | 3–5 days (if scraping) or 1–2 hours (if using substitute) |
| 1 | Text preprocessing (Indonesian NLP) | 2–4 hours |
| 2 | TF feature extraction | 30 min–1 hour |
| 3 | Binary PSO implementation | 2–4 hours |
| 4 | SMOTE integration | 30 min |
| 5 | Random Forest setup | 1 hour |
| 6 | Experiment loop + running all experiments | 1–2 hours |
| 7 | Visualization and analysis | 1–2 hours |
| **Total** | **Coding & experiments (excl. dataset)** | **~1–2 days** |
| **Total** | **Including dataset recreation** | **~4–7 days** |

---

## Key Considerations & Potential Issues

1. **Dataset unavailability**: The biggest challenge. Results will differ with a substitute dataset. If scraping from kompas.com, respect robots.txt and terms of service.

2. **Indonesian NLP**: Stemming and stopword removal require Indonesian-specific tools (Sastrawi). Ensure proper handling of Indonesian text encoding (UTF-8).

3. **Train/test split**: The paper does not specify the exact ratio. Try 80/20 with stratified split. Consider also testing 70/30.

4. **PSO iteration count**: Not explicitly stated in the paper. Start with 100 iterations and adjust based on convergence.

5. **PSO fitness weights (k1, k2)**: Not explicitly specified. Use common defaults (k1=0.9, k2=0.1) to balance accuracy vs feature reduction.

6. **Reproducibility**: Set random seeds throughout for consistent results. The paper's results may vary due to stochastic nature of PSO and Random Forest.

7. **SMOTE underperformance**: The paper found SMOTE actually hurt accuracy (83.75% vs 91%). This is likely due to the small dataset size — SMOTE may introduce noise with only 69 minority samples.
