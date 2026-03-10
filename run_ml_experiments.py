#!/usr/bin/env python3
"""
Traditional ML Experiments for Tourism Sentiment Analysis

Models: Naive Bayes, Logistic Regression, SVM
Oversampling: None, SMOTE, ADASYN, RandomOverSampler

Based on: "Advancing tourism sentiment analysis" (Srianan et al., 2025)
"""

import argparse
import os
import re
import time
import warnings
from datetime import datetime
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix
)

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION (Paper Parameters)
# =============================================================================

CONFIG = {
    # Data split
    'test_size': 0.3,
    'random_state': 42,

    # TF-IDF
    'ngram_range': (1, 2),
    'max_features': 5000,

    # Oversampling
    'smote_k_neighbors': 5,
    'adasyn_n_neighbors': 5,
}


# =============================================================================
# TEXT PREPROCESSING (Paper Section 3.1.2)
# =============================================================================

def download_nltk_data():
    """Download required NLTK resources."""
    for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


class TextPreprocessor:
    """
    Text preprocessing pipeline following the paper's methodology.

    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove extra spaces
    4. Remove stopwords
    5. Lemmatize
    """

    def __init__(self):
        download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)

        # Step 3: Remove extra spaces
        text = ' '.join(text.split())

        # Step 4 & 5: Tokenize, remove stopwords, lemmatize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()

        tokens = [self.lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words]

        return ' '.join(tokens)

    def preprocess_batch(self, texts: pd.Series) -> pd.Series:
        return texts.apply(self.preprocess)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data(
    data_path: str = None,
    text_col: str = 'review_text',
    rating_col: str = 'rating'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load data and prepare for experiments.

    Returns:
        X_train, X_test, y_train, y_test, stats
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    else:
        print("No data file provided. Using sample data for demonstration...")
        df = create_sample_data()
        text_col = 'review_text'
        rating_col = 'rating'

    # Data cleaning
    print("\n1. Cleaning data...")
    initial_count = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=[text_col, rating_col])
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df = df.dropna(subset=[rating_col])
    print(f"   Removed {initial_count - len(df)} invalid rows")

    # Text preprocessing
    print("2. Preprocessing text...")
    preprocessor = TextPreprocessor()
    df['processed_text'] = preprocessor.preprocess_batch(df[text_col])

    # Sentiment labeling (1-3: Negative, 4-5: Positive)
    print("3. Labeling sentiment...")
    df['sentiment'] = (df[rating_col] >= 4).astype(int)

    # Train/test split
    print("4. Splitting data (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'].values,
        df['sentiment'].values,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=df['sentiment'].values
    )

    # Calculate statistics
    stats = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'positive_ratio': (df['sentiment'] == 1).mean(),
        'negative_ratio': (df['sentiment'] == 0).mean(),
    }

    print(f"\n   Dataset Statistics:")
    print(f"   - Total samples: {stats['total_samples']:,}")
    print(f"   - Training: {stats['train_samples']:,}")
    print(f"   - Test: {stats['test_samples']:,}")
    print(f"   - Positive: {stats['positive_ratio']:.2%}")
    print(f"   - Negative: {stats['negative_ratio']:.2%}")

    return X_train, X_test, y_train, y_test, stats


def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    positive_reviews = [
        "Amazing beach! Crystal clear water and beautiful sunset.",
        "Best vacation ever! Will definitely come back.",
        "Wonderful temple, very peaceful and spiritual.",
        "Incredible diving experience with colorful fish!",
        "Beautiful scenery and friendly locals everywhere.",
        "Perfect hotel with excellent service and clean rooms.",
        "Delicious local food, must try the seafood!",
        "Great tour guide, very knowledgeable and fun.",
        "Stunning views from the mountain top.",
        "Relaxing spa treatment, highly recommend!",
    ]

    negative_reviews = [
        "Terrible experience. Dirty hotel and rude staff.",
        "Worst vacation ever. Everything was overpriced.",
        "Disappointing tour, guide was unprofessional.",
        "Found bugs in the room, absolutely disgusting.",
        "Very crowded and touristy, not worth the hype.",
        "Food poisoning from the restaurant, awful!",
        "Scammed by taxi driver, be careful!",
        "Beach was polluted with trash everywhere.",
        "Long waiting times and poor organization.",
        "Noisy hotel room, couldn't sleep at all.",
    ]

    # Create imbalanced dataset (~80% positive, ~20% negative)
    n_positive = 800
    n_negative = 200

    reviews = (positive_reviews * (n_positive // 10)) + (negative_reviews * (n_negative // 10))
    ratings = [5] * (n_positive // 2) + [4] * (n_positive // 2) + \
              [1] * (n_negative // 3) + [2] * (n_negative // 3) + [3] * (n_negative - 2*(n_negative // 3))

    return pd.DataFrame({
        'review_text': reviews,
        'rating': ratings
    })


# =============================================================================
# FEATURE EXTRACTION (TF-IDF)
# =============================================================================

def create_tfidf_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF features.

    Paper parameters:
    - ngram_range: (1, 2)
    - max_features: 5000
    """
    vectorizer = TfidfVectorizer(
        ngram_range=CONFIG['ngram_range'],
        max_features=CONFIG['max_features']
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   TF-IDF shape: {X_train_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# =============================================================================
# OVERSAMPLING
# =============================================================================

def apply_oversampling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply oversampling to training data.

    Methods: 'none', 'smote', 'adasyn', 'random'
    """
    if method == 'none' or method is None:
        return X_train, y_train

    # Convert sparse to dense if needed
    if hasattr(X_train, 'toarray'):
        X_dense = X_train.toarray()
    else:
        X_dense = X_train

    if method == 'smote':
        sampler = SMOTE(
            k_neighbors=CONFIG['smote_k_neighbors'],
            random_state=CONFIG['random_state']
        )
    elif method == 'adasyn':
        sampler = ADASYN(
            n_neighbors=CONFIG['adasyn_n_neighbors'],
            random_state=CONFIG['random_state']
        )
    elif method == 'random':
        sampler = RandomOverSampler(
            random_state=CONFIG['random_state']
        )
    else:
        raise ValueError(f"Unknown oversampling method: {method}")

    try:
        X_resampled, y_resampled = sampler.fit_resample(X_dense, y_train)
        print(f"   Oversampling ({method}): {len(y_train)} → {len(y_resampled)} samples")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"   Warning: {method} failed ({e}), using original data")
        return X_dense, y_train


# =============================================================================
# MODELS
# =============================================================================

def create_model(model_name: str):
    """Create model instance."""
    if model_name == 'nb':
        return MultinomialNB()
    elif model_name == 'lr':
        return LogisticRegression(
            random_state=CONFIG['random_state'],
            max_iter=1000
        )
    elif model_name == 'svm':
        return SVC(
            kernel='linear',
            random_state=CONFIG['random_state']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
    }


def interpret_kappa(kappa: float) -> str:
    """Interpret Cohen's Kappa value."""
    if kappa < 0.00:
        return "Poor"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def print_results(results: Dict, model_name: str, oversampling: str, training_time: float):
    """Print formatted results."""
    print(f"\n   {'─'*50}")
    print(f"   {model_name.upper()} + {oversampling.upper()}")
    print(f"   {'─'*50}")
    print(f"   Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision:         {results['precision']:.4f}")
    print(f"   Recall:            {results['recall']:.4f}")
    print(f"   F1-Score:          {results['f1_score']:.4f}")
    print(f"   Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"   Cohen's Kappa:     {results['cohen_kappa']:.4f} ({interpret_kappa(results['cohen_kappa'])})")
    print(f"   Training Time:     {training_time:.2f}s")


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    X_train_tfidf,
    X_test_tfidf,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    oversampling: str
) -> Dict:
    """Run a single experiment."""

    # Apply oversampling
    X_train_resampled, y_train_resampled = apply_oversampling(
        X_train_tfidf, y_train, oversampling
    )

    # Handle negative values for NB (TF-IDF can have negative values after SMOTE)
    if model_name == 'nb' and oversampling != 'none':
        X_train_resampled = np.abs(X_train_resampled)

    # Create and train model
    model = create_model(model_name)

    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start_time

    # Predict
    if hasattr(X_test_tfidf, 'toarray'):
        X_test_dense = X_test_tfidf.toarray()
    else:
        X_test_dense = X_test_tfidf

    if model_name == 'nb' and oversampling != 'none':
        X_test_dense = np.abs(X_test_dense)

    predictions = model.predict(X_test_dense)

    # Evaluate
    results = evaluate(y_test, predictions)
    results['training_time'] = training_time
    results['model'] = model_name
    results['oversampling'] = oversampling

    # Print results
    print_results(results, model_name, oversampling, training_time)

    return results


def run_all_experiments(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """Run all experiments."""

    models = ['nb', 'lr', 'svm']
    oversamplings = ['none', 'smote', 'adasyn', 'random']

    # Create TF-IDF features
    print("\n5. Creating TF-IDF features...")
    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)

    all_results = []

    print("\n" + "="*60)
    print("RUNNING EXPERIMENTS")
    print("="*60)

    for model_name in models:
        for oversampling in oversamplings:
            print(f"\n>>> Experiment: {model_name.upper()} + {oversampling.upper()}")

            try:
                results = run_experiment(
                    X_train_tfidf, X_test_tfidf,
                    y_train, y_test,
                    model_name, oversampling
                )
                all_results.append(results)
            except Exception as e:
                print(f"   ERROR: {e}")

    return pd.DataFrame(all_results)


def print_summary(results_df: pd.DataFrame):
    """Print summary table."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Format for display
    summary = results_df[['model', 'oversampling', 'accuracy', 'f1_score', 'cohen_kappa', 'balanced_accuracy', 'training_time']].copy()
    summary['model'] = summary['model'].str.upper()
    summary['oversampling'] = summary['oversampling'].str.upper()
    summary['accuracy'] = summary['accuracy'].apply(lambda x: f"{x:.4f}")
    summary['f1_score'] = summary['f1_score'].apply(lambda x: f"{x:.4f}")
    summary['cohen_kappa'] = summary['cohen_kappa'].apply(lambda x: f"{x:.4f}")
    summary['balanced_accuracy'] = summary['balanced_accuracy'].apply(lambda x: f"{x:.4f}")
    summary['training_time'] = summary['training_time'].apply(lambda x: f"{x:.2f}s")

    print(summary.to_string(index=False))

    # Best model
    best_idx = results_df['cohen_kappa'].idxmax()
    best = results_df.loc[best_idx]
    print(f"\n{'─'*80}")
    print(f"BEST MODEL (by Cohen's Kappa): {best['model'].upper()} + {best['oversampling'].upper()}")
    print(f"  Kappa: {best['cohen_kappa']:.4f}")
    print(f"  Accuracy: {best['accuracy']:.4f}")
    print(f"  F1-Score: {best['f1_score']:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Traditional ML Experiments for Tourism Sentiment Analysis')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset (CSV or JSON)')
    parser.add_argument('--text-col', type=str, default='review_text', help='Name of text column')
    parser.add_argument('--rating-col', type=str, default='rating', help='Name of rating column')
    parser.add_argument('--output', type=str, default='outputs/results/', help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("TOURISM SENTIMENT ANALYSIS")
    print("Traditional ML Experiments")
    print("="*60)

    # Load and prepare data
    X_train, X_test, y_train, y_test, stats = load_and_prepare_data(
        args.data, args.text_col, args.rating_col
    )

    # Run all experiments
    results_df = run_all_experiments(X_train, X_test, y_train, y_test)

    # Print summary
    print_summary(results_df)

    # Save results
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output, f'ml_results_{timestamp}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
