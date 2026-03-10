#!/usr/bin/env python3
"""
Main Execution Script for Tourism Sentiment Analysis Experiment

Reproduction of: "Advancing tourism sentiment analysis: a comparative evaluation
of traditional machine learning, deep learning, and transformer models on
imbalanced datasets" (Srianan et al., 2025)

Usage:
    python main.py --model all --oversampling all
    python main.py --model nb --oversampling smote
    python main.py --model lstm --oversampling none
"""

import argparse
import yaml
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.preprocessing import prepare_dataset, TextPreprocessor
from src.models.traditional_ml import TraditionalMLTrainer, TfidfConfig
from src.models.deep_learning import (
    Vocabulary, SentimentDataset, CNNClassifier, LSTMClassifier, GRUClassifier,
    DLTrainer, get_device
)
from src.utils.metrics import evaluate_model, print_results, ExperimentTracker
from src.utils.oversampling import apply_oversampling


def load_config(config_path: str = 'configs/config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(data_path: str) -> pd.DataFrame:
    """Load dataset from file."""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        return pd.read_json(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def run_traditional_ml_experiment(
    X_train, y_train, X_test, y_test,
    model_type: str,
    oversampling: str,
    config: dict,
    tracker: ExperimentTracker
):
    """Run experiment for traditional ML model."""
    print(f"\n{'='*60}")
    print(f"Running: {model_type.upper()} with {oversampling or 'no'} oversampling")
    print('='*60)

    # Apply oversampling to training data
    if oversampling and oversampling != 'none':
        # For TF-IDF based models, we need to vectorize first
        tfidf_config = TfidfConfig(
            ngram_range=tuple(config['tfidf']['ngram_range']),
            max_features=config['tfidf']['max_features']
        )
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = tfidf_config.create_vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_train_resampled, y_train_resampled = apply_oversampling(
            X_train_vec.toarray(), y_train, method=oversampling
        )
        # Train directly on vectorized data
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import time

        if model_type in ['naive_bayes', 'nb']:
            model = MultinomialNB()
        elif model_type in ['logistic_regression', 'lr']:
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'svm':
            model = SVC(kernel='linear', random_state=42)

        start_time = time.time()
        model.fit(X_train_resampled, y_train_resampled)
        training_time = time.time() - start_time

        X_test_vec = vectorizer.transform(X_test)
        predictions = model.predict(X_test_vec)
    else:
        # Train without oversampling using pipeline
        trainer = TraditionalMLTrainer(
            TfidfConfig(
                ngram_range=tuple(config['tfidf']['ngram_range']),
                max_features=config['tfidf']['max_features']
            )
        )
        pipeline, training_time = trainer.train(X_train, y_train, model_type)
        predictions = pipeline.predict(X_test)

    # Evaluate
    results = evaluate_model(y_test, predictions)
    results['training_time'] = training_time

    print_results(results, f"{model_type.upper()} ({oversampling})")

    # Track results
    tracker.add_result(model_type.upper(), oversampling or 'none', results, training_time)

    return results


def run_deep_learning_experiment(
    train_texts, train_labels, test_texts, test_labels,
    model_type: str,
    oversampling: str,
    config: dict,
    tracker: ExperimentTracker
):
    """Run experiment for deep learning model."""
    print(f"\n{'='*60}")
    print(f"Running: {model_type.upper()} with {oversampling or 'no'} oversampling")
    print('='*60)

    # Build vocabulary
    vocab = Vocabulary(max_size=config['deep_learning'].get('max_vocab_size', 50000))
    vocab.build(train_texts)

    # Create datasets
    max_length = config['deep_learning'].get('max_sequence_length', 256)

    # Handle oversampling for text data
    if oversampling and oversampling != 'none':
        # For text, we need to convert to indices first
        train_indices = np.array([vocab.encode(t, max_length) for t in train_texts])
        train_indices_resampled, train_labels_resampled = apply_oversampling(
            train_indices, np.array(train_labels), method=oversampling
        )
        # Convert back to dataset
        train_dataset = SentimentDataset(
            [' '.join([vocab.idx2word.get(idx, '<UNK>') for idx in seq if idx != 0])
             for seq in train_indices_resampled],
            train_labels_resampled.tolist(),
            vocab,
            max_length
        )
    else:
        train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)

    test_dataset = SentimentDataset(test_texts, test_labels, vocab, max_length)

    # Create model
    dl_config = config['deep_learning']
    vocab_size = len(vocab)

    if model_type == 'cnn':
        model = CNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=dl_config['embedding_dim'],
            num_classes=2,
            dropout=dl_config['dropout']
        )
    elif model_type == 'lstm':
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=dl_config['embedding_dim'],
            hidden_dim=dl_config['hidden_dim'],
            num_classes=2,
            dropout=dl_config['dropout']
        )
    elif model_type == 'gru':
        model = GRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=dl_config['embedding_dim'],
            hidden_dim=dl_config['hidden_dim'],
            num_classes=2,
            dropout=dl_config['dropout']
        )

    # Train
    device = get_device()
    trainer = DLTrainer(
        model,
        device,
        learning_rate=dl_config['learning_rate'],
        batch_size=dl_config['batch_size'],
        num_epochs=dl_config['num_epochs']
    )

    trained_model, training_time, history = trainer.train(train_dataset, verbose=True)

    # Predict
    predictions = trainer.predict(test_dataset)

    # Evaluate
    results = evaluate_model(np.array(test_labels), predictions)
    results['training_time'] = training_time

    print_results(results, f"{model_type.upper()} ({oversampling})")

    # Track results
    tracker.add_result(model_type.upper(), oversampling or 'none', results, training_time)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Tourism Sentiment Analysis Experiment'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        help='Model to run: nb, lr, svm, cnn, lstm, gru, roberta, or all'
    )
    parser.add_argument(
        '--oversampling',
        type=str,
        default='all',
        help='Oversampling method: none, smote, adasyn, random, or all'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to dataset file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/results/',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config = load_config(args.config)

    # Initialize tracker
    tracker = ExperimentTracker()

    # Define models and oversampling methods
    traditional_models = ['nb', 'lr', 'svm']
    dl_models = ['cnn', 'lstm', 'gru']
    oversampling_methods = ['none', 'smote', 'adasyn', 'random']

    if args.model == 'all':
        models_to_run = traditional_models + dl_models
    else:
        models_to_run = [args.model]

    if args.oversampling == 'all':
        oversamplings_to_run = oversampling_methods
    else:
        oversamplings_to_run = [args.oversampling]

    # Load or create sample data
    if args.data:
        print(f"Loading data from {args.data}...")
        df = load_data(args.data)
        train_df, test_df, stats = prepare_dataset(df)
    else:
        print("\nNo data file provided. Creating sample data for demonstration...")
        print("To run with real data, use: python main.py --data path/to/your/data.csv\n")

        # Create sample data
        sample_texts = [
            "Amazing beach! Crystal clear water and beautiful sunset views.",
            "Terrible experience. Dirty hotel and rude staff everywhere.",
            "Wonderful temple visit. So peaceful and spiritual.",
            "The food was okay but overpriced. Nothing special.",
            "Best vacation of my life! Will definitely return.",
            "Disappointing tour. Guide was not professional.",
            "Incredible diving experience. Saw many colorful fish!",
            "Worst hotel ever. Found bugs in the room.",
            "Beautiful scenery and friendly locals.",
            "Very crowded and touristy. Not worth the hype.",
        ] * 100  # Repeat for more data

        sample_ratings = [5, 1, 5, 3, 5, 2, 5, 1, 4, 2] * 100

        df = pd.DataFrame({
            'review_text': sample_texts,
            'rating': sample_ratings
        })
        train_df, test_df, stats = prepare_dataset(df)

    # Prepare data
    X_train = train_df['processed_text'].tolist()
    y_train = train_df['sentiment'].tolist()
    X_test = test_df['processed_text'].tolist()
    y_test = test_df['sentiment'].tolist()

    # Run experiments
    print("\n" + "="*60)
    print("STARTING EXPERIMENTS")
    print("="*60)

    for model in models_to_run:
        for oversampling in oversamplings_to_run:
            try:
                if model in traditional_models:
                    run_traditional_ml_experiment(
                        np.array(X_train), np.array(y_train),
                        np.array(X_test), np.array(y_test),
                        model, oversampling, config, tracker
                    )
                elif model in dl_models:
                    run_deep_learning_experiment(
                        X_train, y_train, X_test, y_test,
                        model, oversampling, config, tracker
                    )
                elif model == 'roberta':
                    print(f"\nRoBERTa experiments require transformers library.")
                    print("Run separately with: python src/train/train_transformer.py")
            except Exception as e:
                print(f"\nError running {model} with {oversampling}: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    tracker.print_summary()

    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_df = tracker.get_summary_dataframe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output, f'results_{timestamp}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
