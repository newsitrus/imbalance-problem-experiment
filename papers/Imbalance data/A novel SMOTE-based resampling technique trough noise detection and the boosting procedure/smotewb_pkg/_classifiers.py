"""Classifier registry — internal module."""

import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

try:
    from pyearth import Earth as _Earth
    _MARS_AVAILABLE = True
except ImportError:
    _MARS_AVAILABLE = False

# Registry: name -> (description, class, params)
# Parameters match paper's R package settings as closely as possible.
CLASSIFIER_REGISTRY = {
    'J48': (
        'J48 / C4.5 Decision Tree',
        DecisionTreeClassifier,
        {'criterion': 'entropy', 'min_samples_leaf': 2, 'random_state': 42},
    ),
    'KNN': (
        'K-Nearest Neighbors (k=5)',
        KNeighborsClassifier,
        {'n_neighbors': 5},
    ),
    'CART': (
        'CART Decision Tree',
        DecisionTreeClassifier,
        {'criterion': 'gini', 'ccp_alpha': 0.01, 'random_state': 42},
    ),
    'Radial_SVM': (
        'RBF SVM (sigma=1, C=0.25)',
        SVC,
        {'kernel': 'rbf', 'gamma': 1.0, 'C': 0.25, 'random_state': 42},
    ),
    'Linear_SVM': (
        'Linear SVM (C=0.25)',
        SVC,
        {'kernel': 'linear', 'C': 0.25, 'random_state': 42},
    ),
    'NNet': (
        'Neural Network (size=3, decay=0.001)',
        MLPClassifier,
        {'hidden_layer_sizes': (3,), 'alpha': 0.001, 'max_iter': 1000,
         'solver': 'lbfgs', 'random_state': 42},
    ),
    'NaiveBayes': (
        'Gaussian Naive Bayes',
        GaussianNB,
        {},
    ),
    'LogisticReg': (
        'Logistic Regression',
        LogisticRegression,
        {'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42},
    ),
}

if _MARS_AVAILABLE:
    CLASSIFIER_REGISTRY['MARS'] = (
        'MARS (nprune=10, degree=1)',
        _Earth,
        {'max_terms': 10, 'max_degree': 1},
    )


def get_classifier(name):
    """Return a fresh classifier instance by registry name."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY)}")
    _, cls, params = CLASSIFIER_REGISTRY[name]
    return cls(**params)


def list_classifiers():
    """Return dict of {name: description} for all available classifiers."""
    return {k: v[0] for k, v in CLASSIFIER_REGISTRY.items()}
