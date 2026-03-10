import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

try:
    from pyearth import Earth
    _MARS_AVAILABLE = True
except ImportError:
    _MARS_AVAILABLE = False


CLASSIFIER_REGISTRY = {
    'J48': {
        'description': 'J48 (C4.5 approx)',
        'class': DecisionTreeClassifier,
        'params': {'criterion': 'entropy', 'min_samples_leaf': 2, 'random_state': 42},
    },
    'KNN': {
        'description': 'K-Nearest Neighbors',
        'class': KNeighborsClassifier,
        'params': {'n_neighbors': 5},
    },
    'CART': {
        'description': 'CART Decision Tree',
        'class': DecisionTreeClassifier,
        'params': {'criterion': 'gini', 'ccp_alpha': 0.01, 'random_state': 42},
    },
    'Radial_SVM': {
        'description': 'RBF SVM (sigma=1, C=0.25)',
        'class': SVC,
        'params': {'kernel': 'rbf', 'gamma': 1.0, 'C': 0.25, 'random_state': 42},
    },
    'Linear_SVM': {
        'description': 'Linear SVM (C=0.25)',
        'class': SVC,
        'params': {'kernel': 'linear', 'C': 0.25, 'random_state': 42},
    },
    'NNet': {
        'description': 'Neural Network (size=3, decay=0.001)',
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (3,),
            'alpha': 0.001,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42,
        },
    },
    'NaiveBayes': {
        'description': 'Gaussian Naive Bayes',
        'class': GaussianNB,
        'params': {},
    },
    'LogisticReg': {
        'description': 'Logistic Regression',
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'solver': 'lbfgs', 'random_state': 42},
    },
}

if _MARS_AVAILABLE:
    CLASSIFIER_REGISTRY['MARS'] = {
        'description': 'MARS (nprune=10, degree=1)',
        'class': Earth,
        'params': {'max_terms': 10, 'max_degree': 1},
    }


def get_classifier(name):
    """Return a fresh classifier instance by name."""
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {list(CLASSIFIER_REGISTRY)}"
        )
    cfg = CLASSIFIER_REGISTRY[name]
    return cfg['class'](**cfg['params'])


def list_classifiers():
    """List available classifiers."""
    return {k: v['description'] for k, v in CLASSIFIER_REGISTRY.items()}
