import numpy as np
import pytest
import pandas as pd
from matplotlib import pyplot as plt

import project1 as p1
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron

from utils import plot_toy_data


def create_random_dataset(seed=999):
    """
    This class generates a random dataset for training.
    always 2 features, two possible labels.
    Returns:

    """
    X, y = make_classification(
        n_samples=1000,  # 1000 observations
        n_features=2,  # 5 total features
        n_informative=2,  # 3 'useful' features
        n_repeated=0,
        n_redundant=0,
        n_classes=2,  # binary target/label
        random_state=seed  # if you want the same results as mine
    )

    # Create DataFrame with features as columns
    dataset = pd.DataFrame(X)
    # give custom names to the features
    dataset.columns = ['X1', 'X2']
    toy_features = dataset.to_numpy()
    # Now add the label as a column
    dataset['y'] = y
    toy_labels = y
    dataset.info()

    return toy_features, toy_labels


def test_perceptron():
    """This test will run the perceptron with 10 iterations.
    It runs the skitlearn score function to give a value"""
    max_iterations = 10
    toy_features, toy_labels = create_random_dataset()
    thetas_perceptron = p1.perceptron(toy_features, toy_labels, T=max_iterations)

    p = Perceptron(random_state=42, max_iter=max_iterations)
    p.fit(toy_features, toy_labels)
    sk_predictions_test = p.predict(toy_features)
    sk_test_score = accuracy_score(sk_predictions_test, toy_labels)
    # Forcing the perceptron to fit the previous value
    p.coef_[0] = thetas_perceptron[0]
    p.eta0 = thetas_perceptron[1]
    predictions_test = p.predict(toy_features)
    test_score = accuracy_score(predictions_test, toy_labels)
    print("score on test data: ", test_score)
    assert abs(test_score-sk_test_score) < 0.1, f"The score on p1.perceptron was" \
                                                f" {test_score}, but on skitlearn is {sk_test_score}"
    pass


def test_perceptron_average():
    """This test will run the perceptron with 10 iterations.
    It runs the skitlearn score function to give a value"""
    max_iterations = 10
    toy_features, toy_labels = create_random_dataset()
    thetas_perceptron = p1.average_perceptron(toy_features, toy_labels, T=max_iterations)

    p = Perceptron(random_state=42, max_iter=max_iterations)
    p.fit(toy_features, toy_labels)
    sk_predictions_test = p.predict(toy_features)
    sk_test_score = accuracy_score(sk_predictions_test, toy_labels)
    # Forcing the perceptron to fit the previous value
    p.coef_[0] = thetas_perceptron[0]
    p.eta0 = thetas_perceptron[1]
    predictions_test = p.predict(toy_features)
    test_score = accuracy_score(predictions_test, toy_labels)
    print("score on test data: ", test_score)
    assert abs(test_score-sk_test_score) < 0.1, f"The score on p1.perceptron was" \
                                                f" {test_score}, but on skitlearn is {sk_test_score}"
    pass

def test_perceptron_pegasos():
    """This test will run the perceptron with 10 iterations.
    It runs the skitlearn score function to give a value"""
    max_iterations = 10
    L = 0.2
    toy_features, toy_labels = create_random_dataset()
    thetas_perceptron = p1.pegasos(feature_matrix=toy_features, labels=toy_labels, T=max_iterations, L=L)

    p = Perceptron(random_state=42, max_iter=max_iterations)
    p.fit(toy_features, toy_labels)
    sk_predictions_test = p.predict(toy_features)
    sk_test_score = accuracy_score(sk_predictions_test, toy_labels)
    # Forcing the perceptron to fit the previous value
    p.coef_[0] = thetas_perceptron[0]
    p.eta0 = thetas_perceptron[1]
    predictions_test = p.predict(toy_features)
    test_score = accuracy_score(predictions_test, toy_labels)
    print("score on test data: ", test_score)
    assert abs(test_score-sk_test_score) < 0.1, f"The score on p1.perceptron was" \
                                                f" {test_score}, but on skitlearn is {sk_test_score}"
    pass

test_perceptron()
test_perceptron_average()
test_perceptron_pegasos()