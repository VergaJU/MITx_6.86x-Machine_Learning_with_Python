import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *
import time

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
#plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################



def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error

"""
# Don't run this until the relevant functions in linear_regression.py have been fully implemented.
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=1))
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.1))
print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))

"""
#######################################################################
# 3. Support Vector Machine
#######################################################################


def run_svm_one_vs_rest_on_MNIST(C):
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x,C)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error

#print('SVM one vs. rest test_error, C=0.001:', run_svm_one_vs_rest_on_MNIST(0.001))
#print('SVM one vs. rest test_error, C=0.01:', run_svm_one_vs_rest_on_MNIST(0.01))
#print('SVM one vs. rest test_error, C=0.1:', run_svm_one_vs_rest_on_MNIST(0.1))
#print('SVM one vs. rest test_error, C=1:', run_svm_one_vs_rest_on_MNIST(1))
#print('SVM one vs. rest test_error, C=10:', run_svm_one_vs_rest_on_MNIST(10))


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################




def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()

    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta.pkl.gz")
    return test_error



#print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

def test_errors(temp_parameter=1):
    """
    Evaluate errors with normal labels and mod 3 labels with an already trained model
    :param temp_parameter:
    :return:
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta = read_pickle_data('./theta.pkl.gz')
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    test_error_mod3 = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)
    print(f"my test error  is: {test_error}\nMy test error mod3 is: {test_error_mod3}")


#test_errors()
#for t in [.5,1,2]:
#    print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=t))

temp_parameter = 1
#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    # YOUR CODE HERE
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error_mod3(test_x, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_mod3.pkl.gz")
    return test_error



#print('softmax test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))

#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##



n_components = 18

###Correction note:  the following 4 lines have been modified since release.



# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.

def run_softmax_on_MNIST_pca(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

    plot_PC(train_x[range(000, 100),], pcs, train_y[range(000, 100)],
            feature_means)  # feature_means added since release
    firstimage_reconstructed = reconstruct_PC(train_pca[0,], pcs, n_components, train_x,
                                              feature_means)  # feature_means added since release
    plot_images(firstimage_reconstructed)
    plot_images(train_x[0,])

    secondimage_reconstructed = reconstruct_PC(train_pca[1,], pcs, n_components, train_x,
                                               feature_means)  # feature_means added since release
    plot_images(secondimage_reconstructed)
    plot_images(train_x[1,])

    theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_pca18.pkl.gz")
    return test_error

#print('softmax test_error=', run_softmax_on_MNIST_pca(temp_parameter=1))



## Cubic K`ernel ##

def run_softmax_on_MNIST_pca_cubic(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    n_components = 10
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)


    train_cubic = cubic_features(train_pca)
    test_cubic = cubic_features(test_pca)

    theta, cost_function_history = softmax_regression(train_cubic, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_cubic, test_y, theta, temp_parameter)
    # Save the model parameters theta obtained from calling softmax_regression to disk.
    write_pickle_data(theta, "./theta_pca_cubic.pkl.gz")
    return test_error

#print('softmax test_error=', run_softmax_on_MNIST_pca_cubic(temp_parameter=1))


# TODO: Find the 10-dimensional PCA representation of the training and test set


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

#train_cube = cubic_features(train_pca10)
#test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).


from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss

def skl_svm_poly():
    """
    SVM with cubic polynomial kernel made with sklearn
    :return: error rate
    """
    n_components = 10
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    SVM = SVC(random_state=0, kernel='poly', degree=3)
    SVM.fit(train_pca, train_y)
    pred_y = SVM.predict(test_pca)
    error_rate = zero_one_loss(test_y, pred_y)

    return error_rate

#print(f"SVM with polynomial kernel error rate: {skl_svm_poly()}")

def skl_svm_rbf():
    """
    SVM with radial basis function kerlen made with sklearn
    :return: error rate
    """
    n_components = 10
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_centered, feature_means = center_data(train_x)
    pcs = principal_components(train_x_centered)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    SVM = SVC(random_state=0, kernel='rbf')
    SVM.fit(train_pca, train_y)
    pred_y = SVM.predict(test_pca)
    error_rate = zero_one_loss(test_y, pred_y)

    return error_rate

print(f"SVM with polynomial kernel error rate: {skl_svm_rbf()}")
