import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    def denominator(x,theta,t):
        c = np.dot(np.max(theta, axis=0),x)/t # find max values in theta column wise, compute dot prod and devide by temp_parameter
        theta_dot = np.e ** ((np.dot(theta,x)/t) - c) ## create vector with e^exponent
        denom = 1/np.sum(theta_dot) # compute denominator
        probs = np.dot(denom, theta_dot) # compute softmax for x
        return probs


    # c = np.dot(np.max(theta,axis=0),X.T)/temp_parameter
    # theta_dot = np.e ** ((np.dot(theta, X.T) / temp_parameter) - c)
    # denom = 1 / np.sum(theta_dot)  # compute denominator
    # probs = np.dot(denom, theta_dot)  # compute softmax for x
    # H = probs
    H =np.apply_along_axis(denominator, axis=1, arr=X, theta=theta, t=temp_parameter).T # apply the function row wise of X

    return H



def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HEREx[x!=0] = np.log(x[x!=0])
    probs = compute_probabilities(X=X,theta=theta,temp_parameter=temp_parameter)
    probs[probs !=0] = np.log(probs[probs!=0])
    # create a 7x3 matrix
    iverson = np.zeros((probs.shape[0], probs.shape[1]))
    # create a row index array based on y
    row_index = Y.reshape((-1, 1))
    # create a column index array based on the range of columns in iverson
    col_index = np.arange(iverson.shape[1])
    col_index = col_index.reshape((-1,1))
    # use the row and column index arrays to assign 1 to the corresponding elements of iverson
    iverson[row_index, col_index] = 1
    first_term = np.multiply(probs,iverson)
    first_term = np.sum(first_term) * (-(1/X.shape[0]))
    second_term = np.sum(theta**2)
    second_term = (lambda_factor/2) * second_term
    return first_term + second_term




# def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
#     """
#     Runs one step of batch gradient descent
#
#     Args:
#         X - (n, d) NumPy array (n datapoints each with d features)
#         Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
#             data point
#         theta - (k, d) NumPy array, where row j represents the parameters of our
#                 model for label j
#         alpha - the learning rate (scalar)
#         lambda_factor - the regularization constant (scalar)
#         temp_parameter - the temperature parameter of softmax function (scalar)
#
#     Returns:
#         theta - (k, d) NumPy array that is the final value of parameters theta
#     """
#     probs = compute_probabilities(X=X,theta=theta,temp_parameter=temp_parameter)
#     iverson = np.zeros((probs.shape[0], probs.shape[1]))
#     # create a row index array based on y
#     row_index = Y.reshape((-1, 1))
#     # create a column index array based on the range of columns in iverson
#     col_index = np.arange(iverson.shape[1])
#     col_index = col_index.reshape((-1,1))
#     # use the row and column index arrays to assign 1 to the corresponding elements of iverson
#     iverson[row_index, col_index] = 1
#     diff = iverson - probs
#     def first_term(diff_row):
#         first = np.sum(X * diff_row.reshape((-1,1)), axis =0)
#         return first
#
#     first = np.apply_along_axis(first_term, axis=1,arr=diff)
#     first = first * (-(1)/(temp_parameter*X.shape[0]))
#     second = lambda_factor * theta
#     J = first + second
#     newtheta = theta - (alpha * J)
#     return newtheta

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    # Version provided by course:
    itemp=1./temp_parameter
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    probabilities = compute_probabilities(X, theta, temp_parameter)
    # M[i][j] = 1 if y^(j) = i and 0 otherwise.
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    non_regularized_gradient = np.dot(M-probabilities, X)
    non_regularized_gradient *= -itemp/num_examples
    return theta - alpha * (non_regularized_gradient + lambda_factor * theta)

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    return train_y_mod3, test_y_mod3

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    assigned_labels = assigned_labels % 3
    return 1 - np.mean(assigned_labels == Y)


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
