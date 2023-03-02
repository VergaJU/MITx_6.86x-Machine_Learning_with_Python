import numpy as np
import project1.sentiment_analysis.project1 as utils

x = np.array([[1,0,1],[1,1,1],[1,1,-1],[-1,1,1]])
y = np.array([2,2.7,-0.7,2])
theta = np.array([0,1,2])

def deviation(x,y,theta):
    """
    Return errors vector
    :param x: feature matrix
    :param y: label
    :param theta: theta
    :return: scalar hinge loss
    """
    pred = np.dot(x, theta)
    err = y - pred
    return err

def sq_deviation(err):
    """

    :param err:
    :return:
    """
    err = err**2
    err = err/2
    return err

def hinge_loss(err):
    """
    Return hinge loss
    :param err: error vector
    :return: vector hinge loss
    """
    for e in range(len(err)):
        if err[e] >= 1:
            err[e] = 0
        else:
            err[e] = 1-err[e]
    return err

def emp_risk(err):
    """
    Return empirical risk (average deviation)
    :param err: lossh vector
    :return: scalar empirical risk
    """
    R = np.sum(err) / len(err)
    return R


''''
def squared_err(x,y,theta):
    """
    Return squared error of x*theta vs y
    :param x: feature vector
    :param y: label
    :param theta: theta vector
    :return: return scalar squared error
    """
    err = y-np.dot(x,theta)
    sq_err = err**2
    return sq_err


def R(x,y,theta):
    """
    return average squared error
    :param x: feature matrix
    :param y: label vector
    :param theta: theta
    :return: scalar average error
    """
    tot_err = 0
    for i in range(x.shape[0]):
        sq_err = squared_err(x[i],y[i],theta)
        tot_err += sq_err
        print(f"Tot err until now is {tot_err}")
    return tot_err/x.shape[0]
'''
if __name__ == "__main__":
    err = deviation(x,y,theta)
    #err = sq_deviation(err)
    err = hinge_loss(err)
    risk = emp_risk(err)
    print(f"Empirical risk:{risk}")
