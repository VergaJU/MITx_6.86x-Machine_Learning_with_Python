import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
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
    Return seuqre deviation devided by 2
    :param err: errors vectors
    :return: squared deviation vector
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

def read_ponts(data):
    """
    Read data points from lecture 6
    :param data: path to file
    :return: pd dataframe with label set as 1 and 0
    """
    df = pd.read_csv(data)
    df['label'] = df['labell'].map({'p': 1, 'n': 0})
    return df


def plot_points(df):
    """
    plot 2d scatterplot
    :param df: data points
    :return: plot
    """
    plt.scatter(x=df.x1, y=df.x2, c=df.label)
    plt.show()


def transform(df):
    """
    Transform feature vector and return 3d scatterplot
    :param df: data points
    :return: 3d scatterplot
    """
    df["transformed"] = (df.x1**2) + 2*(df.x2**2)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=df.x1, ys=df.x2, zs=df.transformed, c=df.label)
    plt.show()

def dimensionality(n,k):
    """
    evaluate dimensionality feature transformation
    :param n: number of points (dimension initial vector)
    :param k:degree
    :return:dimensions
    """
    dims = math.factorial(n+k-1)/(math.factorial(n-1)* math.factorial(k))
    print(f"Degree: {n}, Original dimension {k}")
    return dims

def repetition_dims(n,k):
    """
    repeat for all dimension from 1 to k
    :param n: number of points (dimension initial vector)
    :param k:degree
    :return:dimensions (sum)
    """
    dims = n
    print(dims)
    for i in range(2,k+1):
        dims += dimensionality(n,i)
    print(dims)
    return dims



def rbk(x,y):
    """
    Evaluate radial basis kernel function between 2 vectors of equal dimension
    :param x: vector
    :param y: vector
    :return: scalar
    """
    exponent = -.5 * (np.linalg.norm((x-y))**2)
    exponent = np.exp(exponent)
    return exponent
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
    #err = deviation(x,y,theta)
    #err = sq_deviation(err)
    #err = hinge_loss(err)
    #risk = emp_risk(err)
    #print(f"Empirical risk:{risk}")
    #df = read_ponts("points.csv")
    #plot_points(df)
    #transform(df)
    #repetition_dims(150, 3)
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    print(rbk(x,y))
