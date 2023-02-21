import numpy as np

def perceptron(theta, theta0, x, y):
    result = y*(np.dot(theta,x)+theta0)
    if result <= 0:
        theta = theta + (y*x)
        theta0 = theta0 + y
        print(f"Mistake, result:{result}\nNew theta: {theta}\nNew theta0: {theta0}")
        return theta, theta0
    else:
        return theta, theta0
    


# x1
x1 = np.array([-4, 2])
y1 = 1

#x2
x2 = np.array([-2, 1])
y2 = 1


#x3
x3 = np.array([-1, -1])
y3 = -1

#x4
x4 = np.array([2,2])
y4 = -1

#
# x5
x5 = np.array([1, -2])
y5 = -1

### Starting point
theta = np.array([-1,1])
theta0 = -.1
### Starting from x1:
theta, theta0 = perceptron(theta, theta0, x1, y1)
theta, theta0 = perceptron(theta, theta0, x2, y2)
theta, theta0 = perceptron(theta, theta0, x3, y3)
theta, theta0 = perceptron(theta, theta0, x4, y4)
theta, theta0 = perceptron(theta, theta0, x5, y5)
theta, theta0 = perceptron(theta, theta0, x1, y1)
theta, theta0 = perceptron(theta, theta0, x2, y2)
theta, theta0 = perceptron(theta, theta0, x3, y3)
theta, theta0 = perceptron(theta, theta0, x4, y4)
theta, theta0 = perceptron(theta, theta0, x5, y5)
