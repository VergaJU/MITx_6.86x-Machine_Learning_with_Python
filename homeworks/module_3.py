import numpy as np

x1 = np.array([-1, -1])
x2 = np.array([1, -1])
x3 = np.array([-1, 1])
x4 = np.array([1, 1])

#w1 = w11,w21
#w2 = w12, w22

w1 = [1,-1]
w2 = [-1,1]
w01 = 1
w02 = 1
points = [x1,x2,x3,x4]

def f(z):
    return z



def run_function():
    for point in range(len(points)):
        f1 = f(np.dot(w1,points[point]) + w01)
        f2 = f(np.dot(w2,points[point]) + w02)
        print(f"x{point+1}:\nf1: {round(f1,2)}\tf2: {round(f2,2)}")

#run_function()


def relu(nodes, weights, bias):
    z = np.sum(np.multiply(weights,nodes), axis=1) + bias
    return np.where(z > 0, z, 0)


def softmax(z):
    denom = np.e**z
    o1 = denom[0] / np.sum(denom)
    o2 = denom[1] / np.sum(denom)
    return [o1,o2]


x = np.array([3,14])
weights_1 = np.array([[1,0],[0,1],[-1,0],[0,-1]])
bias_1 = np.array([-1,-1,-1,-1])

hidden_layer = relu(x, weights_1, bias_1)
print(hidden_layer)
weights_2 = np.array([[1,1,1,1],[-1,-1,-1,-1]])
bias_2 = np.array([0,2])

hidden_layer = relu(hidden_layer, weights_2, bias_2)

print(softmax(hidden_layer))