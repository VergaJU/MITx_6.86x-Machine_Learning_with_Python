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

run_function()