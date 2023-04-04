import os
import sys
import time
import traceback
import numpy as np
import neural_nets


sys.path.append("..")
import utils

verbose = False

epsilon = 1e-6

def green(s):
    return '\033[1;32m%s\033[m' % s

def yellow(s):
    return '\033[1;33m%s\033[m' % s

def red(s):
    return '\033[1;31m%s\033[m' % s

def log(*m):
    print(" ".join(map(str, m)))

def log_exit(*m):
    log(red("ERROR:"), *m)
    exit(1)

def check_number(ex_name, f, exp_res, *args):
    try:
        res = f(*args)
    except NotImplementedError:
        log(red("FAIL"), ex_name, ": not implemented")
        return True
    if not np.isreal(res):
        log(red("FAIL"), ex_name, ": does not return a real number, type: ", type(res))
        return True
    if not -epsilon < res - exp_res < epsilon:
        log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        return True


def check_ReLu():
    ex_name = "ReLu"
    x = 10
    exp_res = 10
    if check_number(
            ex_name, neural_nets.rectified_linear_unit,
            exp_res, x):
        return

    x = -1
    exp_res = 0

    if check_number(
            ex_name, neural_nets.rectified_linear_unit,
            exp_res, x):
        return

    x = 0
    exp_res = 0

    if check_number(
            ex_name, neural_nets.rectified_linear_unit,
            exp_res, x):
        return

    log(green("PASS"), ex_name, "")

def check_ReLu_derivative():
    ex_name = " derivative"
    x = 10
    exp_res = 1
    if check_number(
            ex_name, neural_nets.rectified_linear_unit_derivative,
            exp_res, x):
        return

    x = -1
    exp_res = 0

    if check_number(
            ex_name, neural_nets.rectified_linear_unit_derivative,
            exp_res, x):
        return

    x = 0
    exp_res = 0

    if check_number(
            ex_name, neural_nets.rectified_linear_unit_derivative,
            exp_res, x):
        return

    log(green("PASS"), ex_name, "")

def main():
    try:
        check_ReLu()
        check_ReLu_derivative()
    except Exception:
        log_exit(traceback.format_exc())

if __name__ == "__main__":
    main()