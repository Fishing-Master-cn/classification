# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:37:49 2022

@author: W'chang
"""

import numpy as np
import math

section = np.array([0, 1])


def derivation(f, x0):
    h = 1e-5
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def f1(x):
    return 4 / (1 + x ** 2)


def f2(x):
    if x == 0:
        return 1
    return math.sin(x) / x


def f3(x):
    return x * math.exp(-x)


def f4(x):
    return x - math.exp(-x)


def fi(x):
    return math.exp(-x)


def Compound_trapezoidal_formula(section, f, n):
    h = (section[1] - section[0]) / n
    x = np.linspace(section[0], section[1], n + 1)
    T = f(x[0]) + f(x[-1])
    for i in range(1, x.shape[0] - 1):
        T += 2 * f(x[i])
    return T * h / 2


def Compound_Simpson_formula(section, f, n):
    h = (section[1] - section[0]) / n
    x = np.linspace(section[0], section[1], n + 1)
    S = f(x[0]) + f(x[-1])
    for i in range(1, x.shape[0] - 1):
        S += 2 * f(x[i])
    for i in range(x.shape[0] - 1):
        S += 4 * f(x[i] + h / 2)
    return S * h / 6


def dichotomization(section, f, accuracy):
    a = section[0]
    b = section[1]
    while (b - a) > accuracy:
        x0 = (a + b) / 2
        if math.fabs(f(x0)) < 1e-5:
            break
        elif f(x0) * f(a) < 0:
            b = x0
        else:
            a = x0

    return x0


def fixed_point_iteration_method(section, f, accuracy):
    x0 = 0.5
    while math.fabs(f(x0) - x0) > accuracy:
        x0 = f(x0)
    return x0


def newton_method(section, f, accuracy):
    x0 = 0.5
    new_x0 = x0 - f(x0) / derivation(f, x0)
    while math.fabs(new_x0 - x0) > accuracy:
        x0 = new_x0
        new_x0 = x0 - f(x0) / derivation(f, x0)
    return new_x0

# T = Compound_trapezoidal_formula(section, f, 20)
# print(T)

# S = Compound_Simpson_formula(section, f2, 4)
# print(S)

# x0 = dichotomization([1.0,1.5], f4, accuracy=1e-5)
# print(x0)

# x0 = fixed_point_iteration_method(section, fi, accuracy=1e-5)
# print(x0)

# x0 = newton_method(section, f4, accuracy=1e-5)
# print(x0)
