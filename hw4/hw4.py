#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 01:08:19 2018

@author: yashtrivedi
"""

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
b = 1
x1 = 1
x2 = 3
x3 = 2
x4 = 1
oTrue = 1

h1 = (1 * b) + (2 * x1) + (3 * x2) + (-2 * x3) + (1 * x4)
h1 = sigmoid(h1)

h2 = (2 * b) + (3 * x1) + (1 * x2) + (4 * x3) + (1 * x4)
h2 = sigmoid(h2)

h3 = (-1 * b) + (1 * x1) + (-2 * x2) + (0 * x3) + (3 * x4)
h3 = sigmoid(h3)

oPred = (1 * b) + (3 * h1) + (2 * h2) + (1 * h3)
oPred = sigmoid(oPred)

print("%.12f %.12f %.12f %.12f" % (h1, h2, h3, oPred))

delta = oTrue - oPred
deltah1 =  h1 * (1 - h1) * (delta * 3)
deltah2 = h2 * (1 - h2) * (delta * 2)
deltah3 = h3 * (1 - h3) * (delta * 1)

print("%.12f %.12f %.12f %.12f" % (delta, deltah1, deltah2, deltah3))

print("%.12f" % (deltah1 * 1))
print("%.12f" % (deltah2 * 1))
print("%.12f" % (deltah3 * 1))

print("%.12f" % (deltah1 * x1))
print("%.12f" % (deltah2 * x1))
print("%.12f" % (deltah3 * x1))

print("%.12f" % (deltah1 * x2))
print("%.12f" % (deltah2 * x2))
print("%.12f" % (deltah3 * x2))

print("%.12f" % (deltah1 * x3))
print("%.12f" % (deltah2 * x3))
print("%.12f" % (deltah3 * x3))

print("%.12f" % (deltah1 * x4))
print("%.12f" % (deltah2 * x4))
print("%.12f" % (deltah3 * x4))

print("%.12f" % (delta * 1))
print("%.12f" % (delta * h1))
print("%.12f" % (delta * h2))
print("%.12f" % (delta * h3))

