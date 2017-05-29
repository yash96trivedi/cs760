#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:05:40 2018

@author: yashtrivedi
"""
import matplotlib.pyplot as plt

X_naive = [0.041666666666666664, 0.08333333333333333, 0.125, 0.16666666666666666, 0.20833333333333334, 0.25, 0.25, 0.2916666666666667, 0.3333333333333333, 0.375, 0.4166666666666667, 0.4583333333333333, 0.5, 0.5, 0.5, 0.5416666666666666, 0.5833333333333334, 0.625, 0.6666666666666666, 0.7083333333333334, 0.75, 0.7916666666666666, 0.8333333333333334, 0.875, 0.875, 0.875, 0.9166666666666666, 0.9583333333333334, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Y_naive = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8571428571428571, 0.875, 0.8888888888888888, 0.9, 0.9090909090909091, 0.9166666666666666, 0.9230769230769231, 0.8571428571428571, 0.8, 0.8125, 0.8235294117647058, 0.8333333333333334, 0.8421052631578947, 0.85, 0.8571428571428571, 0.8636363636363636, 0.8695652173913043, 0.875, 0.84, 0.8076923076923077, 0.8148148148148148, 0.8214285714285714, 0.8275862068965517, 0.8, 0.7741935483870968, 0.75, 0.7272727272727273, 0.7058823529411765, 0.6857142857142857, 0.6666666666666666, 0.6486486486486487, 0.631578947368421, 0.6153846153846154, 0.6, 0.5853658536585366, 0.5714285714285714]

X_tan = [0.041666666666666664, 0.08333333333333333, 0.125, 0.16666666666666666, 0.20833333333333334, 0.25, 0.2916666666666667, 0.3333333333333333, 0.375, 0.375, 0.4166666666666667, 0.4583333333333333, 0.5, 0.5, 0.5416666666666666, 0.5833333333333334, 0.5833333333333334, 0.625, 0.6666666666666666, 0.6666666666666666, 0.7083333333333334, 0.75, 0.7916666666666666, 0.8333333333333334, 0.875, 0.9166666666666666, 0.9166666666666666, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 0.9583333333333334, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
Y_tan = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9090909090909091, 0.9166666666666666, 0.9230769230769231, 0.8571428571428571, 0.8666666666666667, 0.875, 0.8235294117647058, 0.8333333333333334, 0.8421052631578947, 0.8, 0.8095238095238095, 0.8181818181818182, 0.8260869565217391, 0.8333333333333334, 0.84, 0.8461538461538461, 0.8148148148148148, 0.8214285714285714, 0.7931034482758621, 0.7666666666666667, 0.7419354838709677, 0.75, 0.7272727272727273, 0.7058823529411765, 0.6857142857142857, 0.6666666666666666, 0.6486486486486487, 0.631578947368421, 0.6153846153846154, 0.6, 0.5853658536585366, 0.5714285714285714]

f = plt.figure()

plt.plot(X_naive, Y_naive, 'r-')
plt.plot(X_tan, Y_tan, 'b--')
plt.xlim(0.0, 1.1)
plt.ylim(0.0, 1.1)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Naive Bayes vs. TAN')

plt.grid()
plt.show()

f.savefig("/Users/yashtrivedi/hw3plot2c.pdf", bbox_inches='tight')