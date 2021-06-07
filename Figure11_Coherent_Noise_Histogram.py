#!/usr/bin/env python
# coding: utf-8

import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def main():
    # Import datasets
    dat00 = np.load('./Data/stats_c00.npy')
    dat10 = np.load('./Data/stats_c10.npy')
    dat20 = np.load('./Data/stats_c20.npy')
    dat30 = np.load('./Data/stats_c30.npy')
    dat40 = np.load('./Data/stats_c40.npy')
    dat50 = np.load('./Data/stats_c50.npy')

    # Extract only the mean-squared errors
    mean00, mean10, mean20, mean30, mean40, mean50 = dat00[:,0], dat10[:,0], dat20[:,0], dat30[:,0], dat40[:,0], dat50[:,0]

    # Histogram plot for COHERENT
    fig, ax = plt.subplots(figsize=(8.5,6))

    width = 0.15

    # Set position on X-axis
    p1 = np.arange(len(mean00))
    p2 = [i + width for i in p1]
    p3 = [i + width for i in p2]
    p4 = [i + width for i in p3]
    p5 = [i + width for i in p4]
    p6 = [i + width for i in p5]

    # Make the plot
    plt.bar(p1, mean00, color='royalblue', width=width,edgecolor='black', label='0')
    plt.bar(p2, mean10, color='orange', width=width,edgecolor='black', label='10%')
    plt.bar(p3, mean20, color='magenta', width=width,edgecolor='black', label='20%')
    plt.bar(p4, mean30, color='green', width=width,edgecolor='black', label='30')
    plt.bar(p5, mean40, color='cyan', width=width,edgecolor='black', label='40%')
    plt.bar(p6, mean50, color='red', width=width,edgecolor='black', label='50%')

    # Add xticks on the middle of the group bars
    plt.xlabel('parameter', fontname='serif', fontweight='bold')
    plt.ylabel('MAE (%)',fontname='serif', fontweight='bold')
    plt.xticks([j + width+0.22 for j in range(len(mean00))], [r'$x$', r'$y$', r'$z$',r'$v_{p0}$', r'$v_{s0}$',r'$\rho$'])
    plt.title('Variation of anisotropic parameters: Orthorhombic media', fontsize=10,fontname='serif', fontweight='bold')
    plt.legend(loc='best')
    fig.tight_layout()
    plt.show()
    
# This will actually run the code if called stand-alone:
if __name__ == '__main__':
    main()
