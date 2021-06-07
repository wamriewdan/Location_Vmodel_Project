#!/usr/bin/env python
# coding: utf-8

# Import libraries
import tensorflow as tf
tf.__version__

import os
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.utils import plot_model

def main():
    # Load test dataset
    X = np.load('./Data/Xtest.npy')
    y = np.load('./Data/ytest.npy')

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    scalerx = MinMaxScaler()
    scalery = MinMaxScaler()
    X_ = []
    for i in range(X.shape[0]):
        xmid = []
        for j in range(X.shape[-1]):
            x = X[i,:,:,j]
            scalerx.fit(x)
            x_ = scalerx.transform(x)
            xmid.append(x_)
        X_.append(xmid)
    X_ = np.array(X_)   
    X_norm = np.rollaxis(X_, 3,1)
    X_test = np.rollaxis(X_norm, 3,1)

    scalery.fit(y)
    y_test = scalery.transform(y)

    # print(X_test.shape, y_test.shape)

    # Load saved model
    model = load_model("./Best_Models/BestAGJ06_02b.h5")
    scores = model.evaluate(X_test, y_test, verbose=2)

    # Perform predictions
    pred  = model.predict(X_test)
    preds = scalery.inverse_transform(pred)
    y_test= scalery.inverse_transform(y_test)

    # Correlation plots
    # Plot configuration
    plt.ion()
    fig1 = plt.figure(figsize=(9,6))
    gs1  = gridspec.GridSpec(2,3, width_ratios=[1,1,1], hspace=0.4, wspace=0.4)

    # Plot X Coordinates
    ax1  = plt.subplot(gs1[0])
    ax1.scatter(y_test[:,0], preds[:,0],facecolors='none', edgecolors='b', s=10)
    ax1.set_title('X-coordinate', fontname='serif', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Truth', fontname='serif')
    ax1.set_ylabel('Predictions', fontname='serif')

    # Plot Y Coordinates
    ax2  = plt.subplot(gs1[1])
    ax2.scatter(y_test[:,1], preds[:,1],facecolors='none', edgecolors='b', s=10)
    ax2.set_title('Y-coordinate', fontname='serif', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Truth', fontname='serif')
    ax2.set_ylabel('Predictions', fontname='serif')

    # Plot Z Coordinates
    ax3  = plt.subplot(gs1[2])
    ax3.scatter(y_test[:,2], preds[:,2],facecolors='none', edgecolors='b', s=10)
    ax3.set_title('Z-coordinate', fontname='serif', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Truth', fontname='serif')
    ax3.set_ylabel('Predictions', fontname='serif')

    # Plot X Coordinates
    ax4  = plt.subplot(gs1[3])
    ax4.scatter(y_test[:,3], preds[:,3],facecolors='none', edgecolors='b', s=10)
    ax4.set_title('Vp', fontname='serif', fontsize=10, fontweight='bold')
    ax4.set_xlabel('Truth', fontname='serif')
    ax4.set_ylabel('Predictions', fontname='serif')

    # Plot Y Coordinates
    ax5  = plt.subplot(gs1[4])
    ax5.scatter(y_test[:,4], preds[:,4],facecolors='none', edgecolors='b', s=10)
    ax5.set_title('Vs', fontname='serif', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Truth', fontname='serif')
    ax5.set_ylabel('Predictions', fontname='serif')

    # Plot Z Coordinates
    ax6  = plt.subplot(gs1[5])
    ax6.scatter(y_test[:,5], preds[:,5],facecolors='none', edgecolors='b', s=10)
    ax6.set_title('Density', fontname='serif', fontsize=10, fontweight='bold')
    ax6.set_xlabel('Truth', fontname='serif')
    ax6.set_ylabel('Predictions', fontname='serif')
    # plt.savefig('./Figures/XYZPSR_AGJ04_01c.png')
    plt.show()



    # Plot FIGURE 7
    #=========================================================================================================================
    # Plot configuration
    plt.ion()
    fig1 = plt.figure(figsize=(9,2.7))
    gs1  = gridspec.GridSpec(1,3,width_ratios=[1,1,1], hspace=0.2, wspace=0.2)

    # Plot X vs Y Coordinates
    ax1  = plt.subplot(gs1[0])
    ax1.scatter(y_test[:200,0], y_test[:200,1],facecolors='b', edgecolors='b', s=50, label='True')
    ax1.scatter(preds[:200,0], preds[:200,1],facecolors='r', edgecolors='r', s=50, label='Inverted')
    ax1.set_title('Plan view: X vs Y', fontweight='bold', fontname='serif')
    ax1.set_xlabel('x (m)', fontname='serif')
    ax1.set_ylabel('y (m)', fontname='serif')
    plt.legend(loc=1)

    # Plot X vs Z Coordinates
    ax2  = plt.subplot(gs1[1])
    ax2.scatter(y_test[:200,0], y_test[:200,2],facecolors='b', edgecolors='b', s=50, label='True')
    ax2.scatter(preds[:200,0], preds[:200,2],facecolors='r', edgecolors='r', s=50, label='Inverted')
    ax2.set_title('Plan view: X vs Z', fontweight='bold', fontname='serif')
    ax2.set_xlabel('x (m)', fontname='serif')
    ax2.set_ylabel('depth, z (m)', fontname='serif')
    plt.legend(loc=1)

    # Plot Y vs Z Coordinates
    ax3  = plt.subplot(gs1[2])
    ax3.scatter(y_test[:200,1], y_test[:200,2],facecolors='b', edgecolors='b', s=50, label='True')
    ax3.scatter(preds[:200,1], preds[:200,2],facecolors='r', edgecolors='r', s=50, label='Inverted')
    ax3.set_title('Plan view: Y vs Z', fontweight='bold', fontname='serif')
    ax3.set_xlabel('y (m)', fontname='serif')
    ax3.set_ylabel('depth, z (m)', fontname='serif')
    plt.legend(loc=1)
    # plt.savefig('./Figures/Plan_view.png', dpi=500)
    plt.show()

    # Plot FIGURE 8
    #=========================================================================================================================
    # Prepare data for velocity model plots
    vmod = y[:,2:]
    vmod_pred = preds[:,2:]

    vmodels = vmod.reshape((25,200,4))
    vmodels_pred = vmod_pred.reshape((25,200,4))

    # Plot velocity models: Ground-truth vs predictions - FIGURE 8
    for i in range(len(vmodels)):
        vmodel = vmodels[i]
        vmodelsort = vmodel[vmodel[:,0].argsort()]

        vmodel_pred = vmodels_pred[i]
        vmodel_predsort = vmodel_pred[vmodel_pred[:,0].argsort()]

        plt.figure(figsize=(5,4))
        plt.plot(vmodelsort[:,1], vmodelsort[:,0], 'b', linewidth=5, label = 'True $v_p$')
        plt.plot(vmodel_predsort[:,1], vmodel_predsort[:,0], color='brown', linewidth=4, label = 'Inverted $v_p$')
        plt.plot(vmodelsort[:,2], vmodelsort[:,0], 'g', linewidth=5, label = 'True $v_s$')
        plt.plot(vmodel_predsort[:,2], vmodel_predsort[:,0], color='orange',linewidth=4, label = 'Inverted $v_s$')
        plt.plot(vmodelsort[:,3], vmodelsort[:,0], 'k', linewidth=5, label = r'True $\rho$')
        plt.plot(vmodel_predsort[:,3], vmodel_predsort[:,0],'c',linewidth=4, label = r'Inverted $\rho$')
        plt.ylabel('Depth (m)', fontsize=9, fontname='serif')
        plt.xlabel('Velocity ($ms^{-1}$)\t [Density ($kgm^{-3}$)]', fontsize=9, fontname='serif')
        plt.title('True vs Inverted velocity model: Model %d'%(i+1), fontsize=11, fontname='serif', fontweight='bold')
        plt.gca().invert_yaxis()
        plt.legend(loc=10)
    #     plt.savefig('./Figures/vmodels/vmodel_Plot %d.png'%(i+1), dpi=500)
        plt.show()

    # Plot FIGURE 9
    #=========================================================================================================================
    # Calculate mean-squared errors
    errors= (preds - y_test)
    msex, msey, msez = np.mean(np.absolute(errors[:,0]))*100/(np.mean(y_test[:,0])), np.mean(np.absolute(errors[:,1]))*100/(np.mean(y_test[:,1])), np.mean(np.absolute(errors[:,2]))*100/(np.mean(y_test[:,2]))
    msep, mses, mser = np.mean(np.absolute(errors[:,3]))*100/(np.mean(y_test[:,3])), np.mean(np.absolute(errors[:,4]))*100/(np.mean(y_test[:,4])), np.mean(np.absolute(errors[:,5]))*100/(np.mean(y_test[:,5]))
    # print(msex, msey, msez, msep, mses, mser)

    # Calculate standard deviations
    stdx, stdy, stdz = np.std(errors[:,0]), np.std(errors[:,1]), np.std(errors[:,2])
    stdp, stds, stdr = np.std(errors[:,3]), np.std(errors[:,4]), np.std(errors[:,5])
    # print(stdx, stdy, stdz, stdp, stds, stdr)

    # Display the maximum errors
    rx, ry, rz = errors[:,0].max(), errors[:,1].max(), errors[:,2].max()
    rp, rs, rr = errors[:,3].max(), errors[:,4].max(), errors[:,5].max()
    print(rx, ry, rz, rp, rs, rr)

    # Plot Histogram in Figure 9
    fig, ax = plt.subplots(figsize=(7,5))

    mean = [msex, msey, msez, msep, mses, mser]
    std  = [stdx, stdy, stdz, stdp, stds, stdr]
    ermax= [rx, ry, rz, rp, rs, rr]

    width = 0.40

    # Set position on X-axis
    p1 = np.arange(len(mean))
    p2 = [i + width for i in p1]
    p3 = p1+0.2

    # Make the plot
    plt.bar(p1, mean, color='b', width=width,edgecolor='black', label='mean (%)')
    plt.bar(p2, std, color='darkorange', width=width,edgecolor='black', label='std')
    plt.bar(p3, ermax, color='none', width=width*2.01,edgecolor='black', label='max_abs')

    # Add xticks on the middle of the group bars
    plt.xlabel('parameter', fontname='serif', fontweight='bold')
    plt.ylabel('error (log scale)',fontname='serif', fontweight='bold')
    plt.xticks([j + width-0.19 for j in range(len(mean))], [r'$x$', r'$y$', r'$z$',r'$v_p$', r'$v_s$',r'$\rho$'])
    plt.yscale('log')
    plt.title('Error plots', fontsize=10,fontname='serif', fontweight='bold')


    plt.text(-0.17, round(msex,2)+0.05,round(msex,2))
    plt.text(0.2,   round(stdx,2)+0.15,round(stdx,2))
    plt.text(0.01,  round(rx,1)+0.5,round(rx,1))

    plt.text(0.85, round(msey,2)+0.07,round(msey,2))
    plt.text(1.23, round(stdy,2)+0.30,round(stdy,2))
    plt.text(1.05, round(ry,1)+2.0,round(ry,1))

    plt.text(1.83, round(msez,2)+0.01,round(msez,2))
    plt.text(2.21, round(stdz,2)+0.1,round(stdz,2))
    plt.text(2.08, round(rz,1)+1.0,round(rz,1))

    plt.text(2.84, round(msep,2)+0.01,round(msep,2))
    plt.text(3.22, round(stdp,2)+2.0,round(stdp,1))
    plt.text(3.00, round(rp,1)+20,round(rp,1))

    plt.text(3.83, round(mses,2)+0.04,round(mses,2))
    plt.text(4.20, round(stds,1)+2.0,round(stds,1))
    plt.text(4.00, round(rs,1)+20,round(rs,1))

    plt.text(4.82, round(mser,2)+0.01,round(mser,2))
    plt.text(5.20, round(stdr,2)+0.5, round(stdr,2))
    plt.text(5.05, round(rr,1)+7,round(rr,1))

    plt.legend(loc='upper left')
    # plt.savefig('./Figures/Error_plots.png',bbox_inches = "tight", dpi=600)
    plt.show()    


    # Plot FIGURE 10
    #=========================================================================================================================
    # Load data to plot figure 10
    data10, data20 = np.load('./Data/Scores_Noise_10pc.npy'),  np.load('./Data/Scores_Noise_20pc.npy')
    data30, data40 = np.load('./Data/Scores_Noise_30pc.npy'), np.load('./Data/Scores_Noise_40pc.npy')
    data50 = np.load('./Data/Scores_Noise_50pc.npy')

    data = [data10, data20, data30, data40, data50]
    labels  = ['10%', '20%', '30%', '40%', '50%']

    fig = plt.figure(figsize =(8, 6))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, labels=labels, patch_artist = True, vert = 0)

    colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF00FF','#F97306']

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # changing color and linewidth of whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='#8B008B', linewidth = 1.5, linestyle =":")

    # changing color and linewidth of caps
    for cap in bp['caps']:
        cap.set(color ='#8B008B', linewidth = 2)

    # changing color and linewidth of medians
    for median in bp['medians']:
        median.set(color ='red', linewidth = 3)

    # changing style of fliers
    for flier in bp['fliers']:
        flier.set(marker ='D', color ='#e7298a', alpha = 0.5)

    # x-axis labels
    ax.set_yticklabels(labels)

    # Adding title
    plt.title("Random noise robustness test", fontsize=11, fontname='serif', fontweight='bold')
    plt.xlabel('MSE', fontsize=9, fontname='serif')
    plt.ylabel('Noise level', fontsize=9, fontname='serif')

    # Removing top axes and right axes ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    # plt.savefig('./Figures/boxplot.png', dpi=600)
    plt.show(bp)
    
# This will actually run the code if called stand-alone:
if __name__ == '__main__':
    main()