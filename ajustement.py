#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:19:15 2018

@author: 3874345
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('fit_sinus.dat')
data_x = data[:,0]; data_y = data[:,1]

x = np.linspace(-200,200,500)
def frange(x,amplitude,decalage,position,largeur):
    return amplitude*np.cos(2*np.pi*(x+position)/largeur)+decalage


plot(x,frange(x,0.175,0.2,0,200))

plt.plot(data_x,data_y,'o')

p_opt, cor_mat = curve_fit(frange, data_x, data_y,[0.175,0.2,0,200])


plt.plot(x,frange(x,p_opt[0], p_opt[1], p_opt[2], p_opt[3]))
plt.plot(data_x,data_y,'o')

position_centrale = p_opt[2]
incertitude_position_centrale = np.sqrt(cor_mat[2,2])

## CORRELATION ENTRE PARAMETRES

np.random.seed(0)
N = 100
x = np.linspace(2000,2018, N)
y = np.arange(N)*0.2+45+np.random.normal(size=N)

def affine(x,a,b):
    return a*x+b

p_opt,cor_mat = curve_fit(affine, x, y,[1,0])

plt.plot(x,y)
plt.plot(x,p_opt[0]*x+p_opt[1])
plt.xlim(0, None)
plt.ylim(-3000, None)
incertitude_b = np.sqrt(cor_mat[1,1])

valeur_2010 = p_opt[0]*2010+p_opt[1]

incertitude_2010 = np.sqrt(2010**2*cor_mat[0,0]+cor_mat[1,1]+2*cor_mat[0,1])

# Nouvelle fonction de fit

def affine2(x,a,b):
    return a*(x-np.mean(x))+b

p_opt,cor_mat = curve_fit(affine2, x, y, [1,0])

plt.plot(x-np.mean(x),y)
plt.plot(x-np.mean(x),p_opt[0]*(x-np.mean(x))+p_opt[1])

incertitude_b = np.sqrt(cor_mat[1,1])

valeur_2010 = p_opt[0]*2010+p_opt[1]

incertitude_2010 = np.sqrt((2010-np.mean(x))**2*cor_mat[0,0]+cor_mat[1,1]+2*cor_mat[0,1])
 

# FIT D'UNE IMAGE

image = np.loadtxt('double_star.txt')
ny, nx = image.shape
X,Y = np.meshgrid(range(nx),range(ny))
xdata = np.array([X.flatten(),Y.flatten()]).transpose()

def gauss(xdata, amplitude, center_x, center_y, diameter):
    x = xdata[:,0]
    y = xdata[:,1]
    return amplitude*np.exp(-((x-center_x)**2 + (y-center_y)**2)/diameter**2)

p0 = [1,0,0,1]
popt,pcov = curve_fit(gauss,xdata,image.flatten(),p0)
gaussienne1 = gauss(xdata,popt[0],popt[1],popt[2],popt[3])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(xdata[:,0],xdata[:,1],gaussienne1)
