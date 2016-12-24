
from skimage.exposure import rescale_intensity
import cv2
import numpy as np
import scipy
import argparse
import sys
from enum import Enum

def build_filters(w, h,num_theta, fi, sigma_x, sigma_y, psi):
    filters = []
    for i in range(num_theta):
        theta = ((i+1)*1.0 / num_theta) * np.pi
        print i
        for f_var in fi:
            kernel = get_gabor_kernel(w, h,sigma_x, sigma_y, theta, f_var, psi)
            kernel = 2.0*kernel/kernel.sum()
            kernel = cv2.normalize(kernel, kernel, 1, 0, cv2.NORM_L1 )
            filters.append(kernel)
    return filters

def get_gabor_kernel(w, h,sigma_x, sigma_y, theta, fi, psi):
    # Bounding box
    kernel_size_x = w
    kernel_size_y = h
    (y, x) = np.meshgrid(np.arange(0, kernel_size_y ), np.arange(0,kernel_size_x))
    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    #Calculate the gabor kernel according the formulae
    gb = np.exp(-1.0*(x_theta ** 2.0 / sigma_x ** 2.0 + y_theta ** 2.0 / sigma_y ** 2.0)) * np.cos(2 * np.pi * fi * x_theta + psi)
    print gb
    return gb

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    arrays = []
    for kern,params in filters: 
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        arrays.append(fimg)
        # np.maximum(accum, fimg, accum)
    return arrays



def normalize(beta, alpha, img):
        mins = np.ones((img.shape[0], img.shape[1])) * img.min()
        maxs = np.ones((img.shape[0], img.shape[1])) * img.max()
        return (img - mins)*(beta - alpha)/(maxs - mins) + alpha



def main(argv):
    path = argv[0]
    img = cv2.imread(path,0)

    filters = build_filters(img.shape[0],img.shape[1],5,(0.75,1.5),2,1,np.pi/2.0)
    
    fft_filters = [np.fft.fft2(i) for i in filters]
    img_fft = np.fft.fft2(img)
    a = fft_filters * img_fft
    s = [np.fft.ifft2(i) for i in a]
    k = [p.real for p in s]
    # k = process(img, build_filters2())
    print k[0].shape

    cv2.imshow('1',k[0])
    cv2.imshow('2',k[1])
    cv2.imshow('3',k[2])
    cv2.imshow('4',k[3])
    cv2.imshow('5',k[4])
    cv2.imshow('6',k[5])
    cv2.imshow('7',k[6])
    cv2.imshow('8',k[7])
    cv2.imshow('9',k[8])
    cv2.imshow('10',k[9])
    cv2.imshow('orig',img)
    cv2.waitKey()
    

if __name__ == "__main__":
    main(sys.argv[1:])

