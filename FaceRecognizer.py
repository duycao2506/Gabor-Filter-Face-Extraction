import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import pprint
import os
import FaceClassManager
from enum import Enum
import math as mth
from scipy import signal

class FaceRecognizer(object):
    "Class for recognizing faces"
    "There will be a training datasets first"
    "Method: Gabor filter"
    def __init__(self):
        pass


    def build_filters(self, w, h,num_theta, fi, sigma_x, sigma_y, psi):
        "Get set of filters for GABOR"
        filters = []
        for i in range(num_theta):
            theta = ((i+1)*1.0 / num_theta) * np.pi
            for f_var in fi:
                kernel = self.get_gabor_kernel(w, h,sigma_x, sigma_y, theta, f_var, psi)
                kernel = 2.0*kernel/kernel.sum()
                # kernel = cv2.normalize(kernel, kernel, 1.0, 0, cv2.NORM_L2)
                filters.append(kernel)
        return filters

    def get_gabor_kernel(self, w, h,sigma_x, sigma_y, theta, fi, psi):
        "getting gabor kernel with those values"
        # Bounding box
        kernel_size_x = w
        kernel_size_y = h
        (y, x) = np.meshgrid(np.arange(0, kernel_size_y ), np.arange(0,kernel_size_x))
        # Rotation 
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        #Calculate the gabor kernel according the formulae
        gb = np.exp(-1.0*(x_theta ** 2.0 / sigma_x ** 2.0 + y_theta ** 2.0 / sigma_y ** 2.0)) * np.cos(2 * np.pi * fi * x_theta + psi)
        return gb

    def distanceOfFV(self, fv1, fv2):
        "distance of feature vector 1 and feature vector 2"
        normset = []
        for i in range(len(fv1)):
            k = fv1[i]
            p = fv2[i]
            # k = cv2.normalize(fv1[i],k,1.0,0,norm_type=cv2.NORM_L2)
            # p = cv2.normalize(fv2[i],p,1.0,0,norm_type=cv2.NORM_L2)
            normset.append((p-k)**2.0)
        sums = 0
        sums = sum([i.sum() for i in normset])
        return mth.sqrt(sums)/100000

    def avgDist(self, imgFVClass, imgFV):
        "classify the imgFV in the classes"
        distes = [self.distanceOfFV(iFv,imgFV) for iFv in imgFVClass]
        print len(distes)
        return (sum(distes) / len(distes))

    def classify(self, imgFV, imgFVClasses):
        avgdistes = [self.avgDist(imgFVClass, imgFV) for imgFVClass in imgFVClasses]
        print avgdistes
        return avgdistes.index(min(avgdistes))

    def extractFeatures(self, img):
        "A vector of 2n elements where n is the number of theta angles"
        "and 2 is the number of frequencies under consideration"
        filters =  self.build_filters(img.shape[0],img.shape[1],5,(0.75,1.5),2,1,np.pi/2.0)
        fft_filters = [np.fft.fft2(i) for i in filters]
        img_fft = np.fft.fft2(img)
        a =  img_fft * fft_filters
        s = [np.fft.ifft2(i) for i in a]
        k = [p.real for p in s]
        return k
        

    def fft_convolve2d(self, x,y):
        """ 2D convolution, using FFT"""
        fr = fft2(x)
        fr2 = fft2(np.flipud(np.fliplr(y)))
        m,n = fr.shape
        cc = np.real(ifft2(fr*fr2))
        cc = np.roll(cc, -m/2+1,axis=0)
        cc = np.roll(cc, -n/2+1,axis=1)
        return cc
        
    def  recognize(self, imgs, faceClassManager):
        feats = []
        dem = 0 
        for i in imgs :
            feat = self.extractFeatures(i)
            dem += 1
            cv2.imshow( str(dem), i)
            feats.append(feat)
        cv2.waitKey()
        print [self.classify(i,faceClassManager.fvClasses) for i in feats]
        pass