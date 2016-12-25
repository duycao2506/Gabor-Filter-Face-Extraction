import cv2
import numpy as np
import pprint
import os
import FaceClassManager
from enum import Enum

class FaceRecognizer(object):
    "Class for recognizing faces"
    "There will be a training datasets first"
    "Method: Gabor filter"
    def __init__(self, faceClassManager):
        self.classManagers = faceClassManager

    def build_filters(w, h,num_theta, fi, sigma_x, sigma_y, psi):
        "Get set of filters for GABOR"
        filters = []
        for i in range(num_theta):
            theta = ((i+1)*1.0 / num_theta) * np.pi
            for f_var in fi:
                kernel = self.get_gabor_kernel(w, h,sigma_x, sigma_y, theta, f_var, psi)
                kernel = 1.5*kernel/kernel.sum()
                kernel = cv2.normalize(kernel, kernel, 1.0, 0, cv2.NORM_L1 )
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
            k = None
            p = None
            k = cv2.normalize(fv1[i],k,1.0,0,norm_type=cv2.NORM_L2)
            p = cv2.normalize(fv2[i],p,1.0,0,norm_type=cv2.NORM_L2)
            normset.append((p-k)**2.0)
        sums = 0
        sums = sum([i.sum() for i in normset])
        return sums/10

    def classify(self, imgFVClass, imgFV):
        "classify the imgFV in the classes"
        distes = [distanceOfFV(iFv,imgFV) for iFv in imgFVClass]
        return distes.index(min(distes))

    def extractFeatures(self, img):
        "A vector of 2n elements where n is the number of theta angles"
        "and 2 is the number of frequencies under consideration"
        filters =   build_filters(img.shape[0],img.shape[1],5,(0.75,1.5),2,1,np.pi/2.0)
        fft_filters = [np.fft.fft2(i) for i in filters]
        img_fft = np.fft.fft2(img)
        a = fft_filters * img_fft
        s = [np.fft.ifft2(i) for i in a]
        k = [p.real for p in s]
        return k 
        

        
    def  recognize(imgs):
        pass