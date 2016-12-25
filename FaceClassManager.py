import os
import numpy as np
import cv2

class FaceClassManager(object):
    "Handling Face class, there may be some errors with Windows File System"
    def __init__ (self, pathToDataSets):
        self.faceClasses = []
        self.pathSets = pathToDataSets
        folders = [f for f in os.listdir(self.pathSets) if os.path.isdir(os.path.join(self.pathSets,f))]
        folders = [os.path.join(self.pathSets,f) for f in folders]
        self.faceClasses = [[ os.path.join(dir,f) for f in os.listdir(dir)] for dir in folders]
        # print self.faceClasses