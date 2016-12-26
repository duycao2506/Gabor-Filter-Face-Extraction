import os
import numpy as np
import cv2
from FaceRecognizer import FaceRecognizer

class FaceClassManager(object):
    "Handling Face class, there may be some errors with Windows File System"
    def __init__ (self, pathToDataSets):
        self.pathClasses = []
        self.fvClasses = []
        self.fvClassNames = []
        self.pathSets = pathToDataSets
        folders = [f for f in os.listdir(self.pathSets) if os.path.isdir(os.path.join(self.pathSets,f))]
        folders = [os.path.join(self.pathSets,f) for f in folders]
        self.pathClasses = [[ os.path.join(dir,f) for f in os.listdir(dir) if f != ".DS_Store"] for dir in folders]
        # print self.faceClasses

    def trainToFV(self):
        recognizer = FaceRecognizer()
        for i in self.pathClasses:
            imgClass = [cv2.imread(imgPath,0) for imgPath in i]
            # for j in range(len(imgClass)):
            #     cv2.imshow(str(i) + str(j), imgClass[j])
            fvClass = [recognizer.extractFeatures(img) for img in imgClass]
            self.fvClasses.append(fvClass)
            self.fvClassNames.append(i[0])
       
        pass
