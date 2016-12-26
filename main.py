import cv2
import os
import pprint
import sys
import numpy as np
from enum import Enum
from FaceClassManager import FaceClassManager
from FaceRecognizer import FaceRecognizer

def main(argv):
    faceClassManager = None
    dir = "/Users/admin/Desktop/Computer Vision/Projects/Final Project HBD/Final Project Github/Input/"
    inputImagePaths = [os.path.join(dir,f) for f in os.listdir(dir)]
    inputImages = [cv2.imread(i,0) for i in inputImagePaths]
    if (len(argv) == 0):
        faceClassManager = FaceClassManager("./DataSets/")
        # TODO: inputimages default
        # inputImages.append(cv2.imread(ex,0))
    elif len(argv) == 1:
        #TODO: datasets default
        #TODO: input images from commands
        pass
    elif len(argv) == 2:
        #TODO: input images and datasets from commands
        pass
    else:
        #TODO Errors
        pass
    
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(faceClassManager.pathClasses)
    
    

    faceClassManager.trainToFV()


    recognizer = FaceRecognizer()

    recognizer.recognize(inputImages, faceClassManager)    
    pp.pprint(faceClassManager.fvClassNames)

    cv2.imshow("Hello", cv2.imread(faceClassManager.fvClassNames[2],0))
    
    cv2.waitKey()

if __name__ == "__main__":
    main(sys.argv[1:])

