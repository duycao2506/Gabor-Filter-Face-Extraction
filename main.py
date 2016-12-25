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
    if (len(argv) == 0):
        faceClassManager = FaceClassManager("./DataSets/")
    else:
        path = argv[0]    
        faceClassManager = FaceClassManager(path)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(faceClassManager.faceClasses)
    
    recognizer = FaceRecognizer(faceClassManager)
    

if __name__ == "__main__":
    main(sys.argv[1:])

