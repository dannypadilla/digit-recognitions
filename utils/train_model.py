'''
1. This program will train a model on training data passed in and store the model to disk

2. Stores the model to specified directory

3. Training data (images) are expected to be .jpg files

* Usage:
  train_model.py <training/images/path/> <model_name>

* Defaults:

    - Label CSV file:
        ../label.csv

    - Training Images:
        ../digits/

    - Model Path:
        ../models/

* Output:
   ../models/
'''

import cv2
import numpy as np
import pandas as pd
import sys
import glob

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


# constants
DIGITS_LABEL_MAP = pd.read_csv("../label.csv")
DIGIT_IMAGES_PATH = "../digits/"


# **************************** USEFUL METHODS ********************* #


def resize(frame, scale):
    return cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale) ))


def init_hog():
    min_dims = 8
    block_size = (2, 2)
    block_stride = (1, 1)
    cell_size = (2, 2)
    bins = 9
    dims = (8, 8)
    hog = cv2.HOGDescriptor(dims, block_size, block_stride, cell_size, bins)
    return hog

# reads a directory of images, flattens the features to a single row
# returns a list of images as a dataframe
def extract_features(file_path, features, hog=None):
    new_df = []
    for feature in range( len(features) ):
        img_path = file_path + str(feature) + ".jpg"
        img = cv2.imread(img_path)
        
        if (hog is not None):
            feat = hog.compute(img)
            img_flat = feat.flatten()
        else:
            img_flat = img.flatten()
            
        new_df.append(img_flat)
    return pd.DataFrame(new_df)


def model_tests(svm, X_train, X_test, y_train, y_test):
    y_predict = svm.predict(X_test)
    print("Number of MISCLASSIFIED samples: %d" % (y_test != y_predict).sum() )
    print("\nSVM predictions:\n", y_predict)
    
    accuracy = accuracy_score(y_test, y_predict)
    print("\nAccuracy: %.2f" % accuracy)
    print("\n")
    

# ****************************** /END METHODS/ ******************** #

# ***************************************************************** #

if __name__ == '__main__':

    print(__doc__)

    try:
        model_name = sys.argv[1]
    except:
        model_name = "digits_svm"
    
    # init HOG - feature extractor
    hog = init_hog()
    #hog = None

    digit_file_names = DIGITS_LABEL_MAP["name of the file"]

    X_df = extract_features(DIGIT_IMAGES_PATH, digit_file_names, hog=hog)
    y_df = DIGITS_LABEL_MAP["digit"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y_df,
        test_size=0.3,
        random_state=2
    )

    py_vers_label = "py2" # python 2 is default version
    # check python version
    if(sys.version_info >= (3, 0) ): # since joblib/pickle is picky with python versions
        py_vers_label = "py3"

    # init SVM
    svm = None
    model_path = "../models/" # where models are stored (should be)
    model_file_name = model_name + ".pkl" # append file extension to the model_file name
    full_model_path = model_path + py_vers_label + "_" + model_file_name # the entire path to the model appended (naming convention)
    
    # see if MODEL exists... if not TRAIN and STORE to disk
    print("\n MESSAGE(S):")
    try:
        svm = joblib.load(full_model_path)
        print("Model already exists!")
        print("\nLoading...\n")
    except:
        print("Training model...")
        svm = SVC(C=1.0, kernel="linear", probability=True, random_state=2)
        svm.fit(X_train, y_train)
        joblib.dump(svm, full_model_path) # store model object to disk
        print("\n\tStored model to location: " + "\"" + full_model_path + "\"\n")

    
    # print SVM TESTS
    model_tests(svm, X_train, X_test, y_train, y_test)

    
    # ******************************* /END OF CURRENT IMPLEMENTATION/ ************************ #

    
    '''
    # ****************************************************************** #
    # ### ### ### ### IN-WORK: DON'T USE YET ### ### ### ### ### ### ### #
    # * Next part will record from camera source and store to disk       #
    # * Need to determine whether we want to train on preprocessing data #
    # * --- Will need to change above model training to new source       #
    # * --- ALSO determine scale of frame/image source                   #
    # ****************************************************************** #

    # will be implemented as args
    # test for type of params passed... if bool.. etc..
    train_with_video_source = False
    camera_is_upside_down = False

    #video setup - NOT CURRENTLY IMPLEMENTED... YET
    # this will be the camera source - maybe using video would be too much at this point
    video_path = "videos/gate_new.avi" # not used right now
    video = cv2.VideoCapture(video_path) # not used right now
    
    ## for outputting video
    fps = 30.0
    file_name = "./run_jons_.avi"
    
    fourcc  = cv2.VideoWriter_fourcc(*"M", "J", "P", "G") # create write object for mac
    #out = cv2.VideoWriter(file_name, fourcc, fps, (744, 480) ) # has to be frame size of img
    out = cv2.VideoWriter(file_name, fourcc, fps, (640, 480) ) # has to be frame size of img

    

    while( (video.isOpened() ) and train_with_video_source):
        ret, frame = video.read()
        
        if(camera_is_upside_down): # whether camera should be rotated 180 deg
            rows, cols,_ = frame.shape
            rot_trans = cv2.getRotationMatrix2D( (cols/2, rows/2), 180, 1) # rotate image 180
            frame = cv2.warpAffine(frame, rot_trans, (cols, rows) ) # since camera is upside down..

        if(ret):
            #video_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # gray

            # loop here for processing each frame to SVM

            cv2.imshow("gate", frame) # actual frame
            cv2.moveWindow("gate", 0, 0)

            if(cv2.waitKey(1) & 0xFF == ord("q") ):
                break
        else:
            break

    #print(svm.classes_)
    out.release()
    video.release()
    cv2.destroyAllWindows()
    '''
